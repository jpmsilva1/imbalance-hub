import http.server
import threading
import warnings

import pandas as pd

from imbalance_hub.catalog import _is_cacheable, _ref_for, load_catalog


def test_ref_for_latest_resolves_to_main_branch():
    # No git tags exist on the repo, so "latest" (the default version) has no
    # matching ref of its own -- it must resolve to main, not be used
    # literally as a ref, or every un-cached fetch 404s.
    assert _ref_for("latest") == "main"


def test_ref_for_pinned_version_passes_through_unchanged():
    assert _ref_for("v1.0.0") == "v1.0.0"


def test_is_cacheable_false_for_latest():
    # "latest" caching forever was a real bug: a user's first load_catalog()
    # call would silently pin them to whatever was published that moment,
    # with no way back to freshness short of refresh=True.
    assert _is_cacheable("latest") is False


def test_is_cacheable_true_for_pinned_version():
    assert _is_cacheable("v1.0.0") is True


def test_load_catalog_fetches_from_source_and_caches(tmp_path):
    src = tmp_path / "src" / "series.csv"
    src.parent.mkdir()
    pd.DataFrame({"id": ["gluonts:m4_hourly:h1"], "imbalance_level": ["mild"]}).to_csv(src, index=False)
    cache_dir = tmp_path / "cache"

    df = load_catalog(source=str(src), cache_dir=cache_dir, version="v1")

    assert df["id"].tolist() == ["gluonts:m4_hourly:h1"]
    assert (cache_dir / "v1" / "series.csv").exists()


def test_load_catalog_reads_from_cache_without_a_source(tmp_path):
    cache_dir = tmp_path / "cache"
    (cache_dir / "v1").mkdir(parents=True)
    pd.DataFrame({"id": ["tslib:traffic:0"]}).to_csv(cache_dir / "v1" / "series.csv", index=False)

    # No `source` given -- must come from the cache, not attempt a real fetch.
    df = load_catalog(cache_dir=cache_dir, version="v1")

    assert df["id"].tolist() == ["tslib:traffic:0"]


def test_load_catalog_refresh_re_fetches_even_when_cached(tmp_path):
    src = tmp_path / "src" / "series.csv"
    src.parent.mkdir()
    pd.DataFrame({"id": ["new:row:1"]}).to_csv(src, index=False)
    cache_dir = tmp_path / "cache"
    (cache_dir / "v1").mkdir(parents=True)
    pd.DataFrame({"id": ["stale:row:1"]}).to_csv(cache_dir / "v1" / "series.csv", index=False)

    df = load_catalog(source=str(src), cache_dir=cache_dir, version="v1", refresh=True)

    assert df["id"].tolist() == ["new:row:1"]


def test_load_catalog_emits_no_dtype_warning(tmp_path):
    # A column with mixed int/str values across chunks used to trigger
    # pandas' DtypeWarning on every load_catalog() call.
    src = tmp_path / "series.csv"
    rows = [{"id": f"gluonts:m4_hourly:h{i}", "name": i if i % 2 else str(i)} for i in range(3000)]
    pd.DataFrame(rows).to_csv(src, index=False)

    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        load_catalog(source=str(src), cache_dir=tmp_path / "cache", version="v1")

    assert not any("mixed types" in str(w.message) for w in caught)


class _ETagCSVHandler(http.server.BaseHTTPRequestHandler):
    """Serves a fixed CSV body and honors If-None-Match, so load_catalog's
    conditional-GET path can be exercised against a real HTTP round trip."""
    body = b"id\ngluonts:m4_hourly:h1\n"
    etag = "etag-v1"
    request_count = 0

    def do_GET(self):
        type(self).request_count += 1
        if self.headers.get("If-None-Match") == self.etag:
            self.send_response(304)
            self.end_headers()
            return
        self.send_response(200)
        self.send_header("ETag", self.etag)
        self.send_header("Content-Length", str(len(self.body)))
        self.end_headers()
        self.wfile.write(self.body)

    def log_message(self, *args):
        pass


def _run_etag_server():
    server = http.server.HTTPServer(("127.0.0.1", 0), _ETagCSVHandler)
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    return server, thread


def test_load_catalog_latest_serves_cache_on_304(tmp_path):
    _ETagCSVHandler.request_count = 0
    server, thread = _run_etag_server()
    try:
        url = f"http://127.0.0.1:{server.server_port}/series.csv"
        cache_dir = tmp_path / "cache"

        first = load_catalog(version="latest", source=url, cache_dir=cache_dir)
        second = load_catalog(version="latest", source=url, cache_dir=cache_dir)

        assert first["id"].tolist() == second["id"].tolist() == ["gluonts:m4_hourly:h1"]
        assert _ETagCSVHandler.request_count == 2  # both hit the server; the 2nd got a 304
    finally:
        server.shutdown()
        thread.join()


def test_load_catalog_latest_refetches_body_when_etag_changes(tmp_path):
    _ETagCSVHandler.request_count = 0
    _ETagCSVHandler.etag = "etag-v1"
    server, thread = _run_etag_server()
    try:
        url = f"http://127.0.0.1:{server.server_port}/series.csv"
        cache_dir = tmp_path / "cache"

        load_catalog(version="latest", source=url, cache_dir=cache_dir)
        _ETagCSVHandler.etag = "etag-v2"
        _ETagCSVHandler.body = b"id\ngluonts:m4_hourly:h2\n"

        second = load_catalog(version="latest", source=url, cache_dir=cache_dir)

        assert second["id"].tolist() == ["gluonts:m4_hourly:h2"]
    finally:
        server.shutdown()
        thread.join()
        _ETagCSVHandler.etag = "etag-v1"
        _ETagCSVHandler.body = b"id\ngluonts:m4_hourly:h1\n"
