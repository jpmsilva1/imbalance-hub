import pandas as pd

from imbalance_hub.catalog import load_catalog


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
