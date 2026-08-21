"""Load the imbalance-hub catalog -- accepted series only, as a DataFrame."""
import urllib.error
import urllib.request
from pathlib import Path

import pandas as pd

GITHUB_REPO = "jpmsilva1/imbalance-hub"
DEFAULT_CACHE_DIR = Path.home() / ".cache" / "imbalance_hub"


def _ref_for(version: str) -> str:
    """Git ref for a catalog version. No tags exist yet, so "latest" has no
    matching ref of its own -- it resolves to main, the branch that always
    has the newest published catalog."""
    return "main" if version == "latest" else version


def _is_cacheable(version: str) -> bool:
    """"latest" is a moving target -- unlike a pinned version, a locally
    cached copy of it can never be trusted as still current, so it always
    re-fetches. A pinned version is immutable once published, so its cache
    entry is safe to serve without a refresh."""
    return version != "latest"


def _is_url(s: str) -> bool:
    return s.startswith("http://") or s.startswith("https://")


def load_catalog(version: str = "latest", refresh: bool = False, source: str | None = None,
                  cache_dir: Path | str | None = None) -> pd.DataFrame:
    """Load catalog/series.csv as a DataFrame, cached locally per version.

    `source` overrides the default GitHub raw URL with a local path or URL --
    used by tests, and for pre-release use before a version is tagged.
    """
    cache_dir = Path(cache_dir) if cache_dir is not None else DEFAULT_CACHE_DIR
    cache_path = cache_dir / version / "series.csv"
    etag_path = cache_path.with_suffix(".etag")

    if source is None and _is_cacheable(version) and cache_path.exists() and not refresh:
        return pd.read_csv(cache_path, low_memory=False)

    fetch_from = source or f"https://raw.githubusercontent.com/{GITHUB_REPO}/{_ref_for(version)}/catalog/series.csv"

    # "latest" is never served from cache outright (see _is_cacheable), but a
    # conditional GET lets us skip re-downloading the 20+MB body when the
    # remote content hasn't actually changed since our last fetch.
    if (not _is_cacheable(version) and cache_path.exists() and not refresh
            and _is_url(fetch_from) and etag_path.exists()):
        request = urllib.request.Request(fetch_from, headers={"If-None-Match": etag_path.read_text().strip()})
        try:
            response = urllib.request.urlopen(request)
        except urllib.error.HTTPError as exc:
            if exc.code == 304:
                return pd.read_csv(cache_path, low_memory=False)
            raise
    elif _is_url(fetch_from):
        response = urllib.request.urlopen(fetch_from)
    else:
        response = fetch_from

    df = pd.read_csv(response, low_memory=False)
    new_etag = response.headers.get("ETag") if _is_url(fetch_from) else None

    cache_path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(cache_path, index=False)
    if new_etag and not _is_cacheable(version):
        etag_path.write_text(new_etag)
    return df
