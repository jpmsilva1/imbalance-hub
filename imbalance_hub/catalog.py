"""Load the imbalance-hub catalog -- accepted series only, as a DataFrame."""
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


def load_catalog(version: str = "latest", refresh: bool = False, source: str | None = None,
                  cache_dir: Path | str | None = None) -> pd.DataFrame:
    """Load catalog/series.csv as a DataFrame, cached locally per version.

    `source` overrides the default GitHub raw URL with a local path or URL --
    used by tests, and for pre-release use before a version is tagged.
    """
    cache_dir = Path(cache_dir) if cache_dir is not None else DEFAULT_CACHE_DIR
    cache_path = cache_dir / version / "series.csv"

    if source is None and _is_cacheable(version) and cache_path.exists() and not refresh:
        return pd.read_csv(cache_path)

    fetch_from = source or f"https://raw.githubusercontent.com/{GITHUB_REPO}/{_ref_for(version)}/catalog/series.csv"
    df = pd.read_csv(fetch_from)

    cache_path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(cache_path, index=False)
    return df
