"""Pull one or more accepted series from the HF blob mirror by catalog id."""
import hashlib
import posixpath
from concurrent.futures import ThreadPoolExecutor

import numpy as np
import pandas as pd

from .catalog import load_catalog

HF_REPO = "jpms5/imbalance-ts-hub"


def _content_hash(values):
    arr = np.asarray(values, dtype=float)
    return hashlib.sha256(arr.tobytes()).hexdigest()[:16]


def _check_safe_blob_path(blob_path):
    # blob_path comes from the remote catalog CSV, not something we
    # generate locally -- reject anything that isn't a plain relative path
    # before it reaches hf_hub_download, in case the catalog is ever served
    # tampered content.
    normalized = posixpath.normpath(blob_path)
    if blob_path.startswith("/") or normalized == ".." or normalized.startswith("../"):
        raise ValueError(f"unsafe blob_path in catalog: {blob_path!r}")


def _default_download(blob_path, revision):
    from huggingface_hub import hf_hub_download
    return hf_hub_download(repo_id=HF_REPO, filename=blob_path, revision=revision, repo_type="dataset")


def pull(id_: str, catalog: pd.DataFrame | None = None, download_fn=None) -> pd.Series:
    """Fetch one series by catalog id, cached on disk via huggingface_hub's
    own cache, verified against the catalog's content_hash."""
    cat = catalog if catalog is not None else load_catalog()
    matches = cat.loc[cat["id"] == id_]
    if matches.empty:
        raise KeyError(f"id not found in catalog: {id_}")
    row = matches.iloc[0]

    if pd.isna(row["blob_path"]):
        raise ValueError(f"no blob uploaded yet for id: {id_}")
    _check_safe_blob_path(row["blob_path"])

    fetch = download_fn or _default_download
    local_path = fetch(row["blob_path"], row["hf_revision"])
    df = pd.read_parquet(local_path)

    # ponytail: hash covers non-null values only, matching scan.py's dropna-based
    # hash. NaN positions are therefore unverified; length/missing_pct in the
    # catalog cover them loosely. Upgrade = re-scan with a NaN-inclusive hash.
    if _content_hash(df["value"].dropna()) != row["content_hash"]:
        raise ValueError(f"content_hash mismatch for id: {id_} -- blob does not match catalog")

    if "timestamp" in df.columns:
        return pd.Series(df["value"].to_numpy(), index=pd.DatetimeIndex(df["timestamp"]), name=id_)
    return pd.Series(df["value"].to_numpy(), name=id_)


def pull_many(ids, catalog: pd.DataFrame | None = None, download_fn=None, errors: str = "raise") -> dict:
    """Fetch multiple series by catalog id, concurrently.

    `errors="raise"` (default) aborts on the first failure, matching `pull()`.
    `errors="skip"` collects failures instead of raising, returning only the
    series that succeeded -- useful when pulling a large batch where a few
    missing/broken ids shouldn't sink the whole call.
    """
    if errors not in ("raise", "skip"):
        raise ValueError(f"errors must be 'raise' or 'skip', got {errors!r}")

    cat = catalog if catalog is not None else load_catalog()
    unique_ids = list(dict.fromkeys(ids))  # de-dupe, preserve order; repeats would download twice

    def _pull_or_capture(id_):
        try:
            return pull(id_, catalog=cat, download_fn=download_fn), None
        except Exception as exc:
            if errors == "raise":
                raise
            return None, exc

    with ThreadPoolExecutor(max_workers=8) as pool:
        results = list(pool.map(_pull_or_capture, unique_ids))

    return {id_: series for id_, (series, exc) in zip(unique_ids, results) if exc is None}
