#!/usr/bin/env python3
"""Stages accepted series as per-series Parquet blobs and pushes them to the
HF dataset repo, then fills blob_path/hf_revision/size_bytes/pipeline_version/
ingested_at back into catalog/series.csv.

Requires imbalance_eval on the path -- see scripts/scan.py's import fallback,
which this module inherits by importing from it."""
import argparse
import hashlib
import re
import tomllib
from datetime import datetime, timezone
from importlib.metadata import PackageNotFoundError, version
from pathlib import Path

import pandas as pd

from imbalance_hub.download import HF_REPO
from scripts.scan import content_hash, iter_gluonts_series, iter_tslib_series, list_tslib_collections, read_tslib_csv


def blob_path_for(id_: str) -> str:
    """catalog id -> repo-relative parquet path. Sanitizes the key so ids
    containing '/' (tslib:weather:wv-(m/s)) don't become nested directories,
    and shards into a 2-hex-char subdirectory -- git/HF repos reject a
    directory with more than 10,000 files, and gluonts:m4_monthly alone has
    16,998 series. 256-way sharding keeps every directory under ~70 files at
    today's scale with headroom for collections several times larger."""
    source, collection, key = id_.split(":", 2)
    shard = hashlib.md5(id_.encode()).hexdigest()[:2]
    safe_key = re.sub(r'[^a-z0-9._-]', '-', key)
    return f"{source}/{collection}/{shard}/{safe_key}.parquet"


def series_to_frame(series: pd.Series) -> pd.DataFrame:
    """Raw values (NaNs included) as the blob's `value` column, plus
    `timestamp` iff the series carries a DatetimeIndex."""
    if isinstance(series.index, pd.DatetimeIndex):
        return pd.DataFrame({"value": series.to_numpy(), "timestamp": series.index})
    return pd.DataFrame({"value": series.to_numpy()})


def stage_series(stage_dir: Path, id_: str, series: pd.Series, expected_hash: str) -> Path:
    """Write series to stage_dir/blob_path_for(id_) as Parquet, after verifying
    content_hash(series.dropna().values) == expected_hash. Returns the path.
    No-op if the file already exists (resumable staging)."""
    if content_hash(series.dropna().values) != expected_hash:
        raise ValueError(f"content_hash mismatch for id: {id_} -- series does not match catalog")

    path = Path(stage_dir) / blob_path_for(id_)
    if not path.exists():
        path.parent.mkdir(parents=True, exist_ok=True)
        series_to_frame(series).to_parquet(path)
    return path


def finalize_catalog(series_df: pd.DataFrame, blob_sizes: dict, revision: str,
                      pipeline_version: str, ingested_at: str) -> pd.DataFrame:
    """Fill blob_path/size_bytes/hf_revision/pipeline_version/ingested_at for
    ids present in blob_sizes; leave the rest untouched (NA)."""
    result = series_df.copy()
    uploaded = result["id"].isin(blob_sizes)
    result.loc[uploaded, "blob_path"] = result.loc[uploaded, "id"].map(blob_path_for)
    result.loc[uploaded, "size_bytes"] = result.loc[uploaded, "id"].map(blob_sizes)
    result.loc[uploaded, "hf_revision"] = revision
    result.loc[uploaded, "pipeline_version"] = pipeline_version
    result.loc[uploaded, "ingested_at"] = ingested_at
    return result


def pipeline_version() -> str:
    """Installed package version, falling back to pyproject.toml's [project]
    version -- upload_blobs.py is routinely run via `python -m` against an
    uninstalled checkout, where importlib.metadata has no record at all."""
    try:
        return version("imbalance-hub")
    except PackageNotFoundError:
        pyproject = Path(__file__).resolve().parent.parent / "pyproject.toml"
        with open(pyproject, "rb") as f:
            return tomllib.load(f)["project"]["version"]


def _stage_collection(stage_dir, source, collection, wanted_hashes):
    """Fetch one collection from source, stage every id in wanted_hashes
    (id -> content_hash), and return {id: staged_size_bytes}. Ids present in
    the catalog but absent from the fetched collection are silently skipped
    -- they belong to a different run of the source and are picked up when
    that collection is staged."""
    if source == "gluonts":
        try:
            from gluonts.dataset.repository.datasets import get_dataset
        except ImportError:
            from gluonts.dataset.repository import get_dataset
        ds = get_dataset(collection)
        items = iter_gluonts_series(collection, ds.train, getattr(ds.metadata, "freq", ""))
    else:
        repo_path = list_tslib_collections({collection}).get(collection)
        if repo_path is None:
            raise ValueError(f"no TSLib csv found for collection: {collection}")
        items = iter_tslib_series(collection, read_tslib_csv(repo_path))

    sizes = {}
    for id_, *_, series in items:
        if id_ not in wanted_hashes:
            continue
        path = stage_series(stage_dir, id_, series, expected_hash=wanted_hashes[id_])
        sizes[id_] = path.stat().st_size
    return sizes


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--catalog", default="catalog/series.csv")
    parser.add_argument("--stage-dir", default="/tmp/imbalance-hub-blobs")
    parser.add_argument("--collections", default=None, help="comma-separated subset of collections")
    parser.add_argument("--stage-only", action="store_true", help="stage blobs, skip the HF upload")
    args = parser.parse_args()

    series_df = pd.read_csv(args.catalog, low_memory=False)
    unstaged = series_df[series_df["blob_path"].isna()]
    if args.collections:
        wanted = set(args.collections.split(","))
        unstaged = unstaged[unstaged["collection"].isin(wanted)]

    stage_dir = Path(args.stage_dir)
    blob_sizes = {}
    for (source, collection), group in unstaged.groupby(["source", "collection"]):
        wanted_hashes = dict(zip(group["id"], group["content_hash"]))
        sizes = _stage_collection(stage_dir, source, collection, wanted_hashes)
        blob_sizes.update(sizes)
        missing = wanted_hashes.keys() - sizes.keys()
        if missing:
            print(f"{source}:{collection} WARNING: {len(missing)} catalog ids not found in source: {sorted(missing)[:5]}...")
        print(f"{source}:{collection} staged {len(sizes)}/{len(wanted_hashes)}")

    staged_paths = [blob_path_for(id_) for id_ in blob_sizes]
    assert len(staged_paths) == len(set(staged_paths)), "blob_path collision -- sanitizer needs disambiguation"

    if args.stage_only or not blob_sizes:
        print(f"staged {len(blob_sizes)} blobs under {stage_dir}, stopping (--stage-only or nothing new)")
        return

    from huggingface_hub import HfApi
    api = HfApi()
    api.upload_folder(repo_id=HF_REPO, folder_path=str(stage_dir), repo_type="dataset")
    revision = api.list_repo_commits(HF_REPO, repo_type="dataset")[0].commit_id

    updated = finalize_catalog(
        series_df, blob_sizes, revision=revision,
        pipeline_version=pipeline_version(),
        ingested_at=datetime.now(timezone.utc).isoformat(),
    )
    updated.to_csv(args.catalog, index=False)
    print(f"uploaded {len(blob_sizes)} blobs, revision {revision}, wrote {args.catalog}")


if __name__ == "__main__":
    main()
