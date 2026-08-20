import hashlib

import numpy as np
import pandas as pd
import pytest

from imbalance_hub.download import pull, pull_many


def _content_hash(values):
    arr = np.asarray(values, dtype=float)
    return hashlib.sha256(arr.tobytes()).hexdigest()[:16]


def _write_blob(tmp_path, values, timestamps=None):
    path = tmp_path / "blob.parquet"
    data = {"value": values}
    if timestamps is not None:
        data["timestamp"] = timestamps
    pd.DataFrame(data).to_parquet(path)
    return path


def test_pull_downloads_and_returns_series_matching_content_hash(tmp_path):
    values = [1.0, 2.0, 3.0]
    blob_path = _write_blob(tmp_path, values)
    catalog = pd.DataFrame([{
        "id": "gluonts:m4_hourly:h1", "blob_path": "gluonts/m4_hourly/h1.parquet",
        "hf_revision": "main", "content_hash": _content_hash(values),
    }])

    result = pull("gluonts:m4_hourly:h1", catalog=catalog, download_fn=lambda *a, **k: blob_path)

    assert isinstance(result, pd.Series)
    assert result.tolist() == values


def test_pull_returns_datetime_indexed_series_when_blob_has_timestamps(tmp_path):
    values = [1.0, 2.0]
    timestamps = ["2020-01-01", "2020-01-02"]
    blob_path = _write_blob(tmp_path, values, timestamps)
    catalog = pd.DataFrame([{
        "id": "tslib:traffic:0", "blob_path": "tslib/traffic/0.parquet",
        "hf_revision": "main", "content_hash": _content_hash(values),
    }])

    result = pull("tslib:traffic:0", catalog=catalog, download_fn=lambda *a, **k: blob_path)

    assert isinstance(result.index, pd.DatetimeIndex)
    assert result.tolist() == values


def test_pull_rejects_traversal_shaped_blob_path():
    # blob_path/hf_revision come from the remote catalog CSV -- a compromised
    # or malicious catalog shouldn't be able to make hf_hub_download read
    # outside the intended cache directory.
    catalog = pd.DataFrame([{"id": "gluonts:m4_hourly:h1", "blob_path": "../../etc/passwd",
                              "hf_revision": "main", "content_hash": "deadbeef"}])

    with pytest.raises(ValueError, match="blob_path"):
        pull("gluonts:m4_hourly:h1", catalog=catalog, download_fn=lambda *a, **k: None)


def test_pull_rejects_absolute_blob_path():
    catalog = pd.DataFrame([{"id": "gluonts:m4_hourly:h1", "blob_path": "/etc/passwd",
                              "hf_revision": "main", "content_hash": "deadbeef"}])

    with pytest.raises(ValueError, match="blob_path"):
        pull("gluonts:m4_hourly:h1", catalog=catalog, download_fn=lambda *a, **k: None)


def test_pull_raises_when_id_not_in_catalog():
    catalog = pd.DataFrame([{"id": "known:id:1", "blob_path": "x.parquet",
                              "hf_revision": "main", "content_hash": "deadbeef"}])

    with pytest.raises(KeyError):
        pull("unknown:id:1", catalog=catalog, download_fn=lambda *a, **k: None)


def test_pull_raises_when_series_has_no_uploaded_blob_yet():
    # Reflects the current real catalog: blob_path is blank until the HF
    # upload step runs, which hasn't happened yet.
    catalog = pd.DataFrame([{"id": "gluonts:m4_hourly:h1", "blob_path": np.nan,
                              "hf_revision": np.nan, "content_hash": "deadbeef"}])

    with pytest.raises(ValueError, match="no blob"):
        pull("gluonts:m4_hourly:h1", catalog=catalog, download_fn=lambda *a, **k: None)


def test_pull_raises_on_content_hash_mismatch(tmp_path):
    blob_path = _write_blob(tmp_path, [1.0, 2.0, 3.0])
    catalog = pd.DataFrame([{
        "id": "gluonts:m4_hourly:h1", "blob_path": "gluonts/m4_hourly/h1.parquet",
        "hf_revision": "main", "content_hash": "wronghash0000000",
    }])

    with pytest.raises(ValueError, match="content_hash"):
        pull("gluonts:m4_hourly:h1", catalog=catalog, download_fn=lambda *a, **k: blob_path)


def test_pull_verifies_hash_over_non_null_values(tmp_path):
    # Blobs are staged raw (NaNs included, per the agreed blob format), but
    # scan.py's content_hash is computed over the NaN-dropped scoring view --
    # pull() must verify against the same non-null view, not the raw column.
    blob_path = _write_blob(tmp_path, [1.0, np.nan, 3.0])
    catalog = pd.DataFrame([{
        "id": "gluonts:nn5_daily_with_missing:0", "blob_path": "gluonts/nn5_daily_with_missing/0.parquet",
        "hf_revision": "main", "content_hash": _content_hash([1.0, 3.0]),
    }])

    result = pull("gluonts:nn5_daily_with_missing:0", catalog=catalog, download_fn=lambda *a, **k: blob_path)

    assert result.tolist()[0] == 1.0
    assert np.isnan(result.tolist()[1])
    assert result.tolist()[2] == 3.0


def test_pull_many_returns_one_series_per_id(tmp_path):
    values_a, values_b = [1.0, 2.0], [3.0, 4.0]
    blob_a, blob_b = _write_blob(tmp_path, values_a), tmp_path / "b.parquet"
    pd.DataFrame({"value": values_b}).to_parquet(blob_b)
    catalog = pd.DataFrame([
        {"id": "a:x:1", "blob_path": "a.parquet", "hf_revision": "main", "content_hash": _content_hash(values_a)},
        {"id": "b:x:1", "blob_path": "b.parquet", "hf_revision": "main", "content_hash": _content_hash(values_b)},
    ])
    paths = {"a.parquet": blob_a, "b.parquet": blob_b}

    result = pull_many(["a:x:1", "b:x:1"], catalog=catalog,
                        download_fn=lambda blob_path, revision: paths[blob_path])

    assert set(result.keys()) == {"a:x:1", "b:x:1"}
    assert result["a:x:1"].tolist() == values_a
    assert result["b:x:1"].tolist() == values_b
