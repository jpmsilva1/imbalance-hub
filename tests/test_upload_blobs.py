import numpy as np
import pandas as pd
import pytest

from scripts.scan import content_hash
from scripts.upload_blobs import blob_path_for, finalize_catalog, series_to_frame, stage_series


def test_blob_path_for_plain_id():
    assert blob_path_for("gluonts:m4_hourly:h1") == "gluonts/m4_hourly/h1.parquet"


def test_blob_path_for_does_not_nest_on_slash_in_key():
    # tslib:weather:wv-(m/s) -- a literal "/" in the key must not become a
    # nested directory in the blob path.
    path = blob_path_for("tslib:weather:wv-(m/s)")

    assert path == "tslib/weather/wv--m-s-.parquet"
    assert ".." not in path and not path.startswith("/")


def test_series_to_frame_without_datetime_index_has_only_value_column():
    series = pd.Series([1.0, np.nan, 3.0])

    frame = series_to_frame(series)

    assert list(frame.columns) == ["value"]
    assert frame["value"].tolist()[0] == 1.0
    assert np.isnan(frame["value"].tolist()[1])


def test_series_to_frame_with_datetime_index_adds_timestamp_column():
    series = pd.Series([1.0, 2.0], index=pd.DatetimeIndex(["2020-01-01", "2020-01-02"]))

    frame = series_to_frame(series)

    assert list(frame.columns) == ["value", "timestamp"]
    assert frame["timestamp"].tolist() == list(series.index)


def test_stage_series_rejects_hash_mismatch(tmp_path):
    series = pd.Series([1.0, 2.0, 3.0])

    with pytest.raises(ValueError, match="content_hash"):
        stage_series(tmp_path, "gluonts:m4_hourly:h1", series, expected_hash="wronghash0000000")

    assert not (tmp_path / "gluonts" / "m4_hourly" / "h1.parquet").exists()


def test_stage_series_writes_parquet_matching_series_to_frame(tmp_path):
    series = pd.Series([1.0, 2.0, 3.0])
    expected_hash = content_hash(series.dropna().values)

    path = stage_series(tmp_path, "gluonts:m4_hourly:h1", series, expected_hash=expected_hash)

    assert path == tmp_path / "gluonts" / "m4_hourly" / "h1.parquet"
    assert pd.read_parquet(path)["value"].tolist() == [1.0, 2.0, 3.0]


def test_stage_series_skips_already_staged_file(tmp_path):
    series = pd.Series([1.0, 2.0, 3.0])
    expected_hash = content_hash(series.dropna().values)
    path = stage_series(tmp_path, "gluonts:m4_hourly:h1", series, expected_hash=expected_hash)
    mtime_before = path.stat().st_mtime_ns

    stage_series(tmp_path, "gluonts:m4_hourly:h1", series, expected_hash=expected_hash)

    assert path.stat().st_mtime_ns == mtime_before


def test_finalize_catalog_fills_only_uploaded_rows():
    series_df = pd.DataFrame([
        {"id": "gluonts:m4_hourly:h1", "blob_path": pd.NA, "size_bytes": pd.NA,
         "hf_revision": pd.NA, "pipeline_version": pd.NA, "ingested_at": pd.NA},
        {"id": "gluonts:m4_hourly:h2", "blob_path": pd.NA, "size_bytes": pd.NA,
         "hf_revision": pd.NA, "pipeline_version": pd.NA, "ingested_at": pd.NA},
    ])
    blob_sizes = {"gluonts:m4_hourly:h1": 1234}

    result = finalize_catalog(series_df, blob_sizes, revision="abc123",
                               pipeline_version="0.1.0", ingested_at="2026-08-20")

    uploaded = result.loc[result["id"] == "gluonts:m4_hourly:h1"].iloc[0]
    assert uploaded["blob_path"] == "gluonts/m4_hourly/h1.parquet"
    assert uploaded["size_bytes"] == 1234
    assert uploaded["hf_revision"] == "abc123"
    assert uploaded["pipeline_version"] == "0.1.0"
    assert uploaded["ingested_at"] == "2026-08-20"

    untouched = result.loc[result["id"] == "gluonts:m4_hourly:h2"].iloc[0]
    for col in ("blob_path", "size_bytes", "hf_revision", "pipeline_version", "ingested_at"):
        assert pd.isna(untouched[col])
