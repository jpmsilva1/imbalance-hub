import pandas as pd

from scripts.check_catalog_docs import compute_reference, diff_reference


def _catalog_row(**overrides):
    row = {
        "source": "gluonts", "collection": "m4_monthly", "granularity": "M",
        "time_column": None, "dtype": "float64",
        "N": 100, "IR": 5.0, "%Rare": 10.0, "length": 100,
        "missing_pct": 0.0, "seasonal_period": 12, "imbalance_level": "moderate",
    }
    row.update(overrides)
    return row


def test_compute_reference_counts_categoricals():
    df = pd.DataFrame([_catalog_row(), _catalog_row(source="tslib", time_column="date")])

    ref = compute_reference(df)

    assert ref["row_count"] == 2
    assert ref["source"] == {"gluonts": 1, "tslib": 1}
    assert ref["time_column_by_source"] == {"gluonts": None, "tslib": "date"}


def test_compute_reference_null_granularity_is_bucketed_not_dropped():
    df = pd.DataFrame([_catalog_row(granularity=None)])

    assert compute_reference(df)["granularity"] == {"__null__": 1}


def test_compute_reference_numeric_quantiles_and_rounding():
    df = pd.DataFrame([_catalog_row(**{"%Rare": 5.081}), _catalog_row(**{"%Rare": 13.539})])

    numeric = compute_reference(df)["numeric"]["%Rare"]

    assert numeric["min"] == 5.08
    assert numeric["max"] == 13.54


def test_diff_reference_flags_changed_numeric_value():
    diffs = diff_reference({"numeric": {"N": {"max": 100}}}, {"numeric": {"N": {"max": 200}}})

    assert len(diffs) == 1 and "numeric.N.max" in diffs[0]


def test_diff_reference_flags_new_and_missing_keys():
    diffs = diff_reference({"source": {"gluonts": 10}}, {"source": {"gluonts": 10, "tslib": 3}})

    assert any("new in catalog" in d for d in diffs)


def test_diff_reference_no_drift_returns_empty():
    ref = {"row_count": 5, "source": {"gluonts": 5}}

    assert diff_reference(ref, ref) == []
