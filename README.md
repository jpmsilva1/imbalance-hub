# imbalance-hub

Phase 1: scan candidate time series datasets and score each with the
`imbalance_eval` library (`/Users/joaopms/Documents/imbalance_eval/imbalance_eval.py`,
read-only reference, not vendored here).

## Setup

```bash
pip install -r requirements.txt
```

## Run

Full scan (GluonTS + TSLib, ~150k series):

```bash
python scripts/scan.py --out scan_results.csv
```

Pilot run on one collection:

```bash
python scripts/scan.py --out pilot.csv --source gluonts --collections exchange_rate --limit 20
```

Resume an interrupted run (skips ids already in the CSV):

```bash
python scripts/scan.py --out scan_results.csv --resume
```

Add ADF stationarity test (~3x slower, off by default):

```bash
python scripts/scan.py --out scan_results.csv --with-adf
```

The script flushes results after each collection, so a crash never loses
more than the collection in progress. It prints one progress line per
collection and a final summary (status counts, imbalance_level counts, how
many series passed the `n_rare > 0` gate).

## What gets scanned

All GluonTS collections in `dataset_names` except the exclusions in
`GLUONTS_EXCLUDE` (kaggle_web_traffic ×3, dominick, temperature_rain — each
tens to hundreds of thousands of near-identical short series that would swamp
the catalog without adding dataset variety; plus `m5`, which needs a manual
Kaggle download, and `constant`, which is synthetic). Roughly 150k series
remain, dominated by M4 (~100k).

From TSLib, only the forecasting folders (`ETT-small`, `electricity`,
`exchange_rate`, `illness`, `traffic`, `weather`). Its anomaly-detection
(SMD/SWaT/PSM/MSL/SMAP) and UEA classification folders are out of scope.

Expect tens of minutes to a couple of hours, mostly download time, plus 1-2 GB
of data cached under `~/.gluonts` and `~/.cache/huggingface`.

## Known environment issue

Some `brotlicffi` builds fail to decode HuggingFace's larger responses
(`can_accept_more_data() is False`), which breaks `hf_hub_download` for the
bigger TSLib files. `read_tslib_csv` falls back to the plain resolve URL, which
does not request brotli, so this is handled — but it means those files are not
locally cached and are re-fetched if you rerun without `--resume`.

## CSV columns

| column | meaning |
|---|---|
| `id` | stable id, `source:collection:key` (e.g. `gluonts:m4_hourly:h1`) |
| `source` | `gluonts` or `tslib` |
| `collection` | dataset/repo name (e.g. `m4_hourly`, `ETTh1`) |
| `name` | series/column name within the collection |
| `granularity` | frequency string (e.g. `H`, `D`) |
| `time_column` | name of the date column dropped before scoring (tslib only) |
| `length` | number of observations in the raw series (incl. NaNs) |
| `dtype` | pandas dtype of the raw series |
| `content_hash` | `sha256(values)[:16]`, for dedup/change detection |
| `N` | embedded series length scored by `compute_imbalance` |
| `n_normal`, `n_rare` | counts of normal vs. rare (phi >= 0.9) points |
| `IR` | imbalance ratio, `n_normal / n_rare` (`inf` if no rare points) |
| `%Rare` | `100 * n_rare / N` |
| `imbalance_level` | `none`/`extreme`/`severe`/`moderate`/`mild`, derived from `%Rare` (lower %Rare = more imbalanced) |
| `missing_pct` | % NaN in the raw series |
| `mean`, `std`, `cv`, `skewness`, `kurtosis`, `autocorr_lag1` | cheap O(n) series characteristics (on non-null values) |
| `adf_pvalue`, `is_stationary` | only populated with `--with-adf` |
| `status` | `ok` / `skipped_short` (too short to embed) / `error` |
| `note` | error message or skip reason, truncated to ~200 chars |

Every series scanned gets a row, including failures -- the CSV is a
complete audit, not just the series that passed.

Scoring params are fixed to match Moniz, Branco & Torgo (2017):
`rel_thres=0.9, rel_coef=1.5, rel_xtrm_type="both"`, time-delay embedding
with `k=10`.
