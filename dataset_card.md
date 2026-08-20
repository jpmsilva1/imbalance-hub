---
pretty_name: imbalance-hub
license: unknown
task_categories:
  - time-series-forecasting
tags:
  - time-series
  - imbalanced-regression
  - imbalanced-learning
  - forecasting
size_categories:
  - 10K<n<100K
---

# imbalance-hub

A curated, versioned catalog of **imbalanced time series** — "OpenML for imbalanced time
series." Every series here has been scored with
[`imbalance_eval`](https://github.com/jpmsilva1/imbalance_eval) (the relevance-function
methodology from Moniz, Branco & Torgo, 2017) and kept only if it actually has a rare
regime worth studying.

The catalog metadata and client library live in the companion GitHub repo,
**[jpmsilva1/imbalance-hub](https://github.com/jpmsilva1/imbalance-hub)** — start there for
the schema, the scoring methodology, and how the catalog is built. This dataset repo holds
the actual data: one Parquet file per accepted series.

## Quick start

```bash
pip install git+https://github.com/jpmsilva1/imbalance-hub.git
```

```python
from imbalance_hub import load_catalog, pull

catalog = load_catalog()
severe_hourly = catalog[(catalog.imbalance_level == "severe") & (catalog.granularity == "H")]

series = pull(severe_hourly.id.iloc[0])   # -> pd.Series, values ready to use
```

`load_catalog()` fetches the metadata CSV from GitHub (not from this repo) and caches it
locally; `pull()` downloads only the one Parquet blob you asked for from here, verified
against a `content_hash` recorded in the catalog. Browsing the catalog's 59k+ rows never
downloads more than the metadata.

## Dataset structure

One Parquet file per accepted series, at `{source}/{collection}/{shard}/{key}.parquet`
(`shard` is a 2-hex-char directory keeping every folder under Hugging Face's per-directory
file limit — an implementation detail, not part of the catalog id). Each file has:

| Column | Type | Notes |
|---|---|---|
| `value` | float64 | Raw values, in original order. NaNs preserved where the source series has missing points. |
| `timestamp` | datetime64 | Present only when the source series carries real timestamps (TSLib); absent for GluonTS series, which are index-only. |

The catalog id (`source:collection:key`, e.g. `gluonts:m4_hourly:h1`) is what maps an id to
its blob path — see `blob_path_for()` in
[`scripts/upload_blobs.py`](https://github.com/jpmsilva1/imbalance-hub/blob/main/scripts/upload_blobs.py)
in the GitHub repo, or just use the catalog CSV's `blob_path` column directly.

## Dataset summary

- **59,489 accepted series** out of 136,673 scanned (~43.5%) — the rest were excluded by
  the inclusion gate (no rare regime) or failed to embed (too short).
- **55 source collections**, from [GluonTS](https://github.com/awslabs/gluonts)
  (58,354 series — M4, M3, electricity, traffic, tourism, and more) and
  [TSLib](https://huggingface.co/datasets/thuml/Time-Series-Library) (1,135 series — ETT,
  weather, exchange rate, national illness).
- **Severity distribution**: 1,172 `extreme`, 13,256 `severe`, 32,867 `moderate`,
  12,194 `mild` (see the GitHub README for how `imbalance_level` is computed from `%Rare`).

## Licensing

**No single license applies — this catalog spans sources with different terms, checked
against the original loaders rather than assumed.** The curation layer (catalog metadata,
imbalance-scoring methodology, pipeline code) is separate from the underlying values and
is available under the
[imbalance-hub GitHub repo](https://github.com/jpmsilva1/imbalance-hub)'s terms. For the
raw values themselves:

- **Most GluonTS collections** (`electricity`, `traffic`, `tourism_*`, `nn5_*`, `weather`,
  `wind_farms_*`, `m1_*`, and most others fetched via GluonTS's
  [`_tsf_datasets.py`](https://github.com/awslabs/gluonts/blob/dev/src/gluonts/dataset/repository/_tsf_datasets.py))
  come from the [Monash Time Series Forecasting Archive](https://forecastingdata.org/) on
  Zenodo, which licenses its datasets under
  **[CC BY 4.0](https://creativecommons.org/licenses/by/4.0/)** — verified directly against
  the [NN5 Daily record](https://zenodo.org/record/4656110).
- **M4** (`m4_*` — GluonTS's largest single contributor by series count) is fetched from
  [`M4Competition/M4-methods`](https://github.com/M4Competition/M4-methods) on GitHub,
  which **carries no LICENSE file and no license declared** (confirmed via the GitHub API).
  Treat M4 series as unclear rights until the competition organizers state otherwise.
- **TSLib** (`ETT-small`, `electricity`, `exchange_rate`, `illness`, `traffic`, `weather`)
  — the [thuml/Time-Series-Library](https://github.com/thuml/Time-Series-Library) repo
  itself is **MIT-licensed**, but that covers the code; the bundled benchmark CSVs'
  original licenses aren't stated in that repo.

If you plan to use a specific series commercially or redistribute it standalone, verify
that series' actual source collection rather than relying on this summary.

## Citation

If you use this catalog's imbalance scoring, cite the methodology it implements:

```bibtex
@article{moniz2017resampling,
  title={Resampling strategies for imbalanced time series forecasting},
  author={Moniz, Nuno and Branco, Paula and Torgo, Lu{\'\i}s},
  journal={International Journal of Data Science and Analytics},
  volume={3},
  pages={161--181},
  year={2017},
  publisher={Springer}
}
```
