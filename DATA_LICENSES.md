# Data licensing scope

The `LICENSE` file (MIT) in this repo covers the **code only** — the
`imbalance_hub/` client library and the `scripts/` ingestion pipeline. It does
**not** cover the catalog metadata (`catalog/series.csv`) or the redistributed
series values (the Parquet blobs on the companion Hugging Face dataset repo,
[`jpms5/imbalance-ts-hub`](https://huggingface.co/datasets/jpms5/imbalance-ts-hub)).

Those come from third-party sources with their own, differing terms. See
`dataset_card.md`'s "Licensing" section for the full per-collection breakdown
(Monash/Zenodo archive under CC BY 4.0, M4 competition data with no license
declared, TSLib's bundled CSVs with code under MIT but data terms unstated).

`catalog/series.csv` has a `license` column (`cc-by-4.0` / `unlicensed` / `unknown`)
so you can filter programmatically, e.g. `catalog[catalog.license == "cc-by-4.0"]`.

If you plan to use a specific series commercially or redistribute it
standalone, verify that series' actual source collection rather than relying
on this summary.
