# A Hybrid Spatio-Temporal Pipeline for NYC Crash Hotspot Discovery and Severity Prediction

CS 439 — Intro to Data Science final project.

**Authors:** Vishal Nagamalla (vn218), Pranav Arra (psa39), Logan Sharrott (ls1428)

This repository contains the code, processed-data manifest, figures, and
NeurIPS-format report for our CS 439 final project. We pair unsupervised
geospatial clustering of NYC traffic crashes with imbalance-aware supervised
severity prediction, and run a feature-group ablation to isolate the marginal
value of each engineered feature group.

## Headline numbers (test set, 2021–2022)

| Model | Bal. Acc. | PR-AUC | ROC-AUC | Recall (serious) |
|-------|----------:|-------:|--------:|-----------------:|
| Majority baseline                | 0.500 | —     | —     | 0.000 |
| Logistic Regression (balanced)   | **0.585** | **0.045** | **0.615** | **0.623** |
| Random Forest (balanced)         | 0.501 | 0.044 | 0.593 | 0.003 |
| XGBoost (scale_pos_weight)       | 0.544 | 0.042 | 0.588 | 0.231 |
| XGBoost + SMOTE                  | 0.500 | 0.040 | 0.583 | 0.001 |
| XGBoost + Threshold tuned (τ=0.69) | 0.510 | 0.042 | 0.588 | 0.041 |

KMeans (k=4 by silhouette = 0.465) recovers four NYC hotspot regions whose
serious-crash rates differ by 28% relative.

## Repository layout

```
.
├── config.py                          # paths + RANDOM_STATE
├── requirements.txt
├── data/
│   ├── raw/                           # nyc_crashes.csv, nyc_weather_daily.csv (gitignored)
│   └── processed/                     # parquet cache + sqlite mirror (gitignored)
├── src/
│   ├── etl.py                         # CSV -> parquet + sqlite cache
│   ├── features.py                    # cyclical/weather/spatial features
│   ├── cluster.py                     # KMeans + DBSCAN hotspot mining
│   ├── train.py                       # LR / RF / XGBoost ladder + SMOTE + threshold tuning + ablation
│   ├── evaluate.py                    # PR/ROC, SHAP, PCA, calibration, cluster maps
│   └── sql_summary.py                 # hotspot SQL output dumps
├── sql/
│   └── hotspot_queries.sql            # runnable on the sqlite cache
├── notebooks/
│   └── 01_results_walkthrough.ipynb   # reproduces every figure in the paper
├── figures/                           # PNGs referenced from the LaTeX
└── report/
    ├── cs439_final_project.tex        # NeurIPS preprint template
    ├── neurips_2026.sty
    ├── refs.bib
    └── cs439_final_project.pdf        # build artifact
```

## Datasets

Both source CSVs are public. We keep local copies in `data/raw/` for
one-command reproduction; **they are gitignored** because the crash CSV is
~557 MB, well over GitHub's 100 MB per-file limit. To rebuild:

* **NYC Motor Vehicle Collisions, Crashes** — NYC Open Data
  <https://data.cityofnewyork.us/Public-Safety/Motor-Vehicle-Collisions-Crashes/h9gi-nx95>
  (download CSV → save as `data/raw/nyc_crashes.csv`)
* **NYC Hourly Weather (2016–2022)** — Open-Meteo via Kaggle
  <https://www.kaggle.com/datasets/aadimator/nyc-weather-2016-to-2022>
  (download CSV → save as `data/raw/nyc_weather_daily.csv`; despite the
  filename it is hourly and gets aggregated to daily inside `src/etl.py`)

## Reproduction (one venv)

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt

# pipeline
python -m src.etl
python -m src.features
python -m src.cluster
python -m src.train
python -m src.evaluate
python -m src.sql_summary

# report
cd report
pdflatex cs439_final_project.tex
bibtex   cs439_final_project
pdflatex cs439_final_project.tex
pdflatex cs439_final_project.tex
```

The pipeline writes deterministic outputs (fixed `RANDOM_STATE = 42`) into
`data/processed/`:

* `accidents.parquet`, `weather.parquet`, `accidents_joined.parquet`
* `features.parquet`, `features_with_clusters.parquet`
* `traffic.db` (SQLite mirror used by `sql/hotspot_queries.sql`)
* `models/*.joblib` for the saved models
* `metrics/*.csv` and `metrics/test_predictions.npz` for every number cited in
  the paper

Every figure in `figures/` is regenerated from these artifacts by
`src/evaluate.py`.

## Reading the paper

The full report is at [`report/cs439_final_project.pdf`](report/cs439_final_project.pdf).
It follows the NeurIPS 2026 preprint template (8 body pages, references on
page 9).

