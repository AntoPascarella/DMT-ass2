# DMT Assignment 2 — Expedia Hotel Ranking

Group 162: Alessandro Bertoncini, Antonio Pascarella, Vesper Kukler. Vrije Universiteit Amsterdam, 2026.

Learning-to-rank pipeline for the Expedia hotel search dataset, evaluated with NDCG@5.

## Pipeline

1. **EDA** — target distribution, position bias, price, property quality, missingness.
2. **Data preparation** — outlier clipping, per-country q1 imputation, feature engineering (per-query ranks/z-scores, composite features, competitor aggregates), graded `relevance` label (book=5, click=1), 90/10 query-level split.
3. **Modeling** — three base models combined with a per-query z-score weighted ensemble:
   - Logistic Regression (pointwise baseline)
   - LightGBM with LambdaRank (learning-to-rank)
   - Truncated SVD on a `srch_destination_id` × `prop_id` matrix (recommender systems lecture)
4. **Deployment / fairness** — query-level adaptive reranking to reduce under-exposure of independent hotels (`prop_brand_bool=0`).

## Results (validation NDCG@5)

| Model | NDCG@5 |
|---|---|
| Logistic Regression | 0.3466 |
| LightGBM LambdaRank | 0.3935 |
| Matrix Factorization (SVD, k=32) | 0.2719 |
| **Ensemble (LGBM 0.8 / SVD 0.2)** | **0.3999** |
| Ensemble + fairness rerank (α=0.5) | 0.4000 |

## Repo layout

```
main.tex                 LaTeX report (LNCS)
notebook.ipynb           End-to-end pipeline, top to bottom
figures/                 Plots used in the report
outputs/                 Fairness summary CSVs + fair submission
submission.csv           Final ensemble submission
data/                    Raw CSVs (gitignored, download from Kaggle)
processed/               Prepared parquet files (gitignored, generated)
slides/, reference_paper.pdf, assignment_2.pdf   Supporting material
```

## How to run

Place `training_set_VU_DM.csv` and `test_set_VU_DM.csv` from the [Kaggle competition](https://www.kaggle.com/competitions/dmt-2026-2nd-assignment) into `data/`, then:

```bash
pip install pandas numpy scikit-learn lightgbm scipy pyarrow matplotlib seaborn
jupyter nbconvert --to notebook --execute notebook.ipynb --inplace
```

This runs all 16 cells in order and produces:
- `figures/*.pdf,png` — EDA plots
- `processed/{train,val,test}.parquet` — cleaned datasets
- `submission.csv` — ensemble submission
- `outputs/submission_fair.csv` — fairness-aware submission
- `outputs/fairness_*.csv` — fairness metrics

Compile the report with `pdflatex main.tex` (requires the LNCS class).
