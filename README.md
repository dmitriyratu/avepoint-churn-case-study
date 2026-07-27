# AvePoint — SaaS Churn Prediction

Churn prediction case study using the RavenStack synthetic SaaS dataset (Kaggle / Rivalytics).

## Problem

A mid-sized SaaS company wants to understand why users churn and predict it before it happens,
so the team can intervene with targeted retention actions.

## Dataset

Five relational tables, all keyed on `account_id`:

| File | Rows | Description |
|------|------|-------------|
| `ravenstack_accounts.csv` | 500 | One row per customer. Contains `churn_flag` (target). |
| `ravenstack_subscriptions.csv` | 5,000 | Subscription history — plan, MRR, upgrades/downgrades |
| `ravenstack_feature_usage.csv` | 25,000 | Per-feature usage logs linked via `subscription_id` |
| `ravenstack_support_tickets.csv` | 2,000 | Support interactions, resolution times, satisfaction scores |
| `ravenstack_churn_events.csv` | 600 | Logged churn reasons for churned accounts |

Place the raw CSVs in `data/raw/` before running.

## Structure

```
AvePoint/
├── data/
│   ├── raw/                  # original CSVs (not committed)
│   └── processed/            # cleaned tables + feature matrix
├── notebooks/
│   ├── 01_eda.py             # exploratory analysis
│   ├── 02_cleaning.py        # data cleaning walkthrough
│   ├── 03_feature_engineering.py
│   ├── 04_modeling.py        # first pass — static label (superseded by 06)
│   ├── 05_results_validation.py  # SHAP, segment analysis, recommendations
│   └── 06_audit_and_temporal_redesign.py   # <- read this one
├── src/
│   ├── config.py             # cutoff/horizon, leakage exclusion lists
│   ├── load_data.py          # load the 5 tables
│   ├── clean.py              # cleaning + integrity_report
│   ├── labeling.py           # cohort construction, table truncation
│   ├── features.py           # cutoff-aware feature aggregation
│   ├── model.py              # model ladder, permutation test, oof threshold
│   └── evaluate.py           # metrics, ROC/PR plots, SHAP helpers
├── outputs/
│   ├── figures/              # saved plots
│   ├── models/               # saved model artifacts (.joblib)
│   └── reports/              # CSV metrics
└── docs/
    └── ASSUMPTIONS.md        # key decisions and their rationale
```

## Setup

```bash
cd AvePoint
python -m venv .venv
source .venv/bin/activate      # Windows: .venv\Scripts\activate
pip install -r requirements.txt
```

## Running

Notebooks are in [jupytext percent format](https://jupytext.readthedocs.io/).
Open and run them in VS Code (Python Interactive), or convert to `.ipynb`:

```bash
jupytext --to notebook notebooks/01_eda.py
jupyter notebook notebooks/01_eda.ipynb
```

Run all notebooks in order:

```bash
for nb in notebooks/0*.py; do
    jupytext --to notebook --execute $nb
done
```

Or run the src modules directly as a pipeline:

```bash
python -c "
from src.load_data import load_all
from src.clean import clean_all
from src.features import build_model_dataset
from src.model import prep_xy, train_lgb, save_model

tables = clean_all(load_all())
df = build_model_dataset(tables)
# fill nulls for accounts with no tickets/usage
df = df.fillna(0)
X, y = prep_xy(df)
model = train_lgb(X, y)
save_model(model, 'lgb_churn')
print('Done')
"
```

## Problem framing

`accounts.churn_flag` is a static flag with no date attached, so predicting it directly
has no observation window — a customer's post-churn activity ends up in their own feature
vector. This project models a forward-looking target instead:

```
|<------ observation window ------>|<---- prediction window ---->|
2023-01-01                    2024-06-30                    2024-12-27
       features built here              label defined here
```

- **Eligible**: signed up before the cutoff, not already churned at it → 187 accounts
- **Label**: first churn event within 180 days after the cutoff → 88 positives (47%)
- **Features**: computed only from rows dated before the cutoff

## Results

Model ladder, repeated stratified CV (5 folds × 10 repeats, identical folds throughout):

| Rung | Model | CV ROC-AUC | 95% CI |
|------|-------|-----------|--------|
| 0 | Prior (no features) | 0.500 | — |
| 1 | Decision stump | 0.531 | [0.43, 0.60] |
| 2 | Logistic (L2, C=1) | 0.574 | [0.40, 0.71] |
| 3 | Logistic (L2, C=0.05) | 0.583 | [0.37, 0.71] |
| **4** | **Logistic (L1, C=0.1)** | **0.635** | **[0.44, 0.76]** |
| 5 | Random forest (depth 4) | 0.587 | [0.44, 0.72] |
| 6 | LightGBM (shallow) | 0.570 | [0.35, 0.70] |

Tuned LightGBM over a 54-point grid reaches 0.619 — still below the linear model.

**Permutation test** (300 label shuffles): observed 0.636 vs null mean 0.494,
95th percentile 0.605 → **p = 0.013**. The association is real, if modest.

**Operating point** (threshold selected out-of-fold, never on the evaluation set):
threshold 0.45 → recall 0.864, precision 0.547, F1 0.670.

L1 keeps **8 of 76** features.

## Key Findings

1. **Recency beats volume.** `days_since_last_sub_start` is the strongest single
   predictor — accounts that have stopped opening new subscriptions are disengaging,
   regardless of how much they used the product historically.

2. **Complexity does not pay here.** Both tree ensembles score below L1 logistic, and
   grid search doesn't close the gap. At 1.16 events per variable, hard feature
   selection is worth more than model capacity. Shipping the boosted model would have
   been a worse product for more compute.

3. **Leakage was the real story, not "weak data."** Features derived from `churn_events`
   (refund amount, churn reason, reactivation) take CV AUC to **0.997** — they encode the
   answer, since a refund is issued *because* the customer left. They are excluded via
   `config.POST_OUTCOME_COLS`.

4. **The label itself is inconsistent.** `churn_flag` and the `churn_events` table
   disagree for 312 of 500 accounts. The event log is used as ground truth because it
   carries dates.

## Honest limitations

- 187 accounts / 88 positives. The AUC interval [0.44, 0.76] is wide; the point estimate
  should not be quoted alone.
- A single cutoff date. Production evaluation needs rolling-origin backtesting.
- Source data has integrity problems (1,077 tickets predate their account's signup;
  19,128 usage rows predate their subscription's start) — surfaced by
  `clean.integrity_report`, not silently ignored.
- Synthetic data caps feature-target association at |r| = 0.28. Real telemetry typically
  reaches 0.3–0.6, where AUC 0.75+ is achievable with this feature set.

See `notebooks/06_audit_and_temporal_redesign.py` and `docs/ASSUMPTIONS.md` for the
full audit, including the bugs found in the first pass and how they were corrected.
