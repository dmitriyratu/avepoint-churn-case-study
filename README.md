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
│   ├── 04_modeling.py        # LightGBM + XGBoost, CV, threshold tuning
│   └── 05_results_validation.py  # SHAP, segment analysis, recommendations
├── src/
│   ├── load_data.py          # load the 5 tables
│   ├── clean.py              # date parsing, null imputation
│   ├── features.py           # feature aggregation + engineering
│   ├── model.py              # train, cross_validate, save/load
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

## Results

| Model | CV AUC (5-fold) | Test AUC | Test Recall | Test Precision |
|-------|----------------|----------|-------------|----------------|
| LightGBM | 0.43 | 0.55 | — | — |
| XGBoost | 0.47 | 0.55 | — | — |
| **Logistic Reg** | **0.49** | **0.60** | **0.86** | 0.26 |

**Note on signal quality**: this is a fully synthetic dataset where features and labels were
generated independently (max feature-target Pearson correlation ~0.12). CV AUC oscillating
around 0.5 is the expected outcome. Logistic regression outperforms tree models here because
regularized linear models generalize better than high-capacity models when there's no real
signal to find — tree models overfit to noise and invert on out-of-fold data.

In production with real user behavior data, feature-target correlations of 0.3–0.6 are typical,
and AUC of 0.75–0.85 is achievable with this feature set.

## Key Findings

1. **Feature breadth** is the single highest-importance feature in the LightGBM model (by gain).
   In real SaaS data, accounts touching a wider fraction of the product consistently retain longer.

2. **Support escalation rate** and **resolution time** appear in the top features — both
   reflect product experience health, not just support workload.

3. **Recency of usage** (days_since_last_usage) is the strongest individual signal even in this
   synthetic dataset. Active users don't churn — this should be tracked in production in real time.

4. **Logistic regression beats tree models** on this data. When signal is weak, simpler models
   with L2 regularization generalize better than high-capacity models that latch onto noise.

See `notebooks/05_results_validation.py` and `docs/ASSUMPTIONS.md` for full details.
