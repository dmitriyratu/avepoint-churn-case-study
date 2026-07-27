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
│   ├── 01_eda.py             # EDA — quality pass on all rows, target pass on train split only
│   ├── 02_cleaning.py        # data cleaning walkthrough
│   ├── 03_feature_engineering.py  # feature families + buffer sensitivity
│   ├── 04_modeling.py        # model ladder, permutation test, operating point
│   ├── 05_results_validation.py  # recommendations, deployment, monitoring, mentoring
│   ├── 06_leakage_quantification.py   # what each form of leakage is worth
│   └── 07_leakage_audit.py   # automated leakage + cleaning gates
├── src/
│   ├── audit.py              # automated leakage + quality gates
│   ├── config.py             # cutoff/buffer/horizon, leakage exclusion lists
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
    ├── ASSUMPTIONS.md        # key decisions and their rationale
    ├── CLEANING_CHECKLIST.md # cleaning/preprocessing practices, with findings
    ├── DATA_DICTIONARY.md    # field-by-field availability-at-prediction-time audit
    ├── EDA_CHECKLIST.md      # EDA practices followed, with what each one found
    └── FEATURE_ENGINEERING.md # churn FE taxonomy, buffer analysis, what helped
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

Run all notebooks in order (they also execute as plain scripts):

```bash
cd notebooks && for nb in 0*.py; do python "$nb" || echo "FAILED: $nb"; done
```

All seven run clean end-to-end. `03` and `04` assert the leakage suite passes and
will stop the pipeline if it does not.

Or run the src modules directly as a pipeline:

```bash
python -c "
from src.load_data import load_all
from src.clean import clean_all
from src.labeling import build_cohort, truncate_tables
from src.features import build_model_dataset
from src.model import prep_xy, model_ladder, evaluate_ladder, save_model
from src.config import CUTOFF_DATE
import src.audit as audit

tables = clean_all(load_all())
cohort = build_cohort(tables)                       # forward-looking label
obs    = truncate_tables(tables, CUTOFF_DATE)       # observation window only
df     = build_model_dataset(obs, cohort, CUTOFF_DATE)
X, y   = prep_xy(df)                                # NaNs kept for in-fold imputation

_, passed = audit.run_all(X, y, df, obs, CUTOFF_DATE)
assert passed, 'leakage audit failed'               # gate before any score is trusted

print(evaluate_ladder(X, y).to_string(index=False))
best = model_ladder()[4][1].fit(X, y)
save_model(best, 'churn_l1_logistic')
"
```

## Problem framing

`accounts.churn_flag` is a static flag with no date attached, so predicting it
directly has no observation window. This project models a forward-looking target
with an explicit **buffer** between the last observable day and the first day a
churn can count:

```
|<------ observation ------>|<- buffer ->|<------ prediction ------>|
2023-01-01             2024-05-31    2024-06-30              2024-12-27
      features built here             30 days      label defined here
```

- **Eligible**: signed up before the cutoff, not already churned when the
  prediction window opens → 168 accounts
- **Label**: first churn event within 180 days of the prediction start → 74
  positives (44%)
- **Features**: computed only from rows dated before the feature cutoff

The buffer is the operational reality — a CSM needs lead time between the score
landing and the customer leaving — and it turns out to decide the whole result.

## Headline result

**On this dataset there is no actionable churn signal at a realistic
intervention horizon.**

| Buffer | Eligible | Positives | CV ROC-AUC | 95% CI | Permutation p |
|---:|---:|---:|---:|---|---:|
| 0 days | 187 | 88 | 0.616 | [0.50, 0.74] | **0.025** |
| 15 days | 176 | 80 | 0.574 | [0.40, 0.75] | 0.254 |
| **30 days** | **168** | **74** | **0.548** | **[0.38, 0.70]** | **0.249** |
| 60 days | 154 | 67 | 0.545 | [0.35, 0.71] | 0.209 |
| 90 days | 139 | 57 | 0.467 | [0.27, 0.59] | 0.960 |

With no buffer the model beats chance. Give it even two weeks of lead time and it
does not, and it never recovers. The apparent signal is **reactive** — it detects
customers who have effectively already left.

A model shipped without a buffer would have passed cross-validation and been
useless in production. That is the finding, and it is worth more than the score.

## Model ladder (30-day buffer)

Repeated stratified CV, 5 folds x 10 repeats, identical folds throughout:

| Rung | Model | CV ROC-AUC | 95% CI |
|------|-------|-----------|--------|
| 0 | Prior (no features) | 0.500 | — |
| 1 | Decision stump | 0.523 | [0.38, 0.64] |
| 2 | Logistic (L2, C=1) | 0.490 | [0.38, 0.64] |
| 3 | Logistic (L2, C=0.05) | 0.492 | [0.34, 0.64] |
| **4** | **Logistic (L1, C=0.1)** | **0.548** | **[0.38, 0.70]** |
| 5 | Random forest (depth 4) | 0.453 | [0.32, 0.58] |
| 6 | LightGBM (shallow) | 0.484 | [0.35, 0.67] |

Nothing here is distinguishable from chance. Reported as-is.

### Did richer feature engineering help?

| Configuration | CV ROC-AUC |
|---|---:|
| 30-day buffer, original features | 0.551 |
| 30-day buffer, enriched features (window ladder, acceleration, trend slope, gaps, MRR volatility) | 0.548 |

No. Roughly twenty added features moved the score slightly down — at 74
positives, extra columns cost more in variance than they return. The binding
constraint is **data, not feature engineering**. See `docs/FEATURE_ENGINEERING.md`.

### Leakage controls

All numbers are produced under an automated audit suite (`src/audit.py`) that
must pass before any result is reported:

| Gate | Threshold | Result |
|---|---|---|
| Temporal provenance — no datetime >= cutoff, every column | 0 violations | PASS |
| Single-feature AUC | fail >= 0.80 | PASS (max 0.620) |
| Perfect separation | none | PASS |
| Identifier / row-order leakage | AUC < 0.60 | PASS |
| Duplicate rows | none | PASS |
| Constant columns | none | PASS |

The provenance gate caught a leak code review missed: 5 support tickets opened
before the cutoff but closed after it, whose resolution fields were not knowable
at prediction time. See `docs/DATA_DICTIONARY.md`.

## Key Findings

1. **The signal is reactive, not predictive.** This is the main finding — see
   the buffer table above. Anything that looked like a churn model here was
   detecting customers already on their way out.

2. **Complexity does not pay.** Both tree ensembles score below L1 logistic. At
   roughly one event per variable, regularisation is worth more than capacity —
   and no configuration clears chance once a buffer is required.

3. **Leakage was the real story, not "weak data."** Features derived from `churn_events`
   (refund amount, churn reason, reactivation) take CV AUC to **0.997** — they encode the
   answer, since a refund is issued *because* the customer left. Excluded via
   `config.POST_OUTCOME_COLS` and enforced by the audit suite.

4. **The label itself is inconsistent.** `churn_flag` and the `churn_events` table
   disagree for 312 of 500 accounts. The event log is used as ground truth because it
   carries dates.

## Honest limitations

- **I would not deploy this.** At a 30-day buffer the model is not distinguishable from
  chance (p = 0.25). Shipping it would mean sending CSMs a call list ordered by noise.
  The recommendation is to fix the data, not to tune the model.
- 168 accounts / 74 positives. Even the buffer-free variant has a CI whose lower bound
  sits on 0.50.
- A single cutoff date. Production evaluation needs rolling-origin backtesting.
- Source data has integrity problems (1,077 tickets predate their account's signup;
  19,128 usage rows predate their subscription's start) — surfaced by
  `clean.integrity_report`, not silently ignored.
- Synthetic data caps feature-target association at |r| = 0.28. Real telemetry typically
  reaches 0.3–0.6, where AUC 0.75+ is achievable with this feature set.

See `notebooks/06_leakage_quantification.py` and `docs/ASSUMPTIONS.md` for the
full audit, including the bugs found in the first pass and how they were corrected.
