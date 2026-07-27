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
│   ├── 03_feature_engineering.py
│   ├── 04_modeling.py        # model ladder, permutation test, operating point
│   ├── 05_results_validation.py  # recommendations, deployment, monitoring, mentoring
│   ├── 06_audit_and_temporal_redesign.py   # why the first framing was wrong
│   └── 07_leakage_audit.py   # automated leakage + cleaning gates
├── src/
│   ├── audit.py              # automated leakage + quality gates
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
    ├── ASSUMPTIONS.md        # key decisions and their rationale
    ├── CLEANING_CHECKLIST.md # cleaning/preprocessing practices, with findings
    ├── DATA_DICTIONARY.md    # field-by-field availability-at-prediction-time audit
    └── EDA_CHECKLIST.md      # EDA practices followed, with what each one found
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
| 1 | Decision stump | 0.541 | [0.45, 0.63] |
| 2 | Logistic (L2, C=1) | 0.562 | [0.42, 0.69] |
| 3 | Logistic (L2, C=0.05) | 0.553 | [0.38, 0.69] |
| **4** | **Logistic (L1, C=0.1)** | **0.618** | **[0.50, 0.74]** |
| 5 | Random forest (depth 4) | 0.571 | [0.41, 0.68] |
| 6 | LightGBM (shallow) | 0.553 | [0.37, 0.67] |

**Permutation test** (300 label shuffles) → **p = 0.013**. Distinguishable from
chance, though the model is weak.

**Operating point** (threshold selected out-of-fold, never on the evaluation set):
threshold 0.40 → recall 0.943, precision 0.494, F1 0.648.

L1 keeps **5** features out of 62 (75 after in-fold encoding).

All preprocessing that learns from data — imputation, scaling, one-hot encoding —
runs inside the CV pipeline, so it is refit per fold. Moving the encoder in-fold
(from `pd.get_dummies` on the full frame) is what lifted the lower CI bound above
0.50. See `docs/CLEANING_CHECKLIST.md`.

### Leakage controls

All numbers above are produced under an automated audit suite (`src/audit.py`,
`notebooks/07_leakage_audit.py`) that must pass before results are reported:

| Gate | Threshold | Result |
|---|---|---|
| Temporal provenance — no datetime >= cutoff, every column | 0 violations | PASS (7 cols) |
| Single-feature AUC | fail >= 0.80 | PASS (max 0.650) |
| Perfect separation | none | PASS |
| Identifier / row-order leakage | AUC < 0.60 | PASS (0.52, 0.54) |
| Duplicate rows | none | PASS |
| Constant columns | none | PASS |

The provenance gate found a leak that code review had missed: 5 support tickets
were opened before the cutoff but closed after it, so their resolution time,
satisfaction score and first-response time were not knowable at prediction time —
and one of those columns was in the final model. Censoring them cost 0.024 AUC.
See `docs/DATA_DICTIONARY.md` for the field-by-field verdicts.

## Key Findings

1. **Tenure and plan dominate.** `days_since_signup` is the strongest single
   predictor (negative — longer-tenured accounts survive), followed by plan tier.

2. **Complexity does not pay here.** Both tree ensembles score below L1 logistic, and
   grid search doesn't close the gap. At 1.16 events per variable, hard feature
   selection is worth more than model capacity. Shipping the boosted model would have
   been a worse product for more compute.

3. **Leakage was the real story, not "weak data."** Features derived from `churn_events`
   (refund amount, churn reason, reactivation) take CV AUC to **0.997** — they encode the
   answer, since a refund is issued *because* the customer left. Excluded via
   `config.POST_OUTCOME_COLS` and enforced by the audit suite.

4. **The label itself is inconsistent.** `churn_flag` and the `churn_events` table
   disagree for 312 of 500 accounts. The event log is used as ground truth because it
   carries dates.

## Honest limitations

- 187 accounts / 88 positives. The AUC interval [0.50, 0.74] only just clears chance at the
  low end; the point estimate should never be quoted alone.
- **This is a triage ranker, not an action trigger.** At 94% recall / 49% precision it is
  useful for ordering a CSM call list where a wasted call is cheap. It is not usable where
  a false positive carries real cost (discounts, account escalation).
- A single cutoff date. Production evaluation needs rolling-origin backtesting.
- Source data has integrity problems (1,077 tickets predate their account's signup;
  19,128 usage rows predate their subscription's start) — surfaced by
  `clean.integrity_report`, not silently ignored.
- Synthetic data caps feature-target association at |r| = 0.28. Real telemetry typically
  reaches 0.3–0.6, where AUC 0.75+ is achievable with this feature set.

See `notebooks/06_audit_and_temporal_redesign.py` and `docs/ASSUMPTIONS.md` for the
full audit, including the bugs found in the first pass and how they were corrected.
