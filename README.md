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
│   ├── config.py             # cutoff/buffer/horizon, leakage exclusion lists
│   ├── load_data.py          # load the 5 raw tables
│   ├── clean.py              # parsing, dedup, integrity_report
│   ├── labeling.py           # cohort construction, observation-window truncation
│   ├── features/             # one module per feature family
│   │   ├── subscription.py   #   size, direction, tenure, plan movement
│   │   ├── usage.py          #   volume, breadth, recency, momentum, rhythm
│   │   ├── support.py        #   load, responsiveness, escalation, trend
│   │   ├── assemble.py       #   join blocks, prune constant/collinear
│   │   └── _helpers.py       #   window ladder, slope, safe division
│   ├── model.py              # model ladder, permutation test, oof threshold
│   ├── audit.py              # leakage + quality gates
│   └── pipeline.py           # build() — one call for the whole chain
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
from src import pipeline
from src.model import evaluate_ladder, model_ladder, save_model

data = pipeline.build(verify=True)      # asserts the leakage suite before returning
print(data.summary.to_string())
print(evaluate_ladder(data.X, data.y).to_string(index=False))

save_model(model_ladder()[4][1].fit(data.X, data.y), 'churn_l1_logistic')
"
```

`pipeline.build()` runs load → clean → cohort → truncate → features → `prep_xy`
and returns the intermediates alongside `X`/`y`. `build_at_buffer(days)` varies
the lead time, which is how the sensitivity sweep is generated.

## Problem framing

`accounts.churn_flag` is a static flag with no date attached — it cannot be
placed relative to any cutoff, and it disagrees with the `churn_events` log for
**312 of 500 accounts** (37.6% agreement). It is not a usable target.

The modelled target is the standard churn formulation instead:

> Given everything observable on **2024-06-30**, will this account churn within
> the next **90 days**?

```
|<-------- observation -------->|<---- prediction ---->|
2023-01-01                 2024-06-30            2024-09-28
       features built here             label defined here
```

- **Eligible**: signed up before the cutoff, not already churned → 187 accounts
- **Label**: first churn event within 90 days → 59 positives (31.5%)
- **Features**: only from rows dated before the cutoff, with fields that
  *resolve* after it censored

90 days is the usual operational horizon for SaaS churn. Both the horizon and
the lead-time buffer are swept rather than assumed — see Robustness below.

## Results

Model ladder, repeated stratified CV (5 folds × 10 repeats, identical folds):

| Rung | Model | CV ROC-AUC | 95% CI |
|------|-------|-----------|--------|
| 0 | Prior (no features) | 0.500 | — |
| 1 | Decision stump | 0.516 | [0.43, 0.60] |
| **2** | **Logistic (L2, C=1)** | **0.595** | **[0.45, 0.73]** |
| 3 | Logistic (L2, C=0.05) | 0.579 | [0.42, 0.73] |
| 4 | Logistic (L1, C=0.1) | 0.569 | [0.41, 0.70] |
| 5 | Random forest (depth 4) | 0.539 | [0.42, 0.68] |
| 6 | LightGBM (shallow) | 0.517 | [0.37, 0.63] |

**Selected: L2 logistic regression.** Both tree ensembles score below it — at
~0.8 events per variable the ensembles have far more capacity than 59 positives
can support, and regularisation is worth more than boosting. A 54-point LightGBM
grid search does not close the gap.

**Operating point** — threshold chosen out-of-fold, favouring recall because a
missed churner costs more than a wasted outreach call:

| | |
|---|---|
| Recall | **0.864** |
| Precision | 0.359 |
| F1 | 0.508 |
| Base rate | 0.315 |

**Permutation test** (300 label shuffles): **p = 0.086**. Weak evidence of
signal — trending, but not significant at conventional thresholds. Reported as
such rather than rounded in either direction.

## Robustness — where this breaks

The result above is the best honest case. It does not survive stress-testing.

### Horizon and lead time

Prediction window opening 2024-06-30 throughout:

| Horizon | Buffer | Positives | CV ROC-AUC | Permutation p |
|---:|---:|---:|---:|---:|
| 30 d | 0 | 25 | 0.402 | 0.72 |
| 60 d | 0 | 45 | 0.527 | 0.49 |
| **90 d** | **0** | **59** | **0.595** | **0.086** |
| 90 d | 30 | 50 | 0.469 | 0.78 |
| 180 d | 0 | 88 | 0.615 | 0.020 |
| 180 d | 30 | 74 | 0.545 | 0.25 |

**Buffer** = lead time the model must give. Zero is the standard default (score
today, act today); a non-zero buffer forces the model to warn you *before* the
customer is visibly on the way out.

Two things to take from this:

1. **Only the 180-day/no-buffer cell is significant** — and at 180 days the
   positive rate is 47% and the model is dominated by `days_since_signup`. It
   predicts *"this newish account will probably be gone within six months"*: a
   survivorship base rate, not a behavioural early warning.
2. **Requiring 30 days of lead time kills it at every horizon.** Much of the
   apparent signal is the customer already visibly leaving.

### Leakage is worth more than the model

| Design | Features may see | CV ROC-AUC |
|---|---|---:|
| A. correct | observation window only | 0.595 |
| B. leaky | + rows dated after the cutoff | 0.634 |
| C. leaky | + columns derived from `churn_events` | **0.996** |

Design C is the one to recognise: refund amount and churn reason reconstruct the
label, because a refund is issued *because* the customer left. **An AUC near 1.0
on a churn problem is a bug report, not a result.**

### Feature engineering did not help

| | CV ROC-AUC |
|---|---:|
| Baseline feature set | 0.551 |
| + window ladder, acceleration, trend slope, gaps, MRR volatility (~20 features) | 0.548 |

At this sample size extra columns cost more in variance than they return. The
binding constraint is **data, not features**.

### What I would tell the business

Deploy it as a **CSM triage list**, not an automated trigger — at 86% recall and
36% precision it is useful for ordering a call list where a wasted call is cheap,
and unusable where a false positive carries real cost. Revisit once the data
problems below are fixed.

### Leakage controls

Every number above is produced under an automated audit suite (`src/audit.py`) that
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

## Key findings

1. **The label had to be redefined.** `churn_flag` is undated and agrees with the
   event log for only 37.6% of accounts. Found by counting, not modelling.
2. **Simpler beat more complex.** L2 logistic outscored both tree ensembles; a
   54-point grid search on LightGBM did not close the gap.
3. **The signal does not survive a lead-time requirement.** Ask for 30 days of
   warning and every horizon drops to chance.
4. **Post-outcome columns are worth +0.40 AUC.** Excluded and enforced by an
   automated gate, because that is what a leak looks like from the inside.
5. **More feature engineering did not help.** ~20 additional engineered features
   moved the score slightly down.

## Honest limitations

- 187 accounts, 59 positives. Every interval is wide, and p = 0.086 is trending,
  not significant. The point estimate should not be quoted alone.
- Short horizons are underpowered — 25 positives at 30 days cannot detect a
  modest effect, so that row is "cannot tell", not "no signal".
- Single cutoff. Production evaluation needs rolling-origin backtesting.
- No nested CV, so the reported score excludes hyperparameter-selection variance.
- Source data has integrity problems: 1,077 of 2,000 tickets predate their
  account's signup, 19,128 of 24,979 usage rows predate their subscription's
  start. Surfaced by `clean.integrity_report`, not silently repaired.

See `notebooks/06_leakage_quantification.py` and `docs/ASSUMPTIONS.md` for the
full audit, including the bugs found in the first pass and how they were corrected.
