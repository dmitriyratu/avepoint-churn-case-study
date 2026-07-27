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
│   ├── 07_leakage_audit.py   # automated leakage + cleaning gates
│   └── 08_diagnostics.py     # error analysis, learning curves, why 0.58
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

- **Eligible**: signed up before the cutoff, **holding a subscription still open
  at the cutoff**, and not already churned → 177 accounts. An account whose
  subscriptions had all ended cannot churn in the ordinary sense; counting it as
  a negative would pad the denominator with customers already lost.
- **Label**: first churn event within 90 days → 54 positives (30.5%). The window
  is inclusive at both ends so it matches the eligibility rule exactly — a churn
  landing on the opening day counts as a positive rather than falling through.
- **Features**: only from rows dated before the cutoff, with fields that
  *resolve* after it censored

90 days is the usual operational horizon for SaaS churn. Both the horizon and
the lead-time buffer are swept rather than assumed — see Robustness below.

## Results

Model ladder, repeated stratified CV (5 folds × 10 repeats, identical folds):

| Rung | Model | CV ROC-AUC | 95% CI |
|------|-------|-----------|--------|
| 0 | Prior (no features) | 0.500 | — |
| 1 | Decision stump | 0.516 | [0.37, 0.64] |
| **2** | **Logistic (L2, C=1)** | **0.581** | **[0.37, 0.75]** |
| 3 | Logistic (L2, C=0.05) | 0.564 | [0.40, 0.77] |
| 4 | Logistic (L1, C=0.1) | 0.537 | [0.39, 0.69] |
| 5 | Random forest (depth 4) | 0.554 | [0.40, 0.75] |
| 6 | LightGBM (pipelined) | 0.536 | [0.39, 0.74] |
| 7 | LightGBM (native NaN + categoricals) | 0.536 | [0.40, 0.74] |
| 8 | HistGradientBoosting (native NaN) | 0.552 | [0.33, 0.73] |

**Giving the boosters a fair shot.** Routing LightGBM through the same
`SimpleImputer` + `OneHotEncoder` as the linear models handicaps it: gradient
boosters learn a split direction for missingness rather than needing it filled,
and take native categorical splits rather than a one-hot expansion that fragments
the feature. Rung 7 passes raw `NaN` and pandas `category` dtype straight through.

It helps — 0.517 → 0.531 — and a 54-point grid search adds a little more, to
0.541. **It still does not beat a plain regularised linear model.**

One methodological trap worth naming: `GridSearchCV.best_score_` for that tuned
model reads **0.586**, which looks near-parity with logistic. That number is
selection-inflated — it is the score on the very folds used to pick the
hyperparameters. Re-scored on the ladder's independent folds it drops to 0.541.
Comparing a tuned model's `best_score_` against another model's honest CV score
is a common and invisible way to make the complex model look better than it is.

**Selected: L2 logistic regression** — not because boosting was denied a fair
attempt, but because it was given one and lost. At ~0.8 events per variable the
ensembles have far more capacity than 59 positives can support.

**Operating point** — threshold chosen out-of-fold, favouring recall because a
missed churner costs more than a wasted outreach call:

| | |
|---|---|
| Recall | **0.704** |
| Precision | 0.373 |
| F1 | 0.487 |
| Base rate | 0.305 |

**Permutation test** (300 label shuffles): **p = 0.103**. Not significant.
Reported as-is rather than rounded in either direction.

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

### Why 0.58 and not higher — diagnosed, not guessed

`notebooks/08_diagnostics.py` separates the fixable causes from the rest.

| Candidate cause | Evidence | Verdict |
|---|---|---|
| Data leakage | full audit suite passes, incl. a by-name forbidden-column gate | ruled out |
| Feature engineering | ~20 added features moved AUC *down*; cutting to the top 3 costs 0.06 | not the constraint |
| Overfitting | train 0.97–1.00 vs validation 0.54–0.58 | real, but a symptom |
| **Sample size** | learning curve still rising: **+0.09 AUC per 100 rows** | **primary constraint** |
| **Data quality** | label agrees with the event log 37.6% of the time | **major contributor** |
| **Irreducible** | nearest neighbours disagree at 0.410 vs 0.424 for random pairs | **large floor** |

Three results worth singling out:

**Every model memorises.** The boosters reach train AUC **1.000** on 177 rows and
validate at 0.54. Not a tuning problem — far more capacity than 54 positives can
constrain.

**Fewer features makes it worse**, monotonically (3 features → 0.525, all 73 →
0.588). So the signal is not concentrated in a few predictors drowned out by
noise; it is smeared thinly across many weak ones. That rules out the obvious
remedy.

**Neighbours are barely more alike than strangers.** Accounts adjacent in feature
space disagree on outcome 41.0% of the time, against 42.4% for two customers
picked at random. The features hardly locate a customer's risk at all — that is
the Bayes floor, and it belongs to the data, not the model.

**Error analysis**: of 14 false negatives, only 1 sits near the threshold. The
rest are scored confidently safe, so no operating point recovers them. The
churners we catch are *newer, higher-MRR, fewer-seat* accounts; the ones we miss
are *older, larger-seat, lower-MRR* accounts contracting slowly — which follows
directly from a model dominated by `days_since_signup`. That part is reducible,
with telemetry that captures large-account contraction.

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
| Forbidden columns — outcome + point-in-time-unsafe, by name | none present | PASS |
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
