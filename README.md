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

`accounts.churn_flag` is a static flag with no date attached, so predicting it
directly has no observation window. This project models a dated, forward-looking
target instead:

```
|<------ observation ------>|<- buffer ->|<---- prediction ---->|
2023-01-01             2024-05-31    2024-06-30           2024-09-28
      features built here             30 days     label defined here
```

- **Eligible**: signed up before the cutoff, not already churned when the
  prediction window opens → 168 accounts
- **Label**: first churn event within 90 days of the prediction start → 50
  positives (30%)
- **Features**: computed only from rows dated before the feature cutoff

Both the horizon (90 days) and the buffer (30 days) are swept rather than
assumed — see the table below.

## Headline result

**No operationally sensible configuration beats chance on this dataset.**

Two dials, swept independently, prediction window opening 2024-06-30 throughout:

| Horizon | Buffer | n | Positives | CV ROC-AUC | Permutation p |
|---:|---:|---:|---:|---:|---:|
| 30 d | 0 | 187 | 25 | 0.402 | 0.72 |
| 30 d | 30 | 168 | 22 | 0.472 | 0.38 |
| 60 d | 0 | 187 | 45 | 0.527 | 0.49 |
| 60 d | 30 | 168 | 39 | 0.489 | 0.45 |
| 90 d | 0 | 187 | 59 | 0.569 | 0.23 |
| **90 d** | **30** | **168** | **50** | **0.469** | **0.78** |
| 180 d | 0 | 187 | 88 | **0.615** | **0.020** |
| 180 d | 30 | 168 | 74 | 0.545 | 0.25 |

- **Horizon** — how far forward the label looks ("churn in the next N days").
  30–90 days is the normal range for SaaS churn.
- **Buffer** — lead time the model must give. No buffer is the usual default;
  a buffer matters when you cannot act on a score the day it lands.

Exactly one cell clears chance: a **180-day horizon with zero lead time**. And
that cell is not really a churn model — at 180 days the positive rate is 47% and
the dominant feature is `days_since_signup`, so it is predicting *"this newish
account will probably be gone within six months"*. A survivorship base rate, not
a behavioural early warning. Add 30 days of lead time and it goes (p = 0.25).

The project defaults to **90-day horizon, 30-day buffer** — the configuration a
retention team would actually use — and reports that it does not work.

**Caveat, stated plainly:** the short-horizon rows are underpowered (25 positives
at 30 days), so a modest real effect could not be detected there. The defensible
reading is "no signal at 90 days, where 59 positives give reasonable power",
not a confident negative at 30.

## Model ladder (90-day horizon, 30-day buffer)

No rung separates from the prior-only floor, so the ordering below is the
outcome of a coin-flipping contest and should not be read as a ranking.

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
