# Assumptions & Decisions

Each decision below is backed by a check in the notebooks or a gate in
`src/audit.py`. Practice-level rationale lives in `EDA_CHECKLIST.md`,
`CLEANING_CHECKLIST.md` and `FEATURE_ENGINEERING.md`.

## Problem framing

- **The target is a dated, forward-looking event.** `accounts.churn_flag` has no
  date attached, so it cannot be placed relative to a cutoff. The modelled target
  is instead: *did the account's first churn event fall within 180 days of
  2024-06-30?*

- **A 30-day buffer separates the feature cutoff (2024-05-31) from the start of
  the prediction window (2024-06-30).** Without it a model keys on the collapse
  in activity that immediately precedes churn — accurate in cross-validation,
  useless in production because the customer has already gone. The buffer is also
  the lead time a CSM needs to act.

- **`churn_events` is ground truth over `churn_flag`.** The two agree for only
  37.6% of accounts. The event log wins because it carries dates. This should be
  confirmed with whoever owns the upstream pipeline.

- **Eligible accounts** signed up before the cutoff, held a subscription still
  open at the cutoff, and had not already churned when the prediction window
  opened: 177 accounts, 54 positives (30.5%). The open-subscription requirement
  matters — 10 accounts had no live subscription and cannot churn in the ordinary
  sense, so counting them as negatives would pad the denominator with customers
  already lost.

- **The label window is inclusive at both ends**, matching the eligibility rule.
  Eligibility keeps churn dates on or after the window opens, so a churn landing
  on the opening day must count as a positive rather than falling through as a
  zero.

- **`accounts.seats` and `accounts.is_trial` are excluded as features.** The
  accounts table carries no as-of date and the dataset README describes both as
  current state — meaning as of extraction, after any cutoff we model.
  `accounts.seats` matches the seat count on the account's latest pre-cutoff
  subscription only 51.6% of the time, confirming it reflects a later value.
  Point-in-time equivalents built from the truncated subscription history are
  used instead (`latest_seats`, `n_trial_subs`), including for the per-seat
  normalisations.

- **`accounts.churn_flag` is excluded as a feature.** It is the account-level
  outcome. It reached the model in an earlier revision and scored a
  single-feature AUC of 0.51, which is exactly why the statistical gates did not
  catch it — categorically the outcome, statistically invisible.
  `audit.forbidden_columns` now checks the exclusion lists by name.

## Data

- **`arr_amount` dropped** — equals `mrr_amount * 12` for all 5,000 rows.

- **21 duplicate `usage_id` rows dropped** so per-account event counts are not
  inflated.

- **Satisfaction score (41% missing) is imputed inside the CV fold, not during
  cleaning.** A median computed over the whole table would let validation rows
  influence the statistic applied to training rows. Missingness is unrelated to
  churn (t-test p = 0.81) and flat across ticket priority (0.405–0.422), which is
  consistent with MCAR, so median imputation is defensible. A
  `satisfaction_missing` indicator is retained regardless.

- **Missing does not mean zero.** Three families, three treatments:

  | Family | Missing means | Fill |
  |---|---|---|
  | counts (`n_tickets`, `total_usage_events`) | genuinely zero activity | `0` |
  | recency (`days_since_last_usage`) | never happened — maximally stale | observation-window length |
  | rates (`avg_satisfaction`, `error_rate`) | unknown | `NaN`, imputed in-fold |

  An account with no tickets has an undefined average satisfaction; filling it
  with 0 invents a maximally unhappy customer.

- **Missingness disposition keys on cause, not percentage.**
  `subscriptions.end_date` is 90.3% null, which a ">60% missing, drop it" rule
  would discard — but that null is structural (the subscription is open) and is
  among the most informative fields available. Encoded as `n_open_subs` /
  `pct_subs_ended`.

- **Source-data integrity problems are surfaced, not repaired**
  (`clean.integrity_report`): 1,077 of 2,000 tickets predate their account's
  signup, and 19,128 of 24,979 usage rows predate their subscription's start.
  Generator artefacts that cannot be fixed without inventing data; in a real
  engagement they go back to data engineering first.

## Leakage controls

- **All `churn_events`-derived columns are excluded** (`config.POST_OUTCOME_COLS`).
  Refund amount, churn reason and reactivation describe the outcome — a refund is
  issued *because* the customer left. Restoring them takes CV AUC to **0.996**,
  which is the signature of label reconstruction rather than a good model.

- **`subscriptions.churn_flag` is excluded** — the label at a different grain.

- **Tables are truncated before aggregation**, so no post-cutoff row can reach a
  feature.

- **Fields that resolve after the cutoff are censored, not merely filtered.**
  A ticket opened before the cutoff and closed after it still carries a
  resolution time, satisfaction score and first-response time that nobody could
  know at the cutoff. `labeling.truncate_tables` nulls those and adds
  `ticket_open_at_cutoff`, which *is* observable. This leak was found by
  `audit.temporal_provenance`, not by reading the code.

- **Categoricals are encoded inside the CV pipeline**, never with
  `pd.get_dummies` on the full frame. Encoding up front learns category levels
  from validation rows and offers no `handle_unknown` path, so an unseen
  `industry` would change the column set at serving time.

- **Checks are automated and asserted** (`src/audit.py`): temporal provenance
  across every datetime column, single-feature AUC, perfect separation,
  identifier and row-order leakage, duplicate rows, constant columns. The suite
  must pass before any score is reported.

## Modelling

- **Repeated stratified CV (5 folds x 10 repeats), not a single holdout.** With
  168 rows a single split is not a measurement. All scores carry a 95% interval.

- **The decision threshold is chosen out-of-fold**, never on the data used to
  report the score.

- **Class weighting over resampling.** At 44% positive the imbalance is mild;
  `class_weight="balanced"` avoids the artefacts SMOTE introduces with mixed
  feature types.

- **Model selection is a ladder**: prior → stump → logistic (L2, two strengths) →
  logistic (L1) → random forest → LightGBM → LightGBM with native NaN and
  categorical handling → HistGradientBoosting → tuned LightGBM, all on identical
  folds. L2 logistic leads at 0.595.

- **The boosters are not handicapped.** Routing LightGBM through the linear
  models' `SimpleImputer` + `OneHotEncoder` would deny it two things it does well:
  learning a split direction for missingness, and native categorical splits.
  Rung 7 passes raw `NaN` and pandas `category` dtype through instead
  (`model.AsCategory`, with categories learned per fold). That is worth +0.014,
  and tuning a further +0.010 — still short of the linear model.

- **Tuned scores are re-scored on the ladder's folds.** `GridSearchCV.best_score_`
  for the tuned booster reads 0.586, but that is the score on the folds used to
  select the hyperparameters. On independent folds it is 0.541. Quoting the
  former against another model's honest CV score is a common way to make the
  complex model look better than it is.

- **Significance is tested, not assumed.** A 300-shuffle permutation test gives
  p = 0.25 at the 30-day buffer — the model does not beat chance.

## Conclusion and limitations

1. **There is no actionable churn signal in this dataset.** The apparent signal
   exists only with no buffer (AUC 0.616, p = 0.025) and disappears once any
   realistic lead time is required (p = 0.25 at 15 days and beyond).
2. **I would not deploy this model.** Ranking a CSM call list by noise is worse
   than not ranking it.
3. **The constraint is data, not method.** 168 accounts, 74 positives, usage logs
   with incoherent timestamps, and a label that disagrees with its own event log
   for 62% of accounts. Roughly twenty additional engineered features moved the
   score slightly *down*.
4. **Single cutoff, no nested CV.** A production evaluation needs rolling-origin
   backtesting across several cutoffs, and a nested loop so the reported score
   includes hyperparameter-selection variance.

## Scope

Account-level binary classification over a fixed horizon. A discrete-time hazard
model would use *when* rather than only *whether* and would handle the censoring
this data is full of — the natural next step. NLP over
`churn_events.feedback_text` is out of scope but could segment the ~25% of events
coded `unknown`.
