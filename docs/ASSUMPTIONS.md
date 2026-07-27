# Assumptions & Decisions

Every decision below is backed by a check in `notebooks/02_cleaning.py` or
`notebooks/06_audit_and_temporal_redesign.py`. Where an earlier version of this
project stated a rationale the data does not support, that is recorded too.

## Problem framing

- **The target is forward-looking, not static.** `accounts.churn_flag` is a
  point-in-time flag with no date attached, so it cannot be predicted without
  leaking the future. The modelled target is instead:

  > Did the account's first churn event fall within 180 days after 2024-06-30?

  Features come only from rows dated before that cutoff. Accounts that had
  already churned at the cutoff are excluded — they are not at risk.

- **Cutoff = 2024-06-30, horizon = 180 days** balances a usable observation
  window (18 months) against a cohort large enough to model (187 accounts,
  88 positives). Alternatives were checked in notebook 06.

- **`churn_events` is treated as ground truth over `churn_flag`.** The two
  disagree for 312 of 500 accounts (37.6% agreement). The event log wins because
  it carries dates, which the flag does not. This should be confirmed with
  whoever owns the upstream pipeline.

## Data

- **Reference date is derived, not hardcoded.** An earlier version hardcoded
  `2025-07-21` while the data ends `2024-12-31`, which silently zeroed out every
  30- and 90-day recency window. All time arithmetic is now relative to
  `config.CUTOFF_DATE`.

- **`arr_amount` is dropped.** It equals `mrr_amount * 12` for all 5,000 rows —
  perfectly collinear and worthless to any model.

- **21 duplicate `usage_id` rows are dropped** so per-account event counts are
  not inflated.

- **Satisfaction score (41% missing): global median plus a missing indicator.**
  An earlier version imputed the per-priority median, justified by "response
  rates differ by ticket severity." The data does not support that: missing rates
  are 0.405–0.422 across all four priorities, so the per-priority median is the
  global median. A t-test also shows missingness is unrelated to churn
  (p = 0.81), so imputation is safe — but the indicator is kept because
  "did not respond" is free to encode.

- **Accounts with no tickets or no usage** get 0 for count and rate features.
  Recency features are the exception: "never used the product" is not the same as
  "used it today", so those are filled with the observation-window length.

- **Known integrity problems, surfaced not silenced** (`clean.integrity_report`):
  1,077 of 2,000 tickets predate their account's signup date, and 19,128 of
  24,979 usage rows predate their subscription's start. These are artefacts of
  the synthetic generator. In a real engagement they would go back to data
  engineering before any modelling.

## Leakage controls

- **All `churn_events`-derived features are excluded from the model**
  (`config.POST_OUTCOME_COLS`). Refund amount, churn reason, and reactivation
  flags describe the outcome — a refund is issued *because* the customer left.
  Including them takes CV AUC from 0.635 to **0.997**, which is the signature of
  label leakage rather than a good model. They are retained in the frame for
  post-hoc analysis only.

- **`subscriptions.churn_flag` is excluded** for the same reason: it is the label
  at a different grain.

- **Event tables are truncated before aggregation**, not filtered afterwards, so
  no post-cutoff row can reach a feature.

## Modelling

- **Repeated stratified CV (5 folds x 10 repeats), not a single holdout.** With
  187 rows a single split is not a measurement — fold-to-fold AUC ranges from
  0.44 to 0.76. All reported scores carry a 95% interval.

- **The decision threshold is chosen out-of-fold.** An earlier version tuned the
  threshold on the test set and then reported test-set F1 and recall, which is
  optimistically biased.

- **Class weighting over resampling.** The cohort is 47% positive, so imbalance is
  mild; `class_weight="balanced"` is sufficient and avoids the synthetic-sample
  artefacts SMOTE introduces with mixed feature types.

- **Model selection is a ladder, not a single choice.** Prior -> stump ->
  logistic (L2, two strengths) -> logistic (L1) -> random forest -> LightGBM,
  all on identical folds. L1 logistic wins at 0.635; neither ensemble beats it,
  and a 54-point LightGBM grid search does not close the gap. With 1.16 events
  per variable this is the expected outcome, and it is the reason the extra
  capacity is not shipped.

- **Significance is tested, not assumed.** A 300-shuffle permutation test gives
  p = 0.013 against a null mean of 0.494.

## Known limitations

1. **Cohort size.** 187 accounts / 88 positives. The AUC confidence interval is
   roughly [0.44, 0.76]; the point estimate should not be quoted alone.
2. **A single cutoff date.** Production evaluation needs rolling-origin
   backtesting across several cutoffs.
3. **No nested CV**, so the reported score does not include
   hyperparameter-selection variance.
4. **76 candidate features on 88 positives** is over-parameterised before a model
   is even fit. L1 reduces this to 8 in practice.
5. **Synthetic data.** Feature-target associations top out at |r| = 0.28. On real
   product telemetry, 0.3–0.6 is typical and AUC of 0.75+ is a reasonable target
   with this feature set.

## Scope

- Account-level binary classification over a fixed horizon. Survival modelling
  (Cox or discrete-time hazard) would use *when* a customer churns rather than
  only *whether*, and suits this problem better — noted as next work.
- NLP over `churn_events.feedback_text` is out of scope but would likely help
  segment the ~25% of events with reason code `unknown`.
