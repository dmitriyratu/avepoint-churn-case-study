# Assumptions & Decisions

Every decision below is backed by a check in `notebooks/02_cleaning.py` or
`notebooks/06_audit_and_temporal_redesign.py`, or a gate in `src/audit.py`.
Where an earlier version of this
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

- **Satisfaction score (41% missing): imputed inside the CV fold, not globally.**
  An earlier version imputed the per-priority median in the cleaning step,
  justified by "response rates differ by ticket severity." Two problems. The
  rationale is unsupported — missing rates are 0.405–0.422 across all four
  priorities, so the per-priority median *is* the global median, and a t-test
  shows missingness is unrelated to churn (p = 0.81). And imputing during
  cleaning lets validation rows influence the statistic applied to training rows.
  Cleaning now records `satisfaction_missing` and leaves the value as `NaN`;
  `model._pipe` imputes with a `SimpleImputer` fit on the training fold only.

- **Missing does not mean zero.** The first version filled every null with 0,
  which conflates three different situations:

  | Feature type | Missing means | Fill |
  |---|---|---|
  | counts (`n_tickets`, `total_usage_events`) | genuinely zero activity | `0` |
  | recency (`days_since_last_usage`) | never happened — maximally stale | observation-window length |
  | rates/means (`avg_satisfaction`, `error_rate`) | unknown, not zero | `NaN`, imputed in-fold |

  An account with no tickets has an *undefined* average satisfaction, not a
  satisfaction of zero — filling with 0 invents a maximally unhappy customer.

- **Missingness disposition depends on cause, not just percentage.**
  `subscriptions.end_date` is 90.3% null, which a naive ">60% missing, drop it"
  rule would discard. That null is *structural* — the subscription is still open —
  and is among the most informative fields in the table. Encoded as
  `n_open_subs` / `pct_subs_ended` rather than dropped.

- **Categoricals are encoded inside the CV pipeline**, not with `pd.get_dummies`
  on the full frame. Encoding up front learns category levels from validation
  rows, and offers no `handle_unknown` path, so an unseen `industry` value would
  change the column set at serving time. `OneHotEncoder(handle_unknown="ignore")`
  in a `ColumnTransformer` fixes both; it also lifted the lower CI bound from
  0.44 to 0.50. See `docs/CLEANING_CHECKLIST.md`.

- **Near-duplicate features pruned at |r| > 0.98.** `feature_breadth` was
  `unique_features_used / 40`, correlated at exactly 1.000; keeping both splits
  one effect across two coefficients. Dropped: `n_open_subs`, `feature_breadth`,
  `open_ticket_rate`.

- **Known integrity problems, surfaced not silenced** (`clean.integrity_report`):
  1,077 of 2,000 tickets predate their account's signup date, and 19,128 of
  24,979 usage rows predate their subscription's start. These are artefacts of
  the synthetic generator. In a real engagement they would go back to data
  engineering before any modelling.

## Leakage controls

- **All `churn_events`-derived features are excluded from the model**
  (`config.POST_OUTCOME_COLS`). Refund amount, churn reason, and reactivation
  flags describe the outcome — a refund is issued *because* the customer left.
  Including them takes CV AUC from 0.618 to **0.997**, which is the signature of
  label leakage rather than a good model. They are retained in the frame for
  post-hoc analysis only.

- **`subscriptions.churn_flag` is excluded** for the same reason: it is the label
  at a different grain.

- **Event tables are truncated before aggregation**, not filtered afterwards, so
  no post-cutoff row can reach a feature.

- **Fields that resolve after the cutoff are censored, not just filtered.**
  Filtering `support_tickets` on `submitted_at` is insufficient: a ticket opened
  in June and closed in July still carries `closed_at`, `resolution_time_hours`
  and `satisfaction_score` that nobody could know at the cutoff.
  `labeling.truncate_tables` sets those to `NaT`/`NaN` for the 5 affected tickets
  and censors `first_response_time_minutes` when the response lands after the
  cutoff. `ticket_open_at_cutoff` is added as a legitimate substitute — "how many
  tickets is this account still waiting on" *is* knowable.

  This leak was found by `audit.temporal_provenance`, not by reading the code.
  Measured at the time it was fixed, censoring cost 0.024 AUC (0.635 -> 0.611)
  and weakened the permutation test from p = 0.013 to p = 0.040 — part of the
  earlier result was the leak. (Current headline figures are higher again, 0.618
  at p = 0.013, because a later change moved one-hot encoding inside the CV fold.
  The two changes are independent; the leak fix on its own was a real cost.)

- **Leakage checks are automated, not manual** (`src/audit.py`). The suite gates
  temporal provenance across *every* datetime column, single-feature AUC
  (fail >= 0.80), perfect separation, identifier and row-order leakage, duplicate
  rows, and constant columns. It runs in `notebooks/07_leakage_audit.py` and is
  asserted before any score is reported. Field-by-field verdicts are in
  `docs/DATA_DICTIONARY.md`.

## Modelling

- **Repeated stratified CV (5 folds x 10 repeats), not a single holdout.** With
  187 rows a single split is not a measurement — fold-to-fold AUC ranges from
  0.44 to 0.74. All reported scores carry a 95% interval.

- **The decision threshold is chosen out-of-fold.** An earlier version tuned the
  threshold on the test set and then reported test-set F1 and recall, which is
  optimistically biased.

- **Class weighting over resampling.** The cohort is 47% positive, so imbalance is
  mild; `class_weight="balanced"` is sufficient and avoids the synthetic-sample
  artefacts SMOTE introduces with mixed feature types.

- **Model selection is a ladder, not a single choice.** Prior -> stump ->
  logistic (L2, two strengths) -> logistic (L1) -> random forest -> LightGBM,
  all on identical folds. L1 logistic wins at 0.618; neither ensemble beats it,
  and a 54-point LightGBM grid search does not close the gap. With 1.16 events
  per variable this is the expected outcome, and it is the reason the extra
  capacity is not shipped.

- **Significance is tested, not assumed.** A 300-shuffle permutation test gives
  p = 0.013 against a null mean near 0.50. Real, though the model remains weak.

## Known limitations

1. **Cohort size.** 187 accounts / 88 positives. The AUC confidence interval is
   roughly [0.50, 0.74] — it only just clears chance at the low end, so the point
   estimate should never be quoted alone.
2. **Deployment posture.** At 94% recall / 49% precision this is a triage ranker
   for CSM outreach, not an automated action trigger. It should not drive
   anything with a real cost attached to a false positive.
3. **A single cutoff date.** Production evaluation needs rolling-origin
   backtesting across several cutoffs.
4. **No nested CV**, so the reported score does not include
   hyperparameter-selection variance.
5. **75 encoded features on 88 positives** is over-parameterised before a model
   is even fit. L1 reduces this to 5 in practice.
6. **Synthetic data.** Feature-target associations top out at |r| = 0.28. On real
   product telemetry, 0.3–0.6 is typical and AUC of 0.75+ is a reasonable target
   with this feature set.

## Scope

- Account-level binary classification over a fixed horizon. Survival modelling
  (Cox or discrete-time hazard) would use *when* a customer churns rather than
  only *whether*, and suits this problem better — noted as next work.
- NLP over `churn_events.feedback_text` is out of scope but would likely help
  segment the ~25% of events with reason code `unknown`.
