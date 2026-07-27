# Assumptions & Decisions

Each decision below is backed by a check in the notebooks or a gate in
`src/audit.py`. Practice-level rationale lives in `EDA_CHECKLIST.md`,
`CLEANING_CHECKLIST.md` and `FEATURE_ENGINEERING.md`.

## Problem framing

- **The target is a dated, forward-looking event.** `accounts.churn_flag` has no
  date attached, so it cannot be placed relative to a cutoff. The modelled target
  is instead: *did the account's first churn event fall within 90 days of
  2024-06-30?* 90 days is the standard operational horizon for SaaS churn, and
  it was chosen before the sweep rather than picked from it.

- **No buffer in the primary framing** (`BUFFER_DAYS = 0`): score today, act
  today, which is the usual default. A buffer demands lead time by pulling the
  feature cutoff back, and buffered variants are swept in
  `FEATURE_ENGINEERING.md` — that sweep is where the most important negative
  result lives, so the dial is reported rather than assumed in either direction.

- **`churn_events` is ground truth over `churn_flag`.** The two agree for only
  37.6% of accounts. The event log wins because it carries dates. This should be
  confirmed with whoever owns the upstream pipeline.

- **Eligible accounts** signed up before the cutoff, held a subscription still
  open at the cutoff, and had not already churned when the prediction window
  opened: **177 accounts, 54 positives (30.5%)**, from 347 that signed up in
  time.

- **The cohort rule is the weakest link in the framing, and it is not neutral.**
  Eligibility is defined off the *subscriptions* table; the label is defined off
  *churn_events*; and those two sources agree on only 58% of accounts. The
  consequence is concrete: of 335 accounts holding a live subscription at the
  cutoff, **158 (47%) are dropped for having a churn event beforehand — while
  still holding that open subscription.** The exclusion is defensible under the
  chosen label (an account already churned cannot churn again in the ordinary
  sense), but it means the label's own incoherence selects the study population.
  `10_sanity_checks.py` scores three alternative definitions; none rescues the
  result, which is the reason this is documented rather than changed.

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
  churn (t-test p = 0.63) and flat across ticket priority (0.405–0.422), which is
  consistent with MCAR, so median imputation is defensible. A
  `satisfaction_missing` indicator is retained regardless.

- **`satisfaction_score` cannot express dissatisfaction.** Documented as a 1–5
  scale; the data contains only 3, 4 and 5, near-uniformly (396 / 405 / 374).
  Kept, because a flat feature costs nothing, but it should not be expected to
  predict anything and its absence from the model is not evidence about support
  quality.

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
  issued *because* the customer left. Restoring them takes CV AUC from 0.54 to
  **0.91** (`06_leakage_quantification.py`), which is the signature of label
  reconstruction rather than a good model.

- **Exclusion is by provenance, not by effect size.** `n_churn_events` scores
  0.92 alone and any threshold gate catches it. `total_refund_usd` scores 0.64
  and no reasonable threshold would — yet it is just as invalid. Hence
  `audit.forbidden_columns` checking a list of names.

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
  177 rows a single split is not a measurement. All scores carry a 95% interval.

- **Predictions for the operating point are out-of-fold; the threshold is not.**
  `oof_threshold` maximises F1 over the out-of-fold scores and reports at that
  threshold on the same rows, which is mildly optimistic. Measured rather than
  asserted away in `04_modeling.py`: picking the threshold on a held-out half
  costs **0.049 F1**. Stated here because an earlier version of this document
  claimed the threshold was chosen on data not used for reporting, and it is not.
  The threshold itself comes off the PR curve's own breakpoints
  (`model.best_f1_threshold`) rather than a fixed grid.

- **Class weighting over resampling.** At 30.5% positive the imbalance is mild;
  `class_weight="balanced"` avoids the artefacts SMOTE introduces with mixed
  feature types. XGBoost takes `scale_pos_weight` instead, derived from the
  cohort actually being fitted rather than hardcoded — otherwise a sweep at a
  different horizon silently weights it for the wrong base rate.

- **Model selection is a ladder**: prior → stump → logistic (L2, two strengths) →
  logistic (L1) → random forest → LightGBM → LightGBM with native NaN and
  categorical handling → XGBoost → HistGradientBoosting, all on identical folds,
  plus a tuned LightGBM re-scored on those same folds. L2 logistic leads at
  **0.583** — and **no rung clears chance at the lower bound of its interval**,
  which the notebook says out loud rather than ranking regardless.

- **The boosters are not handicapped.** Routing LightGBM through the linear
  models' `SimpleImputer` + `OneHotEncoder` would deny it two things it does well:
  learning a split direction for missingness, and native categorical splits.
  Rung 7 passes raw `NaN` and pandas `category` dtype through instead
  (`model.AsCategory`, with categories learned per fold). It is worth about
  +0.001 here, and a 54-point grid search a further +0.019 — still short of the
  linear model, and all of it inside noise.

- **Tuned scores are re-scored on the ladder's folds** rather than quoted from
  `GridSearchCV.best_score_`. The reason is *not* the textbook one. `best_score_`
  reads 0.527 and the independent re-score reads 0.563 — lower, not higher —
  because the two differ in resampling scheme (one 5-fold split vs 5×10) as well
  as in selection, and at this sample size the scheme is worth more than the
  selection. The rule that survives is the stronger one: **`best_score_` is not
  comparable to another model's CV score in either direction.** Re-score on
  shared folds or do not compare.

- **Significance is tested once, on a pre-specified configuration.** A
  300-shuffle permutation test on the primary cell gives **p = 0.076**. The
  horizon/buffer sweep deliberately carries intervals but no per-cell p-values:
  twelve tests with the smallest highlighted is the selection error documented in
  `09_classifier_sweep.py`.

- **The permutation test is not the last word, because the model was chosen.**
  It holds the estimator fixed while that estimator was picked as the ladder
  maximum. Nested CV moves selection inside the outer loop and is the figure to
  quote: **0.534 ± 0.016** against a ladder maximum of 0.583. Selection was worth
  **0.049 AUC**, and across 25 outer folds **eight of the ten rungs won at least
  once**, which is what selecting on noise looks like.

- **Nested CV is repeated over 5 outer splits, not run once.** A single 5-fold
  nested run moves by ~0.09 AUC on the seed alone, wider than the effect being
  measured, so a single-split figure is not quotable to three decimals. This is
  why `04_modeling.py` takes ~10 minutes.

- **Selection is expressed as an estimator, not a hand-rolled loop.**
  `model.ladder_search` wraps the ladder in a `GridSearchCV` whose grid is the
  rungs, so `cross_validate` over it *is* nested CV — sklearn refits the choice
  inside every outer fold and there is no bespoke loop to get wrong.

- **The single cutoff is a choice, and a costly one.** `08_diagnostics.py` pools
  four quarterly cutoffs with folds grouped by `account_id`, giving 648 rows and
  159 positives from 281 distinct accounts: **0.560, fold sd 0.034** against the
  single cutoff's sd of 0.098. Grouping is not optional — the ungrouped version
  reads 0.576, and that +0.016 is the same account appearing on both sides of a
  split. The headline remains the single-cutoff figure for comparability with
  everything else in this document, but the pooled design is the better
  measurement and should be the default in any follow-up.

## Conclusion and limitations

1. **This dataset cannot resolve whether a churn signal exists.** That is the
   claim the evidence supports, and it is weaker than either "there is signal"
   or "there is none". No cell of the horizon/buffer grid clears chance at the
   lower bound of its interval; nested CV sits at chance; the best single feature
   is indistinguishable from the best of 86 coin flips. But the positive control
   in `10_sanity_checks.py` shows a *genuinely real but weak* planted signal
   scores about the same, so a confident negative is not available either.

2. **I would not deploy this model.** Ranking a CSM call list on a score whose
   interval spans chance is worse than not ranking it.

3. **Lead time is where it breaks most clearly.** Every zero-buffer cell scores
   0.56–0.59; every cell demanding 30 or 60 days of warning drops to 0.42–0.52,
   consistently across four horizons. Whatever association exists sits in the
   period immediately before the customer leaves — too late to act on.

4. **The constraint is data, not method.** 177 accounts, 54 positives, 0.74
   events per variable, usage logs with incoherent timestamps, and a label whose
   three definitions agree on 20% of accounts. Twenty-one additional engineered
   features moved the score slightly *down*.

5. **Still a single cutoff for the headline.** `08_diagnostics.py` now pools four
   quarterly cutoffs with account-grouped folds, which roughly triples the
   positives and tightens the estimate considerably — but the reported headline
   remains the single-cutoff figure, and a production evaluation would want
   proper rolling-origin backtesting with the pooled design as the default.

6. **The label is the thing to fix first.** Not the features, not the algorithm.
   Until `churn_flag`, `churn_events` and subscription end dates agree on what
   churn means, no amount of modelling effort is recoverable.

## Scope

Account-level binary classification over a fixed horizon. A discrete-time hazard
model would use *when* rather than only *whether* and would handle the censoring
this data is full of — the natural next step. NLP over
`churn_events.feedback_text` is out of scope but could segment the ~25% of events
coded `unknown`.
