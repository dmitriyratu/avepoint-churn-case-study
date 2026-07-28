# Field-Level Availability Audit

Every raw column classified by whether its value is knowable **at the prediction
cutoff** (2024-06-30). This is the check that decides what may become a feature.

Legend:
- **OK** — observable at the cutoff, safe to use
- **CENSOR** — the row exists before the cutoff but this field resolves later
- **STALE** — a real value, but as of data extraction rather than as of the
  cutoff; a point-in-time substitute is built instead
- **EXCLUDE** — describes the outcome; using it leaks the label
- **ID** — identifier, never a feature

## accounts.csv

The accounts table carries **no as-of date**. That makes every mutable column on
it suspect: the value present is the value at extraction (2024-12-31), which is
after any cutoff modelled here. Static-at-signup columns are fine; state columns
are not.

| Column | Verdict | Reasoning |
|---|---|---|
| `account_id` | ID | Join key. Tested for ordering leakage. |
| `account_name` | ID | Free text, no signal, dropped. |
| `industry` | OK | Set at signup, immutable. |
| `country` | OK | Set at signup, immutable. |
| `signup_date` | OK | Known at signup; drives `days_since_signup`. |
| `referral_source` | OK | Set at signup, immutable. |
| `plan_tier` | OK | *Initial* plan, so it is a signup-time fact. The current plan is taken from the truncated subscription history as `latest_plan_tier`. |
| `seats` | **STALE** | Documented as current licensed seats. It matches the seat count on the account's latest pre-cutoff subscription only **51.6%** of the time, which confirms it carries a later value. Replaced by `latest_seats`, built from truncated subscriptions — including in the per-seat normalisations, which inherited the problem. |
| `is_trial` | **STALE** | Same reasoning; matches the latest pre-cutoff subscription 70.1% of the time. Replaced by `latest_is_trial` and `n_trial_subs`. |
| `churn_flag` | **EXCLUDE** | The outcome. Also undated, so it cannot be placed relative to any cutoff, and it is statistically unrelated to `churn_events` — 188/500 agreement against 193 expected by chance, κ = −0.016, p = 0.56. Not used as the target — see `labeling.py`. |

Both **STALE** columns are enforced by name through
`config.POINT_IN_TIME_UNSAFE_COLS` and `audit.forbidden_columns`, not by a
statistical gate — `churn_flag`'s own single-feature AUC is ~0.51, so a
threshold test would never have caught it.

## subscriptions.csv

| Column | Verdict | Reasoning |
|---|---|---|
| `subscription_id` | ID | Join key to usage. |
| `account_id` | ID | Join key. |
| `start_date` | OK | Rows with `start_date >= cutoff` are removed entirely. |
| `end_date` | **CENSOR** | An end date after the cutoff has not happened yet; set to `NaT`. 90.3% null overall, which is *structural* (subscription still open), not missing data. |
| `plan_tier` | OK | Observable. |
| `seats` | OK | Observable; drives `seat_growth`. |
| `mrr_amount` | OK | Billed amount, observable. |
| `arr_amount` | **DROP** | Equals `mrr_amount * 12` for all 5,000 rows. Perfectly collinear, zero added information. |
| `is_trial` | OK | Observable. |
| `upgrade_flag` | OK | Records a past plan change. |
| `downgrade_flag` | OK | Records a past plan change. |
| `churn_flag` | **EXCLUDE** | The label at subscription grain. Feeds `n_churned_subs` / `sub_churn_rate` in the original design — both removed. |
| `billing_frequency` | OK | Observable. |
| `auto_renew_flag` | OK | Observable setting. |

## feature_usage.csv

| Column | Verdict | Reasoning |
|---|---|---|
| `usage_id` | ID | 21 duplicates dropped so event counts are not inflated. |
| `subscription_id` | ID | Bridge to `account_id`. |
| `usage_date` | OK | Rows at or after the cutoff removed. |
| `feature_name` | OK | Drives `feature_breadth`. |
| `usage_count` | OK | Logged at event time. |
| `usage_duration_secs` | OK | Logged at event time. |
| `error_count` | OK | Logged at event time. |
| `is_beta_feature` | OK | Property of the feature. |

## support_tickets.csv

| Column | Verdict | Reasoning |
|---|---|---|
| `ticket_id` | ID | |
| `account_id` | ID | Join key. |
| `submitted_at` | OK | Filter column; rows at or after the cutoff removed. |
| `closed_at` | **CENSOR** | 5 tickets are submitted before the cutoff but closed after it. Set to `NaT`. |
| `resolution_time_hours` | **CENSOR** | Undefined for a ticket still open at the cutoff. Set to `NaN`. |
| `priority` | OK | Assigned at submission. |
| `first_response_time_minutes` | **CENSOR** | Censored when `submitted_at + minutes >= cutoff` — the response has not happened yet. |
| `satisfaction_score` | **CENSOR** | Collected at closure, so unknown for open tickets. 41.2% missing overall; imputed **inside the CV fold**, never globally. |
| `escalation_flag` | OK | Observable during the ticket's life. |

`ticket_open_at_cutoff` is added as a legitimate derived feature — "how many
tickets is this account still waiting on" is knowable and plausibly predictive.

**A quality note on `satisfaction_score` that is not a leakage question.** The
schema documents a 1–5 scale; the data contains only 3, 4 and 5, in near-equal
proportions (396 / 405 / 374). A dissatisfaction signal that cannot express
dissatisfaction is not going to predict churn, and the near-uniformity is what
an independent random draw looks like. Surfaced in `02_cleaning.py`.

## churn_events.csv — excluded wholesale from features

| Column | Verdict | Reasoning |
|---|---|---|
| `churn_event_id` | ID | |
| `account_id` | ID | Join key. |
| `churn_date` | **TARGET** | Defines the label. Events before the cutoff define cohort eligibility. |
| `reason_code` | **EXCLUDE** | A reason exists only once a customer has left. |
| `refund_amount_usd` | **EXCLUDE** | A refund is issued *because* the customer left. |
| `preceding_upgrade_flag` | **EXCLUDE** | Defined relative to a churn event. |
| `preceding_downgrade_flag` | **EXCLUDE** | Defined relative to a churn event. |
| `is_reactivation` | **EXCLUDE** | Implies a prior churn. |
| `feedback_text` | **EXCLUDE** | Written at cancellation. |

**Measured impact** (`06_leakage_quantification.py`): restoring these columns is
worth **+0.37 AUC**, taking the model from 0.42 to **0.79**.

That number is the interesting part. 0.79 is a perfectly plausible-looking AUC
for a churn model — it does not announce itself as broken the way 0.99 would. A
leak that lands in the believable range is the dangerous one, which is why these
columns are excluded **by name** in `config.POST_OUTCOME_COLS` rather than left
to a statistical gate to catch.

## Automated enforcement

`src/audit.py` turns the reasoning above into assertions that run over the
built matrix (`notebooks/07_leakage_audit.py`):

| Gate | Threshold | Current |
|---|---|---|
| Temporal provenance | no datetime value >= cutoff, **all** columns | PASS, 7 columns |
| Forbidden columns, by name | none present | PASS |
| Single-feature AUC | fail >= 0.80, warn >= 0.70 | PASS, max 0.622 |
| Perfect separation | none | PASS |
| Identifier / row-order leakage | AUC < 0.60 | PASS (0.50, 0.53) |
| Duplicate rows | none | PASS |
| Constant columns | none | PASS |

The temporal gate checks *every* datetime column rather than the one used for
filtering. That is what caught the `closed_at` censoring issue, which reading the
code had missed.

**The single-feature gate is necessary but not sufficient, and this dataset shows
why in both directions.** `total_refund_usd` scores 0.64 — a genuine leak that no
threshold would flag. And the *legitimate* maximum of 0.622 is itself no evidence
of signal: the max over 86 shuffled-label features averages 0.612
(`10_sanity_checks.py`). A gate that passes tells you nothing about whether the
features are useful, only that no single one is obviously the label.
