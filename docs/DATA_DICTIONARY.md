# Field-Level Availability Audit

Every raw column classified by whether its value is knowable **at the prediction
cutoff** (2024-06-30). This is the check that decides what may become a feature.

Legend:
- **OK** — observable at the cutoff, safe to use
- **CENSOR** — the row exists before the cutoff but this field resolves later
- **EXCLUDE** — describes the outcome; using it leaks the label
- **ID** — identifier, never a feature

## accounts.csv

| Column | Verdict | Reasoning |
|---|---|---|
| `account_id` | ID | Join key. Tested for ordering leakage (AUC 0.52). |
| `account_name` | ID | Free text, no signal, dropped. |
| `industry` | OK | Set at signup, static. |
| `country` | OK | Set at signup, static. |
| `signup_date` | OK | Known at signup; drives `days_since_signup`. |
| `referral_source` | OK | Set at signup. |
| `plan_tier` | OK | Initial plan, known at signup. |
| `seats` | OK | Current licensed seats. |
| `is_trial` | OK | Observable state. |
| `churn_flag` | **EXCLUDE** | The outcome. Also undated, so it cannot be placed relative to any cutoff, and it disagrees with `churn_events` for 312/500 accounts. Not used as the target — see `labeling.py`. |

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

**Measured impact**: restoring these features takes CV AUC from 0.611 to
**0.997**. That is the signature of label reconstruction, not model quality.
Enforced by `config.POST_OUTCOME_COLS`, which `model.prep_xy` drops
unconditionally.

## Automated enforcement

`src/audit.py` turns the reasoning above into assertions that run over the
built matrix (`notebooks/07_leakage_audit.py`):

| Gate | Threshold | Current |
|---|---|---|
| Temporal provenance | no datetime value >= cutoff, **all** columns | PASS, 7 columns |
| Single-feature AUC | fail >= 0.80, warn >= 0.70 | PASS, max 0.650 |
| Perfect separation | none | PASS |
| Identifier / row-order leakage | AUC < 0.60 | PASS (0.52, 0.54) |
| Duplicate rows | none | PASS |
| Constant columns | none | PASS |

The temporal gate checks *every* datetime column rather than the one used for
filtering. That is what caught the `closed_at` censoring issue, which reading the
code had missed.
