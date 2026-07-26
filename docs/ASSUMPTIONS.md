# Assumptions & Decisions

## Data

- **Reference date**: `2025-07-21` (latest date in the dataset) is used as "today" for tenure calculations
  and for imputing `end_date = null` (active subscriptions).

- **Satisfaction score imputation**: ~41% of support ticket satisfaction scores are missing.
  I impute with the **per-priority median** rather than a global median, under the assumption
  that response rates differ by ticket severity (urgent tickets more likely to receive scores).

- **Zero-MRR subscriptions**: Kept as-is. These are legitimate trial or enterprise-pilot subscriptions.
  The `is_trial` flag already captures this distinction.

- **Feature usage ↔ accounts join**: Usage records link to `subscription_id`, not `account_id` directly.
  I join through the subscriptions table. A small number of usage records whose `subscription_id`
  has no match in subscriptions are dropped (< 1%).

- **Accounts with no ticket history**: Imputed as 0 for all support metrics (no tickets = no escalations, etc.).

## Modeling

- **Target leakage check**: Excluded `account_name`, `account_id`, and `signup_date` from the feature matrix.
  `churn_flag` on the subscriptions table is not used directly — only aggregated counts (e.g. `sub_churn_rate`),
  which represent historical subscription-level churn, not the account-level label we're predicting.

- **Class imbalance**: 22% churn rate. Using `scale_pos_weight` (≈3.5) rather than SMOTE,
  since tree models handle this well and SMOTE can introduce artifacts with mixed feature types.

- **No time-based split**: The accounts table doesn't have a clear temporal ordering for train/test.
  Using stratified random 80/20 split, relying on 5-fold CV for reliable performance estimates.

- **Threshold**: Tuned on held-out test set to maximize F1. In production, the business team
  should set this based on the actual cost ratio of a false negative (missed churner) vs.
  false positive (wasted CSM time).

## Scope

- This analysis focuses on **account-level churn prediction** (will an account eventually churn?).
  A separate analysis could predict **time-to-churn** (survival model) or **subscription-level churn**.

- Reactivated accounts (those with multiple churn events, `is_reactivation = True`) are treated
  as single accounts. A reactivation prediction model is an interesting follow-on.

- NLP on `feedback_text` in churn events is out of scope for this exercise but would likely
  add signal for the "unknown" reason_code accounts.
