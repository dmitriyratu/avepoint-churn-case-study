"""Shared configuration for the churn pipeline."""
import pandas as pd

# "Will this account churn in the next 90 days, given what we know on
# 2024-06-30." BUFFER_DAYS is the lead time demanded of the model; zero is the
# standard default. See labeling.py for the window diagram and
# docs/FEATURE_ENGINEERING.md for the sweep over both dials.
CUTOFF_DATE = pd.Timestamp("2024-06-30")
BUFFER_DAYS = 0
PREDICTION_START = CUTOFF_DATE + pd.Timedelta(days=BUFFER_DAYS)
HORIZON_DAYS = 90

TARGET = "churned_next_90d"
ID_COLS = ["account_id", "account_name", "signup_date"]

# Outcome variables, dropped by prep_xy. A refund is issued *because* the
# customer left; restoring these is worth +0.37 AUC (06_leakage_quantification).
# Only churn_flag reaches the feature layer today — the rest are a standing
# guard, so adding a churn_events aggregate later trips the gate.
POST_OUTCOME_COLS = ["churn_flag", "n_churn_events", "total_refund_usd",
                     "had_reactivation", "had_preceding_downgrade",
                     "had_preceding_upgrade"]

# The accounts table has no as-of date and documents these as current state,
# i.e. as of extraction (2024-12-31). accounts.seats matches the latest
# pre-cutoff subscription only 51.6% of the time. Use latest_seats and
# n_trial_subs from the truncated history instead.
POINT_IN_TIME_UNSAFE_COLS = ["seats", "is_trial"]
