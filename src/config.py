"""Shared configuration for the churn pipeline."""
import pandas as pd
from pathlib import Path

ROOT = Path(__file__).parents[1]
RAW_DIR = ROOT / "data" / "raw"
PROCESSED_DIR = ROOT / "data" / "processed"
OUTPUTS_DIR = ROOT / "outputs"

# Temporal design — see labeling.py for the window diagram.
#
# Primary framing: "will this account churn in the next 90 days, given what we
# know on 2024-06-30". 90 days is the usual operational horizon for SaaS churn.
#
# BUFFER_DAYS is the lead time the model must give. Zero is the standard default
# — you score today and act today. Non-zero variants are evaluated as a
# robustness check in docs/FEATURE_ENGINEERING.md.
CUTOFF_DATE = pd.Timestamp("2024-06-30")
BUFFER_DAYS = 0
PREDICTION_START = CUTOFF_DATE + pd.Timedelta(days=BUFFER_DAYS)
HORIZON_DAYS = 90

TARGET = "churned_next_90d"
ID_COLS = ["account_id", "account_name", "signup_date"]

# Outcome variables, dropped by model.prep_xy. The churn_events-derived columns
# describe what happened at churn (a refund is issued *because* the customer
# left); restoring them takes CV AUC to 0.996. `churn_flag` is the account-level
# outcome itself.
POST_OUTCOME_COLS = ["churn_flag", "n_churn_events", "total_refund_usd",
                     "had_reactivation", "had_preceding_downgrade",
                     "had_preceding_upgrade"]

# The accounts table carries no as-of date, and the dataset README describes
# these two as current state — meaning as of extraction (2024-12-31), which is
# after any cutoff we model. `accounts.seats` matches the seat count on the
# account's latest pre-cutoff subscription only 51.6% of the time, confirming it
# reflects a later value. Point-in-time equivalents built from the truncated
# subscription history are used instead: `latest_seats` and `n_trial_subs`.
POINT_IN_TIME_UNSAFE_COLS = ["seats", "is_trial"]
