"""Shared configuration for the churn pipeline."""
import pandas as pd
from pathlib import Path

ROOT = Path(__file__).parents[1]
RAW_DIR = ROOT / "data" / "raw"
PROCESSED_DIR = ROOT / "data" / "processed"
OUTPUTS_DIR = ROOT / "outputs"

# Temporal design — see labeling.py for the window diagram.
CUTOFF_DATE = pd.Timestamp("2024-05-31")
BUFFER_DAYS = 30
PREDICTION_START = CUTOFF_DATE + pd.Timedelta(days=BUFFER_DAYS)
HORIZON_DAYS = 180

TARGET = "churned_next_180d"
ID_COLS = ["account_id", "account_name", "signup_date"]

# Derived from churn_events, so they describe the outcome rather than its
# precursors. Dropped by model.prep_xy; including them takes CV AUC to 0.997.
POST_OUTCOME_COLS = ["n_churn_events", "total_refund_usd", "had_reactivation",
                     "had_preceding_downgrade", "had_preceding_upgrade"]
