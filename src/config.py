"""Central configuration derived from the data rather than hardcoded."""
import pandas as pd
from pathlib import Path

RAW_DIR = Path(__file__).parents[1] / "data" / "raw"
PROCESSED_DIR = Path(__file__).parents[1] / "data" / "processed"
OUTPUTS_DIR = Path(__file__).parents[1] / "outputs"

# Temporal design for the leakage-free cohort.
# Observation window: everything strictly before CUTOFF_DATE.
# Prediction window:  first churn in (CUTOFF_DATE, CUTOFF_DATE + HORIZON_DAYS].
CUTOFF_DATE = pd.Timestamp("2024-06-30")
HORIZON_DAYS = 180

# Columns never fed to a model.
ID_COLS = ["account_id", "account_name", "signup_date"]
TARGET = "churned_next_180d"

# Features derived from the churn_events table describe an outcome that has
# already happened. They are excluded from the modeling matrix and kept only
# for post-hoc analysis. See docs/ASSUMPTIONS.md.
POST_OUTCOME_COLS = [
    "n_churn_events", "total_refund_usd", "had_reactivation",
    "had_preceding_downgrade", "had_preceding_upgrade",
]


def observation_end(df_dates):
    """Latest timestamp present across the raw tables."""
    return max(pd.to_datetime(s).max() for s in df_dates)
