import pandas as pd
from pathlib import Path

RAW_DIR = Path(__file__).parents[1] / "data" / "raw"

TABLES = {
    "accounts": "ravenstack_accounts.csv",
    "subscriptions": "ravenstack_subscriptions.csv",
    "feature_usage": "ravenstack_feature_usage.csv",
    "support_tickets": "ravenstack_support_tickets.csv",
    "churn_events": "ravenstack_churn_events.csv",
}


def load_all(data_dir=None):
    base = Path(data_dir) if data_dir else RAW_DIR
    return {name: pd.read_csv(base / fname) for name, fname in TABLES.items()}
