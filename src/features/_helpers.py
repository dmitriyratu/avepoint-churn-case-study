"""Shared aggregation helpers for the feature blocks."""
import numpy as np
import pandas as pd

WINDOWS = (30, 60, 90, 180)
TREND_WINDOW_DAYS = 180
COLLINEARITY_THRESHOLD = 0.98

# Missing means something different per feature family:
#   counts   no activity is genuinely zero
#   recency  never happened is maximally stale, not "today"
#   rates    unknown — left NaN and imputed inside the CV fold
RECENCY_COLS = ("days_since_last_usage", "days_since_last_ticket",
                "usage_span_days", "days_since_last_sub_start")
COUNT_PREFIXES = ("n_", "total_", "usage_last_", "usage_prior_",
                  "tickets_last_", "active_days_")


def safe_div(a, b):
    return a / b.replace(0, np.nan)


def trailing(events, windows, prefix, column=None):
    """Trailing-window counts per account, one column per window.

    Counts rows, or distinct values of `column` when given.
    """
    def count(w):
        g = events[events["days_ago"] <= w].groupby("account_id")
        return g[column].nunique() if column else g.size()

    return pd.concat({f"{prefix}_{w}d": count(w) for w in windows}, axis=1)


def group_slope(frame, by, x, y):
    """Least-squares slope of y on x within each group, vectorised."""
    g = frame.groupby(by)
    dx = frame[x] - g[x].transform("mean")
    dy = frame[y] - g[y].transform("mean")
    num = (dx * dy).groupby(frame[by]).sum()
    den = (dx ** 2).groupby(frame[by]).sum()
    return safe_div(num, den)
