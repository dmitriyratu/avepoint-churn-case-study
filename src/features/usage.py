"""Product-engagement features: volume, breadth, recency, momentum, rhythm."""
import pandas as pd

from ..config import CUTOFF_DATE
from ._helpers import TREND_WINDOW_DAYS, WINDOWS, group_slope, safe_div, trailing


def usage_features(usage, subs, as_of=CUTOFF_DATE):
    bridge = subs[["subscription_id", "account_id"]].drop_duplicates()
    u = usage.merge(bridge, on="subscription_id", how="inner")
    if u.empty:
        return pd.DataFrame()

    u["days_ago"] = (as_of - u["usage_date"]).dt.days
    g = u.groupby("account_id")

    feats = pd.DataFrame({
        "total_usage_events": g.size(),
        "unique_features_used": g["feature_name"].nunique(),
        "total_usage_duration_mins": g["usage_duration_secs"].sum() / 60,
        "total_errors": g["error_count"].sum(),
        "beta_feature_pct": g["is_beta_feature"].mean(),
        "avg_usage_count": g["usage_count"].mean(),
        "days_since_last_usage": g["days_ago"].min(),
        "usage_span_days": g["days_ago"].max() - g["days_ago"].min(),
    })
    feats["error_rate"] = safe_div(feats["total_errors"], feats["total_usage_events"])
    feats["feature_breadth"] = feats["unique_features_used"] / usage["feature_name"].nunique()

    feats = pd.concat([
        feats,
        trailing(u, WINDOWS, "usage_last"),
        trailing(u, WINDOWS, "active_days_last", column="usage_date"),
    ], axis=1)

    # Acceleration: recent rate against the account's own longer-run baseline.
    # Length-normalised so windows of different sizes compare fairly.
    for short, long in [(30, 90), (30, 180), (90, 180)]:
        feats[f"accel_{short}d_vs_{long}d"] = safe_div(
            feats[f"usage_last_{short}d"] / short, feats[f"usage_last_{long}d"] / long)

    prior = u[u["days_ago"].between(91, 180)].groupby("account_id").size()
    feats["usage_prior_90d"] = prior
    feats["usage_momentum"] = safe_div(feats["usage_last_90d"], feats["usage_prior_90d"])
    feats["usage_delta_90d"] = feats["usage_last_90d"] - feats["usage_prior_90d"]
    feats["recency_ratio_90d"] = safe_div(feats["usage_last_90d"], feats["total_usage_events"])

    # A fitted slope catches a steady decline that window ratios miss.
    recent = u[u["days_ago"] <= TREND_WINDOW_DAYS].copy()
    recent["week"] = (TREND_WINDOW_DAYS - recent["days_ago"]) // 7
    weekly = recent.groupby(["account_id", "week"]).size().rename("n").reset_index()
    feats["usage_trend_slope"] = group_slope(weekly, "account_id", "week", "n")

    # Rhythm: two accounts with equal volume but different spacing are different
    # risks.
    days = (u[["account_id", "days_ago"]].drop_duplicates()
            .sort_values(["account_id", "days_ago"]))
    gaps = days.groupby("account_id")["days_ago"].diff()
    feats[["mean_gap_days", "max_gap_days"]] = gaps.groupby(days["account_id"]).agg(["mean", "max"])

    return feats
