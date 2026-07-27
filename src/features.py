"""Account-level features, evaluated as of a cutoff date.

Input tables must already be truncated to the observation window by
`labeling.truncate_tables`; `as_of` is used only for recency arithmetic.

Every block returns a frame indexed by `account_id` so they compose with
`pd.concat` rather than a chain of merges.

Excluded by design (see docs/DATA_DICTIONARY.md): anything derived from
`churn_events`, `subscriptions.churn_flag`, and `arr_amount`.
"""
import numpy as np
import pandas as pd

from .config import CUTOFF_DATE

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


def _safe_div(a, b):
    return a / b.replace(0, np.nan)


def _trailing(events, windows, prefix, column=None):
    """Trailing-window counts per account, one column per window.

    Counts rows, or distinct values of `column` when given.
    """
    def count(w):
        g = events[events["days_ago"] <= w].groupby("account_id")
        return g[column].nunique() if column else g.size()

    return pd.concat({f"{prefix}_{w}d": count(w) for w in windows}, axis=1)


def _group_slope(frame, by, x, y):
    """Least-squares slope of y on x within each group, vectorised."""
    g = frame.groupby(by)
    dx = frame[x] - g[x].transform("mean")
    dy = frame[y] - g[y].transform("mean")
    num = (dx * dy).groupby(frame[by]).sum()
    den = (dx ** 2).groupby(frame[by]).sum()
    return _safe_div(num, den)


def subscription_features(subs, as_of=CUTOFF_DATE):
    g = subs.groupby("account_id")

    # Tenure runs signup-to-cutoff. Measuring to max(end_date) would stop the
    # clock at whichever subscription closed first, which is wrong for any
    # account holding both open and closed subscriptions.
    feats = pd.DataFrame({
        "n_subscriptions": g.size(),
        "n_upgrades": g["upgrade_flag"].sum(),
        "n_downgrades": g["downgrade_flag"].sum(),
        "n_trial_subs": g["is_trial"].sum(),
        "total_mrr": g["mrr_amount"].sum(),
        "max_mrr": g["mrr_amount"].max(),
        "avg_mrr": g["mrr_amount"].mean(),
        "mrr_std": g["mrr_amount"].std(),
        "auto_renew_pct": g["auto_renew_flag"].mean(),
        "tenure_days": (as_of - g["start_date"].min()).dt.days,
        "n_ended_subs": g["end_date"].count(),
        "n_open_subs": g["end_date"].apply(lambda s: s.isna().sum()),
        "days_since_last_sub_start": (as_of - g["start_date"].max()).dt.days,
    })

    ordered = subs.sort_values("start_date").groupby("account_id")
    latest = ordered.last()[["plan_tier", "billing_frequency", "seats", "mrr_amount"]]
    latest.columns = ["latest_plan_tier", "billing_freq", "latest_seats", "latest_mrr"]
    first = ordered.first()[["seats", "mrr_amount"]]
    first.columns = ["first_seats", "first_mrr"]

    feats = pd.concat([feats, latest, first], axis=1)

    # Direction of travel, not just size.
    feats["seat_growth"] = feats["latest_seats"] - feats["first_seats"]
    feats["mrr_growth"] = feats["latest_mrr"] - feats["first_mrr"]
    feats["mrr_growth_pct"] = _safe_div(feats["mrr_growth"], feats["first_mrr"])
    feats["upgrade_net"] = feats["n_upgrades"] - feats["n_downgrades"]
    feats["mrr_cv"] = _safe_div(feats["mrr_std"], feats["avg_mrr"])
    feats["pct_subs_ended"] = feats["n_ended_subs"] / feats["n_subscriptions"]

    return feats


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
    feats["error_rate"] = _safe_div(feats["total_errors"], feats["total_usage_events"])
    feats["feature_breadth"] = feats["unique_features_used"] / usage["feature_name"].nunique()

    feats = pd.concat([
        feats,
        _trailing(u, WINDOWS, "usage_last"),
        _trailing(u, WINDOWS, "active_days_last", column="usage_date"),
    ], axis=1)

    # Acceleration: recent rate against the account's own longer-run baseline.
    # Length-normalised so windows of different sizes compare fairly.
    for short, long in [(30, 90), (30, 180), (90, 180)]:
        feats[f"accel_{short}d_vs_{long}d"] = _safe_div(
            feats[f"usage_last_{short}d"] / short, feats[f"usage_last_{long}d"] / long)

    prior = u[u["days_ago"].between(91, 180)].groupby("account_id").size()
    feats["usage_prior_90d"] = prior
    feats["usage_momentum"] = _safe_div(feats["usage_last_90d"], feats["usage_prior_90d"])
    feats["usage_delta_90d"] = feats["usage_last_90d"] - feats["usage_prior_90d"]
    feats["recency_ratio_90d"] = _safe_div(feats["usage_last_90d"], feats["total_usage_events"])

    # A fitted slope catches a steady decline that window ratios miss.
    recent = u[u["days_ago"] <= TREND_WINDOW_DAYS].copy()
    recent["week"] = (TREND_WINDOW_DAYS - recent["days_ago"]) // 7
    weekly = recent.groupby(["account_id", "week"]).size().rename("n").reset_index()
    feats["usage_trend_slope"] = _group_slope(weekly, "account_id", "week", "n")

    # Rhythm: two accounts with equal volume but different spacing are different
    # risks.
    days = (u[["account_id", "days_ago"]].drop_duplicates()
            .sort_values(["account_id", "days_ago"]))
    gaps = days.groupby("account_id")["days_ago"].diff()
    feats[["mean_gap_days", "max_gap_days"]] = gaps.groupby(days["account_id"]).agg(["mean", "max"])

    return feats


def support_features(tickets, as_of=CUTOFF_DATE):
    if tickets.empty:
        return pd.DataFrame()

    t = tickets.assign(days_ago=(as_of - tickets["submitted_at"]).dt.days)
    g = t.groupby("account_id")

    feats = pd.DataFrame({
        "n_tickets": g.size(),
        "avg_resolution_hours": g["resolution_time_hours"].mean(),
        "max_resolution_hours": g["resolution_time_hours"].max(),
        "avg_first_response_mins": g["first_response_time_minutes"].mean(),
        "avg_satisfaction": g["satisfaction_score"].mean(),
        "min_satisfaction": g["satisfaction_score"].min(),
        "sat_missing_rate": g["satisfaction_missing"].mean(),
        "n_escalations": g["escalation_flag"].sum(),
        "days_since_last_ticket": (as_of - g["submitted_at"].max()).dt.days,
        "n_urgent_high": g["priority"].apply(lambda s: s.isin(["urgent", "high"]).sum()),
    })
    if "ticket_open_at_cutoff" in t.columns:
        feats["n_open_tickets"] = g["ticket_open_at_cutoff"].sum()

    feats["urgent_pct"] = feats["n_urgent_high"] / feats["n_tickets"]
    feats["escalation_rate"] = feats["n_escalations"] / feats["n_tickets"]

    feats = pd.concat([feats, _trailing(t, (30, 90, 180), "tickets_last")], axis=1)

    # Rising support load is a churn precursor; the ratio matters more than the
    # count, since heavy users open more tickets in absolute terms.
    feats["ticket_accel_30d_vs_90d"] = _safe_div(
        feats["tickets_last_30d"] / 30, feats["tickets_last_90d"] / 90)

    return feats


def drop_collinear(df, threshold=COLLINEARITY_THRESHOLD, protect=()):
    """Drop the later column of each near-duplicate pair.

    Keeping both halves of a pair like `feature_breadth` and
    `unique_features_used` just splits one effect across two coefficients.
    """
    corr = df.select_dtypes(include=[np.number]).corr().abs()
    candidates = [c for c in corr.columns if c not in protect]
    dropped = []
    for i, a in enumerate(candidates):
        if a in dropped:
            continue
        pair = corr.loc[a, candidates[i + 1:]]
        dropped += [b for b in pair[pair > threshold].index if b not in dropped]
    return df.drop(columns=dropped), dropped


def build_model_dataset(tables, cohort, as_of=CUTOFF_DATE, prune=True):
    """Cohort joined to every feature block, ready for `model.prep_xy`."""
    blocks = [
        subscription_features(tables["subscriptions"], as_of),
        usage_features(tables["feature_usage"], tables["subscriptions"], as_of),
        support_features(tables["support_tickets"], as_of),
    ]
    features = pd.concat([b for b in blocks if not b.empty], axis=1)

    df = cohort.set_index("account_id").join(features)
    df["days_since_signup"] = (as_of - df["signup_date"]).dt.days

    seats = df["seats"].replace(0, np.nan)
    df["usage_per_seat"] = _safe_div(df["total_usage_events"], seats)
    df["tickets_per_seat"] = _safe_div(df["n_tickets"], seats)
    df["mrr_per_seat"] = _safe_div(df["total_mrr"], seats)

    window_len = (as_of - tables["feature_usage"]["usage_date"].min()).days
    df[list(RECENCY_COLS)] = df[list(RECENCY_COLS)].fillna(window_len)

    counts = [c for c in df.columns if c.startswith(COUNT_PREFIXES)]
    df[counts] = df[counts].fillna(0)

    df = df.round(4).reset_index()

    # Categoricals stay as strings; the pipeline encodes them per fold.
    protect = tuple(cohort.columns)
    constant = [c for c in df.columns if c not in protect and df[c].nunique() <= 1]
    df = df.drop(columns=constant)

    collinear = []
    if prune:
        df, collinear = drop_collinear(df, protect=protect)

    df.attrs.update(dropped_constant=constant, dropped_collinear=collinear)
    return df
