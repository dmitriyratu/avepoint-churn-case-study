"""Account-level feature engineering, evaluated as of a cutoff date.

Every aggregation below is computed on tables already truncated to the
observation window by `labeling.truncate_tables`, so `as_of` is only used for
recency arithmetic — no filtering happens here.

Deliberately excluded from the modeling matrix:
  - anything derived from `churn_events` (describes an outcome, not a precursor)
  - `subscriptions.churn_flag` (the label at a different grain)
  - `arr_amount` (perfectly collinear with mrr * 12)
"""
import pandas as pd
import numpy as np

from .config import CUTOFF_DATE

# Window ladder for frequency/recency aggregates. Multiple windows plus the
# ratios between them is the standard churn pattern: the level tells you how big
# an account is, the differences tell you which way it is heading.
WINDOWS = (30, 60, 90, 180)
TREND_WINDOW_DAYS = 180


def _safe_div(a, b):
    return a / b.replace(0, np.nan)


def subscription_features(subs, as_of=CUTOFF_DATE):
    grp = subs.groupby("account_id")

    # tenure = signup-to-now, measured to the cutoff. Using .max() of end_date
    # here would silently stop the clock at whichever subscription happened to
    # end first, which is wrong for any account holding a mix of open and
    # closed subscriptions (62% of accounts in this dataset).
    feats = pd.DataFrame({
        "n_subscriptions": grp.size(),
        "n_upgrades": grp["upgrade_flag"].sum(),
        "n_downgrades": grp["downgrade_flag"].sum(),
        "n_trial_subs": grp["is_trial"].sum(),
        "total_mrr": grp["mrr_amount"].sum(),
        "max_mrr": grp["mrr_amount"].max(),
        "avg_mrr": grp["mrr_amount"].mean().round(1),
        "auto_renew_pct": grp["auto_renew_flag"].mean().round(3),
        "tenure_days": (as_of - grp["start_date"].min()).dt.days,
        "n_ended_subs": grp["end_date"].count(),
        "n_open_subs": grp["end_date"].apply(lambda s: s.isna().sum()),
    }).reset_index()

    ordered = subs.sort_values("start_date")
    latest = (ordered.groupby("account_id")
              .last()[["plan_tier", "billing_frequency", "seats", "mrr_amount"]]
              .reset_index()
              .rename(columns={"plan_tier": "latest_plan_tier",
                               "billing_frequency": "billing_freq",
                               "seats": "latest_seats",
                               "mrr_amount": "latest_mrr"}))
    first = (ordered.groupby("account_id")
             .first()[["seats", "mrr_amount"]]
             .reset_index()
             .rename(columns={"seats": "first_seats", "mrr_amount": "first_mrr"}))

    feats = feats.merge(latest, on="account_id", how="left").merge(first, on="account_id", how="left")

    # Expansion / contraction: the direction of the account, not just its size.
    feats["seat_growth"] = feats["latest_seats"] - feats["first_seats"]
    feats["mrr_growth"] = feats["latest_mrr"] - feats["first_mrr"]
    feats["mrr_growth_pct"] = _safe_div(feats["mrr_growth"], feats["first_mrr"]).round(3)
    feats["upgrade_net"] = feats["n_upgrades"] - feats["n_downgrades"]

    # Revenue volatility: an account whose spend swings is a different risk from
    # one paying the same amount every month.
    mrr_std = subs.groupby("account_id")["mrr_amount"].std().reset_index(name="mrr_std")
    feats = feats.merge(mrr_std, on="account_id", how="left")
    feats["mrr_cv"] = _safe_div(feats["mrr_std"], feats["avg_mrr"]).round(3)
    feats["pct_subs_ended"] = (feats["n_ended_subs"] / feats["n_subscriptions"]).round(3)

    # Days since the most recent subscription started — a stalled account stops
    # opening new subscriptions.
    last_start = grp["start_date"].max().reset_index(name="last_start")
    feats = feats.merge(last_start, on="account_id", how="left")
    feats["days_since_last_sub_start"] = (as_of - feats["last_start"]).dt.days
    feats = feats.drop(columns=["last_start"])

    return feats


def feature_usage_features(usage, subs, as_of=CUTOFF_DATE):
    bridge = subs[["subscription_id", "account_id"]].drop_duplicates()
    u = usage.merge(bridge, on="subscription_id", how="inner")
    if u.empty:
        return pd.DataFrame(columns=["account_id"])

    total_features = usage["feature_name"].nunique()
    u["days_ago"] = (as_of - u["usage_date"]).dt.days

    grp = u.groupby("account_id")
    feats = pd.DataFrame({
        "total_usage_events": grp.size(),
        "unique_features_used": grp["feature_name"].nunique(),
        "total_usage_duration_mins": (grp["usage_duration_secs"].sum() / 60).round(1),
        "total_errors": grp["error_count"].sum(),
        "beta_feature_pct": grp["is_beta_feature"].mean().round(3),
        "avg_usage_count": grp["usage_count"].mean().round(2),
        "days_since_last_usage": grp["days_ago"].min(),
        "usage_span_days": (grp["days_ago"].max() - grp["days_ago"].min()),
    }).reset_index()

    feats["error_rate"] = _safe_div(feats["total_errors"], feats["total_usage_events"]).round(4)
    feats["feature_breadth"] = (feats["unique_features_used"] / total_features).round(3)

    # --- Frequency across a window ladder -------------------------------------
    # Standard churn practice is to compute the same metric over several windows
    # and let the *differences between them* carry the trend, rather than relying
    # on one lifetime total that hides recent collapse.
    for w in WINDOWS:
        win = (u[u["days_ago"] <= w].groupby("account_id").size()
               .reset_index(name=f"usage_last_{w}d"))
        feats = feats.merge(win, on="account_id", how="left")
        feats[f"usage_last_{w}d"] = feats[f"usage_last_{w}d"].fillna(0).astype(int)

        act = (u[u["days_ago"] <= w].groupby("account_id")["usage_date"].nunique()
               .reset_index(name=f"active_days_last_{w}d"))
        feats = feats.merge(act, on="account_id", how="left")
        feats[f"active_days_last_{w}d"] = feats[f"active_days_last_{w}d"].fillna(0).astype(int)

    # --- Acceleration: short window vs long window ----------------------------
    # A rate above 1 means the account is more active lately than its own
    # baseline; below 1 means it is winding down. Normalising by window length
    # keeps the comparison fair.
    for short, long in [(30, 90), (30, 180), (90, 180)]:
        s_rate = feats[f"usage_last_{short}d"] / short
        l_rate = feats[f"usage_last_{long}d"] / long
        feats[f"accel_{short}d_vs_{long}d"] = _safe_div(s_rate, l_rate).round(3)

    # Consecutive non-overlapping periods — did last quarter beat the one before?
    prior = (u[(u["days_ago"] > 90) & (u["days_ago"] <= 180)].groupby("account_id").size()
             .reset_index(name="usage_prior_90d"))
    feats = feats.merge(prior, on="account_id", how="left")
    feats["usage_prior_90d"] = feats["usage_prior_90d"].fillna(0).astype(int)
    feats["usage_momentum"] = _safe_div(feats["usage_last_90d"],
                                        feats["usage_prior_90d"].replace(0, np.nan)).round(3)
    feats["usage_delta_90d"] = feats["usage_last_90d"] - feats["usage_prior_90d"]
    feats["recency_ratio_90d"] = _safe_div(feats["usage_last_90d"],
                                           feats["total_usage_events"]).round(3)

    # --- Trend slope ----------------------------------------------------------
    # Ratios are coarse; a fitted slope over weekly counts captures a steady
    # decline that a 90-vs-90 comparison can miss.
    recent = u[u["days_ago"] <= TREND_WINDOW_DAYS].copy()
    if not recent.empty:
        recent["week"] = (TREND_WINDOW_DAYS - recent["days_ago"]) // 7
        weekly = recent.groupby(["account_id", "week"]).size().reset_index(name="n")
        slopes = (weekly.groupby("account_id")
                  .apply(lambda g: np.polyfit(g["week"], g["n"], 1)[0]
                         if len(g) >= 3 else np.nan)
                  .reset_index(name="usage_trend_slope"))
        feats = feats.merge(slopes, on="account_id", how="left")
        feats["usage_trend_slope"] = feats["usage_trend_slope"].round(4)

    # --- Engagement regularity ------------------------------------------------
    # Lumpy usage is a different risk profile from steady usage at the same
    # volume. Mean and max gap between active days capture that.
    gaps = (u.sort_values("days_ago", ascending=False)
            .groupby("account_id")["days_ago"]
            .apply(lambda s: pd.Series(np.diff(np.sort(s.unique()))
                                       if s.nunique() > 1 else [np.nan]))
            .reset_index())
    if not gaps.empty and "days_ago" in gaps.columns:
        g = gaps.groupby("account_id")["days_ago"].agg(["mean", "max"]).reset_index()
        g.columns = ["account_id", "mean_gap_days", "max_gap_days"]
        feats = feats.merge(g, on="account_id", how="left")
        feats[["mean_gap_days", "max_gap_days"]] = feats[["mean_gap_days", "max_gap_days"]].round(1)

    return feats


def support_features(tickets, as_of=CUTOFF_DATE):
    if tickets.empty:
        return pd.DataFrame(columns=["account_id"])

    grp = tickets.groupby("account_id")
    feats = pd.DataFrame({
        "n_tickets": grp.size(),
        "avg_resolution_hours": grp["resolution_time_hours"].mean().round(1),
        "max_resolution_hours": grp["resolution_time_hours"].max().round(1),
        "avg_first_response_mins": grp["first_response_time_minutes"].mean().round(1),
        "avg_satisfaction": grp["satisfaction_score"].mean().round(2),
        "min_satisfaction": grp["satisfaction_score"].min(),
        "sat_missing_rate": grp["satisfaction_missing"].mean().round(3),
        "n_escalations": grp["escalation_flag"].sum(),
        "days_since_last_ticket": (as_of - grp["submitted_at"].max()).dt.days,
    }).reset_index()

    if "ticket_open_at_cutoff" in tickets.columns:
        opn = grp["ticket_open_at_cutoff"].agg(["sum", "mean"]).reset_index()
        opn.columns = ["account_id", "n_open_tickets", "open_ticket_rate"]
        feats = feats.merge(opn, on="account_id", how="left")

    urgent = (tickets[tickets["priority"].isin(["urgent", "high"])]
              .groupby("account_id").size().reset_index(name="n_urgent_high"))
    feats = feats.merge(urgent, on="account_id", how="left")
    feats["n_urgent_high"] = feats["n_urgent_high"].fillna(0)

    feats["urgent_pct"] = (feats["n_urgent_high"] / feats["n_tickets"]).round(3)
    feats["escalation_rate"] = (feats["n_escalations"] / feats["n_tickets"]).round(3)

    t = tickets.copy()
    t["days_ago"] = (as_of - t["submitted_at"]).dt.days
    for w in (30, 90, 180):
        win = (t[t["days_ago"] <= w].groupby("account_id").size()
               .reset_index(name=f"tickets_last_{w}d"))
        feats = feats.merge(win, on="account_id", how="left")
        feats[f"tickets_last_{w}d"] = feats[f"tickets_last_{w}d"].fillna(0).astype(int)

    # Rising support load is a classic churn precursor — the ratio matters more
    # than the count, since heavy users open more tickets in absolute terms.
    feats["ticket_accel_30d_vs_90d"] = _safe_div(
        feats["tickets_last_30d"] / 30, feats["tickets_last_90d"] / 90).round(3)

    return feats


# Missing values do not all mean the same thing, so they are not filled the
# same way.
#
#   counts   an account with no tickets genuinely had zero tickets -> 0
#   recency  "never used the product" is not "used it today" -> observation-
#            window length, so the model sees it as maximally stale
#   rates    an average satisfaction score with no responses is unknown, not 0.
#            Left as NaN and imputed inside the CV fold by model._pipe, so the
#            statistic is fit on training rows only.
_RECENCY_COLS = ["days_since_last_usage", "days_since_last_ticket",
                 "usage_span_days", "days_since_last_sub_start"]

_COUNT_PREFIXES = ("n_", "total_", "usage_last_", "usage_prior_", "tickets_last_")


def _is_count_col(c):
    return c.startswith(_COUNT_PREFIXES)


def drop_collinear(df, threshold=0.98, protect=()):
    """Drop one of each near-duplicate pair, keeping the earlier column.

    feature_breadth is unique_features_used / 40, so the two are correlated at
    exactly 1.0 — keeping both just splits the same coefficient in a linear model.
    """
    num = df.select_dtypes(include=[np.number])
    cm = num.corr().abs()
    cols, dropped = list(cm.columns), []
    for i, a in enumerate(cols):
        if a in dropped or a in protect:
            continue
        for b in cols[i + 1:]:
            if b in dropped or b in protect:
                continue
            v = cm.loc[a, b]
            if pd.notna(v) and v > threshold:
                dropped.append(b)
    return df.drop(columns=dropped), dropped


def build_model_dataset(tables, cohort, as_of=CUTOFF_DATE, prune_collinear=True):
    """Join every feature block onto the cohort. Returns (X_frame, feature_names)."""
    base = cohort.copy()
    base["days_since_signup"] = (as_of - base["signup_date"]).dt.days

    blocks = [
        subscription_features(tables["subscriptions"], as_of),
        feature_usage_features(tables["feature_usage"], tables["subscriptions"], as_of),
        support_features(tables["support_tickets"], as_of),
    ]
    df = base
    for b in blocks:
        if not b.empty:
            df = df.merge(b, on="account_id", how="left")

    seats = df["seats"].replace(0, np.nan)
    df["usage_per_seat"] = _safe_div(df.get("total_usage_events", pd.Series(0, index=df.index)), seats).round(2)
    df["tickets_per_seat"] = _safe_div(df.get("n_tickets", pd.Series(0, index=df.index)), seats).round(3)
    df["mrr_per_seat"] = _safe_div(df.get("total_mrr", pd.Series(0, index=df.index)), seats).round(1)

    window_len = (int((as_of - tables["feature_usage"]["usage_date"].min()).days)
                  if len(tables["feature_usage"]) else 999)
    for c in _RECENCY_COLS:
        if c in df.columns:
            df[c] = df[c].fillna(window_len)

    # Counts fill to zero; rates and means stay NaN for the in-fold imputer.
    for c in df.select_dtypes(include=[np.number]).columns:
        if _is_count_col(c):
            df[c] = df[c].fillna(0)

    # Categoricals are deliberately left as raw strings. Encoding them here with
    # pd.get_dummies would learn the category levels from every row — including
    # the validation fold — and would silently produce a different column set if
    # an unseen category appeared at serving time. One-hot encoding happens
    # inside the model pipeline instead (model._pipe -> OneHotEncoder with
    # handle_unknown="ignore"), so it is fit per fold and safe in production.

    protect = tuple(cohort.columns)

    # Zero-variance columns carry no information and their presence depends on
    # the cutoff — at an earlier cutoff no support ticket is still open, so
    # n_open_tickets collapses to a constant. Drop them rather than let the
    # audit fail on a benign artefact.
    # nunique() ignores NaN, matching audit.constant_columns — a column holding
    # one value plus nulls is still constant as far as a model is concerned.
    const = [c for c in df.columns if c not in protect and df[c].nunique() <= 1]
    df = df.drop(columns=const)
    df.attrs["dropped_constant"] = const

    if prune_collinear:
        df, dropped = drop_collinear(df, threshold=0.98, protect=protect)
        df.attrs["dropped_collinear"] = dropped
        df.attrs["dropped_constant"] = const

    return df
