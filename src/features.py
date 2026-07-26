import pandas as pd
import numpy as np

# Dataset runs through mid-2025; use this as "today" for tenure calcs
REFERENCE_DATE = pd.Timestamp("2025-07-21")

TOTAL_FEATURES = 40  # unique feature_name values in the usage table


def subscription_features(subs):
    grp = subs.groupby("account_id")

    feats = pd.DataFrame({
        "n_subscriptions": grp.size(),
        "n_churned_subs": grp["churn_flag"].sum(),
        "n_upgrades": grp["upgrade_flag"].sum(),
        "n_downgrades": grp["downgrade_flag"].sum(),
        "n_trial_subs": grp["is_trial"].sum(),
        "total_mrr": grp["mrr_amount"].sum(),
        "max_mrr": grp["mrr_amount"].max(),
        "avg_mrr": grp["mrr_amount"].mean().round(1),
        "auto_renew_pct": grp["auto_renew_flag"].mean().round(3),
        "tenure_days": (grp["end_date"].max().fillna(REFERENCE_DATE) - grp["start_date"].min()).dt.days,
    }).reset_index()

    latest = (subs.sort_values("start_date")
              .groupby("account_id")
              .last()[["plan_tier", "billing_frequency", "seats", "mrr_amount"]]
              .reset_index()
              .rename(columns={
                  "plan_tier": "latest_plan_tier",
                  "billing_frequency": "billing_freq",
                  "mrr_amount": "latest_mrr",
              }))
    latest = latest.drop(columns=["seats"])  # already on accounts

    feats = feats.merge(latest, on="account_id", how="left")
    feats["sub_churn_rate"] = feats["n_churned_subs"] / feats["n_subscriptions"]
    feats["upgrade_net"] = feats["n_upgrades"] - feats["n_downgrades"]

    return feats


def feature_usage_features(usage, subs):
    # bridge usage -> account via subscription
    bridge = subs[["subscription_id", "account_id"]].drop_duplicates()
    u = usage.merge(bridge, on="subscription_id", how="left").dropna(subset=["account_id"])

    grp = u.groupby("account_id")
    feats = pd.DataFrame({
        "total_usage_events": grp.size(),
        "unique_features_used": grp["feature_name"].nunique(),
        "total_usage_duration_mins": (grp["usage_duration_secs"].sum() / 60).round(1),
        "total_errors": grp["error_count"].sum(),
        "beta_feature_pct": grp["is_beta_feature"].mean().round(3),
        "avg_usage_count": grp["usage_count"].mean().round(2),
    }).reset_index()

    feats["error_rate"] = (
        feats["total_errors"] / feats["total_usage_events"].replace(0, np.nan)
    ).round(4)

    # breadth: what fraction of the product's 40 features does this account touch?
    feats["feature_breadth"] = (feats["unique_features_used"] / TOTAL_FEATURES).round(3)

    return feats


def support_features(tickets):
    grp = tickets.groupby("account_id")

    feats = pd.DataFrame({
        "n_tickets": grp.size(),
        "avg_resolution_hours": grp["resolution_time_hours"].mean().round(1),
        "avg_first_response_mins": grp["first_response_time_minutes"].mean().round(1),
        "avg_satisfaction": grp["satisfaction_score"].mean().round(2),
        "n_escalations": grp["escalation_flag"].sum(),
    }).reset_index()

    urgent = (
        tickets[tickets["priority"] == "urgent"]
        .groupby("account_id")
        .size()
        .rename("urgent_count")
    )
    feats = feats.merge(urgent, on="account_id", how="left")
    feats["urgent_count"] = feats["urgent_count"].fillna(0)
    feats["urgent_pct"] = (feats["urgent_count"] / feats["n_tickets"]).round(3)
    feats = feats.drop(columns=["urgent_count"])

    feats["escalation_rate"] = (feats["n_escalations"] / feats["n_tickets"]).round(3)

    return feats


def churn_event_features(churn_events):
    """Historical churn event signals per account."""
    grp = churn_events.groupby("account_id")
    feats = pd.DataFrame({
        "n_churn_events": grp.size(),
        "had_reactivation": grp["is_reactivation"].any().astype(int),
        "had_preceding_downgrade": grp["preceding_downgrade_flag"].any().astype(int),
        "had_preceding_upgrade": grp["preceding_upgrade_flag"].any().astype(int),
        "total_refund_usd": grp["refund_amount_usd"].sum().round(2),
    }).reset_index()

    # one-hot the most common churn reason
    most_common_reason = (churn_events.groupby("account_id")["reason_code"]
                          .agg(lambda x: x.mode()[0] if len(x) > 0 else "none")
                          .reset_index(name="primary_churn_reason"))
    feats = feats.merge(most_common_reason, on="account_id", how="left")
    feats = pd.get_dummies(feats, columns=["primary_churn_reason"], drop_first=False)

    return feats


def recency_features(usage, subs):
    """Time-based engagement signals — recency and trend."""
    bridge = subs[["subscription_id", "account_id"]].drop_duplicates()
    u = usage.merge(bridge, on="subscription_id", how="left").dropna(subset=["account_id"])
    u["days_ago"] = (REFERENCE_DATE - u["usage_date"]).dt.days

    last_usage = u.groupby("account_id")["usage_date"].max().reset_index()
    last_usage["days_since_last_usage"] = (REFERENCE_DATE - last_usage["usage_date"]).dt.days
    last_usage = last_usage[["account_id", "days_since_last_usage"]]

    recent_30 = (u[u["days_ago"] <= 30].groupby("account_id").size()
                 .reset_index(name="usage_last_30d"))
    recent_90 = (u[u["days_ago"] <= 90].groupby("account_id").size()
                 .reset_index(name="usage_last_90d"))

    feats = last_usage.merge(recent_30, on="account_id", how="left")
    feats = feats.merge(recent_90, on="account_id", how="left")
    feats["usage_last_30d"] = feats["usage_last_30d"].fillna(0).astype(int)
    feats["usage_last_90d"] = feats["usage_last_90d"].fillna(0).astype(int)

    # trend: recent vs earlier activity ratio (0 = only old activity, 1 = mostly recent)
    total = u.groupby("account_id").size().reset_index(name="total_events")
    feats = feats.merge(total, on="account_id", how="left")
    feats["recency_ratio"] = (feats["usage_last_90d"] / feats["total_events"].replace(0, np.nan)).round(3)
    feats = feats.drop(columns=["total_events"])

    return feats


def active_subscription_features(subs):
    """Whether the account still has an active (open-ended) subscription."""
    active = (subs[subs["end_date"].isna()]
              .groupby("account_id")
              .agg(
                  has_active_sub=("subscription_id", "any"),
                  active_mrr=("mrr_amount", "sum"),
              )
              .reset_index())
    active["has_active_sub"] = 1
    return active


def build_model_dataset(tables):
    accounts = tables["accounts"].copy()
    accounts["days_since_signup"] = (REFERENCE_DATE - accounts["signup_date"]).dt.days

    sub_feats = subscription_features(tables["subscriptions"])
    usage_feats = feature_usage_features(tables["feature_usage"], tables["subscriptions"])
    support_feats = support_features(tables["support_tickets"])
    churn_ev_feats = churn_event_features(tables["churn_events"])
    recency_feats = recency_features(tables["feature_usage"], tables["subscriptions"])
    active_sub_feats = active_subscription_features(tables["subscriptions"])

    df = (accounts
          .merge(sub_feats, on="account_id", how="left")
          .merge(usage_feats, on="account_id", how="left")
          .merge(support_feats, on="account_id", how="left")
          .merge(churn_ev_feats, on="account_id", how="left")
          .merge(recency_feats, on="account_id", how="left")
          .merge(active_sub_feats, on="account_id", how="left"))

    df["has_active_sub"] = df["has_active_sub"].fillna(0).astype(int)
    df["active_mrr"] = df["active_mrr"].fillna(0)

    # seat-normalized signals
    seats = df["seats"].replace(0, np.nan)
    df["usage_per_seat"] = (df["total_usage_events"] / seats).round(2)
    df["tickets_per_seat"] = (df["n_tickets"] / seats).round(3)
    df["mrr_per_seat"] = (df["total_mrr"] / seats).round(1)

    # encode categoricals
    cat_cols = ["industry", "country", "referral_source", "plan_tier",
                "latest_plan_tier", "billing_freq"]
    df = pd.get_dummies(df, columns=cat_cols, drop_first=False)

    return df
