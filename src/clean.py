"""Cleaning with explicit, data-supported decisions.

Every imputation here is justified by a check in notebooks/02_cleaning.py.
Where the data does not support a rationale, the code says so rather than
inventing one.
"""
import pandas as pd
import numpy as np


def clean_accounts(df):
    df = df.copy()
    df["signup_date"] = pd.to_datetime(df["signup_date"])
    return df


def clean_subscriptions(df):
    df = df.copy()
    df["start_date"] = pd.to_datetime(df["start_date"])
    df["end_date"] = pd.to_datetime(df["end_date"])
    # arr_amount == mrr_amount * 12 for all 5,000 rows (verified in nb 02).
    # It is perfectly collinear with mrr and is dropped rather than fed to a model.
    df = df.drop(columns=["arr_amount"])
    return df


def clean_support_tickets(df):
    df = df.copy()
    df["submitted_at"] = pd.to_datetime(df["submitted_at"])
    df["closed_at"] = pd.to_datetime(df["closed_at"])

    # 41% of satisfaction scores are missing. Two checks (nb 02) drive the handling:
    #   1. Missingness is NOT associated with churn (t-test p = 0.81), so it is
    #      safe to impute rather than treat as informative.
    #   2. Missing rate is flat across priority (0.405-0.422), so a per-priority
    #      median is effectively the global median.
    #
    # The indicator is recorded here, but the value is deliberately NOT filled.
    # Imputing with a median computed over the whole table would let validation
    # rows influence the statistic used on training rows. Imputation belongs
    # inside the CV pipeline (see model._pipe -> SimpleImputer), where it is fit
    # on the training fold only.
    df["satisfaction_missing"] = df["satisfaction_score"].isna().astype(int)
    return df


def clean_feature_usage(df):
    df = df.copy()
    df["usage_date"] = pd.to_datetime(df["usage_date"])
    # 21 duplicated usage_id values; drop so per-account event counts are not inflated.
    df = df.drop_duplicates(subset="usage_id", keep="first")
    return df


def clean_churn_events(df):
    df = df.copy()
    df["churn_date"] = pd.to_datetime(df["churn_date"])
    df["feedback_missing"] = df["feedback_text"].isna().astype(int)
    df["feedback_text"] = df["feedback_text"].fillna("no_feedback")
    return df


def clean_all(tables):
    return {
        "accounts": clean_accounts(tables["accounts"]),
        "subscriptions": clean_subscriptions(tables["subscriptions"]),
        "feature_usage": clean_feature_usage(tables["feature_usage"]),
        "support_tickets": clean_support_tickets(tables["support_tickets"]),
        "churn_events": clean_churn_events(tables["churn_events"]),
    }


def integrity_report(tables):
    """Data-quality violations worth surfacing rather than silently ignoring."""
    acc, subs = tables["accounts"], tables["subscriptions"]
    usage, tix, ce = tables["feature_usage"], tables["support_tickets"], tables["churn_events"]

    bridge = subs[["subscription_id", "account_id", "start_date", "end_date"]]
    u = usage.merge(bridge, on="subscription_id", how="left")
    t = tix.merge(acc[["account_id", "signup_date"]], on="account_id", how="left")

    return pd.DataFrame([
        ("tickets dated before account signup", int((t["submitted_at"] < t["signup_date"]).sum()), len(tix)),
        ("usage dated before its subscription starts", int((u["usage_date"] < u["start_date"]).sum()), len(u)),
        ("usage dated after its subscription ends",
         int(((u["usage_date"] > u["end_date"]) & u["end_date"].notna()).sum()),
         int(u["end_date"].notna().sum())),
        ("accounts whose churn_flag disagrees with churn_events",
         int((acc["account_id"].isin(ce["account_id"]) != acc["churn_flag"]).sum()), len(acc)),
    ], columns=["issue", "n_violations", "n_rows"])
