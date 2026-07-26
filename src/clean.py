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
    # Some trial rows have mrr=0 and arr=0 — keep them, they're real observations
    return df


def clean_support_tickets(df):
    df = df.copy()
    df["submitted_at"] = pd.to_datetime(df["submitted_at"])
    df["closed_at"] = pd.to_datetime(df["closed_at"])
    # ~41% of satisfaction scores are missing. Fill with per-priority median
    # since response rates likely differ by ticket severity.
    df["satisfaction_score"] = df.groupby("priority")["satisfaction_score"].transform(
        lambda x: x.fillna(x.median())
    )
    return df


def clean_feature_usage(df):
    df = df.copy()
    df["usage_date"] = pd.to_datetime(df["usage_date"])
    return df


def clean_churn_events(df):
    df = df.copy()
    df["churn_date"] = pd.to_datetime(df["churn_date"])
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
