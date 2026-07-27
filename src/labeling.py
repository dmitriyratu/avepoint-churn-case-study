"""Temporal cohort construction.

The original framing — aggregate every row a customer ever produced, then predict
a static `churn_flag` — has no observation window. Activity generated *after* the
customer left ends up in the feature vector (18.3% of usage rows in this dataset
are dated after the account's last churn event), and the model is asked to
"predict" something that already happened.

This module replaces that with the standard formulation:

    features  <- data strictly before CUTOFF
    label     <- first churn occurring in (CUTOFF, CUTOFF + HORIZON]
    eligible  <- signed up before CUTOFF and not already churned at CUTOFF

Nothing observable after CUTOFF can reach the feature matrix.
"""
import pandas as pd
import numpy as np

from .config import CUTOFF_DATE, HORIZON_DAYS


def first_churn_date(churn_events):
    return churn_events.groupby("account_id")["churn_date"].min()


def build_cohort(tables, cutoff=CUTOFF_DATE, horizon_days=HORIZON_DAYS):
    """Return the eligible accounts and their forward-looking label."""
    acc = tables["accounts"]
    fc = first_churn_date(tables["churn_events"])
    horizon_end = cutoff + pd.Timedelta(days=horizon_days)

    cohort = acc[acc["signup_date"] < cutoff].copy()
    cohort["first_churn_date"] = cohort["account_id"].map(fc)

    # Exclude accounts that had already churned before the cutoff — they are not
    # at risk during the prediction window.
    cohort = cohort[~(cohort["first_churn_date"] < cutoff)].copy()

    cohort["churned_next_180d"] = (
        cohort["first_churn_date"].between(cutoff, horizon_end, inclusive="right")
    ).astype(int)

    return cohort.drop(columns=["first_churn_date"])


def truncate_tables(tables, cutoff=CUTOFF_DATE):
    """Clip every event table to the observation window (strictly before cutoff).

    Filtering a table on its *start* timestamp is not sufficient. A row that
    began before the cutoff can still carry outcome fields resolved after it —
    a support ticket opened in June and closed in July has a resolution time and
    a satisfaction score that nobody could know at the end of June. Those fields
    are censored here so the feature layer cannot see them.
    """
    subs = tables["subscriptions"]
    subs_t = subs[subs["start_date"] < cutoff].copy()
    # An end_date at or after the cutoff has not happened yet.
    subs_t.loc[subs_t["end_date"] >= cutoff, "end_date"] = pd.NaT

    usage_t = tables["feature_usage"][tables["feature_usage"]["usage_date"] < cutoff].copy()
    usage_t = usage_t[usage_t["subscription_id"].isin(subs_t["subscription_id"])]

    tix_t = tables["support_tickets"][tables["support_tickets"]["submitted_at"] < cutoff].copy()

    # Still open at the cutoff -> every resolution-time outcome is unknown.
    still_open = tix_t["closed_at"].isna() | (tix_t["closed_at"] >= cutoff)
    tix_t["ticket_open_at_cutoff"] = still_open.astype(int)
    tix_t.loc[still_open, "closed_at"] = pd.NaT
    tix_t.loc[still_open, ["resolution_time_hours", "satisfaction_score"]] = np.nan

    # First response is a timestamp we only have as an offset; if it lands after
    # the cutoff it is equally unknowable.
    fr_at = tix_t["submitted_at"] + pd.to_timedelta(
        tix_t["first_response_time_minutes"], unit="m"
    )
    tix_t.loc[fr_at >= cutoff, "first_response_time_minutes"] = np.nan

    ce_t = tables["churn_events"][tables["churn_events"]["churn_date"] < cutoff].copy()

    # The cohort already excludes later signups, but the invariant "nothing this
    # function returns is dated at or after the cutoff" should hold on its own so
    # the audit can assert it without exceptions.
    acc_t = tables["accounts"][tables["accounts"]["signup_date"] < cutoff].copy()

    return {
        "accounts": acc_t,
        "subscriptions": subs_t,
        "feature_usage": usage_t,
        "support_tickets": tix_t,
        "churn_events": ce_t,
    }


def cohort_summary(cohort, cutoff=CUTOFF_DATE, horizon_days=HORIZON_DAYS):
    n = len(cohort)
    pos = int(cohort["churned_next_180d"].sum())
    return pd.Series({
        "cutoff": str(cutoff.date()),
        "horizon_days": horizon_days,
        "eligible_accounts": n,
        "positives": pos,
        "positive_rate": round(pos / n, 4) if n else np.nan,
    })
