"""Temporal cohort construction.

    |<---- observation ---->|<- buffer ->|<---- prediction ---->|
    ...                  cutoff    prediction_start        + horizon
          features built here                    label defined here

Features come only from rows dated before the cutoff. The buffer is the lead
time a retention team needs between a score landing and the customer leaving;
without it a model keys on the collapse in activity that immediately precedes
churn, which scores well and arrives too late to act on.
"""
import numpy as np
import pandas as pd

from .config import BUFFER_DAYS, CUTOFF_DATE, HORIZON_DAYS, PREDICTION_START, TARGET


def first_churn_date(churn_events):
    return churn_events.groupby("account_id")["churn_date"].min()


def build_cohort(tables, cutoff=CUTOFF_DATE, horizon_days=HORIZON_DAYS,
                 prediction_start=PREDICTION_START):
    """Accounts at risk at `prediction_start`, with their forward-looking label.

    Accounts that churned before the prediction window opens — including during
    the buffer — are dropped: at scoring time we could not have acted on them.
    """
    horizon_end = prediction_start + pd.Timedelta(days=horizon_days)
    accounts = tables["accounts"]
    churned_on = accounts["account_id"].map(first_churn_date(tables["churn_events"]))

    # At risk means holding a subscription that is open at the cutoff. An account
    # whose subscriptions had all ended cannot churn in the ordinary sense, and
    # including it as a negative would inflate the denominator with customers the
    # business has already lost.
    subs = tables["subscriptions"]
    at_risk = subs.loc[(subs["start_date"] < cutoff)
                       & (subs["end_date"].isna() | (subs["end_date"] >= cutoff)),
                       "account_id"].unique()

    cohort = accounts.assign(churned_on=churned_on)
    cohort = cohort[(cohort["signup_date"] < cutoff)
                    & cohort["account_id"].isin(at_risk)
                    & ~(cohort["churned_on"] < prediction_start)]

    # inclusive="both" so the window matches the eligibility rule exactly:
    # eligibility keeps churn_date >= prediction_start, so a churn landing on the
    # opening day must count as a positive rather than falling through as a zero.
    label = cohort["churned_on"].between(prediction_start, horizon_end, inclusive="both")
    return cohort.assign(**{TARGET: label.astype(int)}).drop(columns="churned_on")


def truncate_tables(tables, cutoff=CUTOFF_DATE):
    """Clip every table to the observation window.

    Filtering on a row's start timestamp is not enough: a ticket opened in June
    and closed in July still carries a resolution time and satisfaction score
    that were unknowable at the end of June. Those fields are censored so the
    feature layer cannot reach them.
    """
    subs = tables["subscriptions"]
    subs = subs[subs["start_date"] < cutoff].copy()
    subs.loc[subs["end_date"] >= cutoff, "end_date"] = pd.NaT

    usage = tables["feature_usage"]
    usage = usage[(usage["usage_date"] < cutoff)
                  & usage["subscription_id"].isin(subs["subscription_id"])]

    tix = tables["support_tickets"]
    tix = tix[tix["submitted_at"] < cutoff].copy()

    still_open = tix["closed_at"].isna() | (tix["closed_at"] >= cutoff)
    tix["ticket_open_at_cutoff"] = still_open.astype(int)
    tix.loc[still_open, "closed_at"] = pd.NaT
    tix.loc[still_open, ["resolution_time_hours", "satisfaction_score"]] = np.nan

    responded_at = tix["submitted_at"] + pd.to_timedelta(tix["first_response_time_minutes"], "m")
    tix.loc[responded_at >= cutoff, "first_response_time_minutes"] = np.nan

    return {
        "accounts": tables["accounts"][tables["accounts"]["signup_date"] < cutoff],
        "subscriptions": subs,
        "feature_usage": usage,
        "support_tickets": tix,
        "churn_events": tables["churn_events"][tables["churn_events"]["churn_date"] < cutoff],
    }


def cohort_summary(cohort, cutoff=CUTOFF_DATE, horizon_days=HORIZON_DAYS):
    positives = int(cohort[TARGET].sum())
    return pd.Series({
        "feature_cutoff": str(cutoff.date()),
        "buffer_days": BUFFER_DAYS,
        "prediction_start": str(PREDICTION_START.date()),
        "horizon_days": horizon_days,
        "eligible_accounts": len(cohort),
        "positives": positives,
        "positive_rate": round(positives / len(cohort), 4) if len(cohort) else np.nan,
    })
