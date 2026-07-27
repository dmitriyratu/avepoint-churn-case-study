"""Temporal cohort construction.

    |<---- observation ---->|<- buffer ->|<---- prediction ---->|
    ...                  cutoff    prediction_start        + horizon
          features built here                    label defined here

Features come only from rows dated before the cutoff. The buffer is the lead
time between a score landing and the customer leaving; without it a model keys
on the collapse in activity that immediately precedes churn, which scores well
and arrives too late to act on.
"""
import numpy as np
import pandas as pd

from .config import CUTOFF_DATE, HORIZON_DAYS, PREDICTION_START, TARGET


def first_churn_date(churn_events):
    return churn_events.groupby("account_id")["churn_date"].min()


def at_risk_accounts(subs, cutoff):
    """Accounts holding a subscription open at the cutoff.

    One whose subscriptions had all ended cannot churn in the ordinary sense;
    counting it as a negative pads the denominator with customers already lost.
    """
    open_now = subs["end_date"].isna() | (subs["end_date"] >= cutoff)
    return subs.loc[(subs["start_date"] < cutoff) & open_now, "account_id"].unique()


def build_cohort(tables, cutoff=CUTOFF_DATE, horizon_days=HORIZON_DAYS,
                 prediction_start=PREDICTION_START):
    """Accounts at risk at `prediction_start`, with their forward-looking label.

    Accounts that churned before the window opens — including during the buffer
    — are dropped: at scoring time we could not have acted on them.
    """
    churned_on = tables["accounts"]["account_id"].map(
        first_churn_date(tables["churn_events"]))
    cohort = tables["accounts"].assign(churned_on=churned_on)
    cohort = cohort[
        (cohort["signup_date"] < cutoff)
        & cohort["account_id"].isin(at_risk_accounts(tables["subscriptions"], cutoff))
        & ~(cohort["churned_on"] < prediction_start)]

    # inclusive="both" matches the eligibility rule: it keeps churn dates on or
    # after the opening, so a churn landing on day zero is a positive.
    label = cohort["churned_on"].between(
        prediction_start, prediction_start + pd.Timedelta(days=horizon_days),
        inclusive="both")
    return cohort.assign(**{TARGET: label.astype(int)}).drop(columns="churned_on")


def truncate_tables(tables, cutoff=CUTOFF_DATE):
    """Clip every table to the observation window.

    Filtering on a row's start timestamp is not enough: a ticket opened in June
    and closed in July still carries a resolution time and satisfaction score
    nobody knew at the end of June. Those fields are censored too.
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

    responded = tix["submitted_at"] + pd.to_timedelta(
        tix["first_response_time_minutes"], "m")
    tix.loc[responded >= cutoff, "first_response_time_minutes"] = np.nan

    accounts, events = tables["accounts"], tables["churn_events"]
    return {"accounts": accounts[accounts["signup_date"] < cutoff],
            "subscriptions": subs,
            "feature_usage": usage,
            "support_tickets": tix,
            "churn_events": events[events["churn_date"] < cutoff]}


def cohort_summary(cohort, cutoff=CUTOFF_DATE, horizon_days=HORIZON_DAYS,
                   prediction_start=PREDICTION_START):
    """Describe the cohort actually built, not the configured default.

    The buffer is derived from the two dates. An earlier version read
    BUFFER_DAYS from config regardless of its arguments, so every row of the
    buffer sweep printed zero.
    """
    positives = int(cohort[TARGET].sum())
    return pd.Series({
        "feature_cutoff": str(cutoff.date()),
        "buffer_days": (prediction_start - cutoff).days,
        "prediction_start": str(prediction_start.date()),
        "horizon_days": horizon_days,
        "eligible_accounts": len(cohort),
        "positives": positives,
        "positive_rate": round(positives / len(cohort), 4) if len(cohort) else np.nan,
    })
