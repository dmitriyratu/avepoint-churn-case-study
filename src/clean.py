"""Cleaning: parsing, deduplication, and structural fixes only.

Nothing here learns a parameter from the data. Imputation is a stateful step and
lives inside the CV pipeline (`model._pipe`), so its statistics are fit on
training rows only. Rationale for each decision is in docs/CLEANING_CHECKLIST.md.
"""
import pandas as pd

DATE_COLS = {
    "accounts": ["signup_date"],
    "subscriptions": ["start_date", "end_date"],
    "feature_usage": ["usage_date"],
    "support_tickets": ["submitted_at", "closed_at"],
    "churn_events": ["churn_date"],
}


def clean_all(tables):
    out = {name: df.assign(**{c: pd.to_datetime(df[c]) for c in cols})
           for name, (df, cols) in
           ((n, (tables[n], DATE_COLS[n])) for n in tables)}

    # arr_amount == mrr_amount * 12 for every row; perfectly collinear.
    out["subscriptions"] = out["subscriptions"].drop(columns="arr_amount")

    # Duplicated usage_id would inflate per-account event counts.
    out["feature_usage"] = out["feature_usage"].drop_duplicates(subset="usage_id")

    # Missingness is recorded, never filled — see module docstring. Only the
    # ticket flag is kept: churn_events supplies no features at all, so a
    # missingness indicator on its feedback text has nowhere to go.
    out["support_tickets"] = out["support_tickets"].assign(
        satisfaction_missing=lambda d: d["satisfaction_score"].isna().astype(int))

    return out


def integrity_report(tables):
    """Source-data violations worth surfacing rather than silently repairing."""
    acc, subs = tables["accounts"], tables["subscriptions"]
    usage, tix, ce = (tables["feature_usage"], tables["support_tickets"],
                      tables["churn_events"])

    u = usage.merge(subs[["subscription_id", "start_date", "end_date"]],
                    on="subscription_id", how="left")
    t = tix.merge(acc[["account_id", "signup_date"]], on="account_id", how="left")
    ended = u["end_date"].notna()

    checks = [
        ("tickets dated before account signup",
         (t["submitted_at"] < t["signup_date"]).sum(), len(tix)),
        ("usage dated before its subscription starts",
         (u["usage_date"] < u["start_date"]).sum(), len(u)),
        ("usage dated after its subscription ends",
         (ended & (u["usage_date"] > u["end_date"])).sum(), ended.sum()),
        ("accounts whose churn_flag disagrees with churn_events",
         (acc["account_id"].isin(ce["account_id"]) != acc["churn_flag"]).sum(), len(acc)),
    ]
    return pd.DataFrame(checks, columns=["issue", "n_violations", "n_rows"]).astype(
        {"n_violations": int, "n_rows": int})
