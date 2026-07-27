"""Support-desk features: load, responsiveness, escalation, trend."""
import pandas as pd

from ..config import CUTOFF_DATE
from ._helpers import safe_div, trailing


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

    feats = pd.concat([feats, trailing(t, (30, 90, 180), "tickets_last")], axis=1)

    # Rising support load is a churn precursor; the ratio matters more than the
    # count, since heavy users open more tickets in absolute terms.
    feats["ticket_accel_30d_vs_90d"] = safe_div(
        feats["tickets_last_30d"] / 30, feats["tickets_last_90d"] / 90)

    return feats
