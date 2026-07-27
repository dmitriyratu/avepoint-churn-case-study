"""Subscription-derived features: size, direction, tenure, plan movement."""
import pandas as pd

from ..config import CUTOFF_DATE
from ._helpers import safe_div


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
    latest = ordered.last()[["plan_tier", "billing_frequency", "seats", "mrr_amount",
                             "is_trial"]]
    latest.columns = ["latest_plan_tier", "billing_freq", "latest_seats", "latest_mrr",
                      "latest_is_trial"]
    first = ordered.first()[["seats", "mrr_amount"]]
    first.columns = ["first_seats", "first_mrr"]

    feats = pd.concat([feats, latest, first], axis=1)

    # Direction of travel, not just size.
    feats["seat_growth"] = feats["latest_seats"] - feats["first_seats"]
    feats["mrr_growth"] = feats["latest_mrr"] - feats["first_mrr"]
    feats["mrr_growth_pct"] = safe_div(feats["mrr_growth"], feats["first_mrr"])
    feats["upgrade_net"] = feats["n_upgrades"] - feats["n_downgrades"]
    feats["mrr_cv"] = safe_div(feats["mrr_std"], feats["avg_mrr"])
    feats["pct_subs_ended"] = feats["n_ended_subs"] / feats["n_subscriptions"]

    return feats
