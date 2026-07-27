"""Join the feature blocks onto the cohort and prune what carries no signal."""
import numpy as np
import pandas as pd

from ..config import CUTOFF_DATE
from ._helpers import COLLINEARITY_THRESHOLD, COUNT_PREFIXES, RECENCY_COLS, safe_div
from .subscription import subscription_features
from .support import support_features
from .usage import usage_features


def drop_collinear(df, threshold=COLLINEARITY_THRESHOLD, protect=()):
    """Drop the later column of each near-duplicate pair.

    Keeping both halves of a pair like `feature_breadth` and
    `unique_features_used` just splits one effect across two coefficients.
    """
    corr = df.select_dtypes(include=[np.number]).corr().abs()
    candidates = [c for c in corr.columns if c not in protect]
    dropped = []
    for i, a in enumerate(candidates):
        if a in dropped:
            continue
        pair = corr.loc[a, candidates[i + 1:]]
        dropped += [b for b in pair[pair > threshold].index if b not in dropped]
    return df.drop(columns=dropped), dropped


def build_model_dataset(tables, cohort, as_of=CUTOFF_DATE, prune=True):
    """Cohort joined to every feature block, ready for `model.prep_xy`."""
    blocks = [
        subscription_features(tables["subscriptions"], as_of),
        usage_features(tables["feature_usage"], tables["subscriptions"], as_of),
        support_features(tables["support_tickets"], as_of),
    ]
    features = pd.concat([b for b in blocks if not b.empty], axis=1)

    df = cohort.set_index("account_id").join(features)
    df["days_since_signup"] = (as_of - df["signup_date"]).dt.days

    seats = df["seats"].replace(0, np.nan)
    df["usage_per_seat"] = safe_div(df["total_usage_events"], seats)
    df["tickets_per_seat"] = safe_div(df["n_tickets"], seats)
    df["mrr_per_seat"] = safe_div(df["total_mrr"], seats)

    window_len = (as_of - tables["feature_usage"]["usage_date"].min()).days
    df[list(RECENCY_COLS)] = df[list(RECENCY_COLS)].fillna(window_len)

    counts = [c for c in df.columns if c.startswith(COUNT_PREFIXES)]
    df[counts] = df[counts].fillna(0)

    df = df.round(4).reset_index()

    # Categoricals stay as strings; the pipeline encodes them per fold.
    protect = tuple(cohort.columns)
    constant = [c for c in df.columns if c not in protect and df[c].nunique() <= 1]
    df = df.drop(columns=constant)

    collinear = []
    if prune:
        df, collinear = drop_collinear(df, protect=protect)

    df.attrs.update(dropped_constant=constant, dropped_collinear=collinear)
    return df
