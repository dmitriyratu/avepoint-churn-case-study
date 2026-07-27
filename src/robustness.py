"""Design-dial sweeps and multi-cutoff pooling.

Every robustness number the project reports is computed here rather than read
from a file. An earlier version kept two of them as committed CSVs with no code
behind them; when the cohort definition changed, the CSVs went on reporting the
old population and nothing flagged it.
"""
from itertools import product

import numpy as np
import pandas as pd
from sklearn.model_selection import GroupKFold, StratifiedKFold, cross_val_score

from . import pipeline
from .config import HORIZON_DAYS, PREDICTION_START
from .model import CV, SEED, model_ladder, permutation_significance

# The families layered on top of the original count/level features.
ENRICHED_PREFIXES = ("usage_last_", "active_days_last_", "tickets_last_",
                     "accel_", "ticket_accel_", "usage_prior_")
ENRICHED_NAMES = ("usage_momentum", "usage_delta_90d", "recency_ratio_90d",
                  "usage_trend_slope", "mean_gap_days", "max_gap_days",
                  "mrr_std", "mrr_cv", "usage_span_days")


def _selected():
    """The rung the project ships, so sweeps and headline share one model."""
    return model_ladder()[2][1]


def _score(X, y, estimator, cv=CV):
    s = cross_val_score(estimator, X, y, cv=cv, scoring="roc_auc")
    return {"n": len(y), "positives": int(y.sum()), "rate": round(float(y.mean()), 3),
            "cv_auc": round(s.mean(), 4),
            "ci_lo": round(float(np.percentile(s, 2.5)), 4),
            "ci_hi": round(float(np.percentile(s, 97.5)), 4)}


def horizon_buffer_sweep(horizons=(30, 60, 90, 180), buffers=(0, 30, 60),
                         prediction_start=PREDICTION_START, estimator=None):
    """Score every (horizon, buffer) cell with the prediction window fixed.

    The buffer is lead time: it pulls the feature cutoff back from the window
    opening, so the model must warn before the customer is visibly leaving.
    Accounts churning during the buffer drop out, which is the point — nobody
    could have acted on them.

    No per-cell permutation test. Twelve p-values with the smallest highlighted
    is the selection error 09_classifier_sweep is about; under a true null the
    minimum of twelve is small by construction. Cells carry intervals instead,
    and one test runs on the pre-specified primary cell.
    """
    estimator = estimator or _selected()

    def cell(horizon, buffer):
        data = pipeline.build(cutoff=prediction_start - pd.Timedelta(days=buffer),
                              prediction_start=prediction_start, horizon_days=horizon)
        return {"horizon": horizon, "buffer": buffer,
                **_score(data.X, data.y, estimator)}

    return pd.DataFrame(cell(h, b) for h, b in product(horizons, buffers))


def primary_significance(X, y, estimator=None, n_permutations=300):
    """One permutation test, on the configuration chosen before looking."""
    return permutation_significance(estimator or _selected(), X, y,
                                    n_permutations=n_permutations)


def feature_set_comparison(X, y, estimator=None, cv=CV):
    """Baseline features against the full enriched set, on identical folds."""
    estimator = estimator or _selected()
    enriched = [c for c in X.columns
                if c.startswith(ENRICHED_PREFIXES) or c in ENRICHED_NAMES]

    def row(label, frame):
        s = cross_val_score(estimator, frame, y, cv=cv, scoring="roc_auc")
        return {"feature_set": f"{label} ({frame.shape[1]} features)",
                "n_features": frame.shape[1], "cv_auc": round(s.mean(), 4),
                "sd": round(s.std(), 4)}

    return pd.DataFrame([row("baseline", X.drop(columns=enriched)),
                         row("enriched", X)]), enriched


def rolling_origin_cutoffs(n=4, last="2024-09-30", freq="QE"):
    """`n` quarter-end cutoffs, most recent last.

    The final one must leave a full horizon before the data ends: 2024-09-30
    plus 90 days is 2024-12-29, inside the 2024-12-31 extract.
    """
    return list(pd.date_range(end=last, periods=n, freq=freq))


def pooled_dataset(cutoffs, horizon_days=HORIZON_DAYS):
    """One row per (account, cutoff), pooled. Returns (X, y, groups).

    The cohort is small because it is cut at a single date. Rebuilding it at
    several dates multiplies labelled rows without inventing any — an account
    contributes once per date it was at risk. `groups` is account_id, because
    those rows are far from independent.

    prune=False so every cutoff yields the same columns; pruning is
    data-dependent and would otherwise drop a different set per date.
    """
    built = [pipeline.build(cutoff=pd.Timestamp(c), prediction_start=pd.Timestamp(c),
                            horizon_days=horizon_days, prune=False) for c in cutoffs]
    shared = sorted(set.intersection(*(set(d.X.columns) for d in built)))
    return (pd.concat([d.X[shared] for d in built], ignore_index=True),
            pd.concat([d.y for d in built], ignore_index=True),
            pd.concat([d.cohort["account_id"] for d in built], ignore_index=True))


def pooled_cv(X, y, groups, estimator=None, n_splits=5):
    """Account-grouped CV over the pooled cohort, with the ungrouped contrast.

    The ungrouped figure is reported alongside on purpose: it is what pooling
    looks like when you forget the same customer appears at several cutoffs.
    """
    estimator = estimator or _selected()
    grouped = cross_val_score(estimator, X, y, scoring="roc_auc",
                              cv=GroupKFold(n_splits).split(X, y, groups))
    ungrouped = cross_val_score(estimator, X, y, scoring="roc_auc",
                                cv=StratifiedKFold(n_splits, shuffle=True,
                                                   random_state=SEED))
    return pd.Series({
        "n": len(y), "positives": int(y.sum()),
        "distinct_accounts": int(groups.nunique()),
        "grouped_auc": round(grouped.mean(), 4),
        "grouped_sd": round(grouped.std(), 4),
        "ungrouped_auc": round(ungrouped.mean(), 4),
        "grouping_optimism": round(ungrouped.mean() - grouped.mean(), 4),
    })
