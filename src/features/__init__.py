"""Account-level features, evaluated as of a cutoff date.

Input tables must already be truncated to the observation window by
`labeling.truncate_tables`; `as_of` is used only for recency arithmetic.

One module per feature family. Each returns a frame indexed by `account_id`, so
`assemble` composes them with `pd.concat` rather than a chain of merges.

Excluded by design (see docs/DATA_DICTIONARY.md): anything derived from
`churn_events`, `subscriptions.churn_flag`, and `arr_amount`.
"""
from ._helpers import COUNT_PREFIXES, RECENCY_COLS, TREND_WINDOW_DAYS, WINDOWS
from .assemble import build_model_dataset, drop_collinear
from .subscription import subscription_features
from .support import support_features
from .usage import usage_features

__all__ = [
    "build_model_dataset", "drop_collinear",
    "subscription_features", "support_features", "usage_features",
    "WINDOWS", "TREND_WINDOW_DAYS", "RECENCY_COLS", "COUNT_PREFIXES",
]
