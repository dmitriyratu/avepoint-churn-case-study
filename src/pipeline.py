"""One-call construction of the modelling dataset.

Every notebook needs the same chain — load, clean, build the cohort, truncate to
the observation window, assemble features, split into X and y. `build()` does it
once and hands back the intermediates, so a notebook that only wants `X, y` does
not have to restate the pipeline to get them.

Parameterising it on the cutoff also makes the buffer sweep in
`03_feature_engineering.py` a one-liner per configuration.
"""
from dataclasses import dataclass

import pandas as pd

from . import audit
from .clean import clean_all
from .config import CUTOFF_DATE, HORIZON_DAYS, PREDICTION_START
from .features import build_model_dataset
from .labeling import build_cohort, cohort_summary, truncate_tables
from .load_data import load_all
from .model import prep_xy


@dataclass(frozen=True)
class Dataset:
    """The modelling frame plus the intermediates used to build it."""

    raw: dict
    tables: dict
    cohort: pd.DataFrame
    observed: dict
    frame: pd.DataFrame
    X: pd.DataFrame
    y: pd.Series
    cutoff: pd.Timestamp

    @property
    def summary(self):
        return cohort_summary(self.cohort, self.cutoff)

    @property
    def dropped(self):
        return {k: self.frame.attrs.get(k, []) for k in ("dropped_constant", "dropped_collinear")}

    def audit(self):
        """Run the full leakage suite. Returns (results, passed)."""
        return audit.run_all(self.X, self.y, self.frame, self.observed, self.cutoff,
                             raw_tables=self.raw)


def build(cutoff=CUTOFF_DATE, prediction_start=PREDICTION_START,
          horizon_days=HORIZON_DAYS, prune=True, verify=False):
    """Assemble the dataset as of `cutoff`.

    Set `verify=True` to assert the leakage suite before returning — worth doing
    anywhere a score will be reported.
    """
    raw = load_all()
    tables = clean_all(raw)
    cohort = build_cohort(tables, cutoff, horizon_days, prediction_start)
    observed = truncate_tables(tables, cutoff)
    frame = build_model_dataset(observed, cohort, cutoff, prune=prune)
    X, y = prep_xy(frame)

    data = Dataset(raw, tables, cohort, observed, frame, X, y, cutoff)
    if verify:
        _, passed = data.audit()
        assert passed, "leakage audit failed — refusing to return a dataset"
    return data


def build_at_buffer(buffer_days, prediction_start=PREDICTION_START,
                    horizon_days=HORIZON_DAYS):
    """Dataset with `buffer_days` of lead time before the prediction window."""
    return build(cutoff=prediction_start - pd.Timedelta(days=buffer_days),
                 prediction_start=prediction_start, horizon_days=horizon_days)
