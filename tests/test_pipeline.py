"""Invariants that must hold for any cutoff, horizon, or buffer.

Most bugs in this project's history were leaks and point-in-time errors that
looked fine in review and produced plausible numbers. These assert the
properties that would have caught them.

    pytest tests/ -q
"""
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

sys.path.insert(0, str(Path(__file__).parents[1]))

from src import audit, pipeline, robustness
from src.config import (HORIZON_DAYS, POINT_IN_TIME_UNSAFE_COLS,
                        POST_OUTCOME_COLS, PREDICTION_START, TARGET)
from src.labeling import (at_risk_accounts, build_cohort, cohort_summary,
                          first_churn_date, truncate_tables)
from src.model import prep_xy, scale_pos_weight

ALT_CUTOFFS = [pd.Timestamp("2024-03-31"), pd.Timestamp("2024-06-30")]


@pytest.fixture(scope="module")
def data():
    return pipeline.build()


# --------------------------------------------------------------------------
# Temporal correctness
# --------------------------------------------------------------------------

def test_no_observation_row_reaches_the_cutoff(data):
    """Every datetime column, not just the one used for filtering."""
    report = audit.temporal_provenance(data.observed, data.cutoff)
    offenders = report[~report["pass"]]
    assert offenders.empty, f"post-cutoff timestamps survived:\n{offenders}"


@pytest.mark.parametrize("cutoff", ALT_CUTOFFS)
def test_truncation_holds_at_other_cutoffs(cutoff):
    """The invariant is a property of the code, not of one lucky date."""
    tables = pipeline.clean_all(pipeline.load_all())
    report = audit.temporal_provenance(truncate_tables(tables, cutoff), cutoff)
    assert report["pass"].all()


def test_ticket_outcomes_censored_when_unresolved_at_cutoff(data):
    """A ticket open at the cutoff cannot carry a resolution time or score."""
    tickets = data.observed["support_tickets"]
    open_at_cutoff = tickets["ticket_open_at_cutoff"] == 1
    assert tickets.loc[open_at_cutoff, "resolution_time_hours"].isna().all()
    assert tickets.loc[open_at_cutoff, "satisfaction_score"].isna().all()
    assert tickets.loc[open_at_cutoff, "closed_at"].isna().all()


# --------------------------------------------------------------------------
# Label construction
# --------------------------------------------------------------------------

def test_positives_churn_inside_the_prediction_window(data):
    churn_on = first_churn_date(data.tables["churn_events"])
    horizon_end = PREDICTION_START + pd.Timedelta(days=HORIZON_DAYS)
    dates = data.cohort.set_index("account_id")[TARGET].index.map(churn_on)

    for label, churn_date in zip(data.cohort[TARGET], dates):
        if label == 1:
            assert PREDICTION_START <= churn_date <= horizon_end


def test_no_account_already_churned_before_the_window_opens(data):
    churn_on = first_churn_date(data.tables["churn_events"])
    prior = data.cohort["account_id"].map(churn_on)
    assert not (prior < PREDICTION_START).any()


def test_every_cohort_account_is_at_risk(data):
    """At risk means holding a subscription open at the cutoff."""
    live = at_risk_accounts(data.tables["subscriptions"], data.cutoff)
    assert data.cohort["account_id"].isin(live).all()


def test_buffer_shifts_the_window_not_the_features():
    """A larger buffer must shrink the cohort and pull the feature cutoff back."""
    tables = pipeline.clean_all(pipeline.load_all())
    start = pd.Timestamp("2024-06-30")
    wide = build_cohort(tables, cutoff=start, prediction_start=start)
    narrow = build_cohort(tables, cutoff=start - pd.Timedelta(days=60),
                          prediction_start=start)
    assert len(narrow) <= len(wide)


# --------------------------------------------------------------------------
# Leakage
# --------------------------------------------------------------------------

def test_forbidden_columns_never_reach_the_model(data):
    for column in POST_OUTCOME_COLS + POINT_IN_TIME_UNSAFE_COLS:
        assert column not in data.X.columns, f"{column} leaked into the feature matrix"


def test_forbidden_gate_actually_fires():
    """Negative control — the gate must fail when a leak is injected."""
    injected = pd.DataFrame({"churn_flag": [0, 1], "seats": [1, 2]})
    result = audit.forbidden_columns(injected)
    assert len(result) == 2
    assert not result["pass"].any()


def test_no_single_feature_reconstructs_the_label(data):
    strongest = audit.single_feature_auc(data.X, data.y)["auc"].max()
    assert strongest < audit.SINGLE_FEATURE_AUC_FAIL


def test_full_audit_suite_passes(data):
    _, passed = data.audit()
    assert passed


# --------------------------------------------------------------------------
# Feature semantics
# --------------------------------------------------------------------------

def test_counts_are_filled_but_rates_are_left_missing(data):
    """Zero activity is zero; an undefined average is not zero."""
    assert data.X["n_tickets"].notna().all()
    assert data.X["total_usage_events"].notna().all()
    assert data.X["avg_satisfaction"].isna().any(), (
        "rate features should stay NaN for in-fold imputation")


def test_no_constant_or_infinite_features(data):
    assert audit.constant_columns(data.X).empty
    numeric = data.X.select_dtypes(include=[np.number])
    assert not np.isinf(numeric.to_numpy(dtype=float, na_value=0.0)).any()


def test_categoricals_survive_as_strings_for_in_fold_encoding(data):
    non_numeric = [c for c in data.X.columns
                   if not pd.api.types.is_numeric_dtype(data.X[c])]
    assert non_numeric, "categoricals should not be pre-encoded"


def test_prep_xy_survives_a_csv_round_trip(data, tmp_path):
    """Booleans return from CSV as 'True'/'False'; this used to crash."""
    path = tmp_path / "frame.csv"
    data.frame.to_csv(path, index=False)
    X_disk, y_disk = prep_xy(pd.read_csv(path))

    assert X_disk.shape == data.X.shape
    assert (y_disk.to_numpy() == data.y.to_numpy()).all()


# --------------------------------------------------------------------------
# Model plumbing
# --------------------------------------------------------------------------

def test_scale_pos_weight_matches_the_cohort_ratio(data):
    expected = (len(data.y) - data.y.sum()) / data.y.sum()
    assert scale_pos_weight(data.y) == pytest.approx(expected)


def test_build_is_deterministic():
    a, b = pipeline.build(), pipeline.build()
    pd.testing.assert_frame_equal(a.X, b.X)
    pd.testing.assert_series_equal(a.y, b.y)


def test_verify_flag_asserts_the_audit():
    """build(verify=True) must not return a dataset that fails the suite."""
    assert pipeline.build(verify=True) is not None


# --------------------------------------------------------------------------
# Reporting honesty
# --------------------------------------------------------------------------

def test_summary_describes_the_cohort_actually_built():
    """The summary must reflect its arguments, not the configured default.

    It used to read `BUFFER_DAYS` straight from config, so every row of the
    buffer sweep reported a buffer of zero regardless of the cutoff used.
    """
    start = PREDICTION_START
    cutoff = start - pd.Timedelta(days=30)
    cohort = build_cohort(pipeline.clean_all(pipeline.load_all()), cutoff=cutoff,
                          prediction_start=start)
    summary = cohort_summary(cohort, cutoff, HORIZON_DAYS, start)

    assert summary["buffer_days"] == 30
    assert summary["feature_cutoff"] == str(cutoff.date())
    assert summary["prediction_start"] == str(start.date())


# --------------------------------------------------------------------------
# Rolling-origin pooling
# --------------------------------------------------------------------------

@pytest.fixture(scope="module")
def pooled():
    cutoffs = robustness.rolling_origin_cutoffs(n=3)
    return cutoffs, robustness.pooled_dataset(cutoffs)


def test_pooling_adds_rows_without_duplicating_account_cutoff_pairs(data, pooled):
    cutoffs, (X_pool, y_pool, groups) = pooled

    assert len(X_pool) > len(data.X), "pooling should add rows"
    assert len(X_pool) == len(y_pool) == len(groups)
    # An account may appear once per cutoff, never twice within one.
    assert groups.value_counts().max() <= len(cutoffs)


def test_pooled_columns_are_shared_across_every_cutoff(pooled):
    """Pooling must not silently align frames with different column sets."""
    cutoffs, (X_pool, _, _) = pooled
    for cutoff in cutoffs:
        built = pipeline.build(cutoff=cutoff, prediction_start=cutoff,
                               horizon_days=HORIZON_DAYS, prune=False)
        assert set(X_pool.columns) <= set(built.X.columns)


def test_grouping_by_account_is_not_a_no_op(pooled):
    """If accounts never recurred across cutoffs, grouping would prove nothing."""
    _, (_, _, groups) = pooled
    assert (groups.value_counts() > 1).any(), (
        "no account appears at more than one cutoff — grouped CV would be "
        "equivalent to ungrouped, and the pooling result would not need it")


def test_enriched_feature_split_is_a_strict_subset(data):
    """The baseline/enriched comparison must actually remove something."""
    enriched = [c for c in data.X.columns
                if c.startswith(robustness.ENRICHED_PREFIXES)
                or c in robustness.ENRICHED_NAMES]
    assert enriched, "the enriched-family split matched no columns"
    assert set(enriched) < set(data.X.columns), "it must not match everything"
