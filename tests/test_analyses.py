"""Invariants for the product-question analyses (notebooks 11-15).

The pipeline suite guards leakage and point-in-time correctness. These guard the
properties that would make the *analysis* wrong rather than the features:
censoring handled correctly, retrospective columns kept out of the model path,
estimators returning quantities inside their own valid range, and every
null-comparison actually comparing against a null.

Several assert bugs that were present in earlier versions of this code and
produced plausible output — a trimming step that silently trimmed nothing, an
AIPW estimate outside [0, 1], a retention triangle filled past the observable
horizon.

    pytest tests/ -q
"""
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

sys.path.insert(0, str(Path(__file__).parents[1]))

from src import audit, causal, drivers, economics, pipeline, reasons, survival
from src.clean import clean_all
from src.config import CUTOFF_DATE, EXTRACT_DATE, POST_OUTCOME_COLS
from src.load_data import load_all


@pytest.fixture(scope="module")
def tables():
    return clean_all(load_all())


@pytest.fixture(scope="module")
def data():
    return pipeline.build()


@pytest.fixture(scope="module")
def surv(tables):
    return survival.survival_frame(tables)


# --------------------------------------------------------------------------
# The label-coherence claim the deck leads with
# --------------------------------------------------------------------------

def test_churn_sources_are_unrelated_not_merely_inconsistent(tables):
    """The deck's headline finding, pinned.

    The claim is specifically *independence*, not low agreement. Raw agreement
    would be a much weaker result: with rates of 22% and 70.4%, unrelated
    columns already agree ~39% of the time, so "37.6% agreement" alone invites
    the rebuttal that the flag is informative-but-inverted. It is not.
    """
    report = audit.label_source_agreement(tables).set_index(["source_a", "source_b"])

    flag_vs_events = report.loc[("churn_flag", "churn_events")]
    assert flag_vs_events["observed"] == pytest.approx(0.376, abs=0.001)
    assert flag_vs_events["expected_if_unrelated"] == pytest.approx(0.386, abs=0.001)

    # Every pair sits on its own chance baseline: no association, in either
    # direction, between any two recordings of the same event.
    assert (report["kappa"].abs() < 0.1).all()
    assert (report["p_value"] > 0.05).all()


def test_inverting_the_flag_does_not_recover_a_signal(tables):
    """Pre-empts the obvious challenge to the slide."""
    accounts, events = tables["accounts"], tables["churn_events"]
    has_event = accounts["account_id"].isin(events["account_id"])
    inverted = ~accounts["churn_flag"].astype(bool)

    agreement = (inverted == has_event).mean()
    pa, pb = inverted.mean(), has_event.mean()
    by_chance = pa * pb + (1 - pa) * (1 - pb)

    assert agreement == pytest.approx(0.624, abs=0.001)
    assert agreement - by_chance < 0.02  # 62.4% observed vs 61.4% by chance


def test_churn_dates_do_not_coincide_with_subscriptions_ending(tables):
    """The form of the argument that never touches churn_flag."""
    windows, meta = audit.churn_date_coherence(tables)

    same_day = windows.loc[windows["window_days"] == 0, "pct_of_comparable"].iloc[0]
    assert same_day < 5.0                      # 1.6% — two systems, same event
    assert meta["median_gap_days"] > 30        # 62 days
    assert meta["comparable"] < meta["events"]  # 214 events have nothing to compare


# --------------------------------------------------------------------------
# Retrospective analysis must not leak into the model path
# --------------------------------------------------------------------------

def test_reason_columns_never_reach_the_feature_matrix(data):
    """`reasons.py` reads churn_events on purpose; `features/` must not."""
    forbidden = {"reason_code", "feedback_text", "refund_amount_usd",
                 "is_reactivation", "preceding_upgrade_flag",
                 "preceding_downgrade_flag"}
    assert not forbidden & set(data.X.columns)
    assert not set(POST_OUTCOME_COLS) & set(data.X.columns)


def test_retrospective_counts_dominate_point_in_time_counts(tables, data):
    """`reasons.account_behaviour` and `features/` share column *names* —
    `n_tickets`, `n_upgrades` — computed over different windows. That is fine
    and easy to confuse, so the invariant is asserted rather than trusted: the
    whole-history count must never be *below* the truncated one, because the
    observation window is a subset of all history.

    A violation would mean the retrospective summary is itself being truncated,
    or the feature is seeing rows past the cutoff.
    """
    behaviour = reasons.account_behaviour(tables)
    features = data.X.copy()
    features.index = pd.Index(data.cohort["account_id"].values, name="account_id")

    shared = sorted(set(behaviour.columns) & set(features.columns))
    assert shared, "expected the two paths to share count columns"

    for column in shared:
        retrospective = behaviour.loc[features.index, column]
        assert (retrospective + 1e-9 >= features[column]).all(), (
            f"{column}: point-in-time value exceeds whole-history value")


def test_retention_triangle_leaves_unobservable_cells_empty(tables):
    """A three-month-old cohort has no month-six number; it must stay NaN."""
    triangle = reasons.retention_triangle(tables, extract_date="2024-12-31")
    last_cohort = triangle.iloc[-1]
    assert last_cohort.isna().any(), "youngest cohort was filled past its horizon"
    # Retention is a share and can only fall as the window widens.
    for _, row in triangle.iterrows():
        observed = row.dropna()
        assert (observed.diff().dropna() <= 1e-9).all()
        assert observed.between(0, 1).all()


# --------------------------------------------------------------------------
# Survival: censoring is the thing that is easy to get wrong
# --------------------------------------------------------------------------

def test_censored_accounts_are_not_counted_as_events(surv, tables):
    churned = set(tables["churn_events"]["account_id"])
    assert int(surv["event"].sum()) == len(churned & set(surv.index))
    assert (surv.loc[surv["event"] == 0, "duration"] > 0).all()


def test_no_duration_extends_past_the_extract(surv, tables):
    """Follow-up cannot run beyond the data. Catches a wrong censoring date."""
    signup = tables["accounts"].set_index("account_id")["signup_date"]
    latest = (signup.loc[surv.index]
              + pd.to_timedelta(surv["duration"], unit="D"))
    assert (latest <= EXTRACT_DATE + pd.Timedelta(days=1)).all()


def test_survival_covariates_are_baseline_not_current_state(surv):
    """`accounts.plan_tier`/`seats`/`is_trial` are as-of-extraction, so using
    them as baseline covariates conditions on the future."""
    assert "base_plan_tier" in surv.columns
    assert "plan_tier" not in surv.columns
    assert "seats" not in surv.columns


def test_cohort_survival_frame_has_no_negative_follow_up(data, tables):
    frame = survival.cohort_survival_frame(data.cohort, tables, CUTOFF_DATE)
    assert (frame["duration"] > 0).all()
    assert frame["event"].isin([0, 1]).all()
    # Every event must land inside the observable follow-up window.
    assert (frame.loc[frame["event"] == 1, "duration"]
            <= frame.attrs["followup_days"]).all()


def test_survival_framing_recovers_more_events_than_the_binary_label(data, tables):
    """The point of the reframing: censoring-aware follow-up beats a 90-day flag."""
    frame = survival.cohort_survival_frame(data.cohort, tables, CUTOFF_DATE)
    assert int(frame["event"].sum()) > int(data.y.sum())


def test_km_survival_is_monotone_and_bounded(surv):
    table = survival.km_summary(surv)
    assert table["survival"].is_monotonic_decreasing
    assert table["survival"].between(0, 1).all()
    assert (table["ci_lo"] <= table["survival"]).all()
    assert (table["survival"] <= table["ci_hi"]).all()


def test_calendar_hazard_never_exceeds_one(tables):
    """events / at_risk is a share; above 1 means the risk set is wrong."""
    hazard = survival.calendar_hazard(tables)
    usable = hazard[hazard["at_risk"] > 0]
    assert usable["hazard"].between(0, 1).all()
    assert (usable["events"] <= usable["at_risk"]).all()


def test_within_cohort_shape_only_uses_cohorts_with_enough_followup(surv):
    within = survival.shape_within_cohorts(surv, min_followup=365)
    for cohort in within["cohort"]:
        assert surv.loc[surv["cohort"] == cohort, "duration"].max() >= 365


# --------------------------------------------------------------------------
# Causal: the estimator must stay inside its own valid range
# --------------------------------------------------------------------------

def test_treatment_derived_columns_are_dropped_from_confounders(data):
    """Conditioning on a function of the treatment blocks the estimated path."""
    for name, spec in causal.TREATMENTS.items():
        _, confounders = causal.make_treatment(data.X, name)
        assert not set(spec["derived"]) & set(confounders.columns)


def test_aipw_returns_rates_inside_the_unit_interval(data):
    """An unbounded AIPW mean produced a churn rate of -0.17 in an earlier
    version; potential-outcome means must be clipped."""
    treatment, confounders = causal.make_treatment(data.X, "upgrade")
    estimate = causal.aipw_ate(confounders, treatment, data.y, n_boot=50)
    assert 0 <= estimate["control_rate"] <= 1
    assert 0 <= estimate["treated_rate"] <= 1
    assert -1 <= estimate["ate"] <= 1
    assert estimate["ci_lo"] <= estimate["ate"] <= estimate["ci_hi"]


def test_trimming_actually_trims(data):
    """Regression: clipping the propensity before testing it against the same
    bounds made trimming a silent no-op that always reported zero."""
    treatment, confounders = causal.make_treatment(data.X, "auto_renew")
    propensity = causal.cross_fit_propensity(confounders, treatment)
    outside = int((~propensity.between(causal.TRIM, 1 - causal.TRIM)).sum())
    estimate = causal.aipw_ate(confounders, treatment, data.y, n_boot=20)
    assert estimate["n_trimmed"] == outside
    assert estimate["n_analysed"] == len(data.y) - outside


def test_placebo_effects_are_centred_on_zero(data):
    """A randomly assigned treatment has a true ATE of exactly zero. If the
    placebo distribution is off-centre, the estimator is biased."""
    treatment, confounders = causal.make_treatment(data.X, "upgrade")
    placebo = causal.placebo_ate(confounders, data.y, float(treatment.mean()),
                                 n_placebo=12)
    assert abs(placebo["mean"]) < 3 * placebo["sd"]
    assert placebo["detectable_effect"] > 0


def test_e_value_is_at_least_one_and_symmetric():
    """E-values are defined on the risk-ratio scale; RR and 1/RR must agree."""
    for rr in (0.5, 0.7, 1.4, 2.0):
        assert causal.e_value(rr)["e_value_point"] >= 1.0
    assert (causal.e_value(0.5)["e_value_point"]
            == causal.e_value(2.0)["e_value_point"])
    # An interval spanning the null needs no confounding to explain it.
    assert causal.e_value(1.3, lower=0.9)["e_value_ci"] == 1.0


def test_uplift_is_out_of_fold(data):
    """In-sample uplift scores manufacture a convincing Qini curve on their own."""
    treatment, confounders = causal.make_treatment(data.X, "upgrade")
    uplift = causal.t_learner_uplift(confounders, treatment, data.y)
    assert len(uplift) == len(data.y)
    assert uplift.between(-1, 1).all()
    # Out-of-fold predictions are not constant and not perfectly separating.
    assert uplift.nunique() > 10


def test_qini_curve_ends_at_the_overall_effect(data):
    treatment, confounders = causal.make_treatment(data.X, "upgrade")
    uplift = causal.t_learner_uplift(confounders, treatment, data.y)
    curve = causal.qini_curve(uplift, treatment, data.y)
    assert curve["fraction"].iloc[-1] == pytest.approx(1.0)
    # By construction the model curve and the random line meet at the endpoint.
    assert curve["qini"].iloc[-1] == pytest.approx(curve["random"].iloc[-1])


# --------------------------------------------------------------------------
# Drivers: every importance measure needs its null
# --------------------------------------------------------------------------

def test_shap_importance_covers_every_encoded_column(data):
    importance, values, encoded = drivers.shap_importance(data.X, data.y)
    assert len(importance) == encoded.shape[1]
    assert values.shape == (len(data.y), encoded.shape[1])
    assert (importance >= 0).all()


def test_concentration_is_bounded(data):
    importance, _, _ = drivers.shap_importance(data.X, data.y)
    gini = drivers.concentration(importance)
    assert 0 <= gini <= 1
    assert np.isnan(drivers.concentration(pd.Series(dtype=float)))


def test_ale_is_centred(data):
    """ALE is reported relative to the average account, so it must integrate to
    roughly zero under its own weights."""
    booster, encoded = drivers.fit_encoded(data.X, data.y)
    importance, _, _ = drivers.shap_importance(data.X, data.y)
    curve = drivers.ale(booster, encoded, importance.index[0])
    assert len(curve) > 1
    assert abs(np.average(curve["ale"], weights=curve["n"])) < 1e-8


# --------------------------------------------------------------------------
# Economics
# --------------------------------------------------------------------------

def test_break_even_precision_matches_its_definition():
    """p * e * V = C at the break-even point, by construction."""
    cost, value, effectiveness = 150.0, 7000.0, 0.2
    p = economics.break_even_precision(cost, value, effectiveness)
    assert p * effectiveness * value == pytest.approx(cost)
    # Cheaper campaigns and more valuable customers both lower the bar.
    assert economics.break_even_precision(50, value, effectiveness) < p
    assert economics.break_even_precision(cost, 2 * value, effectiveness) < p


def test_clv_is_discounted_below_its_undiscounted_value():
    curve = pd.Series(np.linspace(1.0, 0.2, 400), index=np.arange(1, 401))
    clv = economics.clv_from_survival(curve, monthly_revenue=1000.0)
    assert 0 < clv["clv"] < clv["undiscounted"]
    assert clv["expected_days_retained"] <= clv["horizon_days"]


def test_campaign_value_arithmetic():
    y = np.array([1, 1, 0, 0, 1, 0])
    scores = np.array([0.9, 0.8, 0.7, 0.2, 0.1, 0.05])
    result = economics.campaign_value(y, scores, 0.7, cost=100.0, value=1000.0,
                                      effectiveness=0.5)
    assert result["n_targeted"] == 3
    assert result["true_positives"] == 2
    assert result["customers_saved"] == pytest.approx(1.0)
    assert result["net_value"] == pytest.approx(1.0 * 1000.0 - 3 * 100.0)


def test_sample_size_grows_as_the_effect_shrinks():
    sizes = [economics.sample_size(0.3, e) for e in (0.15, 0.10, 0.05, 0.03)]
    assert all(np.isfinite(sizes))
    assert sizes == sorted(sizes)


def test_mde_and_sample_size_are_inverses():
    """Solving each way must land in the same place."""
    baseline, n = 0.30, 400
    mde = economics.minimum_detectable_effect(baseline, n)
    assert economics.sample_size(baseline, mde) == pytest.approx(n, rel=0.02)


def test_decision_curve_treat_all_matches_prevalence_at_zero_threshold():
    y = np.array([1, 0, 1, 0, 1, 0, 0, 0, 1, 0])
    scores = np.linspace(0.05, 0.95, 10)
    dca = economics.decision_curve(y, scores, thresholds=np.array([0.001]))
    assert dca["net_benefit_treat_all"].iloc[0] == pytest.approx(y.mean(), abs=1e-3)
    assert (dca["net_benefit_treat_none"] == 0).all()
