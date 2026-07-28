"""Time-to-event analysis: *when* accounts churn, not just whether.

The classification framing everywhere else in this project asks one question at
one cutoff — "will this account churn in the 90 days after 2024-06-30" — and
answers it with 177 accounts and 54 positives. That throws away most of the
data: 352 of 500 accounts have an observed churn date, and every one of them
carries information about timing that a binary label at a fixed cutoff discards.

Survival analysis uses all of it, handles the right-censoring properly (148
accounts are still active at extraction and are *not* negatives — they are
unobserved), and separates three things the classification framing cannot tell
apart:

    tenure effect   hazard depends on how long the account has been a customer
    cohort effect   hazard depends on when the account signed up
    period effect   hazard depends on the calendar date

They imply completely different actions — fix onboarding, fix acquisition, or
find what changed in the business — so the decomposition is the point. The
functions below run it in that order, and the pooled result does not survive it.

**There is a fourth explanation this module cannot separate from the other
three, and on this dataset it is the right one.** The extract has a hard end
date, and a churn date drawn at random before it manufactures all three patterns
simultaneously. Nothing here tests for that; `src/generator.py` does, and it
withdraws the period-effect finding this module reports. Read `calendar_hazard`
and `km_at_tenure_by_cohort` with `generator.calendar_hazard_null` alongside.

Baseline covariates come from the *first* subscription, never from
`accounts.plan_tier` / `seats` / `is_trial`: those are documented as current
state as of extraction (docs/DATA_DICTIONARY.md), so conditioning on them is
conditioning on the future. That is the survival-analysis form of the same
point-in-time discipline `labeling.truncate_tables` enforces for the classifier.
"""
import numpy as np
import pandas as pd
import statsmodels.api as sm
from lifelines import (CoxPHFitter, ExponentialFitter, KaplanMeierFitter,
                       NelsonAalenFitter, WeibullFitter)
from lifelines.statistics import multivariate_logrank_test, proportional_hazard_test
from lifelines.utils import concordance_index
from scipy import stats
from sklearn.model_selection import KFold
from statsmodels.stats.multitest import multipletests

from .config import EXTRACT_DATE

# Immutable account attributes plus first-subscription terms. Everything here is
# knowable on the signup date, which is what a baseline covariate has to be.
BASELINE_SEGMENTS = ["industry", "country", "referral_source", "base_plan_tier",
                     "base_billing_frequency", "base_is_trial", "base_auto_renew_flag"]
BASELINE_NUMERIC = ["base_seats", "base_mrr_amount"]

HORIZONS = (30, 60, 90, 180, 365)


def survival_frame(tables, extract_date=EXTRACT_DATE):
    """One row per account: duration, event indicator, baseline covariates.

    Duration runs from signup to the first churn event, or to `extract_date` for
    accounts never seen churning. Those are **censored, not retained** — the
    distinction the binary label cannot express, and the reason 148 accounts that
    the classifier scores as negatives appear here as "unknown, still at risk".

    Durations are clipped at half a day: a same-day churn is a real event, and a
    zero duration is undefined for the parametric fits.
    """
    accounts, subs = tables["accounts"], tables["subscriptions"]
    first_churn = tables["churn_events"].groupby("account_id")["churn_date"].min()

    first_sub = (subs.sort_values("start_date").groupby("account_id")
                 .first()[["plan_tier", "seats", "mrr_amount", "is_trial",
                           "billing_frequency", "auto_renew_flag"]]
                 .add_prefix("base_"))

    frame = (accounts.set_index("account_id")[["industry", "country", "signup_date",
                                               "referral_source"]]
             .join(first_sub).join(first_churn.rename("first_churn")))

    frame["event"] = frame["first_churn"].notna().astype(int)
    frame["duration"] = np.where(
        frame["event"] == 1,
        (frame["first_churn"] - frame["signup_date"]).dt.days,
        (extract_date - frame["signup_date"]).dt.days).astype(float)
    frame["cohort"] = frame["signup_date"].dt.to_period("Q").astype(str)

    # Negative durations would mean churning before signing up. The integrity
    # report finds this pattern in the ticket and usage tables; it does *not*
    # occur here, and the guard records that rather than assuming it.
    invalid = int((frame["duration"] < 0).sum())
    frame = frame[frame["duration"] >= 0].copy()
    frame["duration"] = frame["duration"].clip(lower=0.5)
    frame.attrs["dropped_negative_duration"] = invalid
    frame.attrs["extract_date"] = str(pd.Timestamp(extract_date).date())
    return frame


def km_summary(frame, horizons=HORIZONS):
    """Kaplan-Meier survival with 95% intervals at fixed horizons."""
    km = KaplanMeierFitter().fit(frame["duration"], frame["event"])
    band = km.confidence_interval_

    rows = []
    for horizon in horizons:
        if horizon > frame["duration"].max():
            continue
        # KM is a right-continuous step function, so the value at t comes from
        # the last event at or *including* t. side="right" is what makes the
        # "or including" true — with the default "left" a horizon landing exactly
        # on an event day takes the interval from the step before it, which does
        # not match the point estimate `km.predict` returns for the same horizon.
        idx = band.index[max(np.searchsorted(band.index, horizon, side="right") - 1, 0)]
        rows.append({"days": horizon,
                     "survival": round(float(km.predict(horizon)), 4),
                     "ci_lo": round(float(band.loc[idx].iloc[0]), 4),
                     "ci_hi": round(float(band.loc[idx].iloc[1]), 4)})

    out = pd.DataFrame(rows)
    out.attrs.update(n=len(frame), events=int(frame["event"].sum()),
                     censored=int((frame["event"] == 0).sum()),
                     median_survival=float(km.median_survival_time_))
    return out


def logrank_scan(frame, by_cols=BASELINE_SEGMENTS):
    """Log-rank test per baseline segment, BH-corrected across the family.

    The log-rank compares whole survival curves rather than a rate at one
    horizon, so unlike the chi-square in `reasons.segment_scan` it uses the
    timing of every event. It is the strictly better-powered version of the same
    question, which is why a null here means more.
    """
    rows = []
    for col in by_cols:
        groups = frame[col].astype(str)
        result = multivariate_logrank_test(frame["duration"], groups, frame["event"])
        rows.append({"segment": col, "levels": groups.nunique(),
                     "chi2": round(float(result.test_statistic), 3),
                     "p": float(result.p_value)})

    out = pd.DataFrame(rows)
    out["p_bh"] = multipletests(out["p"], method="fdr_bh")[1]
    return out.round({"p": 4, "p_bh": 4})


def _design(frame, segments=BASELINE_SEGMENTS, numeric=BASELINE_NUMERIC):
    """Model matrix for the Cox fits: dummies for segments, raw for numerics."""
    cols = [c for c in segments + numeric if c in frame.columns]
    design = frame[cols + ["duration", "event"]].copy()
    categorical = [c for c in cols if design[c].dtype == object or
                   design[c].dtype == bool or design[c].nunique() <= 2]
    categorical = [c for c in categorical if c in segments]
    design = pd.get_dummies(design, columns=categorical, drop_first=True)
    return design.astype({c: float for c in design.columns if design[c].dtype == bool})


def fit_cox(frame, penalizer=0.1, **design_kwargs):
    """Cox proportional hazards on baseline covariates.

    Ridge-penalised: 22 dummy columns on 500 accounts with several sparse country
    levels is where an unpenalised Cox fit produces enormous hazard ratios with
    intervals spanning three orders of magnitude.

    Returns (fitted_model, summary_table). The summary carries a BH-corrected
    column because reading 22 coefficients and quoting the smallest p is the
    selection error this project measures elsewhere — at 22 tests the smallest
    raw p is below 0.05 roughly two thirds of the time under a pure null.
    """
    design = _design(frame, **design_kwargs)
    model = CoxPHFitter(penalizer=penalizer).fit(design, "duration", "event")

    summary = model.summary[["coef", "exp(coef)", "exp(coef) lower 95%",
                             "exp(coef) upper 95%", "p"]].copy()
    summary.columns = ["coef", "hazard_ratio", "hr_lo", "hr_hi", "p"]
    summary["p_bh"] = multipletests(summary["p"], method="fdr_bh")[1]

    lr = model.log_likelihood_ratio_test()
    summary.attrs.update(
        concordance=round(float(model.concordance_index_), 4),
        global_p=round(float(lr.p_value), 4),
        global_chi2=round(float(lr.test_statistic), 3),
        n=len(design), events=int(design["event"].sum()),
        n_covariates=design.shape[1] - 2)
    return model, summary.sort_values("p").round(4)


def ph_assumption(model, frame, **design_kwargs):
    """Schoenfeld residual test — does the proportional-hazards assumption hold?

    A Cox hazard ratio is one number for all time. If a covariate's effect grows
    or fades, that single number is a weighted average of a changing effect and
    the model is misspecified. Reported rather than assumed, because "we fitted a
    Cox model" is not the same as "the Cox model was applicable".
    """
    design = _design(frame, **design_kwargs)
    result = proportional_hazard_test(model, design, time_transform="rank")
    out = result.summary[["test_statistic", "p"]].copy()
    out["p_bh"] = multipletests(out["p"], method="fdr_bh")[1]
    out.attrs["n_violations"] = int((out["p_bh"] < 0.05).sum())
    return out.round(4)


def hazard_shape(frame):
    """Is the hazard rising, falling or flat with tenure?

    Weibull against Exponential by likelihood ratio on one degree of freedom.
    The Exponential is the memoryless special case (rho = 1): an account at day
    300 is exactly as likely to churn tomorrow as one at day 10. rho < 1 is a
    falling hazard, the shape an onboarding problem produces.

    This is the headline test for "is churn front-loaded", and section
    `shape_within_cohorts` is the one that decides whether to believe it.
    """
    weibull = WeibullFitter().fit(frame["duration"], frame["event"])
    exponential = ExponentialFitter().fit(frame["duration"], frame["event"])

    lr = 2 * (weibull.log_likelihood_ - exponential.log_likelihood_)
    ci = weibull.summary.loc["rho_", ["coef lower 95%", "coef upper 95%"]]
    return {"rho": round(float(weibull.rho_), 4),
            "rho_ci": [round(float(ci.iloc[0]), 4), round(float(ci.iloc[1]), 4)],
            "lambda": round(float(weibull.lambda_), 2),
            "lr_stat": round(float(lr), 3),
            "p_vs_exponential": float(stats.chi2.sf(lr, 1)),
            "shape": "falling" if weibull.rho_ < 1 else "rising",
            "n": len(frame), "events": int(frame["event"].sum())}


def smoothed_hazard(frame, bandwidth=30, at=(15, 45, 90, 180, 365)):
    """Nelson-Aalen smoothed hazard at selected tenures, for the shape plot."""
    na = NelsonAalenFitter().fit(frame["duration"], frame["event"])
    curve = na.smoothed_hazard_(bandwidth=bandwidth)
    rows = []
    for t in at:
        if t > curve.index.max():
            continue
        idx = curve.index[np.argmin(np.abs(curve.index - t))]
        rows.append({"days": t, "hazard_per_day": round(float(curve.loc[idx].iloc[0]), 6)})
    return pd.DataFrame(rows), curve


def shape_within_cohorts(frame, min_followup=365):
    """Refit the hazard shape inside each signup cohort separately.

    This is the decomposition that matters. A pooled falling hazard has two
    explanations, and they are not distinguishable from the pooled curve:

      1. real tenure effect — accounts are fragile early and settle down
      2. composition — recent cohorts churn faster *at every tenure*, and recent
         cohorts are exactly the ones contributing the short tenures

    Within a single cohort, explanation 2 cannot operate. If rho returns to 1
    there, the pooled falling hazard was composition and the "churn is
    front-loaded, fix onboarding" reading is wrong.

    Restricted to cohorts with at least `min_followup` days of observation,
    because a cohort observed for 90 days cannot speak to the shape beyond 90.
    """
    rows = []
    for cohort, group in frame.groupby("cohort"):
        if group["duration"].max() < min_followup or group["event"].sum() < 10:
            continue
        weibull = WeibullFitter().fit(group["duration"], group["event"])
        exponential = ExponentialFitter().fit(group["duration"], group["event"])
        lr = 2 * (weibull.log_likelihood_ - exponential.log_likelihood_)
        rows.append({"cohort": cohort, "n": len(group),
                     "events": int(group["event"].sum()),
                     "rho": round(float(weibull.rho_), 3),
                     "p_vs_exponential": round(float(stats.chi2.sf(lr, 1)), 4)})
    return pd.DataFrame(rows)


def km_at_tenure_by_cohort(frame, tenures=(30, 60, 90)):
    """Survival at *fixed tenure*, cohort by cohort — the decisive comparison.

    Every cohort is compared at the same age, so censoring cannot produce a
    difference: if the 2023Q1 and 2024Q3 cohorts both have 30 days of follow-up
    for most accounts, S(30) is measured on equal terms.

    Flat down this table means the pooled cohort log-rank was driven by unequal
    follow-up. A gradient means the cohorts genuinely differ.
    """
    rows = []
    for cohort, group in frame.groupby("cohort"):
        km = KaplanMeierFitter().fit(group["duration"], group["event"])
        row = {"cohort": cohort, "n": len(group),
               "events": int(group["event"].sum()),
               "max_followup": int(group["duration"].max())}
        for t in tenures:
            row[f"S({t}d)"] = (round(float(km.predict(t)), 3)
                               if group["duration"].max() >= t else np.nan)
        rows.append(row)
    return pd.DataFrame(rows)


def calendar_hazard(tables, extract_date=EXTRACT_DATE, freq="M"):
    """Churn hazard by *calendar month*, over the account base at risk.

    Builds the person-period table the cohort/tenure views cannot: for each
    month, how many accounts were at risk and how many left. This separates a
    period effect (everyone churns more in late 2024, whatever their tenure or
    cohort) from the other two.

    A Poisson trend test with log(at_risk) as offset gives the monthly rate
    ratio. The offset is what makes it a test of the *rate* rather than of the
    raw count, which would rise simply because the customer base grew.

    **Do not read a trend from this without running `generator.calendar_hazard_null`
    first.** The null here is "the rate is flat", which is not the null that
    matters on an extract with a hard end date: churn dates drawn uniformly
    between signup and that boundary produce a hazard of 1/(END - t), so this
    function returns a large, highly significant rising trend on data containing
    nothing at all. On this dataset it does exactly that — see notebook 16.
    """
    accounts = tables["accounts"]
    first_churn = tables["churn_events"].groupby("account_id")["churn_date"].min()
    frame = accounts.set_index("account_id")[["signup_date"]].copy()
    frame["churn_date"] = first_churn
    frame["end"] = frame["churn_date"].fillna(pd.Timestamp(extract_date))

    periods = pd.period_range(frame["signup_date"].min(), extract_date, freq=freq)
    rows = []
    for period in periods:
        start, stop = period.start_time, period.end_time
        at_risk = int(((frame["signup_date"] <= stop) & (frame["end"] >= start)).sum())
        churned = int(frame["churn_date"].between(start, stop).sum())
        rows.append({"period": str(period), "at_risk": at_risk, "events": churned,
                     "hazard": churned / at_risk if at_risk else np.nan})

    out = pd.DataFrame(rows)
    usable = out[out["at_risk"] > 0].reset_index(drop=True)
    usable["t"] = np.arange(len(usable))

    model = sm.GLM(usable["events"], sm.add_constant(usable[["t"]]),
                   family=sm.families.Poisson(),
                   offset=np.log(usable["at_risk"])).fit()
    rate_ratio = float(np.exp(model.params["t"]))
    lo, hi = np.exp(model.conf_int().loc["t"])

    out.attrs.update(
        rate_ratio_per_period=round(rate_ratio, 4),
        rate_ratio_ci=[round(float(lo), 4), round(float(hi), 4)],
        rate_ratio_annual=round(rate_ratio ** 12, 3),
        p_trend=float(model.pvalues["t"]),
        first_year_mean=round(float(usable["hazard"].head(12).mean()), 4),
        last_year_mean=round(float(usable["hazard"].tail(12).mean()), 4))
    return out.round(4)


def cohort_survival_frame(cohort, tables, cutoff, extract_date=EXTRACT_DATE):
    """Time-to-event for the modelling cohort, measured forward from the cutoff.

    The like-for-like comparison against the classifier. Same accounts, same
    point-in-time features, but the outcome is "how long until they left"
    censored at extraction rather than "did they leave inside 90 days". That
    converts the 90-day window's 54 positives into every event observable in the
    184 days of follow-up the extract actually contains.
    """
    first_churn = tables["churn_events"].groupby("account_id")["churn_date"].min()
    frame = cohort[["account_id"]].copy()
    frame["first_churn"] = frame["account_id"].map(first_churn)

    # A churn recorded before the cutoff belongs to the observation window, not
    # the follow-up: the cohort rule already excludes those accounts, and this
    # guard keeps a caller passing a looser cohort from producing negative times.
    after_cutoff = frame["first_churn"] > pd.Timestamp(cutoff)
    frame["event"] = after_cutoff.fillna(False).astype(int)
    frame["duration"] = np.where(
        frame["event"] == 1,
        (frame["first_churn"] - pd.Timestamp(cutoff)).dt.days,
        (pd.Timestamp(extract_date) - pd.Timestamp(cutoff)).days).astype(float)
    frame["duration"] = frame["duration"].clip(lower=0.5)
    frame.attrs["followup_days"] = (pd.Timestamp(extract_date) - pd.Timestamp(cutoff)).days
    return frame.set_index("account_id")


def cv_concordance(X, duration, event, penalizer=1.0, n_splits=5, seed=42):
    """Cross-validated Harrell's C for a Cox model on the feature matrix.

    Concordance is the survival analogue of AUC — the share of comparable pairs
    the model orders correctly — so 0.5 is chance and it reads on the same scale
    as every other number in this project.

    Numeric columns only, median-imputed inside the fold. That is a weaker
    preprocessing story than `model._pipe` runs for the classifier, and it is
    deliberate: the question here is whether the *framing* buys anything, so
    changing the feature handling at the same time would confound the comparison.
    """
    numeric = X.select_dtypes(include=np.number).replace([np.inf, -np.inf], np.nan)
    folds = KFold(n_splits=n_splits, shuffle=True, random_state=seed)
    scores = []

    for train_idx, test_idx in folds.split(numeric):
        train, test = numeric.iloc[train_idx], numeric.iloc[test_idx]
        medians = train.median()
        train, test = train.fillna(medians), test.fillna(medians)

        # Zero-variance columns in a fold break the Cox fit; drop per fold.
        keep = train.columns[train.std() > 0]
        train, test = train[keep], test[keep]

        design = train.assign(duration=duration.iloc[train_idx].values,
                              event=event.iloc[train_idx].values)
        model = CoxPHFitter(penalizer=penalizer).fit(design, "duration", "event")
        risk = model.predict_partial_hazard(test)
        scores.append(concordance_index(duration.iloc[test_idx], -risk,
                                        event.iloc[test_idx]))

    scores = np.array(scores)
    return {"concordance": round(float(scores.mean()), 4),
            "sd": round(float(scores.std()), 4),
            "folds": [round(float(s), 4) for s in scores],
            "n": len(numeric), "events": int(event.sum())}


def rmst_by(frame, by, tau=365):
    """Restricted mean survival time — expected days retained within `tau`.

    A hazard ratio is hard to price. RMST is in days, so "this segment is worth
    12 more days of retention over the first year" converts straight into
    revenue, which is what the retention-economics notebook needs.
    """
    rows = []
    for level, group in frame.groupby(by):
        km = KaplanMeierFitter().fit(group["duration"], group["event"])
        horizon = min(tau, group["duration"].max())
        curve = km.survival_function_.loc[:horizon]
        # Step function, so the integral is a left-Riemann sum over its own jumps.
        widths = np.diff(np.append(curve.index.values, horizon))
        rows.append({by: level, "n": len(group), "events": int(group["event"].sum()),
                     f"rmst_{tau}d": round(float((curve.iloc[:, 0].values * widths).sum()), 1)})
    return pd.DataFrame(rows)


__all__ = ["survival_frame", "km_summary", "logrank_scan", "fit_cox",
           "ph_assumption", "hazard_shape", "smoothed_hazard",
           "shape_within_cohorts", "km_at_tenure_by_cohort", "calendar_hazard",
           "cohort_survival_frame", "cv_concordance", "rmst_by",
           "BASELINE_SEGMENTS", "BASELINE_NUMERIC", "HORIZONS"]
