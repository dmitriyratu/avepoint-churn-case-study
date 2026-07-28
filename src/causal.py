"""Causal estimation and uplift modelling for "what actions improve retention".

**Read this before reading any number this module produces.**

The dataset records no intervention. Nobody was offered a discount, called by a
CSM, or enrolled in an onboarding programme — or if they were, it is not in the
extract. So the question the product team asked ("what actions can improve
retention") is *not identified* by this data under any method, and no amount of
estimator sophistication changes that. Doubly-robust estimation is robust to
misspecifying one of two models; it is not robust to the treatment never having
happened.

What is available is a set of *observational proxies* — things an account did
that a product team might want to encourage or prevent (upgrading, escalating a
ticket, enabling auto-renew). Estimating their association with churn under an
explicit unconfoundedness assumption is a legitimate exercise, and it is the
closest this data gets. The discipline this module enforces is that the
assumption is written down, diagnosed, and stress-tested rather than left
implicit:

    overlap        does every treated account have a comparable control?
    balance        does weighting actually remove the measured differences?
    AIPW           two chances to be right about the nuisance models
    cluster boot   the pooled design repeats accounts; the CI must know
    placebo        run the whole machine on a treatment assigned at random
    E-value        how strong must an unmeasured confounder be to erase this?

The E-value is the one that matters most for honest reporting. It converts
"we assumed no unmeasured confounding" into a number the business can argue
with: *an unmeasured confounder would have to be associated with both treatment
and churn by a risk ratio of at least X to explain this away.* If X is small, the
estimate is worthless whatever its p-value.
"""
import numpy as np
import pandas as pd
from sklearn.base import clone
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import StratifiedKFold

from .model import SEED, _pipe

# Observational proxies for an action a product team could take. Each is
# (column, rule, description, columns that are mechanically derived from it and
# must therefore be dropped from the confounder set).
TREATMENTS = {
    "upgrade": {
        "rule": lambda X: (X["n_upgrades"] > 0).astype(int),
        "question": "Do accounts that upgrade churn less?",
        "action": "guided upgrade offers",
        "derived": ["n_upgrades", "upgrade_net"]},
    "downgrade": {
        "rule": lambda X: (X["n_downgrades"] > 0).astype(int),
        "question": "Is downgrading an early warning worth intervening on?",
        "action": "outreach on downgrade",
        "derived": ["n_downgrades", "upgrade_net"]},
    "escalation": {
        "rule": lambda X: (X["n_escalations"] > 0).astype(int),
        "question": "Does escalating a support ticket help or hurt retention?",
        "action": "escalate more aggressively",
        "derived": ["n_escalations", "escalation_rate"]},
    "auto_renew": {
        "rule": lambda X: (X["auto_renew_pct"] > 0.5).astype(int),
        "question": "Is auto-renew worth pushing at signup?",
        "action": "default new contracts to auto-renew",
        "derived": ["auto_renew_pct"]},
}

TRIM = 0.05          # propensity trimming bound for the overlap region
N_BOOT = 400
N_FOLDS = 5

# Pre-specified confounders, chosen for the *reason* they confound rather than by
# selection on this data. Handing all 73 features to a propensity model on 177
# rows is not a more careful analysis — it is a guaranteed one. It fits treatment
# almost perfectly, propensity scores pile up at 0 and 1, overlap disappears, and
# IPW weights reach 100. The first version of this module did exactly that and
# returned an ATE of -0.53 on a probability scale, i.e. a negative treated
# churn rate; that is what a p >> n propensity model looks like when it fails.
#
# Ten covariates for 177 rows is still generous. Each is here because it plausibly
# drives both the action and churn: size and price of the contract, how long they
# have been around, how much they use the product, and how much support they need.
CONFOUNDERS = ["tenure_days", "n_subscriptions", "latest_mrr", "latest_seats",
               "plan_tier", "industry", "n_tickets", "avg_satisfaction",
               "total_usage_events", "days_since_last_usage"]


def make_treatment(X, name, confounders=None):
    """Split the feature matrix into (treatment, confounders) for `name`.

    Columns mechanically derived from the treatment are dropped even if they
    appear in the confounder list. Conditioning on `upgrade_net` while treating
    on "had an upgrade" would condition on a function of the treatment, which
    blocks the very path being estimated.
    """
    spec = TREATMENTS[name]
    treatment = spec["rule"](X)
    wanted = list(confounders if confounders is not None else CONFOUNDERS)
    keep = [c for c in wanted if c in X.columns and c not in spec["derived"]]
    return treatment.rename(name), X[keep]


def _learner(kind="linear"):
    """Nuisance model, deliberately over-regularised.

    C=0.1 rather than the default 1.0. With ten covariates on 177 rows an
    unpenalised logistic still finds near-separating directions, and a propensity
    score of 0.001 is not a statement about the account — it is the model
    memorising. Shrinking towards the marginal rate is the conservative error to
    make here: it costs precision and protects the overlap the whole method needs.
    """
    if kind == "linear":
        return _pipe(LogisticRegression(C=0.1, max_iter=2000, random_state=SEED))
    return _pipe(GradientBoostingClassifier(max_depth=2, n_estimators=60,
                                            learning_rate=0.05,
                                            random_state=SEED), scale=False)


def cross_fit_propensity(X, T, kind="linear", n_folds=N_FOLDS, seed=SEED):
    """Out-of-fold P(T=1 | X).

    Cross-fitted because an in-sample propensity score is overfitted towards 0
    and 1, which inflates IPW weights exactly where they are already unstable.
    This is the sample-splitting half of double machine learning, and it is
    cheap insurance.
    """
    scores = np.zeros(len(T), dtype=float)
    folds = StratifiedKFold(n_folds, shuffle=True, random_state=seed)
    for train, test in folds.split(X, T):
        model = clone(_learner(kind)).fit(X.iloc[train], T.iloc[train])
        scores[test] = model.predict_proba(X.iloc[test])[:, 1]
    return pd.Series(scores, index=T.index, name="propensity")


def overlap_report(propensity, T, trim=TRIM):
    """Common-support diagnostics.

    Without overlap there is no comparison to make: a treated account whose
    propensity is 0.99 has no control counterpart, and any estimate for it is
    the outcome model extrapolating. Reported before any effect estimate,
    because a beautiful ATE on a population with no overlap is meaningless.
    """
    treated, control = propensity[T == 1], propensity[T == 0]
    inside = propensity.between(trim, 1 - trim)
    return {"n": len(T), "n_treated": int(T.sum()),
            "treated_rate": round(float(T.mean()), 4),
            "ps_treated_range": [round(float(treated.min()), 3),
                                 round(float(treated.max()), 3)],
            "ps_control_range": [round(float(control.min()), 3),
                                 round(float(control.max()), 3)],
            "overlap_lo": round(float(max(treated.min(), control.min())), 3),
            "overlap_hi": round(float(min(treated.max(), control.max())), 3),
            "share_within_trim": round(float(inside.mean()), 4),
            "n_trimmed": int((~inside).sum()),
            "max_ipw_weight": round(float(
                np.maximum(T / propensity.clip(0.01), (1 - T) / (1 - propensity).clip(0.01)
                           ).max()), 1)}


def standardised_differences(X, T, weights=None):
    """SMD per covariate, before and after weighting.

    |SMD| < 0.1 is the usual threshold for "balanced". Numeric columns only;
    categoricals are one-hot expanded first so each level gets its own row.
    """
    design = pd.get_dummies(X, drop_first=False).astype(float)
    w = np.ones(len(T)) if weights is None else np.asarray(weights, dtype=float)

    rows = []
    for col in design.columns:
        values = design[col].values
        treated, control = T.values == 1, T.values == 0

        def stats(mask):
            ww = w[mask]
            mean = np.average(values[mask], weights=ww)
            var = np.average((values[mask] - mean) ** 2, weights=ww)
            return mean, var

        mt, vt = stats(treated)
        mc, vc = stats(control)
        pooled = np.sqrt((vt + vc) / 2)
        rows.append({"covariate": col,
                     "smd": (mt - mc) / pooled if pooled > 0 else 0.0})

    out = pd.DataFrame(rows)
    out["abs_smd"] = out["smd"].abs()
    return out.sort_values("abs_smd", ascending=False).round(4)


def _aipw_scores(X, T, Y, kind="linear", n_folds=N_FOLDS, seed=SEED, trim=TRIM):
    """Per-row augmented inverse-probability-weighted influence values.

    AIPW is consistent if *either* the propensity model or the outcome model is
    right — the "doubly robust" property. Both are cross-fitted, so neither
    borrows strength from the row it is scoring.

    Returns the per-row scores; their mean is the ATE and their standard error
    follows from their spread, which is what makes the cluster bootstrap below
    a resampling of these values rather than a refit of everything.
    """
    # The raw score decides *overlap*; the clipped one only bounds the weights.
    # Clipping first and then testing the clipped values against the same bounds
    # makes trimming a no-op — an earlier version did exactly that and reported
    # "0 trimmed" while `overlap_report` was flagging 16% of rows outside support.
    propensity = cross_fit_propensity(X, T, kind, n_folds, seed)
    bounded = propensity.clip(trim, 1 - trim)
    mu1 = np.zeros(len(Y), dtype=float)
    mu0 = np.zeros(len(Y), dtype=float)

    folds = StratifiedKFold(n_folds, shuffle=True, random_state=seed)
    for train, test in folds.split(X, Y):
        tr = X.iloc[train]
        for arm, target in ((1, mu1), (0, mu0)):
            mask = T.iloc[train] == arm
            # A fold can leave one arm single-class at these sample sizes; fall
            # back to that arm's mean rather than dropping the fold silently.
            if mask.sum() < 2 or Y.iloc[train][mask].nunique() < 2:
                target[test] = Y.iloc[train][mask].mean() if mask.sum() else Y.mean()
                continue
            model = clone(_learner(kind)).fit(tr[mask.values], Y.iloc[train][mask.values])
            target[test] = model.predict_proba(X.iloc[test])[:, 1]

    t, y, p = T.values, Y.values, bounded.values
    # Per-row potential-outcome estimates, each with its own IPW correction. The
    # difference is the AIPW score; keeping them separate is what lets the caller
    # form a risk *ratio* rather than only a difference.
    psi1 = mu1 + t * (y - mu1) / p
    psi0 = mu0 + (1 - t) * (y - mu0) / (1 - p)
    return (pd.Series(psi1 - psi0, index=Y.index), propensity,
            pd.Series(psi1, index=Y.index), pd.Series(psi0, index=Y.index))


def aipw_ate(X, T, Y, groups=None, kind="linear", n_boot=N_BOOT, seed=SEED,
             trim=TRIM):
    """Doubly-robust ATE with a cluster bootstrap interval.

    `groups` must be account_id whenever the pooled multi-cutoff design is used:
    the same customer appears at several cutoffs, those rows are far from
    independent, and an unclustered interval is too narrow. This is the causal
    twin of the `GroupKFold` point in `robustness.pooled_cv`.

    Positive ATE = treatment *increases* churn. For a retention action that is
    the wrong sign, which is worth stating because the intuitive reading of a
    positive number is backwards here.
    """
    scores, propensity, psi1, psi0 = _aipw_scores(X, T, Y, kind=kind, seed=seed,
                                                  trim=trim)

    # Restrict to the overlap region. Outside it there is no counterfactual to
    # estimate — only the outcome model extrapolating — so the estimand becomes
    # "the ATE among accounts that could plausibly have gone either way". That is
    # a narrower claim than the ATE and the honest one to make.
    inside = propensity.between(trim, 1 - trim)
    scores, psi1, psi0 = scores[inside], psi1[inside], psi0[inside]
    T_in, Y_in = T[inside], Y[inside]
    ate = float(scores.mean())

    rng = np.random.default_rng(seed)
    keys = (pd.Series(np.arange(len(scores)), index=scores.index) if groups is None
            else pd.Series(np.asarray(groups), index=T.index)[inside])
    unique = keys.unique()
    lookup = {k: scores.values[np.where(keys.values == k)[0]] for k in unique}

    draws = np.empty(n_boot)
    for i in range(n_boot):
        picked = rng.choice(unique, size=len(unique), replace=True)
        draws[i] = np.concatenate([lookup[k] for k in picked]).mean()

    # Potential-outcome means, clipped to the unit interval: an AIPW mean can
    # land outside [0, 1] when weights are large, and a churn probability of
    # -0.17 should be surfaced as an unstable estimate, not reported as a rate.
    rate_1, rate_0 = float(np.clip(psi1.mean(), 0, 1)), float(np.clip(psi0.mean(), 0, 1))
    risk_ratio = rate_1 / rate_0 if rate_0 > 0 else np.nan

    return {"ate": round(ate, 4),
            "ci_lo": round(float(np.percentile(draws, 2.5)), 4),
            "ci_hi": round(float(np.percentile(draws, 97.5)), 4),
            "boot_sd": round(float(draws.std()), 4),
            "p_two_sided": round(float(2 * min((draws <= 0).mean(),
                                               (draws >= 0).mean())), 4),
            "control_rate": round(rate_0, 4),
            "treated_rate": round(rate_1, 4),
            "risk_ratio": round(float(risk_ratio), 4),
            "naive_diff": round(float(Y_in[T_in == 1].mean()
                                      - Y_in[T_in == 0].mean()), 4),
            "n_analysed": int(inside.sum()), "n_trimmed": int((~inside).sum()),
            "n_treated": int(T_in.sum()),
            "n_clusters": int(len(unique)),
            "mean_propensity": round(float(propensity.mean()), 4)}


def ipw_ate(X, T, Y, kind="linear", seed=SEED, trim=TRIM):
    """Plain stabilised IPW, reported next to AIPW as a specification check.

    If the two disagree materially, at least one nuisance model is wrong and the
    doubly-robust claim is doing real work. If they agree, neither is rescuing
    the other and the estimate rests on overlap alone.
    """
    propensity = cross_fit_propensity(X, T, kind, seed=seed).clip(trim, 1 - trim)
    t, y, p = T.values, Y.values, propensity.values
    w1, w0 = t / p, (1 - t) / (1 - p)
    return {"ate": round(float((w1 * y).sum() / w1.sum()
                               - (w0 * y).sum() / w0.sum()), 4),
            "ess_treated": round(float(w1.sum() ** 2 / (w1 ** 2).sum()), 1),
            "ess_control": round(float(w0.sum() ** 2 / (w0 ** 2).sum()), 1)}


def e_value(risk_ratio, lower=None):
    """Minimum confounder strength needed to explain the estimate away.

    VanderWeele & Ding (2017). The E-value is the risk ratio an unmeasured
    confounder would need with *both* treatment and outcome to account for the
    observed association. Reported for the point estimate and for the confidence
    limit nearest the null, because the second is what actually needs explaining
    away to overturn a claim of significance.

    An E-value near 1 means a trivially weak confounder suffices, which is the
    situation for every estimate in this project.
    """
    def compute(rr):
        if rr is None or not np.isfinite(rr) or rr <= 0:
            return np.nan
        rr = 1 / rr if rr < 1 else rr
        return round(float(rr + np.sqrt(rr * (rr - 1))), 3)

    out = {"e_value_point": compute(risk_ratio)}
    if lower is not None:
        # A limit spanning the null needs no confounding to explain: E = 1.
        crosses = (lower - 1) * (risk_ratio - 1) <= 0
        out["e_value_ci"] = 1.0 if crosses else compute(lower)
        out["ci_crosses_null"] = bool(crosses)
    return out


def placebo_ate(X, Y, treated_rate, groups=None, n_placebo=30, kind="linear", seed=SEED):
    """The whole estimator, run on treatments assigned completely at random.

    A randomly-assigned treatment has a true ATE of exactly zero, so the spread
    of these estimates is this design's noise floor. An observed ATE inside it is
    not evidence of anything, and the width is the honest answer to "how small an
    effect could this study detect".
    """
    rng = np.random.default_rng(seed)
    estimates = []
    for i in range(n_placebo):
        fake = pd.Series((rng.random(len(Y)) < treated_rate).astype(int),
                         index=Y.index, name="placebo")
        if fake.nunique() < 2:
            continue
        scores, _, _, _ = _aipw_scores(X, fake, Y, kind=kind, seed=seed + i)
        estimates.append(float(scores.mean()))

    estimates = np.array(estimates)
    return {"n_placebo": len(estimates),
            "mean": round(float(estimates.mean()), 4),
            "sd": round(float(estimates.std()), 4),
            "p2_5": round(float(np.percentile(estimates, 2.5)), 4),
            "p97_5": round(float(np.percentile(estimates, 97.5)), 4),
            "detectable_effect": round(float(np.percentile(np.abs(estimates), 95)), 4)}


# --------------------------------------------------------------------------
# Uplift modelling
# --------------------------------------------------------------------------

def t_learner_uplift(X, T, Y, kind="linear", n_folds=N_FOLDS, seed=SEED):
    """Out-of-fold per-account uplift: P(churn | treated) - P(churn | control).

    The T-learner fits one outcome model per arm. Out-of-fold, because an
    in-sample uplift score is fitted to the very rows the Qini curve then
    evaluates, and that alone manufactures a convincing curve.

    Negative uplift = treatment reduces churn = the retention win.
    """
    uplift = np.zeros(len(Y), dtype=float)
    folds = StratifiedKFold(n_folds, shuffle=True, random_state=seed)

    for train, test in folds.split(X, Y):
        predictions = {}
        for arm in (0, 1):
            mask = (T.iloc[train] == arm).values
            if mask.sum() < 2 or Y.iloc[train][mask].nunique() < 2:
                predictions[arm] = np.full(len(test),
                                           Y.iloc[train][mask].mean()
                                           if mask.sum() else Y.mean())
                continue
            model = clone(_learner(kind)).fit(X.iloc[train][mask],
                                              Y.iloc[train][mask])
            predictions[arm] = model.predict_proba(X.iloc[test])[:, 1]
        uplift[test] = predictions[1] - predictions[0]

    return pd.Series(uplift, index=Y.index, name="uplift")


def qini_curve(uplift, T, Y):
    """Qini points over the population ranked by predicted uplift.

    Qini(k) = Y_treated(k) - Y_control(k) * n_treated(k) / n_control(k)

    the treated response among the top k minus the control response rescaled to
    the same treated exposure. The diagonal is what random targeting achieves;
    area between the curve and that diagonal is the Qini coefficient.
    """
    order = np.argsort(-uplift.values)
    t, y = T.values[order], Y.values[order]

    n_t, n_c = np.cumsum(t), np.cumsum(1 - t)
    y_t, y_c = np.cumsum(y * t), np.cumsum(y * (1 - t))
    with np.errstate(divide="ignore", invalid="ignore"):
        qini = y_t - np.where(n_c > 0, y_c * n_t / np.maximum(n_c, 1), 0.0)

    total = qini[-1]
    fraction = np.arange(1, len(t) + 1) / len(t)
    return pd.DataFrame({"fraction": fraction, "qini": qini,
                         "random": total * fraction})


def qini_coefficient(uplift, T, Y):
    """Area between the Qini curve and the random-targeting diagonal.

    Normalised by the number of accounts so it reads per-account, and signed so
    that positive means the ranking beats random targeting.
    """
    curve = qini_curve(uplift, T, Y)
    return float(np.trapezoid(curve["qini"] - curve["random"], curve["fraction"]))


def qini_with_null(X, T, Y, n_null=20, kind="linear", seed=SEED):
    """Observed Qini against Qini from uplift models fitted to shuffled outcomes.

    Uplift models are notorious for looking good on their own training
    distribution — the curve is monotone by construction near the top. The null
    is what makes the number interpretable.
    """
    observed = qini_coefficient(t_learner_uplift(X, T, Y, kind=kind, seed=seed), T, Y)

    rng = np.random.default_rng(seed)
    null = []
    for i in range(n_null):
        shuffled = pd.Series(rng.permutation(Y.values), index=Y.index)
        null.append(qini_coefficient(
            t_learner_uplift(X, T, shuffled, kind=kind, seed=seed + i), T, shuffled))

    null = np.array(null)
    return {"observed_qini": round(observed, 4),
            "null_mean": round(float(null.mean()), 4),
            "null_sd": round(float(null.std()), 4),
            "null_p95": round(float(np.percentile(null, 95)), 4),
            "p_value": round(float((null >= observed).mean()), 4),
            "n_null": len(null)}


__all__ = ["TREATMENTS", "make_treatment", "cross_fit_propensity",
           "overlap_report", "standardised_differences", "aipw_ate", "ipw_ate",
           "e_value", "placebo_ate", "t_learner_uplift", "qini_curve",
           "qini_coefficient", "qini_with_null"]
