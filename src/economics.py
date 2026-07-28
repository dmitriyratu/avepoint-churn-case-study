"""Retention economics and experiment design.

Notebooks 11-14 establish that this data cannot say who will churn or what to do
about it. That is a complete answer to two of the three product questions and a
useless one to a product team, who still have to decide what to fund on Monday.

This module supplies the part that does not depend on having a working model:
what a retained customer is worth, what an intervention has to achieve to pay
for itself, and how large an experiment has to be to tell. None of it needs the
AUC to be good — it needs the arithmetic to be right, and it turns "the model is
at chance" into "here is the decision rule that survives the model being at
chance".

Two results here matter more than the rest:

**Break-even precision.** An intervention costing C that saves a customer worth V
with effectiveness e pays off on a targeted account only if the account's true
churn probability exceeds C / (e * V). Compare that to the base rate and the
question "is the model worth deploying" often answers itself before any AUC is
computed — if the base rate already clears the bar, targeting is unnecessary; if
the bar is far above what any ranking could reach, targeting is impossible.

**Minimum detectable effect.** The number that decides whether an experiment is
worth running at all. Quoted alongside the calendar time needed to accrue the
sample, because "we need 900 accounts per arm" and "that is 43 months of
signups" are the same fact and only the second one ends the discussion.
"""
import numpy as np
import pandas as pd
from statsmodels.stats.power import NormalIndPower
from statsmodels.stats.proportion import proportion_effectsize

# Defaults are illustrative and every downstream number scales with them, so
# they are declared here rather than buried in a notebook cell. A real
# engagement replaces these with finance's figures on day one.
DEFAULT_INTERVENTION_COST = 150.0    # CSM time for one retention outreach
DEFAULT_EFFECTIVENESS = 0.20         # share of would-be churners actually saved
DEFAULT_DISCOUNT_ANNUAL = 0.10
DEFAULT_GROSS_MARGIN = 0.80          # SaaS gross margin on MRR


def clv_from_survival(survival_curve, monthly_revenue, gross_margin=DEFAULT_GROSS_MARGIN,
                      discount_annual=DEFAULT_DISCOUNT_ANNUAL, horizon_days=1095):
    """Customer lifetime value by integrating an actual survival curve.

    The textbook shortcut `ARPU / churn_rate` assumes a constant hazard and an
    infinite horizon. Notebook 12 measured the hazard directly, so there is no
    need to assume it — and the constant-hazard assumption is the one that
    turned out to be defensible *within* a cohort but badly wrong across the
    whole base, where the rate is rising 2.8x a year.

    Integrates S(t) * daily_margin * discount(t) over `horizon_days`.
    """
    daily_margin = monthly_revenue * gross_margin / 30.0
    daily_discount = (1 + discount_annual) ** (1 / 365.0)

    days = np.arange(1, horizon_days + 1)
    # Step function held forward: survival at day t is the last value at or below t.
    curve = survival_curve.reindex(
        survival_curve.index.union(days)).sort_index().ffill().reindex(days).fillna(
        survival_curve.iloc[-1] if len(survival_curve) else 0.0)

    discounted = curve.values * daily_margin / (daily_discount ** days)
    return {"clv": round(float(discounted.sum()), 2),
            "undiscounted": round(float((curve.values * daily_margin).sum()), 2),
            "expected_days_retained": round(float(curve.values.sum()), 1),
            "monthly_revenue": monthly_revenue, "horizon_days": horizon_days}


def break_even_precision(cost=DEFAULT_INTERVENTION_COST, value=None,
                         effectiveness=DEFAULT_EFFECTIVENESS):
    """Minimum true churn probability that justifies intervening on an account.

    Targeting one account costs `cost` whether or not it was going to churn. It
    returns `value` only when the account would have churned *and* the
    intervention works, which happens with probability p_churn * effectiveness.
    Break-even is where those meet:

        p_churn * effectiveness * value = cost
        p_churn = cost / (effectiveness * value)

    A number above 1 means no account can ever justify the intervention at these
    economics, which is a finding about the intervention rather than the model.
    """
    if value is None or value <= 0:
        return np.nan
    return float(cost / (effectiveness * value))


def campaign_value(y_true, scores, threshold, cost=DEFAULT_INTERVENTION_COST,
                   value=None, effectiveness=DEFAULT_EFFECTIVENESS):
    """Net value of targeting everyone scored at or above `threshold`.

    Uses realised outcomes, so this is what the campaign *would have* returned on
    this cohort — not a projection. Out-of-fold scores are the only honest input;
    passing in-sample scores here inflates every figure.
    """
    targeted = scores >= threshold
    n_targeted = int(targeted.sum())
    true_positives = int((targeted & (y_true == 1)).sum())

    saved = effectiveness * true_positives
    revenue = saved * (value or 0.0)
    spend = n_targeted * cost

    return {"threshold": round(float(threshold), 4), "n_targeted": n_targeted,
            "true_positives": true_positives,
            "precision": round(true_positives / n_targeted, 4) if n_targeted else np.nan,
            "recall": round(true_positives / max(int((y_true == 1).sum()), 1), 4),
            "customers_saved": round(saved, 2),
            "revenue": round(revenue, 2), "cost": round(spend, 2),
            "net_value": round(revenue - spend, 2),
            "roi": round((revenue - spend) / spend, 3) if spend else np.nan}


def value_curve(y_true, scores, value, cost=DEFAULT_INTERVENTION_COST,
                effectiveness=DEFAULT_EFFECTIVENESS, n_points=50):
    """Net value across every threshold, plus the treat-all and treat-none rows.

    Treat-none is always zero. Treat-all is the policy a company can run today
    with no model at all, and it is the benchmark a model has to beat — not
    chance, and not zero.
    """
    thresholds = np.quantile(scores, np.linspace(0, 1, n_points))
    rows = [campaign_value(y_true, scores, t, cost, value, effectiveness)
            for t in np.unique(thresholds)]
    frame = pd.DataFrame(rows)

    treat_all = campaign_value(y_true, scores, scores.min() - 1e-9, cost, value,
                               effectiveness)
    frame.attrs.update(treat_all_value=treat_all["net_value"], treat_none_value=0.0,
                       best_value=float(frame["net_value"].max()),
                       best_threshold=float(
                           frame.loc[frame["net_value"].idxmax(), "threshold"]))
    return frame


def decision_curve(y_true, scores, thresholds=None):
    """Decision curve analysis: net benefit against treat-all and treat-none.

    Net benefit = TP/n - (FP/n) * odds(p_t), where p_t is the threshold
    probability at which a decision-maker is indifferent. It expresses false
    positives in true-positive units using the decision-maker's own exchange
    rate, so a model can be compared to the two trivial strategies on one axis
    without committing to a specific cost in currency.

    Standard in clinical prediction and under-used in churn, where "the AUC is
    0.7" is routinely reported without ever asking whether acting on the model
    beats acting on everyone.
    """
    thresholds = np.linspace(0.01, 0.60, 60) if thresholds is None else thresholds
    n = len(y_true)
    prevalence = float(np.mean(y_true))

    rows = []
    for p_t in thresholds:
        odds = p_t / (1 - p_t)
        flagged = scores >= p_t
        tp = float(((flagged == 1) & (y_true == 1)).sum())
        fp = float(((flagged == 1) & (y_true == 0)).sum())
        rows.append({"threshold_prob": round(float(p_t), 4),
                     "net_benefit_model": tp / n - (fp / n) * odds,
                     "net_benefit_treat_all": prevalence - (1 - prevalence) * odds,
                     "net_benefit_treat_none": 0.0})
    return pd.DataFrame(rows).round(5)


def sample_size(baseline_rate, absolute_effect, alpha=0.05, power=0.80, ratio=1.0):
    """Accounts per arm for a two-proportion test.

    Cohen's h effect size, which is the arcsine transform the normal
    approximation actually needs — a raw difference of 5pp means something very
    different at a base rate of 0.05 than at 0.50.
    """
    treated_rate = baseline_rate - absolute_effect
    if not 0 < treated_rate < 1:
        return np.nan
    effect = proportion_effectsize(baseline_rate, treated_rate)
    if effect == 0:
        return np.inf
    return float(NormalIndPower().solve_power(
        effect_size=abs(effect), alpha=alpha, power=power, ratio=ratio,
        alternative="two-sided"))


def minimum_detectable_effect(baseline_rate, n_per_arm, alpha=0.05, power=0.80):
    """Smallest absolute reduction detectable with `n_per_arm` per arm.

    The number that decides whether to run the experiment. Solving for the
    effect size and inverting Cohen's h back to a probability difference, since
    an effect size in arcsine units is not something a product team can act on.
    """
    if n_per_arm < 2:
        return np.nan
    effect = NormalIndPower().solve_power(
        effect_size=None, nobs1=n_per_arm, alpha=alpha, power=power, ratio=1.0,
        alternative="two-sided")
    # h = 2*asin(sqrt(p1)) - 2*asin(sqrt(p2))  ->  invert for p2
    phi = 2 * np.arcsin(np.sqrt(baseline_rate)) - effect
    if not 0 <= phi <= np.pi:
        return np.nan
    return float(baseline_rate - np.sin(phi / 2) ** 2)


def experiment_plan(baseline_rate, effects, signups_per_month, alpha=0.05,
                    power=0.80, horizon_days=90):
    """Sample size and calendar time for a range of candidate effect sizes.

    The calendar column is the one that ends most of these discussions. A
    correctly-powered experiment that needs four years of signups is not an
    experiment, and saying so early is more useful than delivering the sample
    size alone.
    """
    rows = []
    for effect in effects:
        per_arm = sample_size(baseline_rate, effect, alpha, power)
        total = per_arm * 2 if np.isfinite(per_arm) else np.nan
        rows.append({
            "absolute_effect": effect,
            "relative_effect": round(effect / baseline_rate, 3),
            "treated_rate": round(baseline_rate - effect, 4),
            "n_per_arm": round(per_arm) if np.isfinite(per_arm) else np.nan,
            "n_total": round(total) if np.isfinite(total) else np.nan,
            "months_to_enrol": round(total / signups_per_month, 1)
            if np.isfinite(total) and signups_per_month else np.nan,
            "months_to_readout": round(total / signups_per_month
                                       + horizon_days / 30.0, 1)
            if np.isfinite(total) and signups_per_month else np.nan})
    return pd.DataFrame(rows)


def required_precision_table(values, costs, effectiveness=DEFAULT_EFFECTIVENESS,
                             base_rate=None):
    """Break-even precision over a grid of customer values and campaign costs.

    `base_rate` marks the cells where an untargeted campaign already clears the
    bar — those are the cases where the correct recommendation is "treat
    everyone and do not build a model", which no amount of AUC improves on.
    """
    rows = []
    for value in values:
        for cost in costs:
            required = break_even_precision(cost, value, effectiveness)
            rows.append({
                "customer_value": value, "campaign_cost": cost,
                "required_precision": round(required, 4)
                if np.isfinite(required) else np.nan,
                "achievable_untargeted": (None if base_rate is None
                                          else bool(base_rate >= required)),
                "verdict": ("impossible" if required > 1 else
                            "treat everyone" if base_rate is not None
                            and base_rate >= required else "targeting needed")})
    return pd.DataFrame(rows)


__all__ = ["clv_from_survival", "break_even_precision", "campaign_value",
           "value_curve", "decision_curve", "sample_size",
           "minimum_detectable_effect", "experiment_plan",
           "required_precision_table", "DEFAULT_INTERVENTION_COST",
           "DEFAULT_EFFECTIVENESS", "DEFAULT_GROSS_MARGIN",
           "DEFAULT_DISCOUNT_ANNUAL"]
