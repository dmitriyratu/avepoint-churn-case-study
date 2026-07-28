"""Is the timing pattern a fact about customers, or a fact about the file?

Every other module in this project asks whether a pattern is *statistically*
real — is it bigger than sampling noise. That is the wrong null for a generated
dataset. A pattern can be enormous, tightly estimated, p = 1e-15, and still be
nothing but an artefact of the rule that wrote the CSV.

The specific worry. Right-truncation: no churn date can land after the last day
of the extract. If a generator draws each churn date uniformly between signup
and that boundary, the hazard it produces is 1 / (END - t), which *rises without
limit* as t approaches the end of the file. Pooled over accounts that signed up
across two years, that looks exactly like "churn is accelerating" — and it also
makes recent cohorts look worse at every age, because their draw is squeezed
into a shorter interval.

So the null this module tests is not "the rate is flat". It is:

    churn_date ~ Uniform(signup_date, EXTRACT_DATE), and nothing else

`uniformity` asks whether the data is distinguishable from that rule.
`calendar_hazard_null` asks the harder question: if it *were* that rule, would
we see the effect we are about to recommend acting on. Those are different
tests and the second is the one that decides.

The same check is worth running on the other date columns, because a table whose
timestamps are unrelated to the account they hang off cannot carry a signal
about that account no matter how many rows it has. `table_linkage` reports that.
"""
import numpy as np
import pandas as pd
from scipy import stats

from . import survival
from .config import CUTOFF_DATE, EXTRACT_DATE, HORIZON_DAYS

# Enough replicates that the 95% band is stable to about a percentage point,
# and still a couple of seconds to run. Raise for a publication-grade band.
N_SIMS = 400


def _as_days(series):
    """Dates as integer days, which is the unit every draw below works in."""
    return pd.to_datetime(series).values.astype("datetime64[D]").astype(np.int64)


# --------------------------------------------------------------------------
# 1. Is a date column distinguishable from a uniform draw?
# --------------------------------------------------------------------------
def uniformity(dates, lower, upper):
    """KS test of `dates` against Uniform(lower, upper), elementwise bounds.

    Returns the rescaled positions too, because the histogram of those is the
    chart that makes the point to a non-statistician: a flat bar chart means
    "a coin toss decided this".

    Rows where the interval has zero width carry no information and are dropped
    rather than divided by zero.
    """
    d, lo, hi = (_as_days(x) for x in (dates, lower, upper))
    width = hi - lo
    keep = width > 0
    u = (d[keep] - lo[keep]) / width[keep]
    result = stats.kstest(u, "uniform")
    return pd.Series(u), {
        "n": int(keep.sum()),
        "mean_u": round(float(u.mean()), 4),
        "ks_stat": round(float(result.statistic), 4),
        "ks_p": round(float(result.pvalue), 4),
        "outside_bounds": int(((u < 0) | (u > 1)).sum()),
    }


def churn_date_uniformity(tables, extract_date=EXTRACT_DATE, by_cohort=True):
    """The headline test: is every churn date just a random date after signup?

    Run per signup quarter as well as pooled. Pooled uniformity could in
    principle arise from a mixture of non-uniform cohorts; uniformity holding
    *within* each cohort closes that off.
    """
    events = tables["churn_events"].merge(
        tables["accounts"][["account_id", "signup_date"]], on="account_id")
    end = pd.Series(pd.Timestamp(extract_date), index=events.index)
    u, overall = uniformity(events["churn_date"], events["signup_date"], end)

    rows = [{"cohort": "all", **overall}]
    if by_cohort:
        quarter = pd.to_datetime(events["signup_date"]).dt.to_period("Q")
        for period, group in events.groupby(quarter):
            if len(group) < 20:      # a KS p-value on 12 points says nothing
                continue
            _, stat = uniformity(group["churn_date"], group["signup_date"],
                                 pd.Series(pd.Timestamp(extract_date), index=group.index))
            rows.append({"cohort": str(period), **stat})
    return pd.DataFrame(rows), u


def date_column_uniformity(tables, extract_date=EXTRACT_DATE):
    """The same test on every date column, against the *whole* extract window.

    A column that is uniform over the full window regardless of when the account
    signed up has been sprayed across the calendar with no reference to the
    entity it belongs to. `signup_correlation` is the confirming view: near zero
    means the timestamp knows nothing about its own account.
    """
    accounts = tables["accounts"][["account_id", "signup_date"]]
    subs = tables["subscriptions"][["subscription_id", "account_id", "start_date"]]

    usage = tables["feature_usage"].merge(subs, on="subscription_id").merge(
        accounts, on="account_id")
    tickets = tables["support_tickets"].merge(accounts, on="account_id")
    subs_full = subs.merge(accounts, on="account_id")

    rows = []
    for label, frame, column in [
        ("feature_usage.usage_date", usage, "usage_date"),
        ("support_tickets.submitted_at", tickets, "submitted_at"),
        ("subscriptions.start_date", subs_full, "start_date"),
    ]:
        values = pd.to_datetime(frame[column])
        # The window is the whole extract, so the lower bound is the first date
        # anyone recorded — not the first signup, which would push the handful of
        # rows that predate every account outside [0, 1] for no useful reason.
        start = min(pd.to_datetime(accounts["signup_date"]).min(), values.min())
        lower = pd.Series(start, index=frame.index)
        upper = pd.Series(pd.Timestamp(extract_date), index=frame.index)
        _, stat = uniformity(values, lower, upper)
        signup = pd.to_datetime(frame["signup_date"])
        rho = stats.spearmanr(values.astype("int64"), signup.astype("int64"))
        rows.append({
            "column": label, **stat,
            "signup_correlation": round(float(rho.correlation), 4),
            "before_signup_pct": round(float((values < signup).mean() * 100), 1),
        })
    return pd.DataFrame(rows)


# --------------------------------------------------------------------------
# 2. Does that rule alone reproduce the effect we were about to act on?
# --------------------------------------------------------------------------
def simulate_churn_dates(tables, rng, extract_date=EXTRACT_DATE):
    """A copy of the tables with every churn date redrawn from the null rule.

    Deliberately minimal. Each account keeps its real signup date and its real
    *number* of churn events; only the dates are replaced, by a uniform draw
    between signup and the extract boundary. Nothing about customers, plans,
    usage or support enters. Anything this reproduces was never about customers.
    """
    events = tables["churn_events"]
    signup = tables["accounts"].set_index("account_id")["signup_date"]
    lower = _as_days(signup.reindex(events["account_id"]))
    upper = np.datetime64(pd.Timestamp(extract_date), "D").astype(np.int64)

    drawn = lower + np.floor(rng.random(len(events)) * (upper - lower))
    replaced = events.copy()
    replaced["churn_date"] = pd.to_datetime(drawn.astype("datetime64[D]"))
    return {**tables, "churn_events": replaced}


def calendar_hazard_null(tables, n_sims=N_SIMS, seed=0, extract_date=EXTRACT_DATE):
    """Observed monthly hazard against the band the null rule produces.

    Runs the *same* `survival.calendar_hazard` on real and simulated tables, so
    the observed rate ratio and the null distribution are computed by identical
    code. If the observed value sits mid-band, the reported effect is the
    generator, not the business.

    Returns (per-month frame, summary dict).
    """
    rng = np.random.default_rng(seed)
    observed = survival.calendar_hazard(tables, extract_date=extract_date)

    events, hazards, ratios, pvalues = [], [], [], []
    for _ in range(n_sims):
        sim = survival.calendar_hazard(
            simulate_churn_dates(tables, rng, extract_date), extract_date=extract_date)
        events.append(sim["events"].to_numpy())
        hazards.append(sim["hazard"].to_numpy())
        ratios.append(sim.attrs["rate_ratio_per_period"])
        pvalues.append(sim.attrs["p_trend"])

    events, hazards = np.array(events), np.array(hazards, dtype=float)
    ratios, pvalues = np.array(ratios), np.array(pvalues)

    frame = observed[["period", "at_risk", "events", "hazard"]].copy()
    frame["null_events"] = events.mean(axis=0).round(1)
    frame["null_lo"] = np.percentile(events, 2.5, axis=0)
    frame["null_hi"] = np.percentile(events, 97.5, axis=0)
    frame["inside_band"] = (frame["events"] >= frame["null_lo"]) & \
                           (frame["events"] <= frame["null_hi"])
    frame["null_hazard"] = np.nanmean(hazards, axis=0).round(4)
    frame["null_hazard_lo"] = np.nanpercentile(hazards, 2.5, axis=0).round(4)
    frame["null_hazard_hi"] = np.nanpercentile(hazards, 97.5, axis=0).round(4)

    observed_rr = observed.attrs["rate_ratio_per_period"]
    summary = {
        "observed_rate_ratio": observed_rr,
        "observed_annual": observed.attrs["rate_ratio_annual"],
        "observed_p_trend": observed.attrs["p_trend"],
        "null_rate_ratio": round(float(ratios.mean()), 4),
        "null_rate_ratio_ci": [round(float(np.percentile(ratios, 2.5)), 4),
                               round(float(np.percentile(ratios, 97.5)), 4)],
        "null_annual": round(float((ratios ** 12).mean()), 3),
        "null_median_p_trend": float(np.median(pvalues)),
        # Where the observed effect sits inside the null. ~50 means the
        # generator alone explains it; >97.5 would mean something real on top.
        "observed_percentile_in_null": round(float((ratios < observed_rr).mean() * 100), 1),
        "months_inside_band": int(frame["inside_band"].sum()),
        "months_total": int(len(frame)),
        "count_correlation": round(float(np.corrcoef(
            frame["events"], frame["null_events"])[0, 1]), 4),
    }
    return frame, summary


def tenure_gradient_null(tables, n_sims=N_SIMS, seed=1, extract_date=EXTRACT_DATE,
                         cutoff=None):
    """Does the null rule also manufacture the tenure effect behind onboarding?

    The onboarding case rests on newer accounts churning sooner. Under the null
    rule they must, for a reason that has nothing to do with onboarding: a recent
    signup has a shorter window for its random date to land in, so a larger share
    of its draws fall inside any fixed 90-day horizon.

    Measures the same thing on real and simulated data — among accounts still
    active at the cutoff, does tenure separate those who churn in the next 90
    days — and reports how often the null alone clears p < 0.05.
    """
    cutoff = pd.Timestamp(cutoff or CUTOFF_DATE)
    # Everything below indexes positionally, so pin one index rather than relying
    # on the caller's frame carrying a default one.
    accounts = tables["accounts"].reset_index(drop=True)
    signup = pd.to_datetime(accounts["signup_date"])
    eligible = signup <= cutoff

    def measure(events):
        first = events.groupby("account_id")["churn_date"].min()
        first = pd.to_datetime(accounts["account_id"].map(first))
        alive = eligible & (first.isna() | (first > cutoff))
        churned = (first > cutoff) & (first <= cutoff + pd.Timedelta(days=HORIZON_DAYS))
        tenure = (cutoff - signup).dt.days[alive]
        y = churned[alive]
        if y.sum() < 5 or (~y).sum() < 5:
            return np.nan, np.nan
        p = stats.mannwhitneyu(tenure[y], tenure[~y]).pvalue
        return float(p), float(y.mean())

    observed_p, observed_rate = measure(tables["churn_events"])

    rng = np.random.default_rng(seed)
    pvalues = np.array([
        measure(simulate_churn_dates(tables, rng, extract_date)["churn_events"])[0]
        for _ in range(n_sims)])
    pvalues = pvalues[~np.isnan(pvalues)]

    return {
        "observed_p": round(observed_p, 5),
        "observed_churn_rate": round(observed_rate, 4),
        "null_median_p": round(float(np.median(pvalues)), 5),
        "null_share_significant": round(float((pvalues < 0.05).mean()), 3),
        "n_sims": int(len(pvalues)),
    }


# --------------------------------------------------------------------------
# 3. Which tables carry any account-level structure at all?
# --------------------------------------------------------------------------
def table_linkage(tables):
    """Do the columns inside each table relate to each other as they should?

    Not a churn test. A far more basic one: does price track the plan, does
    priority track how fast we responded, does usage track the plan or the seat
    count. A table that fails this cannot carry a signal about anything,
    because its columns were drawn independently of one another.
    """
    subs, tickets = tables["subscriptions"], tables["support_tickets"]
    usage = tables["feature_usage"].merge(
        subs[["subscription_id", "plan_tier"]], on="subscription_id")

    def kruskal(frame, group, value):
        groups = [g[value].dropna().to_numpy() for _, g in frame.groupby(group)
                  if g[value].notna().sum() > 1]
        return float(stats.kruskal(*groups).pvalue) if len(groups) > 1 else np.nan

    checks = [
        ("subscriptions", "monthly price follows the plan tier",
         kruskal(subs, "plan_tier", "mrr_amount")),
        ("subscriptions", "monthly price follows the seat count",
         float(stats.spearmanr(subs["mrr_amount"], subs["seats"]).pvalue)),
        ("feature_usage", "usage volume follows the plan tier",
         kruskal(usage, "plan_tier", "usage_count")),
        ("feature_usage", "usage volume differs for beta features",
         kruskal(tables["feature_usage"], "is_beta_feature", "usage_count")),
        ("support_tickets", "resolution time follows ticket priority",
         kruskal(tickets, "priority", "resolution_time_hours")),
        ("support_tickets", "first response follows ticket priority",
         kruskal(tickets, "priority", "first_response_time_minutes")),
        ("support_tickets", "satisfaction is lower on escalated tickets",
         kruskal(tickets, "escalation_flag", "satisfaction_score")),
    ]
    frame = pd.DataFrame(checks, columns=["table", "relationship", "p"])
    frame["holds"] = frame["p"] < 0.05
    frame["p"] = frame["p"].round(4)
    return frame
