"""Retrospective churn analysis: the "why did they leave" question.

Everything here reads `churn_events` — the reason codes, the free text, the
refunds. Those columns are correctly banned from the feature layer
(`config.POST_OUTCOME_COLS`): a reason exists only once a customer has gone, so
using one to predict churn is leakage. Reading them *after the fact* to describe
churn is a different question and a legitimate one, which is why this module
exists separately rather than inside `features/`.

The distinction worth keeping: `features/` answers "what could we have known",
this answers "what happened". Nothing here may feed a model.

The organising principle is the same as the rest of the project — every claimed
pattern gets a null to beat. A reason taxonomy that carries information should
(a) be unevenly distributed and (b) line up with observable behaviour. Both are
testable, and both tests are run below rather than assumed.
"""
import numpy as np
import pandas as pd
from scipy import stats
from statsmodels.stats.multitest import multipletests
from statsmodels.stats.proportion import proportion_confint

# Behavioural measures a stated reason ought to line up with, if the reasons
# mean anything. "support" should track tickets and satisfaction, "pricing" and
# "budget" should track MRR, "features" should track adoption breadth.
BEHAVIOUR_COLS = ["n_tickets", "mean_satisfaction", "n_escalations", "mean_mrr",
                  "n_upgrades", "n_downgrades", "mean_seats", "n_features_used",
                  "total_usage"]

# Which behaviour each reason code predicts, if the taxonomy is real. Stated up
# front so the test is confirmatory rather than a search over nine columns.
REASON_EXPECTATIONS = {
    "support": "n_tickets up, mean_satisfaction down",
    "pricing": "mean_mrr up",
    "budget": "mean_mrr up, n_downgrades up",
    "features": "n_features_used down, total_usage down",
    "competitor": "no specific prediction",
    "unknown": "no specific prediction",
}


def first_churn_event(tables):
    """One row per churned account — the *first* event, matching `labeling`.

    600 events over 352 accounts: 10% are reactivations, so an account can leave
    more than once. Taking the first keeps this consistent with
    `labeling.first_churn_date`, which defines the modelling label.
    """
    events = tables["churn_events"]
    return events.sort_values("churn_date").groupby("account_id").first()


def account_behaviour(tables):
    """Per-account behavioural summary, indexed by account_id.

    Deliberately *not* point-in-time correct: this is retrospective description
    over the whole history, not a feature matrix. `features/` is the
    cutoff-respecting version and the two must not be confused.
    """
    accounts, subs = tables["accounts"], tables["subscriptions"]
    tickets, usage = tables["support_tickets"], tables["feature_usage"]

    usage_by_account = usage.merge(subs[["subscription_id", "account_id"]],
                                   on="subscription_id", how="inner")
    out = pd.DataFrame(index=pd.Index(accounts["account_id"], name="account_id"))

    ticket_groups = tickets.groupby("account_id")
    out["n_tickets"] = ticket_groups.size()
    out["mean_satisfaction"] = ticket_groups["satisfaction_score"].mean()
    out["n_escalations"] = ticket_groups["escalation_flag"].sum()

    sub_groups = subs.groupby("account_id")
    out["mean_mrr"] = sub_groups["mrr_amount"].mean()
    out["n_upgrades"] = sub_groups["upgrade_flag"].sum()
    out["n_downgrades"] = sub_groups["downgrade_flag"].sum()
    out["mean_seats"] = sub_groups["seats"].mean()

    usage_groups = usage_by_account.groupby("account_id")
    out["n_features_used"] = usage_groups["feature_name"].nunique()
    out["total_usage"] = usage_groups["usage_count"].sum()

    # Counts genuinely are zero when a customer has no tickets or no usage;
    # mean_satisfaction is left NaN because "no rating" is not "rated zero".
    counts = ["n_tickets", "n_escalations", "n_upgrades", "n_downgrades",
              "n_features_used", "total_usage"]
    return out.assign(**{c: out[c].fillna(0) for c in counts})


def reason_distribution(tables):
    """Reason-code counts with a test against the uniform.

    A taxonomy carrying information concentrates: real churn has a dominant
    cause. A flat distribution over six codes is what a random assignment looks
    like, so the chi-square against uniform is the first thing to run.
    """
    counts = first_churn_event(tables)["reason_code"].value_counts()
    chi2, p = stats.chisquare(counts.values)
    out = counts.rename("n").to_frame()
    out["share"] = (out["n"] / out["n"].sum()).round(4)
    out.attrs["chi2"] = round(float(chi2), 3)
    out.attrs["p_uniform"] = round(float(p), 4)
    out.attrs["n_events"] = int(out["n"].sum())
    return out


def reason_behaviour_coherence(tables, behaviour_cols=BEHAVIOUR_COLS):
    """Does the stated reason line up with what the account actually did?

    Kruskal-Wallis rather than ANOVA: these distributions are counts and heavily
    skewed, and a rank test does not ask them to be normal. Benjamini-Hochberg
    across the nine measures, because testing nine and reporting the smallest is
    the selection error this project documents elsewhere.
    """
    reasons = first_churn_event(tables)[["reason_code"]]
    frame = account_behaviour(tables).join(reasons, how="inner")

    rows = []
    for col in behaviour_cols:
        groups = [g[col].dropna().values for _, g in frame.groupby("reason_code")]
        groups = [g for g in groups if len(g) > 1]
        h, p = stats.kruskal(*groups)
        spread = frame.groupby("reason_code")[col].median()
        rows.append({"measure": col, "H": round(float(h), 3), "p": float(p),
                     "median_lo": round(float(spread.min()), 3),
                     "median_hi": round(float(spread.max()), 3)})

    out = pd.DataFrame(rows)
    out["p_bh"] = multipletests(out["p"], method="fdr_bh")[1]
    out.attrs["n_accounts"] = len(frame)
    return out.round({"p": 4, "p_bh": 4})


def reason_vs_feedback(tables):
    """Cross-tabulate the coded reason against the free-text field.

    These are two recordings of the same fact. If both are real they agree —
    an account coded `pricing` should not be writing "missing features". The
    chi-square is a test of whether they carry the same information at all.
    """
    events = first_churn_event(tables)
    table = pd.crosstab(events["reason_code"],
                        events["feedback_text"].fillna("(missing)"))
    chi2, p, dof, _ = stats.chi2_contingency(table)

    # Cramer's V, so the effect size is readable next to the p-value.
    n = table.values.sum()
    v = np.sqrt(chi2 / (n * (min(table.shape) - 1)))
    table.attrs.update(chi2=round(float(chi2), 3), p=round(float(p), 4),
                       dof=int(dof), cramers_v=round(float(v), 4))
    return table


def segment_churn_rates(tables, by, target=None):
    """Churn rate per level of `by`, with Wilson intervals and a chi-square.

    Wilson rather than normal-approximation intervals: several of these cells
    hold ~20 accounts, where the normal interval runs past 0 or 1 and understates
    uncertainty exactly where it matters most.
    """
    accounts = tables["accounts"]
    if target is None:
        churned = accounts["account_id"].isin(tables["churn_events"]["account_id"])
        target = pd.Series(churned.values, index=accounts.index, name="ever_churned")

    frame = accounts.assign(_target=target.astype(int))
    grouped = frame.groupby(by)["_target"].agg(["size", "sum"])
    lo, hi = proportion_confint(grouped["sum"], grouped["size"],
                                alpha=0.05, method="wilson")

    out = grouped.rename(columns={"size": "n", "sum": "churned"})
    out["rate"] = (out["churned"] / out["n"]).round(4)
    out["ci_lo"], out["ci_hi"] = lo.round(4), hi.round(4)

    contingency = pd.crosstab(frame[by], frame["_target"])
    chi2, p, dof, _ = stats.chi2_contingency(contingency)
    out.attrs.update(segment=by, chi2=round(float(chi2), 3), p=round(float(p), 4),
                     dof=int(dof), overall_rate=round(float(frame["_target"].mean()), 4),
                     spread=round(float(out["rate"].max() - out["rate"].min()), 4))
    return out


def segment_scan(tables, segments=("industry", "country", "plan_tier",
                                   "referral_source", "is_trial"), target=None):
    """Every segment at once, with BH correction over the family of tests.

    Scanning five segments and quoting the smallest p is how a null dataset
    produces a "finding". The corrected column is the one to read.
    """
    rows = []
    for seg in segments:
        table = segment_churn_rates(tables, seg, target=target)
        rows.append({"segment": seg, "levels": len(table),
                     "min_rate": table["rate"].min(), "max_rate": table["rate"].max(),
                     "spread": table.attrs["spread"], "chi2": table.attrs["chi2"],
                     "p": table.attrs["p"]})

    out = pd.DataFrame(rows)
    out["p_bh"] = multipletests(out["p"], method="fdr_bh")[1]
    return out.round(4)


def max_segment_spread_null(tables, segments=("industry", "country", "plan_tier",
                                              "referral_source", "is_trial"),
                            n_permutations=500, seed=42):
    """How large a segment gap does pure noise produce?

    The chi-square answers this per segment; this answers it for the *scan*,
    which is what a reader actually does with the table — eye down the column
    for the biggest gap. Under a null that gap is not zero, and this measures how
    far from zero.
    """
    accounts = tables["accounts"]
    churned = accounts["account_id"].isin(tables["churn_events"]["account_id"]).astype(int)
    rng = np.random.default_rng(seed)

    observed = max(segment_churn_rates(tables, s, target=churned).attrs["spread"]
                   for s in segments)

    null = np.empty(n_permutations)
    for i in range(n_permutations):
        shuffled = pd.Series(rng.permutation(churned.values), index=accounts.index)
        null[i] = max(segment_churn_rates(tables, s, target=shuffled).attrs["spread"]
                      for s in segments)

    return {"observed_max_spread": round(float(observed), 4),
            "null_mean": round(float(null.mean()), 4),
            "null_p95": round(float(np.percentile(null, 95)), 4),
            "p_value": round(float((null >= observed).mean()), 4),
            "n_permutations": n_permutations}


def retention_triangle(tables, extract_date=None, freq="M", max_periods=12):
    """Classic cohort retention matrix: signup cohort x periods since signup.

    Cells are only defined where the cohort has been observed that long — a
    cohort three months old has no month-six number, and filling it with
    anything is a lie. Those cells stay NaN.
    """
    accounts, events = tables["accounts"], tables["churn_events"]
    extract = pd.Timestamp(extract_date) if extract_date is not None \
        else accounts["signup_date"].max()

    first_churn = events.groupby("account_id")["churn_date"].min()
    frame = accounts[["account_id", "signup_date"]].copy()
    frame["churn_date"] = frame["account_id"].map(first_churn)
    frame["cohort"] = frame["signup_date"].dt.to_period(freq)

    step = pd.Timedelta(days=30 if freq == "M" else 1)
    rows = {}
    for cohort, group in frame.groupby("cohort"):
        # Observable periods are bounded by the *oldest* possible age in the
        # cohort, measured from the cohort's own start.
        observable = int((extract - cohort.start_time) // step)
        retained = {}
        for period in range(min(observable, max_periods) + 1):
            age = period * step
            churned_by = (group["churn_date"] - group["signup_date"] <= age).sum()
            retained[period] = 1 - churned_by / len(group)
        rows[str(cohort)] = retained

    out = pd.DataFrame(rows).T.sort_index()
    out.columns.name = f"periods_since_signup ({freq})"
    out.attrs["cohort_sizes"] = frame.groupby("cohort").size().to_dict()
    return out.round(3)


__all__ = ["first_churn_event", "account_behaviour", "reason_distribution",
           "reason_behaviour_coherence", "reason_vs_feedback",
           "segment_churn_rates", "segment_scan", "max_segment_spread_null",
           "retention_triangle", "BEHAVIOUR_COLS", "REASON_EXPECTATIONS"]
