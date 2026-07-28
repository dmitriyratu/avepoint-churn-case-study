# %% [markdown]
# # 12 — Survival analysis: when do accounts churn?
#
# Notebook 11 found one real pattern — month-3 retention fell by a third between
# the 2023 and 2024 signup cohorts — and stopped, because a retention triangle
# cannot tell three different explanations apart:
#
# | effect | hazard depends on | what you would do |
# |---|---|---|
# | **tenure** | how long they have been a customer | fix onboarding |
# | **cohort** | when they signed up | fix acquisition |
# | **period** | today's calendar date | find what changed in the business |
#
# All three produce a decaying triangle. They imply completely different actions.
# Separating them is what survival analysis is for, and it is the reason this
# notebook exists.
#
# There is a fourth explanation, which none of the three can be separated from by
# any method in this notebook, and which turns out to be the right one here: the
# extract has a hard end date, and a churn date drawn at random before it
# manufactures all three patterns at once. **Notebook 16 tests that and it is
# where this notebook's headline result is withdrawn.** Read them together.
#
# **Why the framing change matters independently.** The classification setup used
# everywhere else keeps 177 accounts and 54 positives from one cutoff. This uses
# all 500 accounts and all 352 observed churn dates, and it handles the 148 still
# active correctly — they are **censored, not retained**. A binary label cannot
# express "we do not know yet"; it has to call them negatives, which is a claim
# the data does not support.

# %%
import sys
sys.path.insert(0, "..")

import warnings
warnings.filterwarnings("ignore")

import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from lifelines import KaplanMeierFitter

from src import pipeline, survival
from src.clean import clean_all
from src.config import CUTOFF_DATE, EXTRACT_DATE
from src.load_data import load_all

sns.set_theme(style="whitegrid", palette="muted")
pd.set_option("display.width", 130)

tables = clean_all(load_all())
frame = survival.survival_frame(tables)

print(f"accounts        {len(frame)}")
print(f"events observed {int(frame['event'].sum())}  (vs 54 positives in the "
      f"classification cohort)")
print(f"censored        {int((frame['event'] == 0).sum())}  still active at "
      f"{frame.attrs['extract_date']}")
print(f"negative durations dropped: {frame.attrs['dropped_negative_duration']}")

# %% [markdown]
# Zero negative durations. Worth stating explicitly given this dataset's history
# — 1,077 tickets predate their account's signup and 19,128 usage rows predate
# their subscription — but **the churn dates are ordered correctly relative to
# signup**. The time-to-event structure is the one part of this data that is
# internally coherent, which is exactly what makes the analysis viable here.

# %% [markdown]
# ## 1. The survival curve
#
# Kaplan-Meier, which is the estimator that handles censoring: an account with
# 40 days of follow-up contributes to the risk set for 40 days and then leaves
# it, rather than being counted as a survivor forever.

# %%
km_table = survival.km_summary(frame)
print(km_table.to_string(index=False))
print(f"\nmedian survival: {km_table.attrs['median_survival']:.0f} days")

# %%
kmf = KaplanMeierFitter().fit(frame["duration"], frame["event"], label="all accounts")
fig, ax = plt.subplots(figsize=(9, 5))
kmf.plot_survival_function(ax=ax, color="steelblue")
ax.axhline(0.5, color="grey", ls=":", lw=1)
ax.axvline(km_table.attrs["median_survival"], color="red", ls="--",
           label=f"median {km_table.attrs['median_survival']:.0f} d")
ax.set_xlabel("days since signup"); ax.set_ylabel("share still active")
ax.set_title("Kaplan-Meier survival — shaded band is the 95% interval")
ax.legend(); plt.tight_layout()
plt.savefig("../outputs/figures/12_km_overall.png", bbox_inches="tight")
plt.show()

# %% [markdown]
# **This is a real answer to a question the product team asked, and the first
# number in the project with a tight interval.** Half of all accounts are gone
# within 151 days. 38% are gone within 90 days of signing up, ±4 points.
#
# Note how much better-determined this is than anything in the classification
# work: `S(90d) = 0.616 [0.571, 0.658]` is a 9-point-wide interval, against the
# model ladder's ±0.20 AUC. Same dataset. The difference is 352 events instead
# of 54, and an estimator that uses censored rows instead of mislabelling them.

# %% [markdown]
# ## 2. Does any segment survive differently?
#
# The log-rank test is the properly-powered version of notebook 11's
# chi-square: it compares whole curves using the timing of every event, rather
# than a single rate at one horizon. BH-corrected across the seven segments.
#
# Baseline covariates only — first-subscription terms, not `accounts.plan_tier`
# or `accounts.seats`. Those are documented as current state as of extraction,
# so conditioning on them would be conditioning on the future.

# %%
logrank = survival.logrank_scan(frame)
print(logrank.to_string(index=False))

# %%
fig, axes = plt.subplots(1, 2, figsize=(14, 5))
for ax, seg in zip(axes, ["referral_source", "base_plan_tier"]):
    for level, group in frame.groupby(seg):
        KaplanMeierFitter().fit(group["duration"], group["event"],
                                label=f"{level} (n={len(group)})").plot_survival_function(
            ax=ax, ci_show=False)
    p = logrank.loc[logrank["segment"] == seg, "p"].iloc[0]
    p_bh = logrank.loc[logrank["segment"] == seg, "p_bh"].iloc[0]
    ax.set_title(f"{seg}  (log-rank p = {p:.3f}, BH {p_bh:.2f})")
    ax.set_xlabel("days since signup"); ax.set_ylabel("share still active")
    ax.legend(fontsize=8)
plt.tight_layout()
plt.savefig("../outputs/figures/12_km_segments.png", bbox_inches="tight")
plt.show()

# %% [markdown]
# `referral_source` comes closest at raw p = 0.056 — the same segment that was
# closest in notebook 11's chi-square (p = 0.14), which is at least consistent.
# It does not survive correction across seven segments (BH p = 0.39), and the
# curves cross. Nothing else is near.

# %% [markdown]
# ## 3. Cox proportional hazards
#
# The multivariate version: all baseline covariates at once, so a segment effect
# cannot hide inside another. Ridge-penalised, because 21 dummy columns over 500
# accounts with several sparse country levels is where an unpenalised fit returns
# hazard ratios spanning three orders of magnitude.

# %%
cox, cox_summary = survival.fit_cox(frame)
print(f"n = {cox_summary.attrs['n']}, events = {cox_summary.attrs['events']}, "
      f"covariates = {cox_summary.attrs['n_covariates']}")
print(f"concordance (Harrell's C) = {cox_summary.attrs['concordance']}   "
      f"(0.5 = chance, the survival analogue of AUC)")
print(f"global likelihood-ratio test: chi2 = {cox_summary.attrs['global_chi2']}, "
      f"p = {cox_summary.attrs['global_p']}")
print()
print(cox_summary.head(8).to_string())

# %% [markdown]
# **The global test is the one to read: p = 0.57.** Every baseline covariate
# together explains nothing about who leaves. Concordance 0.571 is in the same
# place as the classifier's 0.583 AUC, reached independently with a different
# estimator on 6.5× the events.
#
# `referral_source_organic` has raw p = 0.0099 and a hazard ratio of 1.44, which
# is exactly the row that gets quoted from a table like this. Corrected across
# the 21 coefficients it sits at **p = 0.21**. Under a pure null, the smallest of
# 21 p-values is below 0.05 about two thirds of the time — the raw value is what
# you should expect to see here even if nothing is going on.

# %% [markdown]
# ### Is the Cox model even applicable?
#
# A hazard ratio is one number held constant over all time. If a covariate's
# effect grows or fades, that number is a weighted average of a changing effect
# and the model is misspecified. Schoenfeld residual test:

# %%
ph = survival.ph_assumption(cox, frame)
print(f"covariates violating proportional hazards (BH < 0.05): "
      f"{ph.attrs['n_violations']} of {len(ph)}")
print(ph.sort_values("p").head(5).to_string())

# %% [markdown]
# Assumption holds everywhere. The null above is a real null, not a
# misspecification artefact — worth checking before concluding "no effect", since
# a violated PH assumption can hide a genuine time-varying effect behind an
# average of zero.

# %% [markdown]
# ## 4. The hazard shape — and why the obvious reading is wrong
#
# This is the section that changes a recommendation.
#
# Weibull against Exponential. The Exponential is the memoryless case (rho = 1):
# an account at day 300 is exactly as likely to leave tomorrow as one at day 10.
# rho < 1 means a **falling** hazard — fragile early, settling down — which is
# the shape an onboarding problem produces.

# %%
shape = survival.hazard_shape(frame)
for key, value in shape.items():
    print(f"  {key:18s} {value}")

# %%
hazard_points, curve = survival.smoothed_hazard(frame)
fig, ax = plt.subplots(figsize=(9, 4.5))
ax.plot(curve.index, curve.iloc[:, 0], color="steelblue", lw=2)
ax.set_xlabel("days since signup"); ax.set_ylabel("hazard (churns per account-day)")
ax.set_title(f"Pooled hazard falls with tenure — Weibull rho = {shape['rho']}, "
             f"p = {shape['p_vs_exponential']:.1e}")
ax.set_xlim(0, 500); plt.tight_layout()
plt.savefig("../outputs/figures/12_hazard_shape.png", bbox_inches="tight")
plt.show()
print(hazard_points.to_string(index=False))

# %% [markdown]
# Taken at face value this is emphatic and it is the strongest result anywhere in
# this project: **rho = 0.737, 95% CI [0.673, 0.801], p = 1.7e-13**. The hazard
# at day 15 is nearly three times the hazard at day 365. Churn is front-loaded;
# onboarding is where to spend.
#
# That reading is wrong, and the next two cells are why.

# %% [markdown]
# ### The decomposition
#
# A pooled falling hazard has a second explanation that the pooled curve cannot
# rule out. If recent signup cohorts churn faster **at every tenure**, and recent
# cohorts are precisely the ones contributing the short-tenure observations, then
# the pooled curve falls with tenure without any account ever becoming safer.
#
# That is a composition effect, and it is testable: inside a single cohort it
# cannot operate. Refit the shape within each cohort that has a full year of
# follow-up.

# %%
within = survival.shape_within_cohorts(frame)
print(within.to_string(index=False))
print(f"\npooled rho {shape['rho']}  vs  within-cohort rho "
      f"{within['rho'].min():.2f}-{within['rho'].max():.2f}")
print(f"cohorts where the falling shape is significant: "
      f"{(within['p_vs_exponential'] < 0.05).sum()} of {len(within)}")

# %% [markdown]
# **rho returns to 1 in every cohort.** The values scatter around it (0.87, 1.25,
# 1.17, 1.06, 0.88), two of the five point the other way, and not one is
# distinguishable from exponential. Within a signup cohort, churn is memoryless:
# tenure carries no information about who leaves next.
#
# The confirming view — survival at *fixed tenure*, cohort by cohort. Every
# cohort measured at the same age, so unequal follow-up cannot produce a
# difference:

# %%
by_cohort = survival.km_at_tenure_by_cohort(frame)
print(by_cohort.to_string(index=False))

# %%
fig, ax = plt.subplots(figsize=(9, 5))
for col, colour in zip(["S(30d)", "S(60d)", "S(90d)"], ["#4c72b0", "#dd8452", "#55a868"]):
    valid = by_cohort.dropna(subset=[col])
    ax.plot(valid["cohort"], valid[col], "o-", color=colour, label=col)
ax.set_xlabel("signup cohort"); ax.set_ylabel("share surviving to that tenure")
ax.set_title("Same tenure, different signup cohorts")
ax.legend(); plt.tight_layout()
plt.savefig("../outputs/figures/12_cohort_gradient.png", bbox_inches="tight")
plt.show()

# %% [markdown]
# 30-day survival falls from **0.98 for the 2023Q2 cohort to 0.59 for 2024Q4** —
# measured at the identical age, so unequal follow-up is not the explanation.
#
# So the pooled falling hazard was composition, and the conclusion inverts:
#
# > **Churn is not front-loaded. Tenure is not a risk factor.** The apparent
# > tenure effect is recent cohorts — who churn faster at every age —
# > contributing most of the short-tenure observations.
#
# This matters beyond the statistics. Notebook 05's strongest recommendation was
# "onboard the first 6 months harder", resting on `days_since_signup` being the
# top single feature and on a tenure-band table. Both are this same composition
# effect seen through a weaker lens. **A structured onboarding programme is the
# wrong intervention** — within any cohort, a day-300 account is as likely to
# leave as a day-10 account.
#
# > **Read on with notebook 16 open.** "Recent cohorts churn faster at every age"
# > is itself reproduced by a generator that draws each churn date uniformly
# > between signup and the last day of the extract: a recent signup has a shorter
# > window for that draw to land in. Simulated data with no tenure effect in it
# > clears p < 0.05 in 93% of runs. The recommendation above survives — nothing
# > here supports an onboarding programme — but "composition" is a symptom, not
# > the cause. The cause is the shape of the file, and section 5 below is where
# > that becomes unavoidable.

# %% [markdown]
# ## 5. Cohort or period?
#
# One decomposition left. "Recent cohorts are worse" (an acquisition-quality
# problem) and "everyone is churning more lately" (something changed in the
# business) both produce the gradient above, because a recent cohort spends all
# of its short life in recent calendar time.
#
# The person-period table separates them: for each calendar month, how many
# accounts were at risk and how many left — pooling every cohort and every tenure.

# %%
calendar = survival.calendar_hazard(tables)
print(calendar.tail(12).to_string(index=False))
print(f"\nmean monthly hazard, first 12 months: {calendar.attrs['first_year_mean']}")
print(f"mean monthly hazard, last 12 months : {calendar.attrs['last_year_mean']}")

# %% [markdown]
# Poisson regression on calendar month with `log(at_risk)` as offset. The offset
# is what makes this a test of the *rate*: raw event counts would rise simply
# because the customer base grew from 17 accounts to 200.

# %%
print(f"monthly rate ratio : {calendar.attrs['rate_ratio_per_period']} "
      f"95% CI {calendar.attrs['rate_ratio_ci']}")
print(f"implied annual     : x{calendar.attrs['rate_ratio_annual']}")
print(f"trend p-value      : {calendar.attrs['p_trend']:.2e}")

# %%
usable = calendar[calendar["at_risk"] > 0]
fig, ax1 = plt.subplots(figsize=(11, 5))
ax1.bar(usable["period"], usable["at_risk"], color="lightgrey", label="accounts at risk")
ax1.set_ylabel("accounts at risk"); ax1.set_xlabel("calendar month")
ax1.tick_params(axis="x", rotation=90)
ax2 = ax1.twinx()
ax2.plot(usable["period"], usable["hazard"], "o-", color="crimson",
         label="monthly churn hazard")
ax2.set_ylabel("churn hazard (share of at-risk leaving)", color="crimson")
ax2.grid(False)
ax1.set_title(f"Churn hazard by calendar month — x{calendar.attrs['rate_ratio_annual']} "
              f"per year, p = {calendar.attrs['p_trend']:.0e}")
fig.legend(loc="upper left", bbox_to_anchor=(0.1, 0.95))
plt.tight_layout()
plt.savefig("../outputs/figures/12_calendar_hazard.png", bbox_inches="tight")
plt.show()

# %% [markdown]
# **A period effect, and by a wide margin the strongest number in the project.**
#
# The monthly churn hazard rises from ~0.05 through 2023 to **0.225 in December
# 2024** — a rate ratio of 1.089 per month, 95% CI [1.067, 1.112], which
# compounds to **2.8× per year**, p = 2e-16. The at-risk base is flat at ~200
# accounts through all of 2024, so this is not a growth artefact.
#
# Set against everything else in this project, the contrast is stark:
#
# | question | best evidence | verdict |
# |---|---|---|
# | Which *customers* churn? | Cox global p = 0.57, concordance 0.571 | nothing |
# | Which *segment* churns? | 7 log-rank tests, min BH p = 0.39 | nothing |
# | Does *tenure* predict churn? | within-cohort rho ≈ 1 | nothing |
# | Does the *date* predict churn? | RR 1.089/month, p = 2e-16 | **see notebook 16** |
#
# **That asymmetry is the reason to distrust it.** One enormous, exquisitely
# significant result surrounded by nothing but nulls is a pattern that deserves
# explaining before it is acted on, and the tests above cannot explain it. The
# Poisson trend asks "is the rate flat". The cohort view asks "is this unequal
# follow-up". Both are sound and both are answered. Neither can see the failure
# mode that applies to any file with a hard end date:
#
# > No churn date can land after 2024-12-31. A generator that draws each one
# > uniformly between signup and that boundary produces a hazard of
# > `1 / (END - t)` — rising without limit toward the end of the file, on data
# > where nothing whatever happened.
#
# **Notebook 16 runs that test, and this claim does not survive it.** The churn
# dates are uniform on exactly that interval (KS p = 0.92 pooled, and inside
# every signup quarter). Redrawing them from that rule alone — same signup dates,
# same number of events per account, nothing else — and rerunning
# `survival.calendar_hazard` 400 times gives a rate ratio of **1.0885 against the
# 1.0893 observed**, with all 24 months inside the null band. The observed effect
# sits at the 52nd percentile of pure noise, and the null's median trend p-value
# is 2.8e-16 — marginally *more* significant than the real data.
#
# So the fourth row of the table joins the other three. Churn here is not a
# time-varying account-invariant process; it is a random number. Every model in
# notebooks 04–10 sits at chance because there is nothing to find, in the
# cross-section or over time.
#
# What survives is the negative, and it is worth stating precisely. This extract
# cannot answer why customers leave. Not "we need a price-change log to attribute
# the rise" — there is no rise to attribute. Sending the business to hunt for a
# 2024 pricing event or outage would be sending a team after something that did
# not happen.

# %% [markdown]
# ## 6. Does the time-to-event framing rescue prediction? (Q2)
#
# The framing change is worth testing on its own terms. Same 177 accounts, same
# point-in-time features, but the outcome becomes "how long until they left",
# censored at extraction, instead of "did they leave within 90 days".
#
# That alone converts more of the follow-up into observed events:

# %%
data = pipeline.build()
cohort_surv = survival.cohort_survival_frame(data.cohort, data.tables, CUTOFF_DATE)
X = data.X.copy()
X.index = pd.Index(data.cohort["account_id"].values, name="account_id")
cohort_surv = cohort_surv.loc[X.index]

print(f"follow-up available     {cohort_surv.attrs['followup_days']} days "
      f"(cutoff {CUTOFF_DATE.date()} -> extract {EXTRACT_DATE.date()})")
print(f"events, survival framing  {int(cohort_surv['event'].sum())}")
print(f"positives, 90-day binary  {int(data.y.sum())}")
print(f"gain                      +{int(cohort_surv['event'].sum()) - int(data.y.sum())} "
      f"events from the same accounts")

# %%
concordance = survival.cv_concordance(X, cohort_surv["duration"], cohort_surv["event"])
print("cross-validated Cox concordance on the modelling features:")
for key, value in concordance.items():
    print(f"  {key:14s} {value}")

# %% [markdown]
# **0.51, and the folds straddle chance (0.39 to 0.60).** 50% more events, an estimator that
# uses the timing of each one, and no censored row wasted — and the answer is
# the same as the classifier's nested 0.534.
#
# This is the most useful confirmation in the notebook. The negative result for
# Q2 is not an artefact of the binary framing, the 90-day window, or the
# single-cutoff design. Three independent formulations of the prediction problem
# land at chance, which is a far stronger claim than any one of them repeated.

# %% [markdown]
# ## 7. RMST — retention in days, for the economics
#
# Hazard ratios are hard to price. Restricted mean survival time is in days:
# "this segment is worth N more days of retention in the first year", which
# multiplies directly by MRR in notebook 15.

# %%
rmst = survival.rmst_by(frame, "referral_source", tau=365)
print(rmst.to_string(index=False))
print(f"\nspread: {rmst['rmst_365d'].max() - rmst['rmst_365d'].min():.0f} days "
      f"between best and worst channel")

# %% [markdown]
# A 54-day spread between `ads` and `organic` looks substantial and is the number
# a channel-reallocation deck would be built on. Hold it against section 2: the
# log-rank for `referral_source` does not survive correction, so this spread is
# not distinguishable from noise. It is carried into notebook 15 as an
# **upper bound on what channel mix could be worth**, not as an estimate.

# %%
findings = pd.DataFrame([
    {"analysis": "Kaplan-Meier, overall", "result": f"median {km_table.attrs['median_survival']:.0f}d, "
     f"S(90d)={km_table.iloc[2]['survival']}", "verdict": "solid"},
    {"analysis": "Log-rank, 7 baseline segments",
     "result": f"min BH p = {logrank['p_bh'].min():.2f}", "verdict": "no effect"},
    {"analysis": "Cox PH, 21 covariates",
     "result": f"global p = {cox_summary.attrs['global_p']}, "
     f"C = {cox_summary.attrs['concordance']}", "verdict": "no effect"},
    {"analysis": "Hazard shape, pooled",
     "result": f"rho = {shape['rho']}, p = {shape['p_vs_exponential']:.1e}",
     "verdict": "confounded"},
    {"analysis": "Hazard shape, within cohort",
     "result": f"rho {within['rho'].min():.2f}-{within['rho'].max():.2f}, none significant",
     "verdict": "tenure has no effect"},
    {"analysis": "Calendar-time Poisson trend",
     "result": f"RR {calendar.attrs['rate_ratio_per_period']}/month "
     f"(x{calendar.attrs['rate_ratio_annual']}/yr), p = {calendar.attrs['p_trend']:.0e}",
     "verdict": "REAL AND LARGE"},
    {"analysis": "Cox concordance on cohort features",
     "result": f"{concordance['concordance']} +/- {concordance['sd']}",
     "verdict": "chance"},
])
findings.to_csv("../outputs/reports/survival_findings.csv", index=False)
print(findings.to_string(index=False))

# %% [markdown]
# ## Takeaway
#
# 1. **Retention is measurable and the curve is well determined.** Median 151
#    days, S(90d) = 0.62 ± 0.04. The best-estimated quantity in the project.
# 2. **No customer attribute predicts who leaves.** Cox global p = 0.57 over 21
#    baseline covariates, with the PH assumption holding, on 352 events.
# 3. **Tenure is not a risk factor, and the obvious analysis says it is.** The
#    pooled hazard falls at p = 1.7e-13; within cohorts it is flat. This
#    overturns notebook 05's onboarding recommendation.
# 4. **What reads as a period effect** — 2.8× per year, p = 2e-16, on a flat
#    at-risk base — **is right-truncation, not a business change.** Notebook 16
#    reproduces it in full from a uniform random date, at the 52nd percentile of
#    the null. Findings 3 and 4 have the same single cause.
# 5. **The prediction negative is framing-independent.** Binary/90-day, nested
#    CV, and time-to-event all land at chance.
#
# The action this implies is not a model, not an onboarding programme, and not an
# investigation into 2024. It is a different extract — one whose timestamps are
# recorded against the accounts they belong to. Notebook 14 tests what can be
# recovered causally from what is here, notebook 15 prices the options, and
# notebook 16 is where finding 4 is withdrawn.
