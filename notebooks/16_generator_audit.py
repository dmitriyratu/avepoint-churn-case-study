# %% [markdown]
# # 16 — Auditing the one positive finding
#
# Notebook 12 ended on the only strong result in this project: the monthly churn
# hazard rises by a factor of 2.8 a year, rate ratio 1.089 per month,
# p = 2e-16. Everything else in the study is a null. That asymmetry is exactly
# what should make a person suspicious, and this notebook is the check.
#
# The tests already run were the wrong ones. A Poisson trend with `log(at_risk)`
# as offset asks "is the rate flat", and the cohort-gradient view asks "is this
# unequal follow-up". Both are sound. Neither can see the failure mode that
# actually applies to a generated file:
#
# > **Right-truncation.** No churn date can fall after the last day of the
# > extract. If the generator drew each churn date uniformly between signup and
# > that boundary, the hazard it produces is `1 / (END - t)` — which rises
# > without limit as `t` approaches the end of the file, on data where nothing
# > whatever happened.
#
# That shape is indistinguishable by eye from "something changed in the
# business". It also makes recent cohorts look worse at every age, because their
# draw is squeezed into a shorter interval. So it can manufacture *both*
# time-based findings in notebook 12 at once.
#
# The null to test is therefore not "the rate is flat". It is:
#
# > `churn_date ~ Uniform(signup_date, EXTRACT_DATE)`, **and nothing else**
#
# Three questions, in order of how much they cost to be wrong about:
#
# 1. Is the churn date distinguishable from that draw?
# 2. Does that draw alone reproduce the effect we were going to act on?
# 3. Do the other tables link to the accounts they belong to at all?

# %%
import sys
sys.path.insert(0, "..")

import warnings
warnings.filterwarnings("ignore")

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

from src import generator, survival
from src.clean import clean_all
from src.config import EXTRACT_DATE
from src.load_data import load_all

sns.set_theme(style="whitegrid", palette="muted")
pd.set_option("display.width", 130)

tables = clean_all(load_all())

# %% [markdown]
# ## 1. Is the churn date distinguishable from a coin toss?
#
# Rescale every churn date to its position in the account's own available
# window: `u = (churn_date - signup) / (EXTRACT_DATE - signup)`. If the
# generator drew uniformly, `u` is flat on [0, 1] — and it is flat for an
# account that signed up on day 1 as well as one that signed up last month.
#
# Testing it *within* each signup quarter matters. A pooled uniform could in
# principle be a mixture of non-uniform cohorts. Uniformity holding inside every
# cohort closes that off.

# %%
cohorts, u = generator.churn_date_uniformity(tables)
print(cohorts.to_string(index=False))

# %% [markdown]
# **Every row is uniform, and the pooled test is not close to rejecting.**
# 600 events, mean position 0.503 against the 0.500 a uniform draw expects, KS
# p = 0.92. One quarter of eight lands at p = 0.014, which is what eight tests
# produce by chance — Bonferroni takes it to 0.11 and nothing survives. Zero
# events fall outside the interval, which is the generator's `signup <= churn`
# constraint showing through.
#
# For contrast, this is what a real business process looks like on the same
# scale: churn concentrated in the first months of tenure would pile `u` near 0
# for early cohorts, and a genuine late-2024 shock would pile it near 1 for all
# of them. Neither is present. The bar chart in the figure below is flat.

# %% [markdown]
# ## 2. Does that draw alone reproduce the finding?
#
# This is the test that decides, and it is deliberately blunt. Keep every
# account's real signup date and its real *number* of churn events. Replace only
# the dates, with a uniform draw. No customer attribute, no plan, no usage, no
# support record enters the simulation. Then run **the same
# `survival.calendar_hazard`** used in notebook 12 over 400 replicates, so the
# observed number and the null distribution come from identical code.
#
# Anything this reproduces was never a fact about customers.

# %%
hazard_null, summary = generator.calendar_hazard_null(tables, n_sims=400, seed=0)
print(hazard_null.tail(12).to_string(index=False))
print()
for key, value in summary.items():
    print(f"  {key:28s} {value}")

# %% [markdown]
# **The observed effect sits at the middle of the null.**
#
# | | observed | uniform-draw null |
# |---|---|---|
# | monthly rate ratio | 1.0893 | 1.0885, 95% band [1.072, 1.106] |
# | implied annual | x2.79 | x2.78 |
# | trend p-value | 2.2e-16 | median 2.8e-16 |
# | Dec-2024 hazard | 0.225 | 0.238, band [0.204, 0.271] |
#
# All 24 monthly counts fall inside the null band, and observed and simulated
# counts correlate at 0.97. The observed rate ratio sits at the **52nd
# percentile** of what the generator produces on its own — as close to the
# middle of the null as it is possible to land.
#
# The p-value deserves a moment. p = 2e-16 felt like the most secure number in
# the project. It is not evidence of anything, because the null it rejects
# ("the rate is flat") was never the relevant alternative. The uniform draw
# produces a *median* p of 2.8e-16 — it is, if anything, slightly more
# significant than the real data. **The strength of a p-value says nothing about
# whether the null it tests is the one that matters.**
#
# This is the whole lesson of the notebook: an effect can be large, precisely
# estimated, robust to every check you ran, and still be a property of how the
# file was written.

# %%
usable = hazard_null[hazard_null["at_risk"] > 0].reset_index(drop=True)
x = np.arange(len(usable))


def draw_uniformity(ax):
    ax.hist(u, bins=10, range=(0, 1), color="#2E5F8A", alpha=0.85,
            edgecolor="white", weights=np.full(len(u), 100 / len(u)))
    ax.axhline(10, color="#B02E2E", ls="--", lw=2, label="a uniform random draw")
    ax.set_xlabel("where the churn date falls in the account's own window\n"
                  "(0 = the day they signed up,  1 = the last day of the file)")
    ax.set_ylabel("share of churn events (%)")
    ax.set_ylim(0, 16)
    ax.set_title(f"Churn dates are a coin toss\nKS p = {cohorts.iloc[0]['ks_p']:.2f} "
                 f"against a uniform draw, n = {int(cohorts.iloc[0]['n'])}", loc="left")
    ax.legend(loc="upper right", frameon=True)


def draw_hazard(ax, step=3):
    ax.fill_between(x, usable["null_hazard_lo"], usable["null_hazard_hi"],
                    color="#B02E2E", alpha=0.18,
                    label="what the coin toss alone produces (95% of runs)")
    ax.plot(x, usable["null_hazard"], color="#B02E2E", ls="--", lw=1.8,
            label="coin-toss average")
    ax.plot(x, usable["hazard"], "o-", color="#1A1A1A", lw=2, ms=4.5,
            label="what we actually observe")
    ax.set_xticks(x[::step])
    ax.set_xticklabels([p[:7] for p in usable["period"]][::step], rotation=45, ha="right")
    ax.set_ylabel("share of at-risk accounts leaving that month")
    ax.legend(loc="upper left", frameon=True)


# Two panels for the technical deck: the mechanism, then what it manufactures.
fig, axes = plt.subplots(1, 2, figsize=(13.5, 4.6))
draw_uniformity(axes[0])
draw_hazard(axes[1])
axes[1].set_title(
    f"...which manufactures the whole rise\nobserved x{summary['observed_annual']:.2f}/yr, "
    f"coin toss x{summary['null_annual']:.2f}/yr, "
    f"{summary['months_inside_band']}/{summary['months_total']} months inside the band",
    loc="left")
plt.tight_layout()
plt.savefig("../outputs/figures/16_generator_artefact.png", bbox_inches="tight", dpi=150)
plt.show()

# One panel for the executive deck, where the mechanism is a speaker note and
# only the overlay has to survive being read from the back of a room.
fig, ax = plt.subplots(figsize=(10.5, 4.4))
draw_hazard(ax, step=2)
ax.set_title("The rise we see is exactly the rise a random date produces",
             loc="left", fontsize=13, fontweight="bold")
plt.tight_layout()
plt.savefig("../outputs/figures/16_artefact_exec.png", bbox_inches="tight", dpi=150)
plt.show()

# %% [markdown]
# ## 2b. The same rule also manufactures the onboarding case
#
# The calendar trend is not the only timing finding. Notebook 12 also showed
# newer cohorts churning faster at every age, and notebook 05 before it found
# `days_since_signup` to be the top single feature. Both point at onboarding.
#
# Under the null rule they must, and for a reason that has nothing to do with
# onboarding: a recent signup has a shorter window for its random date to land
# in, so a larger share of its draws fall inside any fixed 90-day horizon. Run
# the same measurement on real and simulated data.

# %%
tenure = generator.tenure_gradient_null(tables, n_sims=400, seed=1)
for key, value in tenure.items():
    print(f"  {key:24s} {value}")

# %% [markdown]
# **The null produces a stronger tenure effect than the real data does.**
# Observed p = 0.006; the median simulation gives p = 0.002, and **93% of runs
# with no tenure effect in them at all clear p < 0.05**.
#
# So the onboarding recommendation in notebook 12 was right, and the reason
# given for it was one layer too shallow. It was attributed to cohort
# composition. Composition is a symptom: the uniform draw generates the tenure
# gradient, the cohort gradient and the calendar trend as one thing.

# %% [markdown]
# ## 3. Do the other tables link to their accounts at all?
#
# The same worry generalises. A timestamp that was sprayed across the calendar
# without reference to the account it belongs to cannot carry information about
# that account, however many rows there are. Test each date column against a
# uniform draw over the *whole* extract window, and check whether it correlates
# with its own account's signup date.

# %%
print(generator.date_column_uniformity(tables).to_string(index=False))

# %% [markdown]
# **Two of the five tables are unlinked noise.**
#
# `feature_usage.usage_date` is uniform across the full two years (KS p = 0.48)
# and correlates with its own account's signup date at **r = 0.002**. Ticket
# timestamps are the same: uniform (p = 0.53), r = 0.014. That is what produces
# the headline data-quality numbers reported since notebook 01 — 77% of usage
# rows dated before the subscription they belong to, 53% of tickets before the
# customer existed. Those are not a broken column to repair. They are the
# visible edge of "the date was drawn at random".
#
# Only `subscriptions.start_date` behaves: never before signup, and correlated
# with it at r = 0.68.
#
# The consequence for the modelling is direct. Every recency, trend,
# acceleration and gap feature built on usage or support — 24 of the columns —
# is a function of random numbers. So is every window aggregate.
#
# The check below asks something even more basic than churn: do the columns
# *inside* each table relate to each other the way the schema says they should?

# %%
print(generator.table_linkage(tables).to_string(index=False))

# %% [markdown]
# **Only the subscriptions table was built with rules.** Price tracks the plan
# tier and the seat count, and ARR is exactly 12x MRR in 100% of rows.
#
# Everything else fails. Usage volume does not differ by plan tier (p = 0.89) or
# for beta features (p = 0.44). Ticket priority does not predict resolution time
# (p = 0.33) or first response (p = 0.13). Escalated tickets do not score lower
# on satisfaction (p = 0.33). These are not subtle effects that a bigger sample
# would find — they are columns drawn independently of one another.
#
# So the model was never short of rows. It was asked to predict a random date
# from random numbers, using the one table with real structure — subscriptions —
# whose contents are unrelated to churn. **AUC 0.534 is the correct answer**, and
# the learning curve in notebook 08 that appeared to promise more with more data
# was reading noise.

# %% [markdown]
# ## What this changes, and what it does not
#
# | claim | status |
# |---|---|
# | Churn is rising over calendar time; find what changed in 2024 | **withdrawn** — reproduced exactly by a uniform draw |
# | Recent cohorts churn faster at every age | **withdrawn** — same cause, same simulation |
# | Do not build the onboarding programme | **stands**, and the case is now stronger |
# | The three churn records are mutually unrelated | stands, untouched |
# | No customer characteristic separates leavers from stayers | stands, untouched |
# | Models sit at chance; nested AUC 0.534 | stands, and is now *explained* rather than lamented |
# | Contact every at-risk account, do not rank | stands — it rests on cost arithmetic, not on any of this |
#
# The onboarding recommendation is worth a line. It was reached in notebook 12
# by attributing the tenure pattern to cohort composition. The composition
# reasoning was incomplete — the same uniform draw generates the tenure
# gradient, the cohort gradient and the calendar trend together, and a
# simulation with no tenure effect in it produces a significant tenure effect in
# 93% of replicates. The recommendation was right. The reason given for it was
# one layer too shallow.
#
# **What the business should be told.** Not "churn is rising, go find the
# cause" — that would send a team looking for a price change or an outage that
# does not exist. The finding is that this extract cannot answer why customers
# leave, and no amount of modelling will change that, because the timestamps
# were not recorded against the customers they belong to.

# %%
report = pd.DataFrame([
    {"question": "Is churn_date distinguishable from a uniform draw after signup?",
     "test": "KS, pooled and by signup quarter",
     "result": f"p = {cohorts.iloc[0]['ks_p']}, mean position "
               f"{cohorts.iloc[0]['mean_u']}", "verdict": "no - it is a coin toss"},
    {"question": "Does that draw reproduce the calendar trend?",
     "test": f"{summary['months_total']}-month hazard vs 400 simulations",
     "result": f"RR {summary['observed_rate_ratio']} observed vs "
               f"{summary['null_rate_ratio']} null; "
               f"{summary['months_inside_band']}/{summary['months_total']} inside band",
     "verdict": "YES - the finding is an artefact"},
    {"question": "Is the trend p-value evidence of anything?",
     "test": "Poisson trend p under the null rule",
     "result": f"median null p = {summary['null_median_p_trend']:.0e}",
     "verdict": "no - the null tested was the wrong one"},
    {"question": "Does the null also manufacture the tenure effect?",
     "test": "Mann-Whitney on tenure vs 400 simulations",
     "result": f"observed p = {tenure['observed_p']}, null median p = "
               f"{tenure['null_median_p']}, {tenure['null_share_significant']:.0%} "
               f"of null runs significant",
     "verdict": "YES - onboarding evidence is an artefact too"},
    {"question": "Do usage timestamps relate to their account?",
     "test": "Spearman vs signup_date", "result": "r = 0.002",
     "verdict": "no - unlinked"},
    {"question": "Do ticket timestamps relate to their account?",
     "test": "Spearman vs signup_date", "result": "r = 0.014",
     "verdict": "no - unlinked"},
    {"question": "Do any tables have internal structure?",
     "test": "7 within-table relationships from the schema",
     "result": "2 of 7 hold, both in subscriptions",
     "verdict": "subscriptions only"},
])
report.to_csv("../outputs/reports/generator_audit.csv", index=False)
print(report.to_string(index=False))
