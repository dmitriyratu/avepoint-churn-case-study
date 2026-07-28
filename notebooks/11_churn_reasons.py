# %% [markdown]
# # 11 — Why are users leaving? (retrospective)
#
# The product question is "why are users leaving". Notebooks 01–10 answer a
# different one — *can we predict who will leave* — and answer it negatively.
# This notebook goes after the descriptive question directly, using the table
# that exists to answer it: `churn_events`, which records a reason code and a
# free-text comment for every departure.
#
# **Why this is not leakage.** `reason_code`, `refund_amount_usd` and
# `feedback_text` are banned from the feature layer by name
# (`config.POST_OUTCOME_COLS`) because a reason only exists once the customer has
# gone. Reading them *after the fact* to describe churn is a different question.
# Nothing computed here feeds a model, and `src/reasons.py` is deliberately a
# separate module from `src/features/` so the two cannot be confused.
#
# The bar a descriptive finding has to clear is the same as everywhere else in
# this project: beat a null. For a reason taxonomy that means two things, both
# testable —
#
# 1. the codes should be **unevenly distributed** (real churn has dominant causes)
# 2. they should **line up with observable behaviour** (a "support" churner should
#    look different in the ticket log from a "pricing" churner)
#
# Both tests run below.

# %%
import sys
sys.path.insert(0, "..")

import warnings
warnings.filterwarnings("ignore")

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

from src import reasons
from src.clean import clean_all
from src.load_data import load_all

sns.set_theme(style="whitegrid", palette="muted")
pd.set_option("display.width", 120)

tables = clean_all(load_all())
accounts, events = tables["accounts"], tables["churn_events"]

print(f"accounts        {len(accounts)}")
print(f"churn events    {len(events)} over {events['account_id'].nunique()} accounts "
      f"({events['account_id'].nunique() / len(accounts):.1%} ever churned)")
print(f"reactivations   {events['is_reactivation'].mean():.1%} of events")

# %% [markdown]
# ## 1. The reason codes
#
# Six codes. If they carry information, some causes dominate — real SaaS churn is
# never a flat six-way split.

# %%
dist = reasons.reason_distribution(tables)
print(dist.to_string())
print(f"\nchi-square vs uniform: chi2 = {dist.attrs['chi2']}, "
      f"p = {dist.attrs['p_uniform']}")

# %%
fig, ax = plt.subplots(figsize=(8, 4))
order = dist.index
ax.bar(order, dist["n"], color="steelblue")
expected = dist["n"].sum() / len(dist)
ax.axhline(expected, color="red", ls="--",
           label=f"uniform expectation ({expected:.0f})")
ax.set_ylabel("churn events"); ax.set_xlabel("stated reason")
ax.set_title(f"Reason codes are indistinguishable from uniform "
             f"(p = {dist.attrs['p_uniform']:.2f})")
ax.legend(); plt.tight_layout()
plt.savefig("../outputs/figures/11_reason_distribution.png", bbox_inches="tight")
plt.show()

# %% [markdown]
# Six codes between 50 and 65 first-churn events, **p = 0.70** against the
# uniform. There is no dominant cause of churn in this data — or the field does
# not record one.
#
# (First event per account, matching `labeling.first_churn_date`, so this is 352
# events not 600. Over all 600 the split is just as flat, p = 0.55.)
#
# That alone is not proof the field is meaningless: a genuinely diverse customer
# base could produce a flat split. The second test is the one that decides it.

# %% [markdown]
# ## 2. Do the stated reasons match what the accounts actually did?
#
# This is the test that matters. An account coded `support` should have more
# tickets and worse satisfaction scores than one coded `pricing`. An account
# coded `features` should show narrower adoption. Those are predictions the data
# can refute.
#
# Stated before running, so this is confirmatory rather than a hunt across nine
# columns:

# %%
for code, expectation in reasons.REASON_EXPECTATIONS.items():
    print(f"  {code:12s} -> {expectation}")

# %% [markdown]
# Kruskal-Wallis across the six reason groups for each behavioural measure —
# a rank test, because these are skewed counts and none of them is normal.
# Benjamini-Hochberg across the nine measures, since testing nine and quoting
# the smallest is the selection error notebook 09 is about.

# %%
coherence = reasons.reason_behaviour_coherence(tables)
print(f"n = {coherence.attrs['n_accounts']} churned accounts\n")
print(coherence.to_string(index=False))

# %%
print(f"smallest raw p      : {coherence['p'].min():.3f}")
print(f"smallest BH-adjusted: {coherence['p_bh'].min():.3f}")
print(f"measures below 0.05 : {(coherence['p_bh'] < 0.05).sum()} of {len(coherence)}")

# %% [markdown]
# **Nothing.** Nine behavioural measures, the smallest raw p-value is 0.278, and
# after correction every one sits at 0.91. Accounts that said they left over
# support have the same ticket counts and the same satisfaction scores as
# accounts that said they left over pricing. Accounts that said they left over
# missing features used the same number of features as everyone else.
#
# The median columns make the size of the non-effect concrete — read `median_lo`
# against `median_hi`, the extremes across all six reason groups:

# %%
spread = coherence.assign(gap=lambda d: d["median_hi"] - d["median_lo"])
print(spread[["measure", "median_lo", "median_hi", "gap"]].to_string(index=False))

# %% [markdown]
# The gaps are nil where it matters most: identical median satisfaction (4.0),
# identical escalations (0), identical upgrades and downgrades across all six
# groups. One ticket and one feature separate the extremes on the count measures.
# The `support` group is not a support story and the `features` group is not an
# adoption story.

# %% [markdown]
# ## 3. The free text says something different from the code
#
# Two independent recordings of the same fact. If both are real they agree: an
# account coded `pricing` should not be writing "missing features".

# %%
crosstab = reasons.reason_vs_feedback(tables)
print(crosstab.to_string())
print(f"\nchi-square p = {crosstab.attrs['p']}, "
      f"Cramer's V = {crosstab.attrs['cramers_v']} "
      f"(0 = independent, 1 = perfectly determined)")

# %% [markdown]
# **They are independent.** Accounts coded `budget` write "missing features" as
# often as they write "too expensive". Cramer's V of 0.09 on a table this size is
# no association at all.
#
# The three verdicts stack: the codes are uniform, they do not predict behaviour,
# and they do not agree with the customer's own words. `reason_code` is not a
# record of why anyone left. Treating it as one — building a "top churn reasons"
# slide from the counts in section 1 — would produce a confident, entirely
# fictional answer to the product team's first question. That slide is the most
# likely wrong output of this dataset, which is why the test is here.
#
# Note what this does *not* say. The customers presumably did leave for reasons.
# The claim is about the field, not about them.

# %% [markdown]
# ## 4. Does churn concentrate in any segment?
#
# The other standard route to "why": find the segment that leaves. Rates with
# Wilson intervals — several of these cells hold ~20 accounts, where the normal
# approximation runs past 0 and 1 and understates uncertainty exactly where it
# matters.

# %%
scan = reasons.segment_scan(tables)
print(scan.to_string(index=False))

# %%
fig, axes = plt.subplots(1, 3, figsize=(15, 4), sharex=True)
for ax, seg in zip(axes, ["industry", "plan_tier", "referral_source"]):
    table = reasons.segment_churn_rates(tables, seg)
    y = np.arange(len(table))
    ax.errorbar(table["rate"], y,
                xerr=[table["rate"] - table["ci_lo"], table["ci_hi"] - table["rate"]],
                fmt="o", color="steelblue", capsize=4)
    ax.axvline(table.attrs["overall_rate"], color="red", ls="--",
               label=f"overall {table.attrs['overall_rate']:.3f}")
    ax.set_yticks(y); ax.set_yticklabels(table.index)
    ax.set_xlabel("ever-churn rate"); ax.set_title(f"{seg}  (p = {table.attrs['p']:.2f})")
    ax.legend(fontsize=8)
plt.suptitle("Every segment interval covers the overall rate", y=1.02)
plt.tight_layout()
plt.savefig("../outputs/figures/11_segment_rates.png", bbox_inches="tight")
plt.show()

# %% [markdown]
# Five segments, and after BH correction the smallest adjusted p is 0.71. The
# widest raw gap is country (0.56 to 0.78) — on cells of 22 to 291 accounts, whose
# intervals all overlap.
#
# `referral_source` is the one worth a second look: `ads` at 0.60 against
# `partner` at 0.75, raw p = 0.14. It is also the only segment that will reappear
# in the survival analysis (notebook 12, log-rank p = 0.056), so it gets tracked
# rather than dismissed. It does not survive correction in either place.

# %% [markdown]
# ### What does a null dataset produce here?
#
# The chi-square answers "is this segment associated with churn". It does not
# answer the question a reader actually asks of the table, which is "what is the
# biggest gap on this page". Scanning five segments for the widest gap has its own
# null, and it is not zero.

# %%
null = reasons.max_segment_spread_null(tables, n_permutations=500)
for key, value in null.items():
    print(f"  {key:22s} {value}")

# %% [markdown]
# The widest gap in the real data is **21.6 points**, and the mean widest gap
# under shuffled labels is **22.4** — the observed value is slightly *below* what
# noise typically produces, and 47% of shuffled runs beat it outright.
#
# "Germany churns 22 points less than the UK" is not a finding. It is the
# expected output of eyeing five segments of this size, and it would have gone
# on a slide.

# %% [markdown]
# ## 5. The cohort retention triangle
#
# Standard product analytics, and the first thing here that shows a real
# pattern. Rows are signup cohorts, columns are months since signup, cells are
# the share still active. Cells a cohort has not lived long enough to fill stay
# blank rather than being imputed.

# %%
triangle = reasons.retention_triangle(tables, extract_date="2024-12-31")
print(triangle.iloc[:, :10].to_string())

# %%
fig, ax = plt.subplots(figsize=(11, 7))
sns.heatmap(triangle, annot=False, cmap="RdYlGn", vmin=0, vmax=1,
            cbar_kws={"label": "share still active"}, ax=ax)
ax.set_xlabel("months since signup"); ax.set_ylabel("signup cohort")
ax.set_title("Retention by cohort — the gradient runs down the rows, not across")
plt.tight_layout()
plt.savefig("../outputs/figures/11_retention_triangle.png", bbox_inches="tight")
plt.show()

# %% [markdown]
# Read **down** column 3 rather than across the rows. Month-3 retention:

# %%
m3 = triangle[3].dropna()
print(m3.to_string())
print(f"\n2023 cohorts: mean month-3 retention {m3[m3.index < '2024'].mean():.3f}")
print(f"2024 cohorts: mean month-3 retention {m3[m3.index >= '2024'].mean():.3f}")

# %% [markdown]
# **Month-3 retention fell by a third over two years** — 0.82 for 2023 cohorts
# against 0.54 for 2024 cohorts, at the *same* age, and the late-2024 rows are
# worse still. That is not a tenure effect and not a segment effect. Something
# is changing along the calendar axis.
#
# This is the first thing in the project that looks like a real answer to "why
# are users leaving", and it is the reason notebook 12 exists: separating a
# tenure effect from a cohort effect from a calendar-period effect is what
# survival analysis is for, and doing it by eye on this triangle is exactly the
# mistake that would follow.
#
# **It is also the reason notebook 16 exists, and that is where this pattern
# ends.** There is a fourth explanation none of the three can be separated from
# here: the extract stops on 2024-12-31, and a churn date drawn at random before
# that boundary crowds into the end of the file. Later cohorts have a shorter
# window for the draw to land in, so their retention falls at every age with
# nothing about them being different. Notebook 16 tests it and the pattern does
# not survive — so read the verdict below as "yes, along the calendar axis",
# not as "yes, because of something the business did".

# %%
summary = pd.DataFrame([
    {"question": "Is there a dominant churn reason?",
     "test": "chi-square vs uniform over 6 codes",
     "result": f"p = {dist.attrs['p_uniform']}", "verdict": "no"},
    {"question": "Do stated reasons match behaviour?",
     "test": "Kruskal-Wallis x 9 measures, BH-corrected",
     "result": f"min p_BH = {coherence['p_bh'].min():.2f}", "verdict": "no"},
    {"question": "Do the code and the free text agree?",
     "test": "chi-square independence",
     "result": f"V = {crosstab.attrs['cramers_v']}", "verdict": "no"},
    {"question": "Does churn concentrate in a segment?",
     "test": "5 segments, BH-corrected",
     "result": f"min p_BH = {scan['p_bh'].min():.2f}", "verdict": "no"},
    {"question": "Does churn concentrate in time?",
     "test": "cohort retention triangle",
     "result": f"month-3 retention {m3[m3.index < '2024'].mean():.3f} -> "
               f"{m3[m3.index >= '2024'].mean():.3f}",
     "verdict": "yes, but see notebook 16 - reproduced by a random-date null"},
])
summary.to_csv("../outputs/reports/churn_reasons.csv", index=False)
print(summary.to_string(index=False))

# %% [markdown]
# ## Takeaway
#
# Four of the five standard routes to "why are users leaving" return nothing on
# this data, and the negative results are worth as much as the positive one
# because each closes off an analysis someone would otherwise ship:
#
# - **the reason field is not usable** — uniform, behaviour-independent, and
#   contradicted by the customer's own free text
# - **no segment concentrates churn** — and the widest gap on the page is
#   smaller than shuffled labels routinely produce
# - **churn is concentrated in calendar time** — month-3 retention halved between
#   the 2023 and 2024 signup cohorts
#
# The one surviving pattern is a *when*, not a *who*. Notebook 12 establishes
# whether it is tenure, cohort or period, because the three imply completely
# different actions and the triangle alone cannot tell them apart. Notebook 16
# then asks the question none of the three can answer — whether a file that
# stops on a fixed date produces this pattern on its own — and the answer is
# yes. Read this notebook's positive finding as provisional until 16.
