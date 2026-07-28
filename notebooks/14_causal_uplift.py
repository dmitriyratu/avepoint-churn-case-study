# %% [markdown]
# # 14 — What actions improve retention? (causal inference and uplift)
#
# The third product question is the only *causal* one. "Why are users leaving"
# and "can we predict churn" are questions about association; "what actions
# improve retention" asks what happens if we intervene, and association does not
# answer it at any sample size or with any model.
#
# The blocking fact, stated before any method:
#
# > **This dataset records no intervention.** No account was offered a discount,
# > called by a CSM, or enrolled in an onboarding programme — or if any were, the
# > extract does not say so. There is no treatment variable, so the effect of a
# > retention action is *not identified* here under any estimator.
#
# Doubly-robust estimation is robust to getting one of two nuisance models wrong.
# It is not robust to the treatment never having happened. That is worth being
# blunt about, because the alternative — running the causal machinery on
# whatever binary column is to hand and reporting the output as an effect — is
# both easy and common.
#
# What *is* available is a set of **observational proxies**: things accounts did
# that a product team might want to encourage or prevent. Estimating those under
# an explicit, diagnosed, stress-tested unconfoundedness assumption is the
# legitimate version of this exercise, and it is what follows.

# %%
import sys
sys.path.insert(0, "..")

import warnings
warnings.filterwarnings("ignore")

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

from src import causal, pipeline

sns.set_theme(style="whitegrid", palette="muted")
pd.set_option("display.width", 140)

data = pipeline.build()
X, y = data.X, data.y
print(f"{len(y)} accounts, {int(y.sum())} churned in the 90-day window "
      f"({y.mean():.1%})")

# %% [markdown]
# ## 1. The proxy treatments
#
# Four actions, each with a real product decision behind it.

# %%
for name, spec in causal.TREATMENTS.items():
    treatment, confounders = causal.make_treatment(X, name)
    print(f"{name:11s} {int(treatment.sum()):3d} treated ({treatment.mean():5.1%})  "
          f"| {spec['question']}")
    print(f"{'':11s} action: {spec['action']}")

# %% [markdown]
# **Confounders are pre-specified**, not selected on this data:

# %%
print(causal.CONFOUNDERS)

# %% [markdown]
# Ten covariates, each included because it plausibly drives both the action and
# churn — contract size and price, tenure, product usage, support load.
#
# The temptation is to hand all 73 features to the propensity model on the
# grounds that more adjustment is safer. It is not. The first version of this
# module did exactly that and produced an ATE of **−0.53 on a probability
# scale** — a negative churn rate — because a propensity model with 71 covariates
# on 177 rows separates the arms almost perfectly, propensity scores pile up at 0
# and 1, and the inverse weights reach 100. That failure is preserved in the
# module docstring because it is the characteristic way this method breaks.

# %% [markdown]
# ## 2. What the dashboard would say
#
# The unadjusted comparison, which is what a BI tool reports and what gets
# quoted in a QBR.

# %%
naive = []
for name in causal.TREATMENTS:
    treatment, _ = causal.make_treatment(X, name)
    rate_t, rate_c = y[treatment == 1].mean(), y[treatment == 0].mean()
    naive.append({"treatment": name, "churn_treated": round(rate_t, 4),
                  "churn_control": round(rate_c, 4),
                  "raw_difference": round(rate_t - rate_c, 4)})
naive = pd.DataFrame(naive)
print(naive.to_string(index=False))

# %% [markdown]
# "Accounts that upgrade churn 13 points less — so drive upgrades." That is the
# slide this data produces if you stop here, and the rest of the notebook is
# about why you cannot.

# %% [markdown]
# ## 3. Overlap and balance
#
# Before any effect estimate: is there a comparison to make at all? A treated
# account with propensity 0.99 has no control counterpart, and an estimate for it
# is the outcome model extrapolating rather than any kind of comparison.

# %%
treatment, confounders = causal.make_treatment(X, "upgrade")
propensity = causal.cross_fit_propensity(confounders, treatment)
overlap = causal.overlap_report(propensity, treatment)
for key, value in overlap.items():
    print(f"  {key:20s} {value}")

# %%
fig, ax = plt.subplots(figsize=(9, 4))
for arm, colour, label in [(1, "crimson", "upgraded"), (0, "steelblue", "did not")]:
    ax.hist(propensity[treatment == arm], bins=20, alpha=0.6, color=colour, label=label)
ax.axvline(causal.TRIM, color="grey", ls="--")
ax.axvline(1 - causal.TRIM, color="grey", ls="--", label=f"trim at {causal.TRIM}")
ax.set_xlabel("propensity score  P(upgrade | confounders)"); ax.set_ylabel("accounts")
ax.set_title("Common support — the two arms have to overlap")
ax.legend(); plt.tight_layout()
plt.savefig("../outputs/figures/14_overlap.png", bbox_inches="tight")
plt.show()

# %% [markdown]
# Overlap is adequate after regularisation. Now: does weighting actually remove
# the differences between the arms? Standardised mean differences before and
# after inverse-probability weighting, |SMD| < 0.1 being the usual bar.

# %%
weights = (treatment / propensity.clip(0.05, 0.95)
           + (1 - treatment) / (1 - propensity).clip(0.05, 0.95))
before = causal.standardised_differences(confounders, treatment)
after = causal.standardised_differences(confounders, treatment, weights=weights)

balance = before.merge(after, on="covariate", suffixes=("_raw", "_weighted"))
print(balance[["covariate", "smd_raw", "smd_weighted"]].head(10).to_string(index=False))
print(f"\ncovariates with |SMD| > 0.1 - before: "
      f"{(before['abs_smd'] > 0.1).sum()}, after: {(after['abs_smd'] > 0.1).sum()}")

# %%
fig, ax = plt.subplots(figsize=(8, 6))
order = balance.sort_values("abs_smd_raw")
y_pos = np.arange(len(order))
ax.scatter(order["abs_smd_raw"], y_pos, color="crimson", label="unadjusted")
ax.scatter(order["abs_smd_weighted"], y_pos, color="steelblue", label="IPW-weighted")
ax.axvline(0.1, color="grey", ls="--", label="|SMD| = 0.1")
ax.set_yticks(y_pos); ax.set_yticklabels(order["covariate"], fontsize=8)
ax.set_xlabel("|standardised mean difference|")
ax.set_title("Love plot — upgraders vs non-upgraders")
ax.legend(); plt.tight_layout()
plt.savefig("../outputs/figures/14_love_plot.png", bbox_inches="tight")
plt.show()

# %% [markdown]
# Unadjusted, the arms are enormously different — upgraders have far more
# subscriptions (SMD 0.87), more usage (0.81) and longer tenure (0.73). That is
# the confounding the naive 13-point difference is made of: bigger, older,
# more-engaged accounts both upgrade *and* stay, and neither causes the other.
#
# Weighting removes most of it — 13 covariates above |SMD| 0.1 before, 5 after —
# which is the necessary condition for proceeding and nowhere near a sufficient
# one. It balances the covariates we measured, and says nothing about the rest.

# %% [markdown]
# ## 4. Doubly-robust estimates
#
# AIPW with cross-fitted nuisance models: consistent if *either* the propensity
# model or the outcome model is right. Bootstrap intervals.
#
# **Sign convention:** positive ATE means the treatment *increases* churn. For a
# retention action that is the wrong direction, and the intuitive reading of a
# positive number is backwards.

# %%
results = []
for name in causal.TREATMENTS:
    treatment, confounders = causal.make_treatment(X, name)
    estimate = causal.aipw_ate(confounders, treatment, y)
    simple = causal.ipw_ate(confounders, treatment, y)
    results.append({"treatment": name, "n_treated": estimate["n_treated"],
                    "naive": estimate["naive_diff"], "ipw": simple["ate"],
                    "aipw": estimate["ate"], "ci_lo": estimate["ci_lo"],
                    "ci_hi": estimate["ci_hi"], "p": estimate["p_two_sided"],
                    "risk_ratio": estimate["risk_ratio"],
                    "trimmed": estimate["n_trimmed"]})
results = pd.DataFrame(results)
print(results.to_string(index=False))

# %%
fig, ax = plt.subplots(figsize=(9, 4))
y_pos = np.arange(len(results))
ax.errorbar(results["aipw"], y_pos,
            xerr=[results["aipw"] - results["ci_lo"], results["ci_hi"] - results["aipw"]],
            fmt="o", color="steelblue", capsize=5, label="AIPW (adjusted)")
ax.scatter(results["naive"], y_pos, color="crimson", marker="x", s=80,
           label="naive difference")
ax.axvline(0, color="black", lw=1)
ax.set_yticks(y_pos); ax.set_yticklabels(results["treatment"])
ax.set_xlabel("effect on 90-day churn probability  (negative = retention gain)")
ax.set_title("Every interval crosses zero")
ax.legend(); plt.tight_layout()
plt.savefig("../outputs/figures/14_ate_forest.png", bbox_inches="tight")
plt.show()

# %% [markdown]
# **Every interval crosses zero.** `upgrade` is the closest to an effect —
# adjusted ATE −0.110, p = 0.14 — and adjustment has already eaten part of the
# naive −0.126, exactly as the balance table predicted.
#
# IPW and AIPW agree closely on every row. That is reassuring about
# specification and it is *not* reassuring about the answer: agreement means
# neither model is rescuing the other, so the estimate rests entirely on the
# unconfoundedness assumption.

# %% [markdown]
# ## 5. How much unmeasured confounding would erase this?
#
# The E-value (VanderWeele & Ding, 2017) converts the untestable assumption into
# a number a business can argue with: the risk ratio an unmeasured confounder
# would need with **both** the treatment and the outcome to explain the whole
# association away.

# %%
evalues = []
for _, row in results.iterrows():
    ev = causal.e_value(row["risk_ratio"])
    evalues.append({"treatment": row["treatment"], "risk_ratio": row["risk_ratio"],
                    "e_value": ev["e_value_point"]})
evalues = pd.DataFrame(evalues)
print(evalues.to_string(index=False))

# %% [markdown]
# The `upgrade` E-value is **2.2**. An unmeasured confounder associated with both
# upgrading and not-churning by a risk ratio of 2.2 would account for the entire
# estimate.
#
# Is such a confounder plausible? *"The account was doing well"* — a champion
# user, a successful internal rollout, budget approved for next year — is
# associated with upgrading by far more than 2.2, and with retention by far more
# than 2.2. It is not measured anywhere in this schema and could not be. So the
# answer is yes, comfortably, and the estimate cannot support a decision.
#
# The other three sit between 1.3 and 1.9, which is weaker still.

# %% [markdown]
# ## 6. The noise floor of this design
#
# The most useful diagnostic here, and the cheapest. Assign a *completely random*
# treatment with the same marginal rate, run the entire pipeline — propensity,
# cross-fitting, AIPW — and see what comes out. The true effect is exactly zero
# by construction, so the spread is this study's noise floor.

# %%
treatment, confounders = causal.make_treatment(X, "upgrade")
placebo = causal.placebo_ate(confounders, y, float(treatment.mean()), n_placebo=30)
for key, value in placebo.items():
    print(f"  {key:20s} {value}")

# %%
observed = results.set_index("treatment")["aipw"]
fig, ax = plt.subplots(figsize=(9, 4))
ax.axvspan(placebo["p2_5"], placebo["p97_5"], alpha=0.25, color="grey",
           label="placebo 95% range (true effect = 0)")
for i, (name, value) in enumerate(observed.items()):
    ax.scatter(value, i, s=90, color="crimson", zorder=3)
    ax.text(value, i + 0.15, name, ha="center", fontsize=9)
ax.axvline(0, color="black", lw=1)
ax.set_yticks([]); ax.set_ylim(-0.6, len(observed) - 0.2)
ax.set_xlabel("estimated ATE on 90-day churn")
ax.set_title("Every observed estimate sits inside the placebo band")
ax.legend(loc="lower right"); plt.tight_layout()
plt.savefig("../outputs/figures/14_placebo.png", bbox_inches="tight")
plt.show()

# %% [markdown]
# **A randomly-assigned placebo treatment produces effects up to ±0.15 on this
# sample.** Every one of the four observed estimates — including `upgrade` at
# −0.110 — is inside that band.
#
# This is a stronger statement than "p > 0.05". It says the design cannot detect
# an effect smaller than about 15 percentage points of churn probability, and a
# 15-point true effect from a single product action would be extraordinary. **The
# study is underpowered for any effect worth acting on**, and that is a fact
# about the sample size rather than about the actions.

# %% [markdown]
# ## 7. Uplift modelling
#
# The question underneath "what action improves retention" is usually narrower:
# *which accounts* should get the action. Uplift (heterogeneous treatment effect)
# models estimate a per-account effect and rank by it, so a CSM team can work
# down the list.
#
# T-learner — one outcome model per arm — scored out-of-fold, because an
# in-sample uplift score is fitted to the very rows the Qini curve then
# evaluates, and that alone manufactures a convincing curve.

# %%
treatment, confounders = causal.make_treatment(X, "upgrade")
uplift = causal.t_learner_uplift(confounders, treatment, y)
print(uplift.describe().round(4).to_string())
print(f"\naccounts predicted to benefit (uplift < 0): {(uplift < 0).sum()} of {len(uplift)}")

# %%
curve = causal.qini_curve(uplift, treatment, y)
qini = causal.qini_with_null(confounders, treatment, y, n_null=20)

fig, ax = plt.subplots(figsize=(8, 5))
ax.plot(curve["fraction"], curve["qini"], color="steelblue", lw=2, label="uplift model")
ax.plot(curve["fraction"], curve["random"], color="grey", ls="--", label="random targeting")
ax.set_xlabel("share of accounts targeted (ranked by predicted uplift)")
ax.set_ylabel("incremental churn events")
ax.set_title(f"Qini curve — coefficient {qini['observed_qini']}, "
             f"null p = {qini['p_value']}")
ax.legend(); plt.tight_layout()
plt.savefig("../outputs/figures/14_qini.png", bbox_inches="tight")
plt.show()

for key, value in qini.items():
    print(f"  {key:16s} {value}")

# %% [markdown]
# The curve sits above the diagonal, which is what a working uplift model looks
# like — and the null says **p = 0.30**, with a null spread (sd 1.97) four times
# the observed coefficient. Uplift models fitted to shuffled outcomes routinely
# produce better curves than this one.
#
# There is a second problem that the null does not even test, and it is the
# larger one. Uplift modelling assumes the treatment was randomly assigned within
# levels of the covariates. Here it was not assigned at all — accounts chose to
# upgrade. So the "uplift" being ranked is a mixture of a treatment effect and
# the selection into treatment, and no amount of validation on this data
# separates them. **A Qini curve computed on observational data is not evidence
# that targeting works.**

# %% [markdown]
# ## 8. What would actually answer the question
#
# Three things this notebook establishes, in increasing order of usefulness:

# %%
summary = pd.DataFrame([
    {"finding": "No intervention is recorded in the data",
     "consequence": "the causal question is not identified at all",
     "fix": "log every CSM touch, discount, and campaign with a timestamp"},
    {"finding": f"Every AIPW interval crosses zero (min p = {results['p'].min()})",
     "consequence": "no proxy action shows a detectable effect",
     "fix": "n/a - see below"},
    {"finding": "E-values 1.3-2.2",
     "consequence": "a mild unmeasured confounder erases every estimate",
     "fix": "randomise, which removes confounding by construction"},
    {"finding": f"Placebo band +/-{placebo['detectable_effect']}",
     "consequence": f"cannot detect effects below "
                    f"{placebo['detectable_effect']:.0%} of churn probability",
     "fix": "power the experiment properly; notebook 15 sizes it"},
    {"finding": f"Qini p = {qini['p_value']}, and treatment was self-selected",
     "consequence": "uplift targeting is not supported",
     "fix": "estimate uplift from a randomised pilot, not from history"},
])
summary.to_csv("../outputs/reports/causal_estimates.csv", index=False)
results.to_csv("../outputs/reports/causal_ate.csv", index=False)
print(summary.to_string(index=False))

# %% [markdown]
# ## Takeaway
#
# The honest answer to "what actions can improve retention" from this dataset is
# **that it cannot be answered from this dataset** — and unlike the other two
# questions, this is not a sample-size problem that more rows would fix. It is
# structural: no action was ever recorded, so there is nothing whose effect could
# be estimated.
#
# What the notebook does deliver is the shape of the answer, which is worth more
# than a spurious number:
#
# - the naive comparison overstates the `upgrade` association by about 15%, and
#   what remains is still confounded by account size, tenure and engagement
# - after adjustment, nothing is distinguishable from zero
# - an E-value of 2.2 means "the account was doing well" — unmeasurable, and
#   obviously associated with both sides — is enough to explain the largest
#   estimate away entirely
# - the design's noise floor is ±15pp, so it could not have detected a realistic
#   effect even if one existed
#
# The route from here is not a better estimator. It is a randomised pilot, and
# the reason this analysis is still worth doing is that it produces the design
# inputs for one: which action, which population, what baseline rate, and how
# small an effect matters. Notebook 15 turns those into a sample size and a
# decision rule.
