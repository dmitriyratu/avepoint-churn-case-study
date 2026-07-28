# %% [markdown]
# # 15 — Retention economics and experiment design
#
# Notebooks 11-14 answer the product team's three questions and two of the
# answers are "this data cannot tell you". That is honest and it is not
# actionable: the team still has to decide what to fund.
#
# This notebook supplies the part that does not depend on having a working model
# — what a retained customer is worth, what an intervention must achieve to pay
# for itself, and how big an experiment has to be to tell. None of it needs the
# AUC to be good. It needs the arithmetic to be right.
#
# The order matters and is deliberately the reverse of the usual one. Most churn
# projects build the model, then ask what it is worth. Doing the economics first
# would have told this project something important before a single model was
# fitted, and section 2 is that result.

# %%
import sys
sys.path.insert(0, "..")

import warnings
warnings.filterwarnings("ignore")

import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from lifelines import KaplanMeierFitter

from src import economics, pipeline, survival
from src.clean import clean_all
from src.load_data import load_all
from src.model import load_model, oof_threshold

sns.set_theme(style="whitegrid", palette="muted")
pd.set_option("display.width", 140)

tables = clean_all(load_all())
data = pipeline.build()
X, y = data.X, data.y
base_rate = float(y.mean())

print(f"cohort     {len(y)} accounts, base churn rate {base_rate:.1%}")
print(f"median MRR ${tables['subscriptions']['mrr_amount'].median():,.0f}")

# %% [markdown]
# ## 1. What is a retained customer worth?
#
# The textbook shortcut is `ARPU / churn_rate`, which assumes a constant hazard
# and an infinite horizon. Notebook 12 *measured* the hazard, so there is no need
# to assume it — and the constant-hazard assumption turned out to hold within a
# cohort but fail badly across the base, where the rate is rising 2.8x a year.
#
# Integrating the actual Kaplan-Meier curve instead:

# %%
survival_frame = survival.survival_frame(tables)
km = KaplanMeierFitter().fit(survival_frame["duration"], survival_frame["event"])
curve = km.survival_function_.iloc[:, 0]

median_mrr = float(tables["subscriptions"]["mrr_amount"].median())
clv = economics.clv_from_survival(curve, median_mrr)
for key, value in clv.items():
    print(f"  {key:24s} {value}")

# %% [markdown]
# **CLV ≈ $7,300** at the median MRR, over a three-year horizon at 80% gross
# margin and a 10% discount rate. The expected-days-retained figure (325 of a
# possible 1,095) is where the churn rate does its damage: the average customer
# is present for under a third of the window.
#
# Every number below scales with the assumptions above. They are declared in
# `economics.py` rather than buried in a cell precisely so they can be replaced
# with finance's real figures in one place.

# %% [markdown]
# ## 2. The result that should have come first
#
# An intervention costing `C` that saves a customer worth `V` with effectiveness
# `e` pays off on a targeted account only when that account's true churn
# probability exceeds:
#
# ```
# p_churn > C / (e * V)
# ```
#
# Below that, the outreach costs more than the expected save. Above it, contact
# them. This is a single division and it does not need a model.

# %%
required = economics.break_even_precision(
    cost=economics.DEFAULT_INTERVENTION_COST, value=clv["clv"],
    effectiveness=economics.DEFAULT_EFFECTIVENESS)

print(f"campaign cost per account   ${economics.DEFAULT_INTERVENTION_COST:,.0f}")
print(f"customer value (CLV)        ${clv['clv']:,.0f}")
print(f"assumed effectiveness       {economics.DEFAULT_EFFECTIVENESS:.0%}")
print(f"\nbreak-even churn probability  {required:.1%}")
print(f"cohort base rate              {base_rate:.1%}")
print(f"\n-> {'TARGETING NEEDED' if base_rate < required else 'TREAT EVERYONE'}")

# %% [markdown]
# **The base rate is three times the break-even threshold.** At these economics,
# contacting *every* account in the at-risk cohort is already profitable, and a
# ranking cannot improve on a policy that is correct for everyone. The model is
# not merely weak here — it is answering a question nobody needed answered.
#
# This is worth dwelling on as a process point. Notebooks 03-10 spent the bulk of
# this project's effort establishing that the model does not work. The division
# above takes one line and would have shown, *before any of it*, that a working
# model was not the binding constraint on the decision. Cost-benefit framing
# belongs at the start of a churn project, not in the deployment section at the
# end.
#
# The conclusion is specific to these economics, so the next section maps where
# it flips.

# %% [markdown]
# ## 3. When does targeting actually matter?
#
# Break-even precision across a grid of customer values and campaign costs.

# %%
grid = economics.required_precision_table(
    values=[1000, 3000, clv["clv"], 15000],
    costs=[50, 150, 500, 2000],
    base_rate=base_rate)
pivot = grid.pivot_table(index="customer_value", columns="campaign_cost",
                         values="required_precision")
print(pivot.round(3).to_string())
print(f"\n(cells below the {base_rate:.1%} base rate -> treat everyone; "
      f"cells above 1.0 -> no account can justify the cost)")

# %%
fig, ax = plt.subplots(figsize=(9, 5))
display = pivot.clip(upper=1.0)
sns.heatmap(display, annot=pivot.round(2), fmt="", cmap="RdYlGn_r", vmin=0, vmax=1,
            cbar_kws={"label": "required churn probability"}, ax=ax)
ax.set_title(f"Break-even precision — base rate is {base_rate:.1%}\n"
             f"green = treat everyone, red = targeting cannot pay")
ax.set_xlabel("campaign cost per account ($)"); ax.set_ylabel("customer value ($)")
plt.tight_layout()
plt.savefig("../outputs/figures/15_breakeven_grid.png", bbox_inches="tight")
plt.show()

# %% [markdown]
# Targeting only earns its keep in the top-right region — expensive
# interventions on moderate-value customers. A $2,000 white-glove save programme
# against a $3,000 customer needs a churn probability above 100%, so it can never
# pay whatever the model says. A $50 automated email against a $7,300 customer
# needs 3.4%, which every account in this cohort clears.
#
# **The band where a churn model changes the decision is narrow**, and finding
# out whether you are inside it costs one division.

# %% [markdown]
# ## 4. Would the model have added value anyway?
#
# Suppose the economics were less favourable and targeting did matter. The model
# still has to beat treating everyone. Out-of-fold scores, realised outcomes, net
# value at every threshold.

# %%
model = load_model("churn_model")
threshold, _, oof_scores = oof_threshold(model, X, y)

values = economics.value_curve(y.values, oof_scores, value=clv["clv"])
print(f"treat none        ${values.attrs['treat_none_value']:>12,.0f}")
print(f"treat all         ${values.attrs['treat_all_value']:>12,.0f}")
print(f"best threshold    ${values.attrs['best_value']:>12,.0f}  "
      f"(at t = {values.attrs['best_threshold']:.3f})")
print(f"\ngain from using the model: "
      f"${values.attrs['best_value'] - values.attrs['treat_all_value']:,.0f}")

# %%
fig, ax = plt.subplots(figsize=(9, 5))
ax.plot(values["threshold"], values["net_value"], color="steelblue", lw=2,
        label="target above threshold")
ax.axhline(values.attrs["treat_all_value"], color="crimson", ls="--",
           label=f"treat everyone (${values.attrs['treat_all_value']:,.0f})")
ax.axhline(0, color="grey", ls=":", label="treat nobody")
ax.set_xlabel("score threshold"); ax.set_ylabel("net value on this cohort ($)")
ax.set_title("Net campaign value by targeting threshold")
ax.legend(); plt.tight_layout()
plt.savefig("../outputs/figures/15_value_curve.png", bbox_inches="tight")
plt.show()

# %% [markdown]
# The best threshold barely separates from treat-everyone, and the curve is
# nearly flat across the whole range — which is what a chance-level ranking looks
# like once it is priced. The apparent gain is also optimistic: the threshold was
# chosen by maximising value on the same out-of-fold predictions it is evaluated
# on, the same +0.049 F1 selection effect notebook 04 measures.

# %% [markdown]
# ### Decision curve analysis
#
# The threshold-free version, standard in clinical prediction and under-used in
# churn. Net benefit expresses false positives in true-positive units using the
# decision-maker's own exchange rate, so the model, treat-all and treat-none can
# be compared on one axis without committing to a dollar figure.

# %%
dca = economics.decision_curve(y.values, oof_scores)
fig, ax = plt.subplots(figsize=(9, 5))
ax.plot(dca["threshold_prob"], dca["net_benefit_model"], color="steelblue", lw=2,
        label="churn model")
ax.plot(dca["threshold_prob"], dca["net_benefit_treat_all"], color="crimson",
        ls="--", label="treat everyone")
ax.axhline(0, color="grey", ls=":", label="treat nobody")
ax.set_ylim(-0.3, 0.35)
ax.set_xlabel("threshold probability (decision-maker's indifference point)")
ax.set_ylabel("net benefit")
ax.set_title("Decision curve — does acting on the model beat acting on everyone?")
ax.legend(); plt.tight_layout()
plt.savefig("../outputs/figures/15_decision_curve.png", bbox_inches="tight")
plt.show()

# %%
crossover = dca[dca["net_benefit_model"] > dca["net_benefit_treat_all"]]
print(f"threshold probabilities where the model beats treat-all: "
      f"{len(crossover)} of {len(dca)}")
if len(crossover):
    print(f"  range {crossover['threshold_prob'].min():.2f} - "
          f"{crossover['threshold_prob'].max():.2f}")
print(f"\nthe relevant threshold here is the break-even probability, "
      f"{required:.1%},")
print(f"where treat-all net benefit = "
      f"{dca.iloc[(dca['threshold_prob'] - required).abs().idxmin()]['net_benefit_treat_all']:.3f} "
      f"and the model = "
      f"{dca.iloc[(dca['threshold_prob'] - required).abs().idxmin()]['net_benefit_model']:.3f}")

# %% [markdown]
# At the threshold probability that this business's economics actually imply
# (10.3%), treat-everyone dominates. The model only competes at threshold
# probabilities far above the break-even point — i.e. in a world where
# intervention is much more expensive relative to customer value than it is here.

# %% [markdown]
# ## 5. Designing the experiment that would answer Q3
#
# Notebook 14's conclusion was that the causal question needs a randomised
# pilot. This is its specification.
#
# Baseline is the observed 90-day churn rate. Signup rate comes from the data.

# %%
signups_per_month = len(tables["accounts"]) / 24
print(f"signups per month (500 accounts / 24 months): {signups_per_month:.1f}")

plan = economics.experiment_plan(
    baseline_rate=base_rate,
    effects=[0.15, 0.10, 0.07, 0.05, 0.03],
    signups_per_month=signups_per_month)
print()
print(plan.to_string(index=False))

# %%
fig, ax = plt.subplots(figsize=(9, 5))
ax.plot(plan["absolute_effect"] * 100, plan["months_to_readout"], "o-",
        color="steelblue")
ax.axhline(12, color="green", ls="--", label="1 year")
ax.axhline(36, color="orange", ls="--", label="3 years")
ax.set_yscale("log")
ax.set_xlabel("absolute reduction in 90-day churn (percentage points)")
ax.set_ylabel("months from launch to readout (log scale)")
ax.set_title("What it costs in calendar time to detect an effect")
ax.legend(); plt.tight_layout()
plt.savefig("../outputs/figures/15_experiment_power.png", bbox_inches="tight")
plt.show()

# %% [markdown]
# **Only a very large effect is testable at this company's scale.** A 15pp
# absolute reduction — cutting churn by half — reads out in about 15 months. A
# 5pp reduction, which would be an excellent result for a real retention
# programme, needs 2,527 accounts and **over ten years** of signups.
#
# That is the single most useful number in this notebook for planning purposes,
# and it is not a statement about the analysis. It is a statement about running
# an experiment on 500 customers.

# %% [markdown]
# ### Cross-check against notebook 14
#
# Two independent routes to the same quantity. Notebook 14 ran the full AIPW
# estimator on randomly-assigned placebo treatments and found a noise floor of
# ±15pp. The power calculation below asks the same question analytically.

# %%
per_arm = len(y) // 2
mde = economics.minimum_detectable_effect(base_rate, per_arm)
print(f"splitting this cohort 50/50 gives {per_arm} accounts per arm")
print(f"analytic MDE at 80% power:        {mde:.1%}")
print("empirical placebo band (nb 14):   ~15%")

# %% [markdown]
# **17.2% against ~15%.** A closed-form power calculation and a 30-run simulation
# through a doubly-robust estimator agree to within a couple of points, which is
# reassuring about both. Neither is a borderline finding: any effect this study
# could have detected would have been implausibly large.

# %% [markdown]
# ## 6. What to actually do
#
# Ordered by expected value, with the evidence behind each and what would
# change the recommendation.

# %%
actions = pd.DataFrame([
    {"rank": 1,
     "action": "Investigate the calendar-time churn increase",
     "evidence": "hazard x2.8/yr on a flat at-risk base, p = 2e-16 (nb 12)",
     "strength": "strong",
     "why": "the only large effect in the data; dwarfs anything account-level"},
    {"rank": 2,
     "action": "Fix the churn label and usage timestamps",
     "evidence": "3 definitions agree on 20% of accounts; 19,128/24,979 usage "
                 "rows predate their subscription",
     "strength": "strong",
     "why": "blocks every re-measurement; cheap relative to its leverage"},
    {"rank": 3,
     "action": "Contact the whole at-risk cohort; do not rank it",
     "evidence": f"base rate {base_rate:.0%} vs break-even {required:.0%}",
     "strength": "strong",
     "why": "profitable without a model, and the model is at chance"},
    {"rank": 4,
     "action": "Instrument interventions (CSM touch, discount, campaign)",
     "evidence": "no treatment variable exists anywhere in the schema (nb 14)",
     "strength": "structural",
     "why": "without it Q3 stays unanswerable at any sample size"},
    {"rank": 5,
     "action": "Run a randomised pilot on the one testable effect size",
     "evidence": f"15pp detectable in ~{plan.iloc[0]['months_to_readout']:.0f} "
                 f"months; 5pp needs 10+ years",
     "strength": "moderate",
     "why": "only worth launching for an intervention expected to halve churn"},
    {"rank": 6,
     "action": "Do NOT build a structured onboarding programme",
     "evidence": "within-cohort hazard is flat, rho ~ 1 (nb 12)",
     "strength": "strong",
     "why": "the tenure effect that motivated it is a composition artefact"},
])
actions.to_csv("../outputs/reports/retention_actions.csv", index=False)
print(actions.to_string(index=False))

# %%
economics_summary = pd.DataFrame([
    {"quantity": "CLV (median MRR, 3yr, discounted)", "value": f"${clv['clv']:,.0f}"},
    {"quantity": "Expected days retained of 1095", "value": clv["expected_days_retained"]},
    {"quantity": "Break-even churn probability", "value": f"{required:.1%}"},
    {"quantity": "Cohort base rate", "value": f"{base_rate:.1%}"},
    {"quantity": "Treat-all net value on cohort",
     "value": f"${values.attrs['treat_all_value']:,.0f}"},
    {"quantity": "Model best-threshold net value",
     "value": f"${values.attrs['best_value']:,.0f}"},
    {"quantity": "MDE, 88 per arm", "value": f"{mde:.1%}"},
    {"quantity": "Months to detect 5pp", "value": plan.set_index('absolute_effect')
     .loc[0.05, 'months_to_readout']},
])
economics_summary.to_csv("../outputs/reports/retention_economics.csv", index=False)
print(economics_summary.to_string(index=False))

# %% [markdown]
# ## Takeaway
#
# 1. **A retained customer is worth ~$7,300**, integrated from the measured
#    survival curve rather than assumed from a constant hazard.
# 2. **Break-even churn probability is 10.3%; the base rate is 30.5%.** Targeting
#    everyone is already profitable, so the ranking problem this project spent
#    most of its effort on was not the binding constraint on the decision. One
#    division, run first, would have said so.
# 3. **The model adds nothing on top of treat-all** by net value or by decision
#    curve, at any threshold this business's economics imply.
# 4. **Only a churn-halving effect is testable here.** 15pp reads out in ~15
#    months; 5pp needs a decade of signups. Two independent methods put the
#    detectable effect at 15-17pp.
# 5. **The highest-value action is not a model at all** — it is finding out what
#    changed between 2023 and 2024, because that effect is an order of magnitude
#    larger than anything account-level in this data.
#
# The general lesson is the ordering. Cost-benefit framing is usually the last
# section of a churn project, after the modelling is done. Here it would have
# redirected the whole effort on day one, and that is not specific to this
# dataset.
