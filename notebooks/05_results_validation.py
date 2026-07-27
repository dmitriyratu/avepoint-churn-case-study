# %% [markdown]
# # 05 — Results, Recommendations & Scalability
#
# Covers assignment Parts 4 and 5: strategic recommendations with a testing
# approach, mentorship, deployment architecture, and monitoring.
#
# Everything here is read against the numbers notebook 04 actually produced, and
# those numbers are weak: the ladder's best rung does not clear chance at the
# lower bound of its interval, and the nested estimate — the one that accounts
# for having *chosen* that rung — sits at chance. The recommendations are
# written to match that. A ranker this weak earns a call list at most, and the
# honest framing is that the data cannot answer the question yet.

# %%
import sys
sys.path.insert(0, "..")

import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import json
import warnings
warnings.filterwarnings("ignore")

from sklearn.metrics import recall_score, precision_score

from src import pipeline
from src.model import load_model, oof_threshold
from src.config import CUTOFF_DATE, TARGET

sns.set_theme(style="whitegrid", palette="muted")

data = pipeline.build()
df, X, y = data.frame, data.X, data.y
model = load_model("churn_model")          # persisted by notebook 04
config = json.load(open("../outputs/models/config.json"))

threshold, _, oof = oof_threshold(model, X, y)
pred = (oof >= threshold).astype(int)

print(f"model          {config['model']}")
print(f"cohort         {config['cohort_n']} accounts, {config['positives']} positives "
      f"({config['positives'] / config['cohort_n']:.1%})")
print(f"CV AUC         {config['cv_auc']:.3f}  CI {config['cv_auc_ci']}  "
      f"p={config['permutation_p']}")
print(f"nested CV AUC  {config['nested_cv_auc']:.3f}   <- the number to quote")
print(f"operating pt   t={threshold}: recall {recall_score(y, pred):.3f}, "
      f"precision {precision_score(y, pred, zero_division=0):.3f}")

# %% [markdown]
# ## What the model actually keys on
#
# The selected rung is linear, so coefficients on standardised inputs are
# directly readable — no SHAP needed, and a simpler explanation is a better one
# when it is available.
#
# The caveat matters more than the list. This is an **L2** model: it shrinks
# coefficients but never sets them to zero, so all of them survive and "made the
# list" means nothing on its own.

# %%
from src.model import feature_names
coef = pd.Series(model.named_steps["clf"].coef_[0], index=feature_names(model, X))
ranked = coef.reindex(coef.abs().sort_values(ascending=False).index)
print(f"{(coef != 0).sum()} non-zero of {len(coef)} encoded features "
      "(L2 shrinks, it does not select)\n")
print(ranked.head(10).round(4).to_string())

# %% [markdown]
# Reading it honestly:

# %%
top = ranked.head(6)
for name, value in top.items():
    direction = "higher risk" if value > 0 else "lower risk"
    print(f"  {name:34s} {value:+.3f}   {direction}")

print("\nSanity-check the ones with a clear domain expectation:")
expectations = {
    "num__unique_features_used": "broader adoption should mean LOWER risk",
    "num__days_since_signup": "longer tenure should mean LOWER risk",
    "num__pct_subs_ended": "more ended subscriptions should mean HIGHER risk",
    "num__n_upgrades": "more upgrades should mean LOWER risk",
}
for name, expected in expectations.items():
    if name in coef.index:
        print(f"  {name:30s} {coef[name]:+.3f}  ({expected})")

# %% [markdown]
# Two of the largest terms point the way domain knowledge says they should, and
# several do not. On a model whose interval spans chance, that is what you would
# expect either way — the coefficient ranking is not stable enough to interpret
# term by term.
#
# What *is* stable across every model in the project is tenure: `days_since_signup`
# and `tenure_days` are the strongest single features in the audit, and the
# tenure bands below are the one relationship that survives every cut. The rest
# of this list should be read as "the model is using everything a little," not as
# a driver analysis.

# %% [markdown]
# ## Where the model is confident, and whether it is right there
#
# A weak average AUC can still be useful if the top of the ranking is reliable —
# that is all a triage list needs.

# %%
rank = pd.DataFrame({"proba": oof, "actual": y.values}).sort_values("proba", ascending=False)
base = y.mean()
print(f"base rate: {base:.3f}\n")
print(" top-K   churn rate   lift")
for k in [10, 20, 30, 50, 80]:
    r = rank.head(k)["actual"].mean()
    print(f"  {k:>4}     {r:.3f}      {r/base:.2f}x")

# %%
deciles = pd.qcut(rank["proba"], 10, labels=False, duplicates="drop")
by_dec = rank.groupby(deciles)["actual"].mean().sort_index(ascending=False)
fig, ax = plt.subplots(figsize=(8, 4))
by_dec.plot(kind="bar", ax=ax, color="steelblue")
ax.axhline(base, color="red", ls="--", label=f"base rate {base:.2f}")
ax.set_xlabel("risk decile (0 = highest)"); ax.set_ylabel("actual churn rate")
ax.set_title("Does the ranking separate at the top?")
ax.legend(); plt.tight_layout()
plt.savefig("../outputs/figures/05_decile_lift.png", bbox_inches="tight")
plt.show()

# %% [markdown]
# ## Part 4 — Strategic recommendations
#
# Each is stated with the evidence behind it, the strength of that evidence, and
# how it would be tested. Where the evidence is weak I say so.

# %% [markdown]
# ### 1. Onboard the first 6 months harder  — *strongest evidence*
#
# Tenure is the one relationship that survives every cut in this project:
# `days_since_signup` is the top single feature in the leakage audit, it is the
# feature the error analysis says the model is really using, and the independent
# published analysis of this dataset reached the same conclusion without
# building a model at all.

# %%
from src.audit import single_feature_auc
sf = single_feature_auc(X, y).set_index("feature")["auc"]
print(f"days_since_signup single-feature AUC: {sf['days_since_signup']:.4f} "
      f"(rank 1 of {len(sf)})")

cohort = data.cohort
tmp = cohort.copy()
tmp["days_since_signup"] = (CUTOFF_DATE - tmp["signup_date"]).dt.days
tmp["tenure_band"] = pd.cut(tmp["days_since_signup"], [-1, 90, 180, 365, 10000],
                            labels=["<3mo", "3-6mo", "6-12mo", "12mo+"])
band = tmp.groupby("tenure_band", observed=True)[TARGET].agg(["size", "sum", "mean"]).round(3)
band.columns = ["n_accounts", "churners", "churn_rate"]
print()
print(band.to_string())

# %% [markdown]
# The gradient runs the right way overall — the 12-month-plus band churns at
# roughly a third the rate of the under-3-month band — but it is **not
# monotone**, and each band holds only 30–60 accounts. Directionally sound,
# individually noisy.
#
# **Action.** Structured onboarding through day 180: milestone checks at 30/60/90,
# with a CSM touch for Enterprise. Target the behaviour, not the tenure number.
#
# **Test.** Randomise new signups 50/50 into the enhanced track. Primary metric:
# retention at 180 days. The minimum detectable effect at this cohort size is
# large, so this needs several months of signups before it reads out — roughly
# 300+ accounts per arm for a 10pp difference at 80% power.

# %% [markdown]
# ### 2. Treat plan tier as a retention lever — *weak evidence, stated as such*
#
# The raw segment rates differ a little, so this is worth a look — but the
# differences are inside the noise at these group sizes, and I would present it
# as a hypothesis rather than a finding.

# %%
print(cohort.groupby("plan_tier")[TARGET].agg(["size", "sum", "mean"]).round(3).to_string())
print("\nGroup sizes are small — read as directional, not settled.")

# %% [markdown]
# **Action.** For Basic accounts showing Pro-level usage breadth, a guided upgrade
# offer. The causal claim ("upgrading causes retention") is *not* established
# here — the association could easily run the other way.
#
# **Test.** This one needs a genuine experiment precisely because the causality is
# ambiguous. Randomise the offer among eligible Basic accounts and measure
# retention over the modelling horizon plus net revenue, so a retention gain that
# costs more in discount than it returns is visible.

# %% [markdown]
# ### 3. Instrument disengagement properly — *infrastructure, not a finding*
#
# On real telemetry, engagement features are normally among the strongest churn
# predictors. Here they carry almost nothing — so the question is whether that is
# a fact about engagement or a fact about this synthetic usage log.

# %%
# Coefficient names are post-encoding ("num__usage_last_30d"); strip the
# transformer prefix before matching against raw column names. An earlier
# version compared the two directly, so this always reported zero.
bare = coef.rename(lambda c: c.split("__", 1)[-1])
engagement = [c for c in X.columns
              if any(k in c for k in ["usage", "feature", "error", "momentum", "accel"])]

share = bare[bare.index.isin(engagement)].abs().sum() / bare.abs().sum()
print(f"engagement features offered      : {len(engagement)} of {X.shape[1]}")
print(f"share of total |coefficient| mass: {share:.1%}")
print(f"strongest engagement term        : "
      f"{bare[bare.index.isin(engagement)].abs().idxmax()} "
      f"({bare[bare.index.isin(engagement)].abs().max():.3f})")
print(f"best engagement single-feature AUC: "
      f"{sf[sf.index.isin(engagement)].max():.4f}  (max over all features "
      f"{sf.max():.4f})")

# %% [markdown]
# They are used, but no engagement feature stands out — the strongest scores
# barely above the rest, on a scale where the best feature in the whole matrix is
# indistinguishable from noise (notebook 10). Given that 19,128 of 24,979 usage
# rows predate their own subscription, the most likely explanation is the
# telemetry, not the concept.
#
# **Action.** Before another modelling pass, fix the inputs: event-level product
# telemetry with reliable timestamps, session depth, and seat-level activation
# rather than account-level totals.
#
# **Test.** Not an A/B test — a data-quality milestone. Re-run this pipeline once
# telemetry is trustworthy and compare against the nested CV figure recorded
# here.

# %% [markdown]
# ## Part 5 — Deployment architecture

# %% [markdown]
# ```
#  ┌──────────────────────────────────────────────────────────────┐
#  │ Sources                                                      │
#  │  billing/CRM · product telemetry · support desk              │
#  └───────────────────────────┬──────────────────────────────────┘
#                              │ nightly batch
#  ┌───────────────────────────▼──────────────────────────────────┐
#  │ Feature pipeline  (the same code path as training)           │
#  │  - AS-OF semantics: every aggregate takes a cutoff argument   │
#  │  - censors fields that resolve after the cutoff               │
#  │  - the audit suite runs here and FAILS the job on violation   │
#  └───────────────────────────┬──────────────────────────────────┘
#                              │
#  ┌───────────────────────────▼──────────────────────────────────┐
#  │ Scoring  (batch; daily is ample for a 90-day horizon)        │
#  │  - churn_model.joblib                                         │
#  │  - writes account_id, score, decile, top contributing terms   │
#  └───────────────────────────┬──────────────────────────────────┘
#                              │
#  ┌───────────────────────────▼──────────────────────────────────┐
#  │ Consumption                                                  │
#  │  - CSM queue ordered by score (NOT auto-triggered actions)     │
#  │  - scores written back to CRM for context                     │
#  └──────────────────────────────────────────────────────────────┘
# ```
#
# The load here is trivial — 500 accounts, a 90-day horizon. Real-time serving
# would be over-engineering; a nightly batch job is the right answer and saying
# so is part of the design.
#
# To be explicit: this is the architecture I would build *if the model were worth
# deploying*. On these results it is not, and the recommendation below says so.
# The design is included because the question was asked, not because the number
# earned it.
#
# The point worth defending: **training and serving share one feature code path,
# parameterised by cutoff date.** Training passes 2024-06-30, production passes
# today. That is the structural defence against training/serving skew, and it is
# why `build_model_dataset` takes `as_of` rather than assuming "now".

# %% [markdown]
# ## Monitoring
#
# | Layer | Check | Cadence | Trigger |
# |---|---|---|---|
# | Input | Feature PSI vs training distribution | weekly | PSI > 0.2 → investigate |
# | Input | Null-rate and row-count deltas | daily | job fails on schema change |
# | Input | **Leakage suite** (`src/audit.py`) | every run | any violation → fail the job |
# | Output | Score distribution drift | weekly | mean shift > 2 sd |
# | Outcome | AUC on matured cohorts | quarterly | drop > 0.05 → retrain |
# | Outcome | Calibration on matured cohorts | quarterly | — |
#
# Labels take a full horizon to mature, so performance monitoring is inherently
# lagged. Input drift is the early warning; outcome metrics confirm it later.
#
# Running the leakage suite in production is the unusual entry and the one I would
# argue for hardest: the censoring bug in this project was a *pipeline* bug, and
# pipeline bugs recur whenever someone adds a feature.

# %% [markdown]
# ## Mentoring a junior engineer on this project
#
# I would hand over the leakage work, because it is where the transferable
# judgement lives:
#
# 1. **Start with the label, not the model.** The first version of this project
#    modelled an undated flag that disagreed with the event log for 62% of
#    accounts. No algorithm recovers from that. "Check your n's" found it.
# 2. **Ask of every column: would I have this at prediction time?** Then write
#    the check down so it runs automatically. Reasoning caught the obvious leak;
#    the automated gate caught the one reasoning missed.
# 3. **A high AUC is a hypothesis, not a result.** Their first strong number
#    should prompt a leakage hunt, not a commit.
# 4. **Establish the floor before celebrating.** `DummyClassifier` first, always.
# 5. **Report the interval, not the point.** A CI spanning chance communicates
#    something the point estimate does not.
# 6. **Choosing a model is itself a fitting step.** The gap between this
#    project's ladder maximum and its nested estimate is the whole apparent
#    signal. That is the lesson I would spend the most time on.
#
# I would have them re-run the pipeline with `POST_OUTCOME_COLS` emptied and
# watch the AUC jump (notebook 06 measures it) — that lesson lands far better as
# an experiment than a lecture.

# %% [markdown]
# ## Summary

# %%
print(f"""
  cohort            {config['cohort_n']} accounts, {config['positives']} positives \
({config['positives'] / config['cohort_n']:.0%})
  model             {config['model']}
  CV ROC-AUC        {config['cv_auc']:.3f}   95% CI {config['cv_auc_ci']}
  nested CV AUC     {config['nested_cv_auc']:.3f}   <- the honest figure
  permutation p     {config['permutation_p']}   (holds the model fixed; it was chosen)
  operating point   t={config['oof_threshold']} -> recall {config['oof_recall']:.3f}, \
precision {config['oof_precision']:.3f}
  encoded terms     {config['n_features_selected']} (L2 shrinks, does not select) \
from {X.shape[1]} raw columns

  Verdict: not deployable. The ladder maximum does not clear chance at the lower
  bound of its interval, and once model selection is cross-validated the estimate
  sits at chance. The binding constraint is data, not algorithm — \
{config['positives']} positives,
  a label whose three definitions agree on 20% of accounts, and telemetry whose
  timestamps do not order events correctly.

  What I would say to the business: do not rank a CSM call list on this. Fix the
  label definition and the usage timestamps first, then re-run — the pipeline is
  parameterised so that is a one-command re-measurement.
""")
