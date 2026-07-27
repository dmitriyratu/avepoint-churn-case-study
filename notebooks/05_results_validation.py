# %% [markdown]
# # 05 — Results, Recommendations & Scalability
#
# Covers assignment Parts 4 and 5: strategic recommendations with a testing
# approach, mentorship, deployment architecture, and monitoring.
#
# Everything here is read against a model with CV AUC 0.611 and a confidence
# interval that touches 0.44. The recommendations are written to match that
# strength — a weak ranker earns a call list, not an automated intervention.

# %%
import sys
sys.path.append("..")

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import json
import warnings
warnings.filterwarnings("ignore")

from sklearn.metrics import roc_auc_score, recall_score, precision_score

from src.load_data import load_all
from src.clean import clean_all
from src.labeling import build_cohort, truncate_tables
from src.model import prep_xy, load_model, oof_threshold, model_ladder
from src.config import CUTOFF_DATE, TARGET

sns.set_theme(style="whitegrid", palette="muted")

df = pd.read_csv("../data/processed/features_temporal.csv")
X, y = prep_xy(df)
model = load_model("churn_l1_logistic")
config = json.load(open("../outputs/models/config.json"))

_, best_est = model_ladder()[4]
threshold, _, oof = oof_threshold(best_est, X, y)
pred = (oof >= threshold).astype(int)

print(f"CV AUC {config['cv_auc']:.3f}  CI {config['cv_auc_ci']}  p={config['permutation_p']}")
print(f"out-of-fold @ t={threshold}: recall {recall_score(y,pred):.3f}, "
      f"precision {precision_score(y,pred,zero_division=0):.3f}")

# %% [markdown]
# ## What the model actually keys on
#
# With a linear model on standardised inputs the coefficients are directly
# readable — no SHAP needed to explain seven terms, and a simpler explanation is
# a better one when it is available.

# %%
from src.model import feature_names
coef = pd.Series(model.named_steps["clf"].coef_[0], index=feature_names(model, X))
nz = coef[coef != 0].sort_values(key=abs, ascending=False)
print(nz.round(4).to_string())

# %% [markdown]
# Reading the signs:
#
# - **`days_since_signup` (negative, strongest)** — longer-tenured accounts churn
#   less. Standard survivorship: the risky period is early.
# - **`latest_plan_tier_Pro` (negative)** — Pro accounts are stickier than the
#   Basic baseline.
# - **`n_trial_subs` (negative)** — more trial subscriptions in history
#   associates with *lower* churn here, which is counter-intuitive and worth
#   flagging rather than explaining away. Likely an artefact of the synthetic
#   generator; on real data I would want this checked before acting on it.
# - **`avg_usage_count` (positive)** — mildly counter-intuitive too.
#
# Two of seven coefficients point the wrong way relative to domain expectation.
# On a model this weak that is what you would expect from noise, and it is a
# reason to treat the ranking as triage rather than explanation.

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
# `days_since_signup` is the largest coefficient and the strongest single feature
# (AUC 0.650). Early-tenure accounts carry materially more risk.

# %%
tables = clean_all(load_all())
cohort = build_cohort(tables)
obs = truncate_tables(tables, CUTOFF_DATE)
tmp = cohort.copy()
tmp["days_since_signup"] = (CUTOFF_DATE - tmp["signup_date"]).dt.days
tmp["tenure_band"] = pd.cut(tmp["days_since_signup"], [0, 180, 365, 550, 10000],
                            labels=["<6mo", "6-12mo", "12-18mo", "18mo+"])
band = tmp.groupby("tenure_band")[TARGET].agg(["size", "mean"]).round(3)
band.columns = ["n_accounts", "churn_rate"]
print(band.to_string())

# %% [markdown]
# **Action.** Structured onboarding through day 180: milestone checks at 30/60/90,
# with a CSM touch for Enterprise. Target the behaviour, not the tenure number.
#
# **Test.** Randomise new signups 50/50 into the enhanced track. Primary metric:
# 180-day retention. Minimum detectable effect at this cohort size is large, so
# this needs to run across several months of signups before it reads out —
# roughly 300+ accounts per arm for a 10pp difference at 80% power.

# %% [markdown]
# ### 2. Treat plan tier as a retention lever — *moderate evidence*
#
# Pro carries a negative coefficient relative to Basic.

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
# 180-day retention plus net revenue, so a retention gain that costs more in
# discount than it returns is visible.

# %% [markdown]
# ### 3. Instrument disengagement properly — *infrastructure, not a finding*
#
# The honest observation is that the engagement features barely contributed:
# usage recency and breadth were largely dropped by the L1 penalty. On real
# telemetry these are normally among the strongest churn predictors, so the most
# likely explanation is that this synthetic usage data carries little signal
# rather than that engagement does not matter.

# %%
eng = [c for c in X.columns if any(k in c for k in
       ["usage", "feature", "error", "recency", "momentum"])]
kept = [c for c in eng if c in nz.index]
print(f"engagement features offered : {len(eng)}")
print(f"engagement features retained: {len(kept)}  {kept}")

# %% [markdown]
# **Action.** Before another modelling pass, fix the inputs: event-level product
# telemetry with reliable timestamps, session depth, and seat-level activation
# rather than account-level totals. The integrity problems found in EDA (19,128
# usage rows predating their subscription) would have to be resolved first.
#
# **Test.** Not an A/B test — a data-quality milestone. Re-run this pipeline once
# telemetry is trustworthy and compare CV AUC against the 0.611 baseline recorded
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
#  │ Scoring  (batch; daily is ample for a 180-day horizon)       │
#  │  - churn_l1_logistic.joblib                                   │
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
# The load here is trivial — 500 accounts, a 180-day horizon. Real-time serving
# would be over-engineering; a nightly batch job is the right answer and saying
# so is part of the design.
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
# Labels take 180 days to mature, so performance monitoring is inherently
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
# 5. **Report the interval, not the point.** [0.44, 0.74] communicates something
#    "0.611" does not.
#
# I would have them re-run the pipeline with `POST_OUTCOME_COLS` emptied and watch
# AUC hit 0.997 — that lesson lands far better as an experiment than a lecture.

# %% [markdown]
# ## Summary

# %%
print(f"""
  cohort            187 accounts, 88 positives (47%)
  model             {config['model']}
  CV ROC-AUC        {config['cv_auc']:.3f}   95% CI {config['cv_auc_ci']}
  permutation p     {config['permutation_p']}
  operating point   t={config['oof_threshold']} -> recall {config['oof_recall']:.3f}, precision {config['oof_precision']:.3f}
  features used     {config['n_features_selected']} of {X.shape[1]}

  Verdict: a weak but statistically real ranker. Deploy as a CSM triage list;
  do not attach automated actions or revenue decisions to it. The binding
  constraint is data, not algorithm — 88 positives and synthetic telemetry.
""")
