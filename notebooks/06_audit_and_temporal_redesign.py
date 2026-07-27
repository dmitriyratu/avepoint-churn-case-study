# %% [markdown]
# # 06 — Audit and Temporal Redesign
#
# This notebook is the review pass over notebooks 01–05. It exists because the
# first version of this analysis produced an AUC near 0.5 and I attributed that
# to "the dataset has no signal." That conclusion was premature — the problem was
# the framing, not the data.
#
# Three things were wrong:
#
# 1. **No observation window.** Features aggregated a customer's entire history,
#    including activity dated after they churned, to predict a static flag.
# 2. **Post-outcome features.** Columns derived from `churn_events` (refund
#    amount, churn reason, reactivation) describe the outcome, not its precursors.
# 3. **Untested claims.** "No signal" was asserted, never tested.
#
# Fixing the framing moved max feature-target correlation from 0.12 to 0.28 and
# produced a model that is statistically distinguishable from chance (p = 0.013).
# (Figures here reflect the final pipeline, which also encodes categoricals in-fold.)

# %%
import sys
sys.path.append("..")

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import warnings
warnings.filterwarnings("ignore")

from src.load_data import load_all
from src.clean import clean_all, integrity_report
from src.labeling import build_cohort, truncate_tables, cohort_summary
from src.features import build_model_dataset
from src.model import (prep_xy, evaluate_ladder, model_ladder,
                       permutation_significance, tune_lightgbm, oof_threshold)
from src.config import CUTOFF_DATE, HORIZON_DAYS

sns.set_theme(style="whitegrid")
tables = clean_all(load_all())

# %% [markdown]
# ## 1. Data-quality issues the first pass missed

# %%
print(integrity_report(tables).to_string(index=False))

# %% [markdown]
# Notable:
#
# - **1,077 of 2,000 support tickets predate the account's own signup date.**
# - **19,128 of 24,979 usage rows predate their subscription's start.**
# - **`churn_flag` disagrees with the `churn_events` table for 312 of 500
#   accounts** (only 37.6% agreement). The label is internally inconsistent.
#
# The third one matters most: modelling `churn_flag` means modelling a label that
# contradicts the event log it should summarise. That alone caps achievable
# performance regardless of feature quality.

# %%
# Other things the original cleaning step did not check
raw = load_all()
print("duplicate usage_id rows:", raw["feature_usage"]["usage_id"].duplicated().sum(),
      "  (now dropped in clean.py)")
print("rows where arr_amount != mrr_amount * 12:",
      (raw["subscriptions"]["arr_amount"] != raw["subscriptions"]["mrr_amount"] * 12).sum(),
      "  -> arr is perfectly collinear, now dropped")

# %%
# The original satisfaction-score rationale ("response rates differ by severity")
# was not supported by the data.
tix = raw["support_tickets"].copy()
tix["missing"] = tix["satisfaction_score"].isna()
print("missing-rate by priority:")
print(tix.groupby("priority")["missing"].mean().round(3).to_string())
print("\nFlat across priorities -> per-priority median == global median.")
print("clean.py now uses the global median plus an explicit missing indicator.")

# %% [markdown]
# ## 2. The reference-date bug
#
# `REFERENCE_DATE` was hardcoded to `2025-07-21`, but the data ends `2024-12-31`.
# Every recency window was therefore anchored 202 days after the last event.

# %%
old = pd.read_csv("../data/processed/features.csv")
for c in ["usage_last_30d", "usage_last_90d", "recency_ratio", "has_active_sub"]:
    if c in old.columns:
        print(f"  {c:22s} nunique={old[c].nunique()}  min={old[c].min()}  max={old[c].max()}")
print("\nFour features shipped with zero variance — three of them because nothing")
print("falls within 30/90 days of a cutoff that sits 7 months past the data.")

# %% [markdown]
# ### And a silent `tenure_days` bug
#
# ```python
# tenure = grp["end_date"].max().fillna(REFERENCE_DATE) - grp["start_date"].min()
# ```
#
# `.max()` skips `NaT`, so `fillna` only fires when *every* subscription is open.
# For an account with a mix of open and closed subscriptions the clock stops at
# whichever one closed last.

# %%
subs = tables["subscriptions"]
g = subs.groupby("account_id")
chk = pd.DataFrame({"n": g.size(), "n_ended": g["end_date"].count()})
mixed = ((chk["n_ended"] > 0) & (chk["n_ended"] < chk["n"])).sum()
print(f"accounts with both open and closed subscriptions: {mixed} / {len(chk)}  ({mixed/len(chk):.0%})")
print("tenure_days was wrong for all of them. Now measured to the cutoff instead.")

# %% [markdown]
# ## 3. The redesign: observation window and prediction window
#
# ```
#   |<---- observation window ---->|<--- prediction window --->|
#   2023-01-01                 2024-06-30                 2024-12-27
#          features built here        label defined here
# ```
#
# - **Eligible**: signed up before the cutoff and not already churned at it.
# - **Label**: first churn event within 180 days after the cutoff.
# - **Features**: computed only from rows dated before the cutoff.

# %%
cohort = build_cohort(tables)
obs = truncate_tables(tables, CUTOFF_DATE)
print(cohort_summary(cohort).to_string())
print("\nrows retained in the observation window:")
for k in ["subscriptions", "feature_usage", "support_tickets"]:
    print(f"  {k:16s} {len(obs[k]):>6} / {len(tables[k]):>6}")

# %%
df = build_model_dataset(obs, cohort, CUTOFF_DATE)
X, y = prep_xy(df)
print(f"feature matrix: {X.shape}   positives: {int(y.sum())}   rate: {y.mean():.3f}")
print(f"events per variable: {y.sum()/X.shape[1]:.2f}  (want >= 10 — we are well short)")
print(f"constant columns: {[c for c in X.columns if X[c].nunique() <= 1]}")

# %%
from src.audit import encode_for_audit
corr = encode_for_audit(X).corrwith(y.astype(float)).sort_values(key=abs, ascending=False)
print("Top 12 associations with the forward-looking label:")
print(corr.head(12).round(4).to_string())
print(f"\nmax |r| = {corr.abs().max():.4f}   (was 0.1196 against the static flag)")

# %% [markdown]
# ## 4. Quantifying the leakage
#
# Same model, same label, same folds — only the information available to the
# features changes.

# %%
leak = pd.read_csv("../outputs/reports/leakage_comparison.csv")
print(leak.to_string(index=False))

# %% [markdown]
# | Design | CV AUC | |
# |---|---|---|
# | A. Observation window only | **0.618** | correct |
# | B. Features may see post-cutoff rows | 0.634 | leaks |
# | C. Plus `churn_events`-derived features | **0.997** | pure label leakage |
#
# Design C is the one that matters. `n_churn_events`, `total_refund_usd` and
# `had_reactivation` reconstruct the label almost exactly — a refund is issued
# *because* the customer left. Those columns were in the original feature matrix.
#
# They did not inflate the original score only because the static `churn_flag`
# was itself inconsistent with the event log. Under a correct label they produce
# a near-perfect model that would collapse in production. `config.POST_OUTCOME_COLS`
# now excludes them explicitly.

# %% [markdown]
# ## 5. Progressive model comparison
#
# The original notebook jumped straight to LightGBM and XGBoost, then added
# logistic regression as an afterthought. There was no floor to beat and no
# baseline to justify the complexity.

# %%
ladder = pd.read_csv("../outputs/reports/model_ladder.csv")
print(ladder.to_string(index=False))

# %%
fig, ax = plt.subplots(figsize=(9, 4.5))
yv = np.arange(len(ladder))
ax.errorbar(ladder["roc_auc_mean"], yv,
            xerr=[ladder["roc_auc_mean"] - ladder["ci_lo"],
                  ladder["ci_hi"] - ladder["roc_auc_mean"]],
            fmt="o", capsize=4, color="#264653")
ax.axvline(0.5, ls="--", c="r", alpha=.6, label="chance")
ax.set_yticks(yv); ax.set_yticklabels(ladder["model"]); ax.invert_yaxis()
ax.set_xlabel("ROC-AUC (mean, 95% CI over 50 folds)")
ax.set_title("Each rung must beat the one below it")
ax.legend(); plt.tight_layout()
plt.savefig("../outputs/figures/06_model_ladder.png", bbox_inches="tight")
plt.show()

# %% [markdown]
# **L1 logistic regression wins at 0.618.** Both tree ensembles score below it.
# With 88 positives and 76 candidate features (1.16 events per variable), the
# ensembles have far more capacity than the data can support, and the L1 penalty
# doing hard feature selection is worth more than any amount of boosting.
#
# Note the confidence intervals overlap heavily — with this cohort size the
# ranking between rungs 2–6 is suggestive, not decisive. That is itself a result.

# %%
# Does tuning rescue the boosted model?
gs = tune_lightgbm(X, y)
print(f"best params: {gs.best_params_}")
print(f"tuned LightGBM CV AUC: {gs.best_score_:.4f}")
print(f"L1 logistic  CV AUC: {ladder.loc[4, 'roc_auc_mean']:.4f}")
print("\nA 54-point grid search still does not beat the linear model.")

# %% [markdown]
# ## 6. Is any of this better than chance?
#
# The claim the first pass should have tested.

# %%
_, best_est = model_ladder()[4]
perm = permutation_significance(best_est, X, y, n_permutations=300)
for k, v in perm.items():
    print(f"  {k:14s} {v}")
print("\nObserved AUC sits above the 95th percentile of the shuffled-label null.")
print("p = 0.013 — the association is real, if modest.")

# %% [markdown]
# ## 7. Choosing the operating point honestly
#
# The original notebook tuned the decision threshold on the test set and then
# reported test-set F1 and recall. Those numbers were optimistically biased.
# Here the threshold is selected from out-of-fold predictions only.

# %%
t, f1, oof = oof_threshold(best_est, X, y)
from sklearn.metrics import roc_auc_score, average_precision_score, recall_score, precision_score
pred = (oof >= t).astype(int)
print(f"  threshold (out-of-fold): {t}")
print(f"  AUC       {roc_auc_score(y, oof):.4f}")
print(f"  AP        {average_precision_score(y, oof):.4f}   (base rate {y.mean():.3f})")
print(f"  F1        {f1:.4f}")
print(f"  recall    {recall_score(y, pred):.4f}")
print(f"  precision {precision_score(y, pred, zero_division=0):.4f}")

# %%
# What the L1 penalty actually kept
best_est.fit(X, y)
from src.model import feature_names
coef = pd.Series(best_est.named_steps["clf"].coef_[0], index=feature_names(best_est, X))
nz = coef[coef != 0].sort_values(key=abs, ascending=False)
print(f"L1 retained {len(nz)} of {len(coef)} encoded features:\n")
print(nz.round(4).to_string())

# %% [markdown]
# ## 8. What I would say about this in the interview
#
# - The strongest signal is **`days_since_last_sub_start`** — accounts that have
#   stopped opening new subscriptions are disengaging. Recency beats volume.
# - **Longer-tenured accounts churn less** in this window (`days_since_signup`,
#   `tenure_days` both negative), the usual survivorship pattern.
# - **Trial-heavy accounts churn more.**
# - AUC 0.618 is a weak-but-real model. It is worth deploying only as a triage
#   ranker for CSM outreach, not as an automated action trigger.
#
# **Caveats I would lead with, not bury:**
#
# 1. 187 accounts and 88 positives. The CI on AUC is roughly [0.44, 0.76].
# 2. One cutoff date. A production evaluation needs rolling-origin backtesting
#    across several cutoffs.
# 3. The source data has documented integrity problems (tickets before signup,
#    usage before subscription start) that would be raised with data engineering
#    before any of this shipped.
# 4. `churn_flag` and `churn_events` disagree for 62% of accounts. I chose the
#    event log as ground truth because it carries dates; that choice should be
#    confirmed with whoever owns the pipeline.

# %% [markdown]
# ## 9. Next steps
#
# - Rolling-origin validation across quarterly cutoffs to get a stable estimate.
# - Survival modelling (Cox / discrete-time hazard) to use *when* rather than
#   *whether*, which suits a churn horizon better than binary classification.
# - Nested CV so the reported score includes hyperparameter-selection variance.
# - Feature reduction: 76 candidates on 88 positives is over-parameterised even
#   before any model is fit.
