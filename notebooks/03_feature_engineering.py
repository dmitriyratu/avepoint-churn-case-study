# %% [markdown]
# # 03 — Feature Engineering
#
# Builds the account-level feature matrix as of the prediction cutoff.
#
# Two rules govern everything here:
#
# 1. **Nothing dated at or after the cutoff may reach a feature.** Tables are
#    truncated first (`labeling.truncate_tables`), and fields that *resolve*
#    after the cutoff are censored even when the row itself predates it.
# 2. **Nothing derived from `churn_events` becomes a feature.** Those columns
#    describe the outcome. See `docs/DATA_DICTIONARY.md`.
# 3. **A 30-day buffer separates the last observable day from the first day a
#    churn can count.** Without it a model keys on the collapse in activity that
#    happens days before someone leaves — accurate and useless. See the
#    sensitivity analysis at the end of this notebook.
#
# Feature families follow the standard churn taxonomy (`docs/FEATURE_ENGINEERING.md`):
#
# | Family | Purpose |
# |---|---|
# | **Recency** | how long since the last meaningful action |
# | **Frequency** | activity counts over a 30/60/90/180-day ladder |
# | **Monetary** | MRR level, growth, volatility |
# | **Acceleration** | short window vs long window — the direction of travel |
# | **Trend** | fitted slope over weekly activity |
# | **Regularity** | gaps between active days — rhythm, not just volume |
# | **Support** | ticket load and its trend |
# | **Account** | industry, country, referral, plan tier, seats |
#
# The level tells you how big an account is. The differences between windows tell
# you where it is heading, which is what a churn model needs.

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
from src.clean import clean_all
from src.labeling import build_cohort, truncate_tables, cohort_summary
from src.features import (build_model_dataset, subscription_features,
                          usage_features, support_features)
from src.model import prep_xy
from src.config import CUTOFF_DATE

sns.set_theme(style="whitegrid", palette="muted")

tables = clean_all(load_all())
cohort = build_cohort(tables)
obs = truncate_tables(tables, CUTOFF_DATE)

print(cohort_summary(cohort).to_string())

# %% [markdown]
# ## How much data survives truncation
#
# Roughly two thirds of the event rows are dated after the cutoff and are
# correctly unavailable. Seeing this number is a useful sanity check — if it were
# near zero, the truncation would not be doing anything.

# %%
for k in ["subscriptions", "feature_usage", "support_tickets", "churn_events"]:
    print(f"  {k:16s} {len(obs[k]):>6} / {len(tables[k]):>6}  "
          f"({len(obs[k])/len(tables[k]):.0%} retained)")

# %% [markdown]
# ## Block 1 — Subscription features
#
# Level tells you how big an account is; **direction** tells you where it is
# going. `seat_growth`, `mrr_growth_pct` and `upgrade_net` capture the second.

# %%
sub_feats = subscription_features(obs["subscriptions"], CUTOFF_DATE)
print(sub_feats.shape)
sub_feats.describe().T.round(2)

# %% [markdown]
# `tenure_days` is measured signup-to-cutoff. An earlier version measured it to
# `end_date.max()`, which silently stops the clock at whichever subscription
# closed first — wrong for the 62% of accounts holding both open and closed
# subscriptions.

# %% [markdown]
# ## Block 2 — Engagement features
#
# Recency and momentum matter more than lifetime totals: an account that used
# the product heavily last year and nothing this quarter looks healthy on
# volume alone.

# %%
usage_feats = usage_features(obs["feature_usage"], obs["subscriptions"], CUTOFF_DATE)
print(usage_feats.shape)
usage_feats[["total_usage_events", "unique_features_used", "days_since_last_usage",
             "usage_last_30d", "usage_last_90d", "usage_momentum", "error_rate"]].describe().T.round(2)

# %% [markdown]
# The windowed columns are anchored to the cutoff. In the first version they were
# anchored to a hardcoded date seven months past the end of the data, which made
# `usage_last_30d` and `usage_last_90d` identically zero for every account.

# %%
print("non-zero windowed activity (a zero-variance column would mean a bug):")
for c in ["usage_last_30d", "usage_last_90d", "usage_last_180d"]:
    if c in usage_feats.columns:
        print(f"  {c:18s} nunique={usage_feats[c].nunique():>4}  mean={usage_feats[c].mean():.1f}")

# %% [markdown]
# ## Block 3 — Support features
#
# `resolution_time_hours` and `satisfaction_score` are censored for tickets still
# open at the cutoff, so these aggregates skip them rather than counting a
# resolution that has not happened. `n_open_tickets` replaces that lost signal
# with something genuinely observable.

# %%
support_feats = support_features(obs["support_tickets"], CUTOFF_DATE)
print(support_feats.shape)
support_feats.describe().T.round(2)

# %% [markdown]
# ## Assemble

# %%
df = build_model_dataset(obs, cohort, CUTOFF_DATE)
X, y = prep_xy(df)
print(f"feature matrix : {X.shape}  (categoricals still raw; encoded in-fold)")
from src.model import categorical_columns
print(f"categorical cols: {categorical_columns(X)}")
print(f"positives      : {int(y.sum())} ({y.mean():.1%})")
print(f"pruned as collinear: {df.attrs.get('dropped_collinear')}")
print(f"events per variable: {y.sum()/X.shape[1]:.2f}   (want >= 10)")

# %% [markdown]
# 1.16 events per variable is severely under-powered. This is the number that
# predicts the modelling result in notebook 04: with 62 raw columns (75 after
# in-fold encoding) and 88 positives, regularisation matters more than capacity.

# %% [markdown]
# ## Missing values: three meanings, three treatments
#
# The first version filled everything with 0, which conflates them.

# %%
na = X.isna().sum()
na = na[na > 0]
print("columns left as NaN for in-fold imputation:")
print(na.to_string() if len(na) else "  (none)")
print(f"\ntotal NaNs retained: {int(X.isna().sum().sum())}")
print("\ncounts -> 0 (no activity is genuinely zero)")
print("recency -> observation-window length (never used != used today)")
print("rates   -> NaN, imputed inside the CV fold by model._pipe")

# %% [markdown]
# ## Association with the target
#
# Reported here for orientation only — feature *selection* is done by the L1
# penalty inside cross-validation, not by picking winners off this list.

# %%
from src.audit import encode_for_audit
Xe = encode_for_audit(X)   # diagnostic view only; the model encodes in-fold
corr = Xe.corrwith(y.astype(float)).sort_values(key=abs, ascending=False)
print(corr.head(15).round(4).to_string())
print(f"\nmax |r| = {corr.abs().max():.4f}")

# %%
top = corr.abs().head(15).index
fig, ax = plt.subplots(figsize=(8, 6))
vals = corr[top]
vals.plot(kind="barh", ax=ax, color=["salmon" if v > 0 else "steelblue" for v in vals])
ax.axvline(0, color="black", lw=.8)
ax.set_title("Top 15 features by |correlation| with 180-day churn")
ax.set_xlabel("Pearson r")
plt.tight_layout()
plt.savefig("../outputs/figures/03_feature_correlations.png", bbox_inches="tight")
plt.show()

# %% [markdown]
# ## Leakage gate before anything is modelled

# %%
import src.audit as audit
res, passed = audit.run_all(X, y, df, obs, CUTOFF_DATE)
print(f"max single-feature AUC : {res['single_feature_auc']['auc'].max():.4f}")
print(f"temporal provenance    : {'PASS' if res['temporal_provenance']['pass'].all() else 'FAIL'}")
print(f"\nSUITE: {'PASS' if passed else 'FAIL'}")
assert passed, "leakage audit failed — do not proceed to modelling"

# %%
df.to_csv("../data/processed/features_temporal.csv", index=False)
print(f"saved features_temporal.csv  {df.shape}")

# %% [markdown]
# ## The buffer, and why it decides everything
#
# Standard practice puts a gap between the last observable day and the first day
# a churn counts. The reason: without it, a model learns the collapse in activity
# that happens immediately before someone leaves. That scores well and arrives
# too late to act on.
#
# This sweep is the most important result in the project.

# %%
sens = pd.read_csv("../outputs/reports/buffer_sensitivity.csv")
print(sens.to_string(index=False))

# %%
fig, ax1 = plt.subplots(figsize=(8, 4.5))
ax1.plot(sens["buffer_days"], sens["cv_auc"], marker="o", color="#264653", label="CV ROC-AUC")
ax1.fill_between(sens["buffer_days"], sens["ci_lo"], sens["ci_hi"], alpha=.15, color="#264653")
ax1.axhline(0.5, ls="--", c="r", alpha=.6, label="chance")
ax1.set_xlabel("buffer (days of lead time required)")
ax1.set_ylabel("CV ROC-AUC")
ax2 = ax1.twinx()
ax2.plot(sens["buffer_days"], sens["perm_p"], marker="s", ls=":", color="#e76f51", label="permutation p")
ax2.axhline(0.05, ls=":", c="#e76f51", alpha=.5)
ax2.set_ylabel("permutation p-value")
ax1.set_title("Signal disappears once the model must be actionable")
ax1.legend(loc="upper right")
plt.tight_layout()
plt.savefig("../outputs/figures/03_buffer_sensitivity.png", bbox_inches="tight")
plt.show()

# %% [markdown]
# **Read this carefully.** At buffer = 0 the model beats chance (p = 0.025). At
# 15 days it does not (p = 0.25), and it never recovers.
#
# So the earlier 0.618 was not a weak-but-real churn model. It was a detector for
# customers who had effectively already left. On this dataset there is **no
# actionable churn signal at a realistic intervention horizon**.
#
# The project defaults to a 30-day buffer because that is the question the
# business actually has, and the honest answer to it is "not from this data."

# %% [markdown]
# ## Did the richer feature families help?
#
# The window ladder, acceleration ratios, trend slope, gap statistics and MRR
# volatility added roughly twenty features.

# %% [markdown]
# | Configuration | CV ROC-AUC |
# |---|---:|
# | 30-day buffer, original features | 0.551 |
# | 30-day buffer, enriched features | **0.548** |
#
# They bought nothing — slightly negative, within noise. With 74 positives, extra
# columns cost more in variance than they return, and L1 shrinks to three terms
# either way.
#
# That is worth stating plainly rather than quietly keeping the bigger set: the
# binding constraint here is **data, not feature engineering**. 168 accounts, 74
# positives, usage logs where 19,128 of 24,979 rows predate their own
# subscription, and a `churn_flag` that disagrees with the event log for 62% of
# accounts. No feature work fixes any of that.
