# %% [markdown]
# # 01 — Exploratory Data Analysis
#
# Structured against `docs/EDA_CHECKLIST.md`.
#
# The notebook is deliberately split into two passes:
#
# - **Part A — data quality**, on all rows. Shapes, dtypes, missingness,
#   duplicates, outliers, referential integrity. None of this touches the target,
#   so using every row is safe.
# - **Part B — target relationships**, on the **exploration split only**. Churn
#   rates by segment, feature-target associations, anything that could steer a
#   modelling decision.
#
# The split matters. Every feature idea that comes out of Part B is chosen with
# knowledge of the labels it was derived from. If that is the whole dataset, the
# eventual score is optimistic in a way no cross-validation can undo. The
# first version of this notebook computed all of Part B on all 500 accounts.

# %%
import sys
sys.path.append("..")

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import warnings
warnings.filterwarnings("ignore")

from sklearn.model_selection import train_test_split

from src.load_data import load_all, TABLES
from src.clean import clean_all, integrity_report
from src.labeling import build_cohort, truncate_tables, cohort_summary
from src.config import CUTOFF_DATE, HORIZON_DAYS

sns.set_theme(style="whitegrid", palette="muted")
pd.set_option("display.max_columns", 40)

raw = load_all()
tables = clean_all(raw)

# %% [markdown]
# ## Checklist §1 — Formulate the question first
#
# "Why do users churn" is not answerable as stated: it has no unit of analysis
# and no time reference. Sharpened to
#
# > Given what is known about an account on **2024-06-30**, will it churn within
# > the next **90 days**?
#
# That immediately forces an observation window, a prediction window, and an
# eligibility rule — the structure the first pass was missing.

# %%
print(f"cutoff  : {CUTOFF_DATE.date()}")
print(f"horizon : {HORIZON_DAYS} days")
print(cohort_summary(build_cohort(tables)).to_string())

# %% [markdown]
# # Part A — Data quality (safe on all rows)

# %% [markdown]
# ## §2 — Check the packaging
#
# Do the shapes match what the dataset README claims?

# %%
expected = {"accounts": 500, "subscriptions": 5000, "feature_usage": 25000,
            "support_tickets": 2000, "churn_events": 600}
pack = pd.DataFrame([
    {"table": k, "file": TABLES[k], "rows": len(raw[k]), "cols": raw[k].shape[1],
     "expected_rows": expected[k], "matches": len(raw[k]) == expected[k]}
    for k in raw
])
print(pack.to_string(index=False))

# %% [markdown]
# ## §3 — Structure, dtypes, and the top *and* bottom
#
# Peng stresses looking at both ends: sorted data hides its problems in the tail.

# %%
for name in ["accounts", "subscriptions"]:
    print(f"--- {name} ---")
    print(raw[name].dtypes.to_string())
    print()

# %%
subs_sorted = raw["subscriptions"].sort_values("start_date")
print("EARLIEST subscriptions:")
print(subs_sorted.head(3).to_string(index=False))
print("\nLATEST subscriptions:")
print(subs_sorted.tail(3).to_string(index=False))

# %% [markdown]
# Two things visible from the dtypes alone: every date arrived as a string, and
# `arr_amount` looks like a deterministic function of `mrr_amount`.

# %%
print("rows where arr_amount != mrr_amount * 12:",
      int((raw["subscriptions"]["arr_amount"] != raw["subscriptions"]["mrr_amount"] * 12).sum()),
      f"of {len(raw['subscriptions'])}")
print("-> perfectly collinear, dropped in clean.py")

# %% [markdown]
# ## §4 — Check your "n"s
#
# Count the same concept three different ways and see whether the answers agree.
# This is the cheapest bug detector in the checklist, and it found the most
# important problem in the project.

# %%
acc = tables["accounts"]
ce = tables["churn_events"]

n_accounts = len(acc)
n_flagged = int(acc["churn_flag"].sum())
n_with_events = int(acc["account_id"].isin(ce["account_id"]).sum())
n_events = len(ce)

print(f"accounts total                       : {n_accounts}")
print(f"accounts with churn_flag = True      : {n_flagged}")
print(f"accounts appearing in churn_events   : {n_with_events}")
print(f"churn_event rows                     : {n_events}")

agree = (acc["account_id"].isin(ce["account_id"]) == acc["churn_flag"]).mean()
print(f"\nagreement between the two definitions: {agree:.1%}")
print(pd.crosstab(acc["churn_flag"],
                  acc["account_id"].isin(ce["account_id"]).rename("has_churn_event")))

# %% [markdown]
# **These cannot all describe the same thing, and they don't.** `churn_flag`
# agrees with the event log for 37.6% of accounts — worse than a coin flip.
#
# This is a labelling decision that has to be made explicitly, not discovered
# after the model underperforms. `churn_events` is used as ground truth because
# it carries dates and `churn_flag` does not; an undated flag cannot be placed
# relative to any cutoff, so it is unusable for a forward-looking target.

# %% [markdown]
# ## §5 — Validate against an external source
#
# The dataset README claims referential integrity and "signup ≤ subscription ≤
# churn". Worth testing rather than trusting.

# %%
print(integrity_report(tables).to_string(index=False))

# %% [markdown]
# The generator does not honour its own temporal claims. These are documented
# and worked around rather than silently repaired — in a real engagement they
# would go back to data engineering first.

# %% [markdown]
# ## §6 — Variation: missingness, and *why*

# %%
import src.audit as audit
print(audit.missingness_report(raw).to_string(index=False))

# %% [markdown]
# Disposition depends on the cause, not the percentage:
#
# - **`end_date`, 90.3%** — structural. The subscription is still open. A naive
#   ">60% missing, drop it" rule would discard one of the most informative
#   columns in the table.
# - **`satisfaction_score`, 41.2%** — genuinely absent; the customer did not
#   respond. Imputed **inside the CV fold**, never globally.
# - **`feedback_text`, 24.7%** — free text, not used as a feature.

# %%
# Is the missingness itself informative? If so it must be encoded, not filled.
tix = raw["support_tickets"].copy()
tix["missing"] = tix["satisfaction_score"].isna()
print("missing-rate by priority:")
print(tix.groupby("priority")["missing"].mean().round(3).to_string())
print("\nFlat across priorities -> no per-priority signal to exploit.")

# %% [markdown]
# ## §6 — Variation: duplicates, constants, impossible values

# %%
print("duplicate keys:")
for name, key in [("accounts", "account_id"), ("subscriptions", "subscription_id"),
                  ("feature_usage", "usage_id"), ("support_tickets", "ticket_id")]:
    d = int(raw[name][key].duplicated().sum())
    print(f"  {name:16s} {key:18s} {d}" + ("   <- dropped in clean.py" if d else ""))

print("\nimpossible values:")
t = raw["support_tickets"]
s = raw["subscriptions"]
checks = [
    ("negative resolution_time_hours", int((t["resolution_time_hours"] < 0).sum())),
    ("negative first_response_minutes", int((t["first_response_time_minutes"] < 0).sum())),
    ("satisfaction outside 1-5", int(((t["satisfaction_score"] < 1) | (t["satisfaction_score"] > 5)).sum())),
    ("negative mrr", int((s["mrr_amount"] < 0).sum())),
    ("seats <= 0", int((raw["accounts"]["seats"] <= 0).sum())),
]
for label, n in checks:
    print(f"  {label:34s} {n}")

# %% [markdown]
# ## §6 — Variation: distributions and outliers

# %%
fig, axes = plt.subplots(1, 3, figsize=(15, 4))
s["mrr_amount"].plot(kind="hist", bins=50, ax=axes[0], edgecolor="white")
axes[0].set_title("MRR — raw (heavy right tail)")
axes[0].set_xlabel("MRR ($)")

np.log1p(s["mrr_amount"]).plot(kind="hist", bins=40, ax=axes[1], edgecolor="white", color="coral")
axes[1].set_title("log1p(MRR) — tail controlled")

raw["accounts"]["seats"].plot(kind="hist", bins=40, ax=axes[2], edgecolor="white", color="seagreen")
axes[2].set_title("Seats per account")
plt.tight_layout()
plt.savefig("../outputs/figures/01_distributions.png", bbox_inches="tight")
plt.show()

# %%
# Quantify the tails rather than eyeballing them.
q = s["mrr_amount"].quantile([.5, .9, .99, 1.0])
iqr = s["mrr_amount"].quantile(.75) - s["mrr_amount"].quantile(.25)
fence = s["mrr_amount"].quantile(.75) + 1.5 * iqr
print(q.round(0).to_string())
print(f"\nTukey upper fence: {fence:,.0f}")
print(f"rows above it    : {int((s['mrr_amount'] > fence).sum())} ({(s['mrr_amount'] > fence).mean():.1%})")
print("\nThese are large enterprise contracts, not data errors — kept. The tree")
print("models are scale-invariant; the linear rungs get StandardScaler in-fold.")

# %% [markdown]
# ## §7 — Covariation between features (target not involved)

# %%
num = s.select_dtypes(include=[np.number])
fig, ax = plt.subplots(figsize=(6, 5))
sns.heatmap(num.corr(), annot=True, fmt=".2f", cmap="RdBu_r", center=0, ax=ax)
ax.set_title("Subscriptions — numeric correlation")
plt.tight_layout()
plt.savefig("../outputs/figures/01_feature_correlation.png", bbox_inches="tight")
plt.show()
print("arr_amount ~ mrr_amount at r = 1.00 — the redundancy is visible here too.")

# %% [markdown]
# # Part B — Target relationships (exploration split only)
#
# Everything below uses the target, so from here on we work on a held-out
# *exploration* split. The confirmation split is not looked at during EDA or
# feature design.

# %%
cohort = build_cohort(tables)
explore_idx, confirm_idx = train_test_split(
    cohort.index, test_size=0.3, stratify=cohort["churned_next_90d"], random_state=42
)
explore = cohort.loc[explore_idx]
confirm = cohort.loc[confirm_idx]

print(f"exploration split : {len(explore)} accounts, {int(explore['churned_next_90d'].sum())} positives")
print(f"confirmation split: {len(confirm)} accounts  <- sealed, not inspected here")

# %% [markdown]
# ## §8 — Understand the target

# %%
y = explore["churned_next_90d"]
print(y.value_counts().rename({0: "retained", 1: "churned"}).to_string())
print(f"\npositive rate: {y.mean():.3f}")
print(f"majority-class accuracy: {max(y.mean(), 1 - y.mean()):.3f}")
print("\n-> Imbalance is mild (47/53), so class weights suffice; no SMOTE needed.")
print("-> Accuracy is useless as a metric here. Use ROC-AUC and average")
print("   precision read against the base rate.")

# %% [markdown]
# ## §7 — Covariation with the target, by segment

# %%
explore_full = explore.copy()
base = y.mean()

fig, axes = plt.subplots(1, 3, figsize=(16, 4))
for ax, col in zip(axes, ["plan_tier", "industry", "referral_source"]):
    rates = explore_full.groupby(col)["churned_next_90d"].agg(["mean", "size"])
    rates = rates[rates["size"] >= 5].sort_values("mean")
    rates["mean"].plot(kind="barh", ax=ax, color="steelblue")
    ax.axvline(base, color="red", ls="--", alpha=.7, label=f"base {base:.2f}")
    ax.set_title(f"churn rate by {col}")
    ax.set_xlabel("")
    ax.legend()
plt.tight_layout()
plt.savefig("../outputs/figures/01_churn_by_segment.png", bbox_inches="tight")
plt.show()
print("Group sizes are small — bars without a base-rate reference line and an")
print("n >= 5 filter would be noise dressed as insight.")

# %%
# Segment sizes, so the bars above can be read honestly
print(explore_full.groupby("plan_tier")["churned_next_90d"].agg(["size", "sum", "mean"]).round(3).to_string())

# %% [markdown]
# ## §9 — Leakage screen, done here rather than after modelling
#
# For every candidate column the question is: **would I have this value on
# 2024-06-30?** Anything that answers "no" is excluded before a model is fit.
# Full per-field verdicts in `docs/DATA_DICTIONARY.md`.

# %%
from sklearn.metrics import roc_auc_score

probe = explore_full[["account_id", "churned_next_90d"]].merge(
    ce.groupby("account_id").agg(
        n_churn_events=("churn_event_id", "count"),
        total_refund_usd=("refund_amount_usd", "sum"),
    ).reset_index(), on="account_id", how="left").fillna(0)

print("Post-outcome columns, screened on the exploration split:")
for c in ["n_churn_events", "total_refund_usd"]:
    a = roc_auc_score(probe["churned_next_90d"], probe[c])
    print(f"  {c:20s} single-feature AUC = {max(a, 1 - a):.4f}")
print("\nA single raw column at this level is the label wearing a different name.")
print("Excluded via config.POST_OUTCOME_COLS; enforced by src/audit.py.")

# %% [markdown]
# ## Takeaways carried into feature engineering
#
# 1. **The label had to be redefined.** `churn_flag` is undated and agrees with
#    the event log only 37.6% of the time. The target is now a dated,
#    forward-looking event.
# 2. **`churn_events` cannot supply features** — those columns reconstruct the
#    answer.
# 3. **Missingness needs three different treatments** — structural, genuinely
#    absent, and not-a-feature. One blanket fill would have been wrong for all
#    three.
# 4. **Segment effects are small and the groups are tiny.** Nothing here
#    justifies a high-capacity model, which is what the model ladder later
#    confirms.
# 5. **MRR is heavy-tailed but the tail is real** — enterprise contracts, kept.
#
# Next: `02_cleaning.py`, then `06_leakage_quantification.py` for the
# corrected modelling path.
