# %% [markdown]
# # 10 — Is the result real, or did we make a mistake?
#
# An AUC near 0.5 should not be accepted on trust. It has two very different
# explanations:
#
# 1. The pipeline is broken and is destroying signal that exists.
# 2. The signal genuinely is not recoverable from this data.
#
# This notebook tries hard to establish the first before accepting the second.
# Three lines of attack: a positive control on the pipeline, alternative cohort
# and label definitions, and a check on whether "churn" is even coherent here.

# %%
import sys
sys.path.insert(0, "..")

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import warnings
warnings.filterwarnings("ignore")

from sklearn.model_selection import cross_val_score

from src import pipeline
from src.config import HORIZON_DAYS, TARGET
from src.features import build_model_dataset
from src.labeling import at_risk_accounts, first_churn_date
from src.model import CV, model_ladder, prep_xy

sns.set_theme(style="whitegrid", palette="muted")

data = pipeline.build()
X, y = data.X, data.y
_, model = model_ladder()[2]
rng = np.random.default_rng(0)

# %% [markdown]
# ## 1. Positive control — can the pipeline find a signal we know is there?
#
# Replace the churn label with targets of known strength and re-run the whole
# thing. If the pipeline cannot recover a planted signal, it is broken.

# %%
controls = {}

strong = (X["total_mrr"] > X["total_mrr"].median()).astype(int)
controls["strong (label = mrr > median)"] = strong

noisy = strong.copy()
flip = rng.choice(len(noisy), size=int(0.25 * len(noisy)), replace=False)
noisy.iloc[flip] = 1 - noisy.iloc[flip]
controls["strong, 25% of labels flipped"] = noisy

z = (0.6 * (X["days_since_signup"] - X["days_since_signup"].mean()) / X["days_since_signup"].std()
     - 0.5 * (X["total_usage_events"] - X["total_usage_events"].mean()) / X["total_usage_events"].std())
prob = 1 / (1 + np.exp(-z))
controls["weak but REAL (2 features + noise)"] = pd.Series(
    (rng.random(len(prob)) < prob).astype(int), index=y.index)

controls["null (shuffled labels)"] = pd.Series(rng.permutation(y.values), index=y.index)
controls["ACTUAL churn label"] = y

rows = [{"target": name,
         "cv_auc": round(cross_val_score(model, X, target, cv=CV, scoring="roc_auc").mean(), 4)}
        for name, target in controls.items()]
control_results = pd.DataFrame(rows)
print(control_results.to_string(index=False))

# %% [markdown]
# **The pipeline is not broken.** It recovers a planted strong signal at ~0.96 and
# correctly returns ~0.49 on shuffled labels.
#
# The row that matters is the third one. A *genuinely real but weak* signal —
# a logistic function of two standardised features plus noise — scores about the
# same as the actual churn label at this sample size.
#
# That is an important correction to how this result should be described. Our
# number is **not** evidence that nothing is there — it is equally consistent
# with a real-but-weak relationship at this sample size. What it cannot do is
# tell the two apart, which is the actual finding.

# %%
fig, ax = plt.subplots(figsize=(8, 4))
colours = ["#264653", "#2a9d8f", "#e9c46a", "#adb5bd", "#e76f51"]
ax.barh(control_results["target"], control_results["cv_auc"], color=colours)
ax.axvline(0.5, ls="--", c="k", alpha=.6, label="chance")
ax.set_xlabel("CV ROC-AUC")
ax.set_title("Positive controls — the pipeline recovers signal when signal exists")
ax.legend()
plt.tight_layout()
plt.savefig("../outputs/figures/10_positive_controls.png", bbox_inches="tight")
plt.show()

# %% [markdown]
# ## 2. Is the cohort definition throwing away the signal?
#
# Our cohort requires a live subscription at the cutoff and excludes accounts
# that had already churned. That is a large exclusion, and reactivations are
# common here — so it could plausibly be removing exactly the churn-prone
# accounts. Where do the accounts go?

# %%
tables, CUTOFF = data.tables, data.cutoff
accounts, events, subs = (tables["accounts"], tables["churn_events"],
                          tables["subscriptions"])
live = at_risk_accounts(subs, CUTOFF)
first_churn = first_churn_date(events)

signed = accounts[accounts["signup_date"] < CUTOFF]
with_sub = signed[signed["account_id"].isin(live)]
prior = with_sub["account_id"].map(first_churn) < CUTOFF

print(f"  all accounts                       {len(accounts):>4}")
print(f"  signed up before the cutoff        {len(signed):>4}")
print(f"  ...holding a live subscription     {len(with_sub):>4}")
print(f"  ...with no prior churn event       {int((~prior).sum()):>4}   <- the cohort")
print(f"\n  dropped as 'already churned': {int(prior.sum())} "
      f"({prior.mean():.0%} of the eligible pool)")
print("  — and every one of them is still holding an open subscription.")

# %% [markdown]
# That last line is worth sitting with. Nearly half the eligible pool has a
# churn event *and* a live subscription at the same moment. Eligibility is
# defined off the subscriptions table; the label is defined off `churn_events`;
# and section 3 below shows those two sources agree on only 58% of accounts.
#
# So the cohort rule is not a neutral filter — it is the label's own incoherence
# deciding who gets studied. That makes the alternatives below a necessary
# check rather than a formality.
#
# Four definitions, same features, same folds.

# %%
HORIZON_END = CUTOFF + pd.Timedelta(days=HORIZON_DAYS)
in_window = lambda dates: dates.between(CUTOFF, HORIZON_END, inclusive="both")

eligible = signed
churned_in_window = events.loc[in_window(events["churn_date"]), "account_id"].unique()
ended_in_window = subs.loc[in_window(subs["end_date"]), "account_id"].unique()


def score_cohort(label, cohort, positives):
    frame = build_model_dataset(
        data.observed, cohort.assign(**{TARGET: cohort["account_id"].isin(positives).astype(int)}),
        CUTOFF)
    Xc, yc = prep_xy(frame)
    auc = cross_val_score(model, Xc, yc, cv=CV, scoring="roc_auc").mean()
    return {"definition": label, "n": len(yc), "positives": int(yc.sum()),
            "cv_auc": round(auc, 4)}


# A is the shipped cohort taken from the pipeline, so this table is anchored to
# the real definition rather than a copy of it that could drift.
variants = [
    {"definition": "A. current (live sub, exclude prior churners)",
     "n": len(y), "positives": int(y.sum()),
     "cv_auc": round(cross_val_score(model, X, y, cv=CV, scoring="roc_auc").mean(), 4)},
    score_cohort("B. live sub, ANY churn counts (reactivations in)",
                 eligible[eligible["account_id"].isin(live)], churned_in_window),
    score_cohort("C. no live-sub requirement", eligible, churned_in_window),
    score_cohort("D. event = subscription ends, not churn_events",
                 eligible[eligible["account_id"].isin(live)], ended_in_window),
]
print(pd.DataFrame(variants).to_string(index=False))

# %% [markdown]
# **No.** Letting reactivated accounts back in nearly doubles the cohort and the
# score goes *down*. Using subscription endings as the event lands below chance.
# The current definition is both the most defensible and the best-performing, and
# no reasonable alternative rescues the result.

# %% [markdown]
# ## 3. Is "churn" even a coherent concept in this data?
#
# The decisive check. This dataset offers three independent ways to say an
# account churned:
#
# - `accounts.churn_flag`
# - a row in `churn_events`
# - a subscription with an `end_date`
#
# If churn is real, these should largely agree, and a churn date should land near
# a subscription ending.

# %%
ended = subs.dropna(subset=["end_date"])
definitions = pd.DataFrame({
    "churn_flag": accounts["churn_flag"].values,
    "has_churn_event": accounts["account_id"].isin(events["account_id"]).values,
    "has_ended_subscription": accounts["account_id"].isin(ended["account_id"]).values,
})

print("pairwise agreement:")
print(f"  churn_flag  vs churn_events        {(definitions.churn_flag == definitions.has_churn_event).mean():.1%}")
print(f"  churn_flag  vs ended subscription  {(definitions.churn_flag == definitions.has_ended_subscription).mean():.1%}")
print(f"  churn_event vs ended subscription  {(definitions.has_churn_event == definitions.has_ended_subscription).mean():.1%}")
print(f"\nall three agree on {(definitions.nunique(axis=1) == 1).mean():.1%} of accounts")

# %%
ends_by_account = ended.groupby("account_id")["end_date"].apply(list)
gaps = pd.Series([
    min((abs((row.churn_date - end).days) for end in ends_by_account.get(row.account_id, [])),
        default=np.nan)
    for row in events.itertuples()
]).dropna()

print(f"churn events comparable to a subscription end: {len(gaps)} of {len(events)}\n")
for window in (0, 7, 30, 90):
    print(f"  within {window:>3} days of a subscription ending: "
          f"{(gaps <= window).sum():>4}  ({(gaps <= window).mean():.1%})")
print(f"\n  median gap: {gaps.median():.0f} days")

# %%
fig, ax = plt.subplots(figsize=(8, 4))
ax.hist(gaps.clip(upper=365), bins=40, color="#e76f51", alpha=.8)
ax.axvline(30, ls="--", c="k", alpha=.6, label="30 days")
ax.set_xlabel("days between a churn event and the nearest subscription ending")
ax.set_ylabel("count")
ax.set_title("Churn dates do not coincide with subscriptions ending")
ax.legend()
plt.tight_layout()
plt.savefig("../outputs/figures/10_churn_coherence.png", bbox_inches="tight")
plt.show()

# %% [markdown]
# ## 4. The cheapest version of the whole argument
#
# Everything above is expensive. There is a two-line version.
#
# The leakage audit reports the strongest single feature at ~0.62 AUC and calls
# it fine. The question it does not ask: **what is the strongest of 86 features
# when the labels are pure noise?** If the two match, no individual column
# carries signal, and nothing downstream can conjure it.

# %%
from src.audit import single_feature_auc

observed_max = single_feature_auc(X, y)["auc"].max()
rng2 = np.random.default_rng(0)
null_max = np.array([
    single_feature_auc(X, pd.Series(rng2.permutation(y.values), index=y.index))["auc"].max()
    for _ in range(200)])

print(f"  observed max single-feature AUC        : {observed_max:.4f}")
print(f"  shuffled-label max, mean               : {null_max.mean():.4f} "
      f"(sd {null_max.std():.4f})")
print(f"  P(noise beats the observed max)        : {(null_max >= observed_max).mean():.3f}")

# %% [markdown]
# The best feature in the matrix is indistinguishable from the best of 86 coin
# flips. Screening many columns and reporting the winner has exactly the same
# problem as screening many models and reporting the winner — notebook 09's
# point, one level down.

# %% [markdown]
# ## Verdict
#
# | Question | Answer |
# |---|---|
# | Is the pipeline destroying signal? | **No** — planted signal recovers at ~0.96, null at ~0.49 |
# | Is the cohort definition the problem? | **No** — three alternatives score the same or worse |
# | Is our result "no signal"? | **Not provably** — it also matches a real-but-weak planted signal |
# | Does any single feature carry signal? | **No** — the best is what noise produces |
# | Is the label coherent? | **No** — three definitions agree on 20% of accounts |
#
# Only a handful of churn events fall on the day a subscription ends, and the
# median gap is two months. The three available definitions of "churned" agree
# on a fifth of accounts. **This dataset has no single, consistent notion of what
# churn is.**
#
# That is the ceiling, and it is not a modelling problem. You cannot predict an
# event whose own definition is 20% self-consistent — the label noise alone caps
# achievable AUC far below anything useful, regardless of features or algorithm.
#
# ### How this should be stated
#
# Carefully, because two readings survive and the data cannot separate them:
#
# - The positive control shows our result is *also* what a genuine but weak
#   relationship looks like at this sample size. So "there is definitely nothing
#   here" overstates it.
# - The single-feature null shows no column beats noise, the nested CV sits at
#   chance, and the label contradicts itself. So "there is something here"
#   overstates it just as much.
#
# The accurate statement is the one that commits to neither: **if a weak
# relationship exists, this dataset cannot resolve it** — 54 positives, a label
# that contradicts itself, and timestamps that do not order events correctly.
# Everything measured here is consistent with that and with nothing else.
