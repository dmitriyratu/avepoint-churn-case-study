# %% [markdown]
# # 08 — Diagnostics: where the performance goes
#
# The model sits at ~0.58 AUC. This notebook asks *why*, separating causes that
# are fixable from causes that are not:
#
# 1. **Are the dropped features dropped correctly**, or is information being
#    thrown away?
# 2. **Error analysis** — what do the false negatives look like, and how much of
#    the error is reducible?
# 3. **Train vs validation** — is the gap overfitting, too little data, weak
#    features, or absent signal?

# %%
import sys
sys.path.append("..")

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import warnings
warnings.filterwarnings("ignore")

from sklearn.feature_selection import SelectKBest, f_classif
from sklearn.impute import SimpleImputer
from sklearn.model_selection import cross_validate, cross_val_score, learning_curve, validation_curve
from sklearn.neighbors import NearestNeighbors
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

from src import pipeline
from src.features import build_model_dataset
from src.model import CV, model_ladder, oof_threshold

sns.set_theme(style="whitegrid", palette="muted")

data = pipeline.build()
X, y, frame = data.X, data.y, data.frame
_, model = model_ladder()[2]          # L2 logistic — the selected rung

# %% [markdown]
# ## 1. Were the dropped features dropped correctly?

# %%
print("dropped as constant :", frame.attrs["dropped_constant"])
print("dropped as collinear:", frame.attrs["dropped_collinear"])

# %%
unpruned = build_model_dataset(data.observed, data.cohort, data.cutoff, prune=False)
num = unpruned.select_dtypes(include=[np.number])

print("what each dropped column was collinear WITH:\n")
for col in frame.attrs["dropped_collinear"]:
    if col in num:
        partner = num.corr()[col].drop(col).abs().sort_values(ascending=False)
        print(f"  {col:24s} r={partner.iloc[0]:.4f} with {partner.index[0]}")

# %% [markdown]
# `feature_breadth` is `unique_features_used / 40` — literally the same variable
# rescaled, so dropping it loses nothing.
#
# `active_days_last_*d` is the interesting one. Active *days* and event *counts*
# are conceptually different: same volume spread over more days is a different
# customer. Their ratio — events per active day — is "intensity", which is
# orthogonal to both parents and would be worth keeping if it varied.

# %%
for w in (30, 90, 180):
    days, events = f"active_days_last_{w}d", f"usage_last_{w}d"
    if days in num and events in num:
        intensity = num[events] / num[days].replace(0, np.nan)
        print(f"  {w:>3}d  events per active day: mean={intensity.mean():.2f} "
              f"sd={intensity.std():.2f}  distinct values={intensity.round(2).nunique()}")

# %% [markdown]
# Intensity is 1.00 with almost no variance — this generator emits at most one
# usage row per account per day, so "active days" and "event count" really are
# the same measurement. The pruning is correct and the ratio would be a constant.
#
# **One genuine loss, now recovered.** `accounts.is_trial` was dropped as
# point-in-time-unsafe (it is current-as-of-extraction). But the same information
# *is* available safely from the latest pre-cutoff subscription, so
# `latest_is_trial` was added rather than accepting the loss.

# %%
print("latest_is_trial present:", "latest_is_trial" in X.columns)
print("latest_seats present   :", "latest_seats" in X.columns)
print("\nExclusions that are correct and not recoverable:")
print("  churn_flag        — the outcome itself")
print("  accounts.seats    — matches pre-cutoff subscription seats only 51.6% of")
print("                      the time, so it carries a post-cutoff value")

# %% [markdown]
# ## 2. Error analysis

# %%
threshold, _, oof = oof_threshold(model, X, y)
pred = (oof >= threshold).astype(int)

err = pd.DataFrame({"proba": oof, "actual": y.values, "pred": pred})
err["kind"] = np.select(
    [(err.actual == 1) & (err.pred == 1), (err.actual == 1) & (err.pred == 0),
     (err.actual == 0) & (err.pred == 1)], ["TP", "FN", "FP"], "TN")
print(f"threshold {threshold}\n")
print(err["kind"].value_counts().to_string())

# %%
print(err.groupby("kind")["proba"].describe()[["count", "mean", "min", "max"]].round(3).to_string())

# %% [markdown]
# ### Are the false negatives recoverable by moving the threshold?

# %%
fn = err[err.kind == "FN"]
near = int((fn["proba"] > threshold - 0.05).sum())
print(f"FN just below the threshold : {near} / {len(fn)}")
print(f"FN max score                : {fn['proba'].max():.3f}  (threshold {threshold})")
print(f"\n-> Only {near} of {len(fn)} sit near the boundary. The rest are scored as")
print("   confidently safe, so no threshold choice recovers them. This is a")
print("   signal problem, not a calibration problem.")

# %%
fig, ax = plt.subplots(figsize=(8, 4))
for kind, colour in [("TN", "#2a9d8f"), ("FP", "#e9c46a"), ("FN", "#e76f51"), ("TP", "#264653")]:
    sub = err[err.kind == kind]
    ax.scatter(sub["proba"], np.random.default_rng(0).normal(0, .04, len(sub)) +
               {"TN": 0, "FP": 1, "FN": 2, "TP": 3}[kind],
               s=22, alpha=.7, color=colour, label=kind)
ax.axvline(threshold, ls="--", c="k", alpha=.6, label=f"threshold {threshold}")
ax.set_yticks(range(4)); ax.set_yticklabels(["TN", "FP", "FN", "TP"])
ax.set_xlabel("out-of-fold predicted probability")
ax.set_title("Errors by score — FNs cluster far below the threshold")
ax.legend(ncol=5, fontsize=8)
plt.tight_layout()
plt.savefig("../outputs/figures/08_error_scores.png", bbox_inches="tight")
plt.show()

# %% [markdown]
# ### What separates the churners we miss from the ones we catch?

# %%
numeric = X.select_dtypes(include=[np.number])
profile = (numeric.assign(kind=err["kind"].values)
           .query("kind in ['TP','FN']").groupby("kind").mean().T)
profile["diff_in_sd"] = ((profile["TP"] - profile["FN"]).abs()
                         / numeric.std().replace(0, np.nan))
print(profile.sort_values("diff_in_sd", ascending=False).head(8).round(2).to_string())

# %% [markdown]
# A consistent picture: the churners we **catch** are newer, higher-MRR accounts
# with fewer seats. The ones we **miss** are older, larger-seat, lower-MRR
# accounts whose usage was already drifting down.
#
# That follows directly from what the model keys on — `days_since_signup`
# dominates, so it is effectively a young-account detector. Any churner who does
# not fit that profile is invisible to it. **This is the reducible part**: a
# feature that captures large-account contraction would address it, if the
# underlying telemetry supported one.

# %% [markdown]
# ### How much of the error is irreducible?
#
# If customers who look alike in feature space share outcomes, the features carry
# signal. If they do not, no model can separate them — that is the Bayes floor.
#
# The right comparison is against the disagreement you would get from *random*
# pairs, which for a base rate p is 2p(1-p).

# %%
Z = StandardScaler().fit_transform(SimpleImputer(strategy="median").fit_transform(numeric))
_, idx = NearestNeighbors(n_neighbors=6).fit(Z).kneighbors(Z)
neighbour_disagreement = np.mean([(y.values[row[1:]] != y.values[i]).mean()
                                  for i, row in enumerate(idx)])
p = y.mean()
random_disagreement = 2 * p * (1 - p)

print(f"  base rate                              : {p:.3f}")
print(f"  disagreement among random pairs  2p(1-p): {random_disagreement:.3f}")
print(f"  disagreement among 5 nearest neighbours : {neighbour_disagreement:.3f}")
print(f"  informativeness ratio                   : {neighbour_disagreement/random_disagreement:.3f}")
print("\n  A ratio near 1.0 means neighbours in feature space are no more alike in")
print("  outcome than two customers picked at random — the features barely locate")
print("  a customer's risk at all. That is the irreducible component, and it is")
print("  a property of the data, not of the model.")

# %% [markdown]
# ## 3. Train vs validation — diagnosing the gap

# %%
rows = []
for name, est in model_ladder():
    r = cross_validate(est, X, y, cv=CV, scoring="roc_auc", return_train_score=True)
    rows.append({"model": name, "train": r["train_score"].mean(),
                 "validation": r["test_score"].mean(),
                 "gap": r["train_score"].mean() - r["test_score"].mean()})
gaps = pd.DataFrame(rows).round(3)
print(gaps.to_string(index=False))

# %%
fig, ax = plt.subplots(figsize=(9, 4.5))
yv = np.arange(len(gaps))
ax.barh(yv - .2, gaps["train"], height=.38, label="train", color="#e76f51")
ax.barh(yv + .2, gaps["validation"], height=.38, label="validation", color="#2a9d8f")
ax.axvline(0.5, ls="--", c="k", alpha=.5, label="chance")
ax.set_yticks(yv); ax.set_yticklabels(gaps["model"], fontsize=8); ax.invert_yaxis()
ax.set_xlabel("ROC-AUC"); ax.set_title("Every model memorises the training set")
ax.legend()
plt.tight_layout()
plt.savefig("../outputs/figures/08_train_val_gap.png", bbox_inches="tight")
plt.show()

# %% [markdown]
# The boosters reach train AUC **1.000** on 177 rows and validate at 0.54. That is
# not a tuning problem — it is far more capacity than 54 positives can constrain.
# Even the L2 logistic memorises to 0.97.

# %% [markdown]
# ### Would more rows help?

# %%
sizes, train_s, val_s = learning_curve(
    model, X, y, cv=CV, scoring="roc_auc",
    train_sizes=np.linspace(0.3, 1.0, 6))
lc = pd.DataFrame({"n_train": sizes.astype(int),
                   "train": train_s.mean(1).round(3),
                   "validation": val_s.mean(1).round(3)})
print(lc.to_string(index=False))
slope_per_100 = np.polyfit(sizes, val_s.mean(1), 1)[0] * 100
print(f"\nvalidation slope: {slope_per_100:+.3f} AUC per 100 additional training rows")

# %%
fig, ax = plt.subplots(figsize=(8, 4.5))
ax.plot(sizes, train_s.mean(1), marker="o", label="train", color="#e76f51")
ax.plot(sizes, val_s.mean(1), marker="o", label="validation", color="#2a9d8f")
ax.fill_between(sizes, val_s.mean(1) - val_s.std(1), val_s.mean(1) + val_s.std(1),
                alpha=.15, color="#2a9d8f")
ax.axhline(0.5, ls="--", c="k", alpha=.5)
ax.set_xlabel("training rows"); ax.set_ylabel("ROC-AUC")
ax.set_title("Validation is still climbing — the curve has not plateaued")
ax.legend()
plt.tight_layout()
plt.savefig("../outputs/figures/08_learning_curve.png", bbox_inches="tight")
plt.show()

# %% [markdown]
# **The validation curve is still rising at the largest sample available.** More
# rows of this same data would help. That is the clearest single piece of
# evidence that sample size, not method, is the binding constraint.

# %% [markdown]
# ### Is it p ≫ n? Test it directly.
#
# 73 features on 54 positives is roughly 0.7 events per variable, so the obvious
# hypothesis is too many features. If that were the story, cutting features would
# raise validation.

# %%
rows = []
for k in [3, 5, 10, 20, 40, X.shape[1]]:
    reduced = Pipeline(model.steps[:-1]
                       + [("select", SelectKBest(f_classif, k=min(k, X.shape[1])))]
                       + [model.steps[-1]])
    s = cross_val_score(reduced, X, y, cv=CV, scoring="roc_auc")
    rows.append({"n_features": min(k, X.shape[1]), "cv_auc": round(s.mean(), 4)})
print(pd.DataFrame(rows).to_string(index=False))

# %% [markdown]
# **It gets monotonically worse.** Cutting to the 3 strongest features costs
# ~0.06 AUC. So the signal is not concentrated in a few good predictors that the
# rest are drowning — it is smeared thinly across many weak ones, and removing
# any of them removes signal.
#
# That rules out the obvious remedy. Regularisation is already doing what
# feature selection would, and doing it better.

# %% [markdown]
# ### Regularisation sweep

# %%
Cs = [0.001, 0.01, 0.05, 0.1, 0.5, 1, 10]
train_s, val_s = validation_curve(model, X, y, param_name="clf__C",
                                  param_range=Cs, cv=CV, scoring="roc_auc")
vc = pd.DataFrame({"C": Cs, "train": train_s.mean(1).round(3),
                   "validation": val_s.mean(1).round(3)})
print(vc.to_string(index=False))
print(f"\nvalidation range across three orders of magnitude of C: "
      f"{vc['validation'].max() - vc['validation'].min():.3f} AUC")

# %% [markdown]
# Train moves from 0.75 to 0.99 across the sweep while validation barely moves.
# When the regularisation strength hardly matters, there is little signal for it
# to trade against.

# %% [markdown]
# ## Verdict
#
# | Candidate cause | Evidence | Verdict |
# |---|---|---|
# | **Data leakage** | full audit suite passes; forbidden-column gate by name | ruled out |
# | **Feature engineering** | ~20 added features moved AUC down; cutting features also hurts | not the constraint |
# | **Overfitting** | train 0.97–1.00 vs validation 0.54–0.58 | real, but a symptom |
# | **Sample size** | learning curve still rising, +0.09 AUC per 100 rows | **primary constraint** |
# | **Data quality** | label agrees with event log 37.6%; 19k usage rows predate their subscription | **major contributor** |
# | **Irreducible** | neighbours disagree at ~the random-pair rate | **large floor** |
#
# The overfitting is real but downstream: with 54 positives and genuinely weak
# features, any model flexible enough to fit will memorise. The fixes that would
# normally apply — fewer features, stronger regularisation — are already
# exhausted or actively harmful here.
#
# **What would actually move it**, in order of expected value:
#
# 1. **More labelled accounts.** The learning curve is still climbing. This is
#    the only lever with direct evidence behind it.
# 2. **A trustworthy label.** 37.6% agreement between `churn_flag` and the event
#    log caps everything downstream.
# 3. **Telemetry with coherent timestamps.** 19,128 of 24,979 usage rows predate
#    their own subscription's start, so recency and trend features are built on
#    incoherent history.
# 4. **Features aimed at the profile we miss** — large-seat, lower-MRR accounts
#    contracting slowly. Seat-level activation and licence utilisation would be
#    the first things I would ask for.
#
# Not on the list: a better algorithm.
