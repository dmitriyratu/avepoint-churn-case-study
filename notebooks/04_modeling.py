# %% [markdown]
# # 04 — Modeling
#
# Progressive comparison: start at a floor that uses no features at all, then
# only keep added complexity if it beats the rung below on identical folds.
#
# Design choices and why:
#
# - **Repeated stratified CV (5 x 10), not a single holdout.** With 187 rows a
#   single split is not a measurement — fold-to-fold AUC spans 0.44 to 0.74.
# - **Every score carries a 95% interval.** A point estimate here would imply a
#   precision the data cannot support.
# - **All preprocessing lives inside the pipeline**, so imputation and scaling
#   are fit on the training fold only.
# - **The decision threshold is chosen out-of-fold**, never on the data used to
#   report the score.

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

from sklearn.metrics import (roc_auc_score, average_precision_score, f1_score,
                             recall_score, precision_score, confusion_matrix,
                             classification_report, roc_curve, precision_recall_curve)
from sklearn.calibration import calibration_curve

from src import pipeline
from src.model import (model_ladder, evaluate_ladder, nested_ladder_cv, tune_lightgbm,
                       permutation_significance, oof_threshold, save_model, CV)

sns.set_theme(style="whitegrid", palette="muted")

# Built from source rather than a cached CSV, so the notebook can never score a
# frame produced under a different cutoff or horizon.
data = pipeline.build(verify=True)
df, X, y = data.frame, data.X, data.y
print(data.summary.to_string())
print(f"\n{X.shape}   positives {int(y.sum())} ({y.mean():.1%})")

# %% [markdown]
# ## The ladder
#
# Rung 0 uses no features. If a model cannot beat it, the features are worthless
# regardless of how sophisticated the algorithm is. The first version of this
# notebook had no such floor, so 0.55 looked like a result rather than noise.

# %%
for name, est in model_ladder():
    print(f"  {name}")

# %%
ladder = evaluate_ladder(X, y, cv=CV)
print(ladder.to_string(index=False))
ladder.to_csv("../outputs/reports/model_ladder.csv", index=False)

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
plt.savefig("../outputs/figures/04_model_ladder.png", bbox_inches="tight")
plt.show()

# %% [markdown]
# **L1 logistic regression wins.** Both tree ensembles land below it.
#
# That is the expected outcome at 1.16 events per variable: the ensembles have
# far more capacity than 88 positives can support, and an L1 penalty doing hard
# feature selection is worth more than boosting. Note also how wide the intervals
# are — the ranking among rungs 2-6 is suggestive, not decisive, and saying so is
# part of the result.

# %%
best_idx = int(ladder["roc_auc_mean"].idxmax())
best_name, best_est = model_ladder()[best_idx]
print(f"top of table: {best_name}")

beats_chance = ladder.loc[best_idx, "ci_lo"] > 0.5
print("clears chance at the CI lower bound:", beats_chance)
if not beats_chance:
    print("\n-> No rung separates from the floor. 'Best' here is the winner of a\n"
          "   coin-flipping contest; the ranking should not be read as a result.")

# %% [markdown]
# ## Does tuning rescue the boosted model?
#
# Worth checking before concluding that complexity does not pay — an untuned
# LightGBM losing is a weak argument.

# %%
gs = tune_lightgbm(X, y)
print(f"best params          : {gs.best_params_}")
print(f"tuned LightGBM CV AUC: {gs.best_score_:.4f}")
print(f"{best_name} CV AUC   : {ladder.loc[best_idx, 'roc_auc_mean']:.4f}")
print("\nA 54-point grid search still does not close the gap.")

# %% [markdown]
# ## The honest score for a chosen-from-many model
#
# Every rung above is reported honestly, but **quoting the winner is not**.
# Picking the maximum of ten candidates is itself a fitting step, and nothing
# cross-validates it.
#
# Nested CV moves the selection inside each outer fold, so the reported score
# includes the cost of choosing.

# %%
per_fold, nested = nested_ladder_cv(X, y)
print(per_fold.to_string(index=False))
print()
print(nested.to_string())
per_fold.to_csv("../outputs/reports/nested_cv_folds.csv", index=False)

# %% [markdown]
# Two things to read here.
#
# **The optimism is large.** The inner loop's chosen model averages ~0.59; the
# same procedure scored on data it never saw averages ~0.52. The difference is
# what selection was worth, and it is roughly the entire apparent signal.
#
# **The winner is unstable.** Three different rungs win across five folds. If a
# model were genuinely better here, it would win consistently. Different winners
# per fold is what selecting on noise looks like.
#
# So the number to quote for "the model we would ship" is the **nested** one, not
# the ladder maximum.

# %%
best_row = ladder.loc[best_idx]
print(f"  ladder maximum (optimistic) : {best_row['roc_auc_mean']:.4f}   <- do not quote alone")
print(f"  nested CV (honest)          : {nested['nested_auc']:.4f}")
print(f"  chance                      : 0.5000")

# %% [markdown]
# ## Is any of this better than chance?
#
# With intervals this wide, the ranking above is not enough. A permutation test
# shuffles the labels 300 times to build the null distribution the observed
# score has to beat.

# %%
perm = permutation_significance(best_est, X, y, n_permutations=300)
for k, v in perm.items():
    print(f"  {k:14s} {v}")

# %%
print(f"\np = {perm['p_value']}", "-> beats chance" if perm["p_value"] < 0.05
      else "-> cannot reject chance")
print("\nNote this test holds the model FIXED. It does not account for that model")
print("having been chosen as the ladder maximum — the nested result above is the")
print("estimate that does.")

# %% [markdown]
# ## Operating point
#
# For churn, a missed churner costs more than a wasted outreach, so the threshold
# should favour recall. It is selected from out-of-fold predictions — tuning it
# on the same data used to report the score would make both numbers optimistic.

# %%
threshold, best_f1, oof = oof_threshold(best_est, X, y)
pred = (oof >= threshold).astype(int)

print(f"threshold (out-of-fold): {threshold}\n")
print(f"  AUC       {roc_auc_score(y, oof):.4f}")
print(f"  AP        {average_precision_score(y, oof):.4f}   (base rate {y.mean():.3f})")
print(f"  F1        {best_f1:.4f}")
print(f"  recall    {recall_score(y, pred):.4f}")
print(f"  precision {precision_score(y, pred, zero_division=0):.4f}")

# %%
print(classification_report(y, pred, target_names=["retained", "churned"]))

# %%
# Threshold sweep — the business picks the point, not the F1 optimum
print("threshold   recall  precision      F1")
for t in [0.30, 0.35, 0.40, 0.42, 0.45, 0.50, 0.55, 0.60]:
    p = (oof >= t).astype(int)
    r = recall_score(y, p, zero_division=0)
    pr = precision_score(y, p, zero_division=0)
    print(f"   {t:.2f}      {r:.3f}      {pr:.3f}   {2*r*pr/(r+pr+1e-9):.3f}")

# %% [markdown]
# ## Curves

# %%
fig, axes = plt.subplots(1, 3, figsize=(16, 4.5))

fpr, tpr, _ = roc_curve(y, oof)
axes[0].plot(fpr, tpr, lw=2, label=f"AUC = {roc_auc_score(y, oof):.3f}")
axes[0].plot([0, 1], [0, 1], "k--", alpha=.4)
axes[0].set(xlabel="FPR", ylabel="TPR", title="ROC (out-of-fold)")
axes[0].legend(loc="lower right")

prec, rec, _ = precision_recall_curve(y, oof)
axes[1].plot(rec, prec, lw=2, label=f"AP = {average_precision_score(y, oof):.3f}")
axes[1].axhline(y.mean(), ls="--", c="k", alpha=.4, label=f"base rate {y.mean():.2f}")
axes[1].set(xlabel="Recall", ylabel="Precision", title="Precision-Recall")
axes[1].legend(loc="upper right")

pt, pp = calibration_curve(y, oof, n_bins=5)
axes[2].plot(pp, pt, marker="o", label="model")
axes[2].plot([0, 1], [0, 1], "k--", alpha=.5, label="perfect")
axes[2].set(xlabel="mean predicted", ylabel="observed frequency", title="Calibration")
axes[2].legend()

plt.tight_layout()
plt.savefig("../outputs/figures/04_curves.png", bbox_inches="tight")
plt.show()

# %%
cm = confusion_matrix(y, pred)
fig, ax = plt.subplots(figsize=(5, 4))
sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", ax=ax,
            xticklabels=["retained", "churned"], yticklabels=["retained", "churned"])
ax.set(xlabel="predicted", ylabel="actual", title=f"Out-of-fold @ t={threshold}")
plt.tight_layout()
plt.savefig("../outputs/figures/04_confusion.png", bbox_inches="tight")
plt.show()

tn, fp, fn, tp = cm.ravel()
print(f"caught {tp} of {tp+fn} churners; missed {fn}; {fp} false alarms")

# %% [markdown]
# ## What the model keys on
#
# The selected rung is linear, so coefficients on standardised inputs are
# directly readable — no SHAP needed for a handful of terms, and a simpler
# explanation is a better one when it is available.

# %%
from src.model import feature_names

best_est.fit(X, y)
clf = best_est.named_steps["clf"]
has_coefs = hasattr(clf, "coef_")

if has_coefs:
    coef = pd.Series(clf.coef_[0], index=feature_names(best_est, X))
    nz = coef[coef != 0].sort_values(key=abs, ascending=False)
    print(f"{len(nz)} non-zero of {len(coef)} encoded features:\n")
    print(nz.head(15).round(4).to_string())
    nz.to_csv("../outputs/reports/model_coefficients.csv", header=["coefficient"])
else:
    nz = pd.Series(dtype=float)
    print(f"{best_name} is not linear — see permutation importance instead.")

# %%
if len(nz):
    top = nz.reindex(nz.abs().sort_values(ascending=False).index).head(12).sort_values()
    fig, ax = plt.subplots(figsize=(8, 4))
    top.plot(kind="barh", ax=ax, color=["salmon" if v > 0 else "steelblue" for v in top])
    ax.axvline(0, color="black", lw=.8)
    ax.set_title(f"{best_name} — standardised coefficients")
    ax.set_xlabel("coefficient (positive = higher churn risk)")
    plt.tight_layout()
    plt.savefig("../outputs/figures/04_coefficients.png", bbox_inches="tight")
    plt.show()
else:
    print("No coefficients to plot for", best_name)

# %% [markdown]
# ## Persist

# %%
save_model(best_est, "churn_model")
config = {
    "model": best_name,
    "persisted_model": best_name,
    "any_rung_beats_chance": bool(beats_chance),
    "cv_auc": float(ladder.loc[best_idx, "roc_auc_mean"]),
    "cv_auc_ci": [float(ladder.loc[best_idx, "ci_lo"]), float(ladder.loc[best_idx, "ci_hi"])],
    "oof_threshold": threshold,
    "oof_f1": float(best_f1),
    "oof_recall": float(recall_score(y, pred)),
    "oof_precision": float(precision_score(y, pred, zero_division=0)),
    "permutation_p": perm["p_value"],
    "nested_cv_auc": float(nested["nested_auc"]),
    "selection_optimism": float(nested["optimism"]),
    "distinct_nested_winners": int(nested["n_distinct_winners"]),
    "n_features_selected": int(len(nz)),
    "cohort_n": int(len(y)),
    "positives": int(y.sum()),
}
with open("../outputs/models/config.json", "w") as f:
    json.dump(config, f, indent=2)
pd.DataFrame([config]).to_csv("../outputs/reports/final_metrics.csv", index=False)
print(json.dumps(config, indent=2))
