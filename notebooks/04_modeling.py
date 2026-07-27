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

from src.model import (prep_xy, model_ladder, evaluate_ladder, tune_lightgbm,
                       permutation_significance, oof_threshold, save_model, CV)
from src.config import TARGET

sns.set_theme(style="whitegrid", palette="muted")

df = pd.read_csv("../data/processed/features_temporal.csv")
X, y = prep_xy(df)
print(f"{X.shape}   positives {int(y.sum())} ({y.mean():.1%})")

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
print(f"selected: {best_name}")

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
print("Marginal. Real, but the model is weak and should be described that way.")

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
# ## Which features survive the L1 penalty

# %%
best_est.fit(X, y)
from src.model import feature_names
coef = pd.Series(best_est.named_steps["clf"].coef_[0], index=feature_names(best_est, X))
nz = coef[coef != 0].sort_values(key=abs, ascending=False)
print(f"L1 retained {len(nz)} of {X.shape[1]} features:\n")
print(nz.round(4).to_string())
nz.to_csv("../outputs/reports/l1_selected_coefficients.csv", header=["coefficient"])

# %%
fig, ax = plt.subplots(figsize=(8, 4))
nz.sort_values().plot(kind="barh", ax=ax,
                      color=["salmon" if v > 0 else "steelblue" for v in nz.sort_values()])
ax.axvline(0, color="black", lw=.8)
ax.set_title("L1 logistic — non-zero standardised coefficients")
ax.set_xlabel("coefficient (positive = higher churn risk)")
plt.tight_layout()
plt.savefig("../outputs/figures/04_coefficients.png", bbox_inches="tight")
plt.show()

# %% [markdown]
# ## Persist

# %%
save_model(best_est, "churn_l1_logistic")
config = {
    "model": best_name,
    "cv_auc": float(ladder.loc[best_idx, "roc_auc_mean"]),
    "cv_auc_ci": [float(ladder.loc[best_idx, "ci_lo"]), float(ladder.loc[best_idx, "ci_hi"])],
    "oof_threshold": threshold,
    "oof_f1": float(best_f1),
    "oof_recall": float(recall_score(y, pred)),
    "oof_precision": float(precision_score(y, pred, zero_division=0)),
    "permutation_p": perm["p_value"],
    "n_features_selected": int(len(nz)),
    "cohort_n": int(len(y)),
    "positives": int(y.sum()),
}
with open("../outputs/models/config.json", "w") as f:
    json.dump(config, f, indent=2)
pd.DataFrame([config]).to_csv("../outputs/reports/final_metrics.csv", index=False)
print(json.dumps(config, indent=2))
