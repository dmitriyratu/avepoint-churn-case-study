# %% [markdown]
# # 09 — Broad Classifier Sweep, and Why It Needs a Correction
#
# The "run everything and see what wins" approach: R's **caret**
# (`train`/`resamples`), and in Python **PyCaret** (`compare_models()`) or the
# lighter **LazyPredict** (`LazyClassifier`). Useful for a fast read on whether
# any model family does something the others cannot.
#
# It also has a trap that matters a great deal at this sample size. Fitting ~30
# classifiers and reporting the best is **30 chances to get lucky**. With
# fold-to-fold standard deviation around 0.09, the maximum of 30 draws sits well
# above 0.5 even when nothing has any signal at all.
#
# So this notebook does three things:
#
# 1. Runs LazyPredict as-is — the tool as people actually use it.
# 2. Re-runs the same families under repeated stratified CV, which is the only
#    comparison worth trusting here.
# 3. Repeats the whole sweep on **shuffled labels**, to measure what the winning
#    score looks like when the answer is known to be noise.

# %%
import sys
sys.path.append("..")

import numpy as np
import pandas as pd
from pathlib import Path
import matplotlib.pyplot as plt
import seaborn as sns
import warnings
warnings.filterwarnings("ignore")

from sklearn.model_selection import cross_val_score, train_test_split
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.ensemble import (AdaBoostClassifier, ExtraTreesClassifier,
                              GradientBoostingClassifier, RandomForestClassifier)
from sklearn.linear_model import LogisticRegression, RidgeClassifier, SGDClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import LinearSVC
from sklearn.tree import DecisionTreeClassifier, ExtraTreeClassifier
import lightgbm as lgb
import xgboost as xgb

from src import pipeline
from sklearn.model_selection import RepeatedStratifiedKFold
from src.model import CV, _pipe, model_ladder

# The observed sweep uses the project's 5x10 CV. The null repeats the whole sweep
# many times over, so it uses a cheaper 5x2 — enough to locate the null's centre
# and spread without ten thousand refits.
NULL_CV = RepeatedStratifiedKFold(n_splits=5, n_repeats=2, random_state=7)

sns.set_theme(style="whitegrid", palette="muted")

data = pipeline.build()
X, y = data.X, data.y
print(f"{X.shape}, {int(y.sum())} positives ({y.mean():.1%})")

# %% [markdown]
# ## 1. LazyPredict, used the way it is normally used
#
# A single stratified train/test split, ~30 classifiers, ranked by test score.

# %%
from lazypredict.Supervised import LazyClassifier

X_enc = pd.get_dummies(X, drop_first=True).fillna(X.median(numeric_only=True)).fillna(0)
X_tr, X_te, y_tr, y_te = train_test_split(X_enc, y, test_size=0.3, stratify=y, random_state=42)

lazy = LazyClassifier(verbose=0, ignore_warnings=True, predictions=False)
scores, _ = lazy.fit(X_tr, X_te, y_tr, y_te)
print(scores[["ROC AUC", "F1 Score"]].head(12).round(3).to_string())

# %%
best_lazy = scores["ROC AUC"].max()
print(f"best single-split ROC AUC across {len(scores)} classifiers: {best_lazy:.3f}")
print(f"our repeated-CV estimate for the selected model            : 0.583")
print("\nThe gap is not a better model. It is one lucky split, chosen from ~30 tries.")

# %% [markdown]
# ## 2. The same families under repeated stratified CV
#
# Identical folds for every candidate, 50 fits each. Slower, and the only version
# of this comparison that means anything at 177 rows.

# %%
CANDIDATES = {
    "Logistic (L2)": LogisticRegression(class_weight="balanced", max_iter=2000, random_state=42),
    "Ridge": RidgeClassifier(class_weight="balanced", random_state=42),
    "SGD (log loss)": SGDClassifier(loss="log_loss", class_weight="balanced", random_state=42),
    "LinearSVC": LinearSVC(class_weight="balanced", random_state=42),
    "LDA": LinearDiscriminantAnalysis(),
    "GaussianNB": GaussianNB(),
    "kNN (k=15)": KNeighborsClassifier(n_neighbors=15),
    "Decision tree (d3)": DecisionTreeClassifier(max_depth=3, class_weight="balanced", random_state=42),
    "Extra tree": ExtraTreeClassifier(max_depth=3, class_weight="balanced", random_state=42),
    "Random forest": RandomForestClassifier(n_estimators=100, max_depth=4, min_samples_leaf=8,
                                            class_weight="balanced", random_state=42),
    "Extra trees": ExtraTreesClassifier(n_estimators=100, max_depth=4, min_samples_leaf=8,
                                        class_weight="balanced", random_state=42),
    "AdaBoost": AdaBoostClassifier(n_estimators=100, learning_rate=0.1, random_state=42),
    "Gradient boosting": GradientBoostingClassifier(n_estimators=100, max_depth=3,
                                                    learning_rate=0.03, random_state=42),
    "LightGBM": lgb.LGBMClassifier(n_estimators=150, learning_rate=0.03, num_leaves=7,
                                   max_depth=3, class_weight="balanced",
                                   random_state=42, verbose=-1),
    "XGBoost": xgb.XGBClassifier(n_estimators=150, learning_rate=0.03, max_depth=3,
                                 scale_pos_weight=2.28, tree_method="hist",
                                 random_state=42, verbosity=0),
}


def sweep(y_target, cv=CV):
    rows = []
    for name, clf in CANDIDATES.items():
        s = cross_val_score(_pipe(clf), X, y_target, cv=cv, scoring="roc_auc")
        rows.append({"model": name, "cv_auc": s.mean(), "sd": s.std()})
    return pd.DataFrame(rows).sort_values("cv_auc", ascending=False).reset_index(drop=True)


observed = sweep(y)
print(observed.round(4).to_string(index=False))
observed.to_csv("../outputs/reports/classifier_sweep.csv", index=False)

# %%
print(f"best under repeated CV : {observed['cv_auc'].max():.4f}  ({observed.iloc[0]['model']})")
print(f"best under LazyPredict : {best_lazy:.4f}  (single split)")
print(f"spread across {len(observed)} models: "
      f"{observed['cv_auc'].min():.3f} to {observed['cv_auc'].max():.3f}")
print(f"typical fold-to-fold sd  : {observed['sd'].mean():.3f}")
print("\nThe spread across models is smaller than the noise within any one of them.")

# %% [markdown]
# ## 3. The correction — what does "best of 16" look like under pure noise?
#
# Shuffle the labels so there is definitionally nothing to learn, run the whole
# sweep, and record the winner. Repeat. The resulting distribution is what
# "the best model scored X" is worth when X was chosen by maximisation.

# %%
# The null is 20 sweeps and is by far the most expensive cell here, so its result
# is cached. Set REGENERATE = True to recompute from scratch (a few minutes).
REGENERATE = False
NULL_PATH = Path("../outputs/reports/selection_null.csv")

if REGENERATE or not NULL_PATH.exists():
    rng = np.random.default_rng(42)
    records = []
    for trial in range(20):
        y_shuffled = pd.Series(rng.permutation(y.values), index=y.index)
        res = sweep(y_shuffled, cv=NULL_CV)
        records += [{"trial": trial, "model": r.model, "cv_auc": r.cv_auc}
                    for r in res.itertuples()]
        print(f"  shuffle {trial + 1:>2}: best of {len(res)} = {res['cv_auc'].max():.4f}")
    null_df = pd.DataFrame(records)
    null_df.to_csv(NULL_PATH, index=False)
else:
    null_df = pd.read_csv(NULL_PATH)
    print(f"loaded cached null: {null_df['trial'].nunique()} shuffles x "
          f"{null_df['model'].nunique()} models  (set REGENERATE=True to recompute)")

null_best = null_df.groupby("trial")["cv_auc"].max().to_numpy()
null_all = null_df["cv_auc"].to_numpy()

# %%
observed_best = observed["cv_auc"].max()
p_corrected = (null_best >= observed_best).mean()

print(f"\n  individual model under shuffled labels : mean {null_all.mean():.4f}")
print(f"  BEST-OF-N under shuffled labels        : mean {null_best.mean():.4f}, "
      f"max {null_best.max():.4f}")
print(f"  observed best                           : {observed_best:.4f}")
print(f"\n  selection-corrected p-value             : {p_corrected:.3f}")

# %%
fig, ax = plt.subplots(figsize=(8, 4.5))
ax.hist(null_all, bins=20, alpha=.45, label="individual model, shuffled labels", color="#2a9d8f")
ax.hist(null_best, bins=8, alpha=.75, label="best-of-N, shuffled labels", color="#e9c46a")
ax.axvline(0.5, ls="--", c="k", alpha=.6, label="chance")
ax.axvline(observed_best, ls="-", c="#e76f51", lw=2, label=f"observed best ({observed_best:.3f})")
ax.set_xlabel("CV ROC-AUC"); ax.set_ylabel("count")
ax.set_title("Maximising over models shifts the null to the right")
ax.legend(fontsize=8)
plt.tight_layout()
plt.savefig("../outputs/figures/09_selection_null.png", bbox_inches="tight")
plt.show()

# %% [markdown]
# ## What this tells us
#
# | | CV ROC-AUC |
# |---|---:|
# | individual model, shuffled labels | 0.498 |
# | **best-of-15, shuffled labels** | **0.567** (max 0.624) |
# | observed best (AdaBoost) | 0.594 |
# | **selection-corrected p** | **0.300** |
#
# An individual model on shuffled labels averages 0.498, exactly as it should.
# **The best of fifteen averages 0.567 and reached 0.624** — on labels with no
# signal whatsoever. Maximising over candidates is itself a fitting procedure,
# and nothing regularises it.
#
# So the sweep's winner, 0.594, is beaten by **30% of pure-noise runs**. It is not
# evidence of anything.
#
# ### This applies to our own headline number too
#
# The reported L2 logistic result carries a permutation p of 0.076. That test
# holds the model fixed — but the model was chosen as the top of a ten-rung
# ladder. The same selection effect applies, less severely with 10 candidates
# than 15, but it applies.
#
# **The honest reading is that the selection-corrected p is in the 0.2–0.3 range,
# not 0.076.** The single-model permutation test understates it, and quoting 0.076
# without that caveat would be the same error this notebook is about.
#
# What would fix it properly: nested cross-validation, with model selection
# happening inside the outer loop so the reported score never sees the selection.
# That is the honest way to report a chosen-from-many model, and it is the main
# methodological gap left in this project.
#
# ### Using the sweep well
#
# - **Do use it** to check whether some family behaves qualitatively differently,
#   to catch a data problem that shows up as one model wildly outperforming, or
#   for a fast read before committing.
# - **Do not use it** to pick the final model, or report the winner's score as
#   the model's performance.
#
# The LazyPredict figure illustrates the same trap one level worse: a single 30%
# split of 177 rows leaves ~53 accounts and ~16 positives, and the best of thirty
# such splits is worth less than nothing without a correction.
#
# The substantive conclusion is unchanged and now better supported: **no model
# family separates from the others, and the spread between them (0.520–0.594) is
# smaller than the fold-to-fold noise within any one of them (sd ≈ 0.090).**
