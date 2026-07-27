"""Model ladder: establish a floor, then only add capacity that earns its keep.

Each rung is evaluated with the same repeated stratified CV so the comparison is
apples-to-apples, and every score carries a confidence interval — with ~190 rows
a single split is not a measurement.
"""
import numpy as np
import pandas as pd
from pathlib import Path

from sklearn.dummy import DummyClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.model_selection import (
    RepeatedStratifiedKFold, StratifiedKFold, cross_val_score,
    cross_val_predict, GridSearchCV, permutation_test_score,
)
from sklearn.metrics import roc_auc_score, average_precision_score, f1_score
import lightgbm as lgb
import joblib

from .config import ID_COLS, TARGET, POST_OUTCOME_COLS

MODELS_DIR = Path(__file__).parents[1] / "outputs" / "models"
MODELS_DIR.mkdir(parents=True, exist_ok=True)

CV = RepeatedStratifiedKFold(n_splits=5, n_repeats=10, random_state=42)

# Only 4 cores here. Estimators stay single-threaded so the outer CV loop owns
# the parallelism — nested n_jobs=-1 oversubscribes and stalls.
N_JOBS_OUTER = 1  # joblib multiprocessing stalls in this sandbox; CV here is fast serially


def prep_xy(df, target=TARGET):
    """Split into (X, y) with dtypes coerced robustly.

    A CSV round-trip turns booleans into the strings "True"/"False", and any
    fillna(0) applied to a boolean column leaves a three-valued mix of
    {True, False, 0}. Both are handled here so the pipeline behaves identically
    whether the frame came from memory or from disk.
    """
    drop = [c for c in ID_COLS + POST_OUTCOME_COLS + [target] if c in df.columns]
    X = df.drop(columns=drop).copy()
    y = df[target].astype(int)

    for c in X.columns:
        if pd.api.types.is_bool_dtype(X[c]):
            X[c] = X[c].astype(int)
        elif not pd.api.types.is_numeric_dtype(X[c]):
            X[c] = (X[c].astype(str)
                    .str.strip()
                    .replace({"True": "1", "False": "0", "nan": "0", "": "0"}))
            X[c] = pd.to_numeric(X[c], errors="coerce").fillna(0)

    # NaN is preserved deliberately — every pipeline in the ladder imputes inside
    # the fold (see _pipe), so the fill statistic never sees validation rows.
    X = X.replace([np.inf, -np.inf], np.nan)
    return X, y


def _pipe(clf, scale=True):
    steps = [("impute", SimpleImputer(strategy="median"))]
    if scale:
        steps.append(("scale", StandardScaler()))
    steps.append(("clf", clf))
    return Pipeline(steps)


def model_ladder(random_state=42):
    """Ordered rungs, cheapest and least flexible first."""
    return [
        ("0. Prior (no features)",
         _pipe(DummyClassifier(strategy="prior"), scale=False)),
        ("1. Single decision stump",
         _pipe(DecisionTreeClassifier(max_depth=1, class_weight="balanced",
                                      random_state=random_state), scale=False)),
        ("2. Logistic (L2, C=1)",
         _pipe(LogisticRegression(class_weight="balanced", max_iter=2000,
                                  random_state=random_state))),
        ("3. Logistic (L2, C=0.05)",
         _pipe(LogisticRegression(C=0.05, class_weight="balanced", max_iter=2000,
                                  random_state=random_state))),
        ("4. Logistic (L1, C=0.1)",
         _pipe(LogisticRegression(C=0.1, penalty="l1", solver="liblinear",
                                  class_weight="balanced", random_state=random_state))),
        ("5. Random forest (depth 4)",
         _pipe(RandomForestClassifier(n_estimators=400, max_depth=4,
                                      min_samples_leaf=8, class_weight="balanced",
                                      random_state=random_state, n_jobs=1), scale=False)),
        ("6. LightGBM (shallow)",
         _pipe(lgb.LGBMClassifier(n_estimators=250, learning_rate=0.03, num_leaves=7,
                                  max_depth=3, min_child_samples=15,
                                  subsample=0.8, subsample_freq=1, colsample_bytree=0.7,
                                  reg_lambda=5.0, class_weight="balanced",
                                  random_state=random_state, verbose=-1), scale=False)),
    ]


def evaluate_ladder(X, y, cv=CV, scoring="roc_auc"):
    rows = []
    for name, est in model_ladder():
        s = cross_val_score(est, X, y, cv=cv, scoring=scoring, n_jobs=N_JOBS_OUTER)
        rows.append({
            "model": name,
            f"{scoring}_mean": round(float(s.mean()), 4),
            "sd": round(float(s.std()), 4),
            "ci_lo": round(float(np.percentile(s, 2.5)), 4),
            "ci_hi": round(float(np.percentile(s, 97.5)), 4),
        })
    return pd.DataFrame(rows)


def tune_lightgbm(X, y, random_state=42):
    grid = {
        "clf__num_leaves": [3, 7, 15],
        "clf__max_depth": [2, 3, 4],
        "clf__learning_rate": [0.02, 0.05],
        "clf__reg_lambda": [1.0, 5.0, 20.0],
    }
    base = _pipe(lgb.LGBMClassifier(n_estimators=300, class_weight="balanced",
                                    subsample=0.8, subsample_freq=1,
                                    colsample_bytree=0.7, min_child_samples=15,
                                    random_state=random_state, verbose=-1), scale=False)
    gs = GridSearchCV(base, grid, scoring="roc_auc",
                      cv=StratifiedKFold(5, shuffle=True, random_state=random_state),
                      n_jobs=N_JOBS_OUTER)
    gs.fit(X, y)
    return gs


def permutation_significance(estimator, X, y, n_permutations=300, random_state=42):
    """Is the observed CV score distinguishable from chance?"""
    score, perm_scores, pvalue = permutation_test_score(
        estimator, X, y,
        cv=StratifiedKFold(5, shuffle=True, random_state=random_state),
        scoring="roc_auc", n_permutations=n_permutations,
        random_state=random_state, n_jobs=N_JOBS_OUTER,
    )
    return {
        "observed_auc": round(float(score), 4),
        "null_mean": round(float(perm_scores.mean()), 4),
        "null_sd": round(float(perm_scores.std()), 4),
        "null_p95": round(float(np.percentile(perm_scores, 95)), 4),
        "p_value": round(float(pvalue), 4),
    }


def oof_threshold(estimator, X, y, cv=None, metric="f1"):
    """Choose the decision threshold out-of-fold, never on the evaluation set."""
    cv = cv or StratifiedKFold(5, shuffle=True, random_state=42)
    proba = cross_val_predict(estimator, X, y, cv=cv, method="predict_proba")[:, 1]
    best_t, best_s = 0.5, -1.0
    for t in np.linspace(0.05, 0.95, 91):
        s = f1_score(y, (proba >= t).astype(int), zero_division=0)
        if s > best_s:
            best_t, best_s = float(t), float(s)
    return round(best_t, 3), round(best_s, 4), proba


def save_model(model, name):
    path = MODELS_DIR / f"{name}.joblib"
    joblib.dump(model, path)
    return path


def load_model(name):
    return joblib.load(MODELS_DIR / f"{name}.joblib")
