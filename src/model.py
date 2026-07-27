"""Model ladder and evaluation.

Rungs are ordered by flexibility, cheapest first, and share one CV splitter so
the comparison is like-for-like. Every score carries an interval: with ~170 rows
a point estimate implies precision the data cannot support.

All preprocessing that learns a parameter — imputation, scaling, encoding — sits
inside the pipeline so it is refit per fold.
"""
import numpy as np
import pandas as pd
from pathlib import Path

import joblib
import lightgbm as lgb
from sklearn.compose import ColumnTransformer, make_column_selector
from sklearn.dummy import DummyClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import f1_score
from sklearn.model_selection import (GridSearchCV, RepeatedStratifiedKFold,
                                     StratifiedKFold, cross_val_predict,
                                     cross_val_score, permutation_test_score)
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.tree import DecisionTreeClassifier

from .config import ID_COLS, POST_OUTCOME_COLS, TARGET

MODELS_DIR = Path(__file__).parents[1] / "outputs" / "models"
MODELS_DIR.mkdir(parents=True, exist_ok=True)

CV = RepeatedStratifiedKFold(n_splits=5, n_repeats=10, random_state=42)
INNER_CV = StratifiedKFold(5, shuffle=True, random_state=42)
SEED = 42

_BOOLISH = {"True", "False", "0", "1", "0.0", "1.0"}


def prep_xy(df, target=TARGET):
    """Split into (X, y), coercing dtypes but preserving NaN and categoricals.

    A CSV round-trip turns booleans into "True"/"False" strings, and filling a
    boolean column with 0 leaves a three-valued mix; both are coerced back to
    numeric here. Genuine categoricals stay as strings for the pipeline encoder,
    and NaN is preserved for in-fold imputation.
    """
    X = df.drop(columns=[c for c in ID_COLS + POST_OUTCOME_COLS + [target]
                         if c in df.columns]).copy()
    y = df[target].astype(int)

    for c in X.columns:
        if pd.api.types.is_bool_dtype(X[c]):
            X[c] = X[c].astype(int)
        elif not pd.api.types.is_numeric_dtype(X[c]):
            text = X[c].astype(str).str.strip()
            X[c] = (pd.to_numeric(text.replace({"True": "1", "False": "0"}), errors="coerce")
                    if set(text.dropna()) <= _BOOLISH else text)

    numeric = X.select_dtypes(include=[np.number]).columns
    X[numeric] = X[numeric].replace([np.inf, -np.inf], np.nan)
    return X, y


def categorical_columns(X):
    return [c for c in X.columns if not pd.api.types.is_numeric_dtype(X[c])]


def _pipe(clf, scale=True):
    """Preprocessing plus estimator.

    OneHotEncoder(handle_unknown="ignore") keeps an unseen category from changing
    the column set at serving time, which `pd.get_dummies` cannot do.
    """
    numeric = [("impute", SimpleImputer(strategy="median"))]
    if scale:
        numeric.append(("scale", StandardScaler()))

    pre = ColumnTransformer([
        ("num", Pipeline(numeric), make_column_selector(dtype_include=np.number)),
        ("cat", Pipeline([("impute", SimpleImputer(strategy="most_frequent")),
                          ("onehot", OneHotEncoder(handle_unknown="ignore", drop="first",
                                                   sparse_output=False))]),
         make_column_selector(dtype_exclude=np.number)),
    ])
    return Pipeline([("pre", pre), ("clf", clf)])


def feature_names(fitted_pipe, X):
    """Post-encoding column names, for reading coefficients off a fitted model."""
    return list(fitted_pipe.named_steps["pre"].get_feature_names_out(X.columns))


def model_ladder():
    """Rungs in increasing flexibility. Rung 0 uses no features at all."""
    return [
        ("0. Prior (no features)",
         _pipe(DummyClassifier(strategy="prior"), scale=False)),
        ("1. Single decision stump",
         _pipe(DecisionTreeClassifier(max_depth=1, class_weight="balanced",
                                      random_state=SEED), scale=False)),
        ("2. Logistic (L2, C=1)",
         _pipe(LogisticRegression(class_weight="balanced", max_iter=2000,
                                  random_state=SEED))),
        ("3. Logistic (L2, C=0.05)",
         _pipe(LogisticRegression(C=0.05, class_weight="balanced", max_iter=2000,
                                  random_state=SEED))),
        ("4. Logistic (L1, C=0.1)",
         _pipe(LogisticRegression(C=0.1, penalty="l1", solver="liblinear",
                                  class_weight="balanced", random_state=SEED))),
        ("5. Random forest (depth 4)",
         _pipe(RandomForestClassifier(n_estimators=400, max_depth=4, min_samples_leaf=8,
                                      class_weight="balanced", random_state=SEED),
               scale=False)),
        ("6. LightGBM (shallow)",
         _pipe(lgb.LGBMClassifier(n_estimators=250, learning_rate=0.03, num_leaves=7,
                                  max_depth=3, min_child_samples=15, subsample=0.8,
                                  subsample_freq=1, colsample_bytree=0.7, reg_lambda=5.0,
                                  class_weight="balanced", random_state=SEED, verbose=-1),
               scale=False)),
    ]


def evaluate_ladder(X, y, cv=CV, scoring="roc_auc"):
    """Score every rung on identical folds, with a 95% interval."""
    rows = []
    for name, est in model_ladder():
        s = cross_val_score(est, X, y, cv=cv, scoring=scoring)
        rows.append({"model": name, f"{scoring}_mean": s.mean(), "sd": s.std(),
                     "ci_lo": np.percentile(s, 2.5), "ci_hi": np.percentile(s, 97.5)})
    return pd.DataFrame(rows).round(4)


def tune_lightgbm(X, y, cv=INNER_CV):
    grid = {"clf__num_leaves": [3, 7, 15],
            "clf__max_depth": [2, 3, 4],
            "clf__learning_rate": [0.02, 0.05],
            "clf__reg_lambda": [1.0, 5.0, 20.0]}
    base = _pipe(lgb.LGBMClassifier(n_estimators=300, class_weight="balanced",
                                    subsample=0.8, subsample_freq=1, colsample_bytree=0.7,
                                    min_child_samples=15, random_state=SEED, verbose=-1),
                 scale=False)
    return GridSearchCV(base, grid, scoring="roc_auc", cv=cv).fit(X, y)


def permutation_significance(estimator, X, y, n_permutations=300, cv=INNER_CV):
    """Compare the observed score against a shuffled-label null."""
    score, null, p = permutation_test_score(
        estimator, X, y, cv=cv, scoring="roc_auc",
        n_permutations=n_permutations, random_state=SEED)
    return {"observed_auc": round(score, 4), "null_mean": round(null.mean(), 4),
            "null_sd": round(null.std(), 4), "null_p95": round(np.percentile(null, 95), 4),
            "p_value": round(p, 4)}


def oof_threshold(estimator, X, y, cv=INNER_CV):
    """Pick the decision threshold out-of-fold. Returns (threshold, f1, proba)."""
    proba = cross_val_predict(estimator, X, y, cv=cv, method="predict_proba")[:, 1]
    grid = np.linspace(0.05, 0.95, 91)
    scores = [f1_score(y, proba >= t, zero_division=0) for t in grid]
    best = int(np.argmax(scores))
    return round(float(grid[best]), 3), round(scores[best], 4), proba


def save_model(model, name):
    path = MODELS_DIR / f"{name}.joblib"
    joblib.dump(model, path)
    return path


def load_model(name):
    return joblib.load(MODELS_DIR / f"{name}.joblib")
