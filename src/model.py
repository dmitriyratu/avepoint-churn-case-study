"""Model ladder, nested selection, and evaluation.

Rungs run cheapest first and share one splitter, so the comparison is
like-for-like. Anything that learns a parameter — imputation, scaling, encoding —
lives inside the pipeline and is refit per fold.
"""
import os
from pathlib import Path

import joblib
import lightgbm as lgb
import numpy as np
import pandas as pd
import xgboost as xgb
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.compose import ColumnTransformer, make_column_selector
from sklearn.dummy import DummyClassifier
from sklearn.ensemble import HistGradientBoostingClassifier, RandomForestClassifier
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import precision_recall_curve
from sklearn.model_selection import (GridSearchCV, RepeatedStratifiedKFold,
                                     StratifiedKFold, cross_val_predict,
                                     cross_val_score, cross_validate,
                                     permutation_test_score)
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.tree import DecisionTreeClassifier

from .config import ID_COLS, POINT_IN_TIME_UNSAFE_COLS, POST_OUTCOME_COLS, TARGET

MODELS_DIR = Path(__file__).parents[1] / "outputs" / "models"
MODELS_DIR.mkdir(parents=True, exist_ok=True)

SEED = 42
CV = RepeatedStratifiedKFold(n_splits=5, n_repeats=10, random_state=SEED)
INNER_CV = StratifiedKFold(5, shuffle=True, random_state=SEED)

# Serial by default: n_jobs=-1 deadlocks under some container runtimes. Set
# CHURN_N_JOBS to opt in — parallelism stays at the outer loop, estimators at 1.
N_JOBS = int(os.environ.get("CHURN_N_JOBS", "1"))

LGBM_PARAMS = dict(n_estimators=250, learning_rate=0.03, num_leaves=7, max_depth=3,
                   min_child_samples=15, subsample=0.8, subsample_freq=1,
                   colsample_bytree=0.7, reg_lambda=5.0, class_weight="balanced",
                   random_state=SEED, verbose=-1)

_BOOLISH = {"True", "False", "0", "1", "0.0", "1.0"}


def scale_pos_weight(y):
    """XGBoost's equivalent of class_weight='balanced'."""
    positives = int(y.sum())
    return (len(y) - positives) / positives if positives else 1.0


def _coerce(col):
    """A CSV round-trip returns booleans as 'True'/'False'; genuine text stays text."""
    if pd.api.types.is_bool_dtype(col):
        return col.astype(int)
    if pd.api.types.is_numeric_dtype(col):
        return col.replace([np.inf, -np.inf], np.nan)
    text = col.astype(str).str.strip()
    if set(text) <= _BOOLISH:
        return pd.to_numeric(text.replace({"True": "1", "False": "0"}), errors="coerce")
    return text


def prep_xy(df, target=TARGET):
    """(X, y) with identifiers, outcomes and point-in-time-unsafe columns dropped.

    Categoricals stay as strings and NaN stays NaN; both are handled per fold.
    """
    excluded = ID_COLS + POST_OUTCOME_COLS + POINT_IN_TIME_UNSAFE_COLS + [target]
    X = df.drop(columns=[c for c in excluded if c in df.columns])
    return X.apply(_coerce), df[target].astype(int)


def categorical_columns(X):
    return [c for c in X.columns if not pd.api.types.is_numeric_dtype(X[c])]


def _pipe(clf, scale=True):
    """Preprocessing plus estimator.

    handle_unknown='ignore' keeps an unseen category from changing the column set
    at serving time, which get_dummies cannot do.
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


class AsCategory(BaseEstimator, TransformerMixin):
    """Cast object columns to pandas `category`, leaving NaN intact.

    Levels are learned on the training fold; anything unseen becomes NaN, which
    a booster routes like any other missing value.
    """

    def fit(self, X, y=None):
        self.categories_ = {c: pd.Index(X[c].dropna().unique())
                            for c in categorical_columns(X)}
        return self

    def transform(self, X):
        return X.assign(**{c: pd.Categorical(X[c], categories=levels)
                           for c, levels in self.categories_.items()})


def _native_pipe(clf):
    """Booster with raw NaN and native categoricals — no imputation, no one-hot."""
    return Pipeline([("cat", AsCategory()), ("clf", clf)])


def model_ladder(pos_weight=2.28):
    """Rungs in increasing flexibility. Rung 0 uses no features at all.

    `pos_weight` is XGBoost's imbalance lever; callers pass `scale_pos_weight(y)`
    so a different horizon does not leave rung 8 weighted for another cohort.
    """
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
        ("6. LightGBM (pipelined)",
         _pipe(lgb.LGBMClassifier(**LGBM_PARAMS), scale=False)),
        ("7. LightGBM (native NaN + categoricals)",
         _native_pipe(lgb.LGBMClassifier(**LGBM_PARAMS))),
        ("8. XGBoost (native NaN + categoricals)",
         _native_pipe(xgb.XGBClassifier(
             n_estimators=250, learning_rate=0.03, max_depth=3, min_child_weight=5,
             subsample=0.8, colsample_bytree=0.7, reg_lambda=5.0,
             scale_pos_weight=pos_weight, enable_categorical=True,
             tree_method="hist", random_state=SEED, verbosity=0))),
        ("9. HistGradientBoosting (native NaN)",
         _pipe(HistGradientBoostingClassifier(
             max_depth=3, max_leaf_nodes=7, learning_rate=0.03, max_iter=250,
             min_samples_leaf=15, l2_regularization=5.0,
             class_weight="balanced", random_state=SEED), scale=False)),
    ]


def evaluate_ladder(X, y, cv=CV, scoring="roc_auc"):
    """Score every rung on identical folds, with a 95% interval."""
    rows = []
    for name, est in model_ladder(scale_pos_weight(y)):
        s = cross_val_score(est, X, y, cv=cv, scoring=scoring, n_jobs=N_JOBS)
        rows.append({"model": name, f"{scoring}_mean": s.mean(), "sd": s.std(),
                     "ci_lo": np.percentile(s, 2.5), "ci_hi": np.percentile(s, 97.5)})
    return pd.DataFrame(rows).round(4)


def ladder_search(y, cv=INNER_CV, scoring="roc_auc"):
    """The ladder as a single estimator: a search whose grid is the rungs.

    Making selection an estimator is what turns nested CV into one
    `cross_validate` call — the choice is refit inside every outer fold.

    The wrapper is seeded with rung 0 rather than "passthrough" so the pipeline
    reads as a classifier; every grid candidate replaces it anyway.
    """
    rungs = model_ladder(scale_pos_weight(y))
    return GridSearchCV(Pipeline([("rung", rungs[0][1])]),
                        [{"rung": [est]} for _, est in rungs],
                        scoring=scoring, cv=cv, n_jobs=N_JOBS)


def nested_ladder_cv(X, y, n_repeats=5, scoring="roc_auc"):
    """Cross-validate the procedure "pick the best rung", not a fixed model.

    Quoting the ladder maximum is optimistic: choosing among ten candidates is
    itself a fitting step, and nothing cross-validates it. Repeated because a
    single 5-fold split swings ~0.09 AUC on the seed alone — see
    `repeat_spread` in the summary.
    """
    n_splits = 5
    names = [name for name, _ in model_ladder()]
    outer = RepeatedStratifiedKFold(n_splits=n_splits, n_repeats=n_repeats,
                                    random_state=SEED)
    run = cross_validate(ladder_search(y, scoring=scoring), X, y, cv=outer,
                         scoring=scoring, return_estimator=True, n_jobs=N_JOBS)

    per_fold = pd.DataFrame({
        "repeat": np.repeat(np.arange(1, n_repeats + 1), n_splits),
        "fold": np.tile(np.arange(1, n_splits + 1), n_repeats),
        "selected": [names[s.best_index_] for s in run["estimator"]],
        "inner_auc": [s.best_score_ for s in run["estimator"]],
        "outer_auc": run["test_score"],
    }).round(4)

    by_repeat = per_fold.groupby("repeat")["outer_auc"].mean()
    summary = pd.Series({
        "nested_auc": per_fold["outer_auc"].mean(),
        "nested_se": by_repeat.sem(),
        "repeat_spread": by_repeat.max() - by_repeat.min(),
        "mean_inner_auc": per_fold["inner_auc"].mean(),
        "optimism": per_fold["inner_auc"].mean() - per_fold["outer_auc"].mean(),
        "n_distinct_winners": per_fold["selected"].nunique(),
    }).round(4)
    return per_fold, summary


def tune_lightgbm(X, y, cv=INNER_CV):
    grid = {"clf__num_leaves": [3, 7, 15],
            "clf__max_depth": [2, 3, 4],
            "clf__learning_rate": [0.02, 0.05],
            "clf__reg_lambda": [1.0, 5.0, 20.0]}
    base = _native_pipe(lgb.LGBMClassifier(
        n_estimators=300, class_weight="balanced", subsample=0.8, subsample_freq=1,
        colsample_bytree=0.7, min_child_samples=15, random_state=SEED, verbose=-1))
    return GridSearchCV(base, grid, scoring="roc_auc", cv=cv, n_jobs=N_JOBS).fit(X, y)


def permutation_significance(estimator, X, y, n_permutations=300, cv=INNER_CV,
                             scoring="roc_auc"):
    """Observed score against a shuffled-label null, on any scoring scale.

    The metric name travels with the numbers. A bare 0.58 is not a result until
    the reader knows which scale it sits on and where that scale's null falls —
    0.50 for ROC-AUC whatever the base rate, but the base rate itself for
    average precision.
    """
    score, null, p = permutation_test_score(
        estimator, X, y, cv=cv, scoring=scoring,
        n_permutations=n_permutations, random_state=SEED, n_jobs=N_JOBS)
    return {"metric": scoring, "observed": round(score, 4),
            "null_mean": round(null.mean(), 4),
            "null_sd": round(null.std(), 4), "null_p95": round(np.percentile(null, 95), 4),
            "p_value": round(p, 4)}


def best_f1_threshold(y, proba):
    """Threshold maximising F1, read off the PR curve's own breakpoints."""
    precision, recall, thresholds = precision_recall_curve(y, proba)
    total = precision + recall
    f1 = np.divide(2 * precision * recall, total, out=np.zeros_like(total),
                   where=total > 0)[:-1]
    best = int(f1.argmax())
    return float(thresholds[best]), float(f1[best])


def oof_threshold(estimator, X, y, cv=INNER_CV):
    """Pick the decision threshold out-of-fold. Returns (threshold, f1, proba)."""
    proba = cross_val_predict(estimator, X, y, cv=cv, method="predict_proba",
                              n_jobs=N_JOBS)[:, 1]
    threshold, f1 = best_f1_threshold(y, proba)
    return round(threshold, 3), round(f1, 4), proba


def save_model(model, name):
    path = MODELS_DIR / f"{name}.joblib"
    joblib.dump(model, path)
    return path


def load_model(name):
    return joblib.load(MODELS_DIR / f"{name}.joblib")
