import numpy as np
import pandas as pd
from pathlib import Path

from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import roc_auc_score, average_precision_score, f1_score
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
import lightgbm as lgb
import xgboost as xgb
import joblib

MODELS_DIR = Path(__file__).parents[1] / "outputs" / "models"
MODELS_DIR.mkdir(parents=True, exist_ok=True)

# columns to exclude from feature matrix
_DROP = ["account_id", "account_name", "churn_flag", "signup_date"]


def prep_xy(df, target="churn_flag"):
    drop = [c for c in _DROP if c in df.columns]
    X = df.drop(columns=drop)
    y = df[target].astype(int)
    X = X.copy()
    bool_cols = X.select_dtypes(include="bool").columns.tolist()
    X[bool_cols] = X[bool_cols].astype(int)
    obj_cols = X.select_dtypes(include="object").columns.tolist()
    if obj_cols:
        X[obj_cols] = X[obj_cols].astype(float)
    return X, y


def _class_ratio(y):
    counts = y.value_counts()
    return counts[0] / counts[1]


def train_lgb(X, y, params=None):
    ratio = _class_ratio(y)
    default_params = {
        "objective": "binary",
        "metric": "auc",
        "scale_pos_weight": ratio,
        "n_estimators": 400,
        "learning_rate": 0.04,
        "num_leaves": 31,
        "min_child_samples": 8,
        "feature_fraction": 0.75,
        "bagging_fraction": 0.75,
        "bagging_freq": 5,
        "lambda_l1": 0.1,
        "verbose": -1,
        "random_state": 42,
    }
    if params:
        default_params.update(params)
    model = lgb.LGBMClassifier(**default_params)
    model.fit(X, y)
    return model


def train_xgb(X, y, params=None):
    ratio = _class_ratio(y)
    default_params = {
        "objective": "binary:logistic",
        "eval_metric": "auc",
        "scale_pos_weight": ratio,
        "n_estimators": 400,
        "learning_rate": 0.04,
        "max_depth": 4,
        "min_child_weight": 5,
        "subsample": 0.75,
        "colsample_bytree": 0.75,
        "reg_alpha": 0.1,
        "random_state": 42,
        "verbosity": 0,
    }
    if params:
        default_params.update(params)
    model = xgb.XGBClassifier(**default_params)
    model.fit(X, y)
    return model


def train_logistic(X, y, params=None):
    pipe = Pipeline([
        ("scaler", StandardScaler()),
        ("clf", LogisticRegression(
            class_weight="balanced",
            max_iter=1000,
            C=0.1,
            random_state=42,
        )),
    ])
    pipe.fit(X, y)
    return pipe


def cross_validate(train_fn, X, y, n_splits=5):
    skf = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=42)
    results = []
    for fold, (tr, val) in enumerate(skf.split(X, y), 1):
        X_tr, X_val = X.iloc[tr], X.iloc[val]
        y_tr, y_val = y.iloc[tr], y.iloc[val]
        m = train_fn(X_tr, y_tr)
        proba = m.predict_proba(X_val)[:, 1]
        threshold = 0.5
        pred = (proba >= threshold).astype(int)
        results.append({
            "fold": fold,
            "roc_auc": roc_auc_score(y_val, proba),
            "avg_precision": average_precision_score(y_val, proba),
            "f1": f1_score(y_val, pred, zero_division=0),
        })
    return pd.DataFrame(results)


def save_model(model, name):
    path = MODELS_DIR / f"{name}.joblib"
    joblib.dump(model, path)
    return path


def load_model(name):
    return joblib.load(MODELS_DIR / f"{name}.joblib")
