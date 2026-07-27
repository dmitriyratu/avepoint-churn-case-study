"""Leakage and data-quality gates.

Each check returns a frame with a `pass` column so the suite can be asserted
rather than eyeballed. `run_all` is called before any score is reported.
"""
import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score

SINGLE_FEATURE_AUC_WARN = 0.70
SINGLE_FEATURE_AUC_FAIL = 0.80
COLLINEARITY_THRESHOLD = 0.95
IDENTIFIER_AUC_MAX = 0.60


def _auc(y, values):
    """Direction-agnostic: a perfectly inverted feature leaks just as much."""
    score = roc_auc_score(y, values)
    return round(max(score, 1 - score), 4)


def encode_for_audit(X):
    """Flat numeric view for the checks below. Never feeds a model."""
    categorical = X.columns[~X.apply(pd.api.types.is_numeric_dtype)]
    return pd.get_dummies(X, columns=list(categorical), drop_first=True, dtype=int)


def temporal_provenance(tables, cutoff):
    """No datetime anywhere in the truncated tables may reach the cutoff.

    Checks every datetime column, not just the one used for filtering — a table
    filtered on `submitted_at` can still carry a `closed_at` in the future.
    """
    rows = [{"table": name, "column": col, "max_value": str(df[col].max())[:10],
             "violations": int((df[col] >= cutoff).sum())}
            for name, df in tables.items()
            for col in df.columns if pd.api.types.is_datetime64_any_dtype(df[col])]
    return pd.DataFrame(rows).assign(**{"pass": lambda d: d["violations"] == 0})


def single_feature_auc(X, y):
    """Standalone discriminative power. A lone strong column is usually the label."""
    cols = encode_for_audit(X).apply(pd.to_numeric, errors="coerce").astype(float)
    rows = [{"feature": c, "auc": _auc(y, s.fillna(s.median()))}
            for c, s in cols.items() if s.notna().any() and s.std() > 0]

    out = pd.DataFrame(rows).sort_values("auc", ascending=False, ignore_index=True)
    out["verdict"] = pd.cut(
        out["auc"], [0, SINGLE_FEATURE_AUC_WARN, SINGLE_FEATURE_AUC_FAIL, 1],
        labels=["ok", "WARN — inspect", "FAIL — near-certain leak"])
    return out


def perfect_separation(X, y, max_levels=10):
    """Low-cardinality columns where every level maps to a single class."""
    rows = [{"feature": c, "levels": int(X[c].nunique()), "pass": False}
            for c in X.columns
            if 1 < X[c].nunique() <= max_levels
            and ((pd.crosstab(X[c], y) > 0).sum(axis=1) == 1).all()]
    return pd.DataFrame(rows, columns=["feature", "levels", "pass"])


def identifier_leakage(df, y, id_col="account_id"):
    """Ids and row order must carry no signal."""
    rows = [{"probe": "row order", "auc": _auc(y, np.arange(len(y)))}]
    if id_col in df.columns:
        as_int = (df[id_col].astype(str).str.extract(r"([0-9a-fA-F]+)")[0]
                  .map(lambda s: int(s, 16) if isinstance(s, str) else np.nan))
        if as_int.notna().any():
            rows.insert(0, {"probe": f"{id_col} as integer",
                            "auc": _auc(y, as_int.fillna(as_int.median()))})
    return pd.DataFrame(rows).assign(**{"pass": lambda d: d["auc"] < IDENTIFIER_AUC_MAX})


def duplicate_rows(X, df=None, id_col="account_id"):
    rows = [{"check": "identical feature vectors", "n": int(X.duplicated().sum())}]
    if df is not None and id_col in df.columns:
        rows.append({"check": f"duplicate {id_col}",
                     "n": int(df[id_col].duplicated().sum())})
    return pd.DataFrame(rows).assign(**{"pass": lambda d: d["n"] == 0})


def collinear_pairs(X, threshold=COLLINEARITY_THRESHOLD):
    corr = encode_for_audit(X).corr(numeric_only=True).abs()
    pairs = (corr.where(np.triu(np.ones(corr.shape, dtype=bool), k=1))
             .stack().rename("abs_r").rename_axis(["feature_a", "feature_b"])
             .reset_index())
    return (pairs[pairs["abs_r"] > threshold]
            .sort_values("abs_r", ascending=False, ignore_index=True).round(4))


def forbidden_columns(X):
    """Outcome and point-in-time-unsafe columns must never reach the model.

    The single-feature gate only catches a *strongly* predictive leak.
    `churn_flag` scores 0.51 here — categorically the outcome, statistically
    invisible — so it needs a check by name.
    """
    from .config import POINT_IN_TIME_UNSAFE_COLS, POST_OUTCOME_COLS

    forbidden = ({c: "outcome variable" for c in POST_OUTCOME_COLS}
                 | {c: "not knowable as of the cutoff"
                    for c in POINT_IN_TIME_UNSAFE_COLS})
    rows = [{"column": c, "reason": why, "pass": False}
            for c, why in forbidden.items() if c in X.columns]
    return pd.DataFrame(rows, columns=["column", "reason", "pass"])


def constant_columns(X):
    rows = [{"feature": c, "n_unique": int(X[c].nunique())}
            for c in X.columns if X[c].nunique() <= 1]
    return pd.DataFrame(rows, columns=["feature", "n_unique"])


def missingness_report(tables, structural=None):
    """Per-column missingness with a disposition keyed on cause, not percentage.

    `structural` names columns where NaN encodes a real state: a null
    `end_date` means the subscription is still open, which is information.
    """
    structural = structural or {"subscriptions": ["end_date"]}

    def disposition(table, column, pct):
        if column in structural.get(table, []):
            return "structural — NaN is a state, encode as a flag"
        if pct > 60:
            return "drop — too sparse to impute honestly"
        return "impute in-fold" + (" + missing indicator" if pct > 20 else "")

    rows = [{"table": name, "column": col, "missing_pct": round(pct, 1),
             "disposition": disposition(name, col, pct)}
            for name, df in tables.items()
            for col, pct in (df.isna().mean() * 100).items() if pct > 0]
    return pd.DataFrame(rows).sort_values("missing_pct", ascending=False)


def run_all(X, y, df, tables, cutoff, raw_tables=None):
    """Full suite. Returns (results, passed).

    `collinear_pairs` and `missingness` are reported but not gated: they inform
    a judgement call rather than define a violation.
    """
    results = {
        "temporal_provenance": temporal_provenance(tables, cutoff),
        "forbidden_columns": forbidden_columns(X),
        "single_feature_auc": single_feature_auc(X, y),
        "perfect_separation": perfect_separation(X, y),
        "identifier_leakage": identifier_leakage(df, y),
        "duplicate_rows": duplicate_rows(X, df),
        "collinear_pairs": collinear_pairs(X),
        "constant_columns": constant_columns(X),
    }
    if raw_tables is not None:
        results["missingness"] = missingness_report(raw_tables)

    passed = (
        all(results[k]["pass"].all() for k in
            ("temporal_provenance", "identifier_leakage", "duplicate_rows"))
        and (results["single_feature_auc"]["auc"] < SINGLE_FEATURE_AUC_FAIL).all()
        and all(results[k].empty for k in
                ("perfect_separation", "forbidden_columns", "constant_columns"))
    )
    return results, passed
