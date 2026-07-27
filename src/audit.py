"""Automated leakage and data-quality gates.

Reasoning about leakage catches the obvious cases; these tests catch the rest.
The ticket-censoring leak in this project was found by `temporal_provenance`,
not by reading the code.

Every check returns a frame with a `pass` column so the suite can be asserted in
CI rather than eyeballed.
"""
import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score

# A single column that separates the classes this well is almost always leakage,
# not a discovery.
SINGLE_FEATURE_AUC_WARN = 0.70
SINGLE_FEATURE_AUC_FAIL = 0.80
COLLINEARITY_THRESHOLD = 0.95


def temporal_provenance(truncated_tables, cutoff):
    """No row feeding a feature may carry a timestamp at or after the cutoff.

    Checks every datetime column, not just the one used for filtering — a table
    filtered on `submitted_at` can still carry a `closed_at` in the future.
    """
    rows = []
    for tname, t in truncated_tables.items():
        for c in t.columns:
            if not pd.api.types.is_datetime64_any_dtype(t[c]):
                continue
            v = int((t[c] >= cutoff).sum())
            rows.append({
                "table": tname, "column": c,
                "max_value": str(t[c].max())[:10],
                "violations": v, "pass": v == 0,
            })
    return pd.DataFrame(rows)


def single_feature_auc(X, y):
    """Rank columns by standalone discriminative power."""
    rows = []
    for c in X.columns:
        v = pd.to_numeric(X[c], errors="coerce").astype(float)
        if v.notna().sum() == 0 or np.nanstd(v) == 0:
            continue
        v = v.fillna(v.median())
        a = roc_auc_score(y, v)
        rows.append({"feature": c, "auc": round(max(a, 1 - a), 4)})
    out = pd.DataFrame(rows).sort_values("auc", ascending=False).reset_index(drop=True)
    out["verdict"] = np.where(out["auc"] >= SINGLE_FEATURE_AUC_FAIL, "FAIL — near-certain leak",
                       np.where(out["auc"] >= SINGLE_FEATURE_AUC_WARN, "WARN — inspect", "ok"))
    return out


def perfect_separation(X, y, max_levels=10):
    """Low-cardinality columns where every level maps to exactly one class."""
    rows = []
    for c in X.columns:
        u = X[c].nunique()
        if 1 < u <= max_levels:
            ct = pd.crosstab(X[c], y)
            pure = ((ct > 0).sum(axis=1) == 1).all()
            if pure:
                rows.append({"feature": c, "levels": int(u), "pass": False})
    return pd.DataFrame(rows) if rows else pd.DataFrame(columns=["feature", "levels", "pass"])


def identifier_leakage(df, y, id_col="account_id"):
    """Ids and row order must not predict the target."""
    rows = []
    if id_col in df.columns:
        num = df[id_col].astype(str).str.extract(r"([0-9a-fA-F]+)")[0]
        num = num.apply(lambda s: int(s, 16) if isinstance(s, str) else np.nan)
        if num.notna().any():
            a = roc_auc_score(y, num.fillna(num.median()))
            rows.append({"probe": f"{id_col} as integer", "auc": round(max(a, 1 - a), 4)})
    a = roc_auc_score(y, np.arange(len(y)))
    rows.append({"probe": "row order", "auc": round(max(a, 1 - a), 4)})
    out = pd.DataFrame(rows)
    out["pass"] = out["auc"] < 0.60
    return out


def duplicate_rows(X, df=None, id_col="account_id"):
    rows = [{"check": "identical feature vectors", "n": int(X.duplicated().sum())}]
    if df is not None and id_col in df.columns:
        rows.append({"check": f"duplicate {id_col}", "n": int(df[id_col].duplicated().sum())})
    out = pd.DataFrame(rows)
    out["pass"] = out["n"] == 0
    return out


def collinear_pairs(X, threshold=COLLINEARITY_THRESHOLD):
    num = X.select_dtypes(include=[np.number])
    cm = num.corr().abs()
    cols = list(cm.columns)
    rows = []
    for i, a in enumerate(cols):
        for b in cols[i + 1:]:
            v = cm.loc[a, b]
            if pd.notna(v) and v > threshold:
                rows.append({"feature_a": a, "feature_b": b, "abs_r": round(float(v), 4)})
    return pd.DataFrame(rows).sort_values("abs_r", ascending=False) if rows else \
        pd.DataFrame(columns=["feature_a", "feature_b", "abs_r"])


def constant_columns(X):
    rows = [{"feature": c, "n_unique": int(X[c].nunique())}
            for c in X.columns if X[c].nunique() <= 1]
    return pd.DataFrame(rows) if rows else pd.DataFrame(columns=["feature", "n_unique"])


def missingness_report(tables, structural=None):
    """Per-column missingness with an explicit disposition.

    `structural` names columns where NaN encodes a real state rather than an
    absent measurement — `subscriptions.end_date` is 90% null because those
    subscriptions are still open, which is information, not a gap.
    """
    structural = structural or {"subscriptions": ["end_date"]}
    rows = []
    for tname, t in tables.items():
        for c in t.columns:
            pct = float(t[c].isna().mean() * 100)
            if pct == 0:
                continue
            if c in structural.get(tname, []):
                disp = "structural — NaN is a state, encode as a flag"
            elif pct > 60:
                disp = "drop — too sparse to impute honestly"
            elif pct > 20:
                disp = "impute in-fold + missing indicator"
            else:
                disp = "impute in-fold"
            rows.append({"table": tname, "column": c,
                         "missing_pct": round(pct, 1), "disposition": disp})
    return pd.DataFrame(rows).sort_values("missing_pct", ascending=False)


def run_all(X, y, df, truncated_tables, cutoff, raw_tables=None):
    """Full suite. Returns (results dict, all_passed)."""
    res = {
        "temporal_provenance": temporal_provenance(truncated_tables, cutoff),
        "single_feature_auc": single_feature_auc(X, y),
        "perfect_separation": perfect_separation(X, y),
        "identifier_leakage": identifier_leakage(df, y),
        "duplicate_rows": duplicate_rows(X, df),
        "collinear_pairs": collinear_pairs(X),
        "constant_columns": constant_columns(X),
    }
    if raw_tables is not None:
        res["missingness"] = missingness_report(raw_tables)

    passed = (
        bool(res["temporal_provenance"]["pass"].all())
        and (res["single_feature_auc"]["auc"] < SINGLE_FEATURE_AUC_FAIL).all()
        and len(res["perfect_separation"]) == 0
        and bool(res["identifier_leakage"]["pass"].all())
        and bool(res["duplicate_rows"]["pass"].all())
        and len(res["constant_columns"]) == 0
    )
    return res, passed
