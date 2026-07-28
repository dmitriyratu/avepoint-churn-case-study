"""Driver analysis: which features move the prediction, and is that real?

SHAP is the standard answer to "why does the model say that", and it will
happily produce a clean, confident, well-ordered bar chart from a model that
scores at chance. It has to: Shapley values decompose whatever the model
learned, and a model that learned noise still has an exact decomposition. The
attribution is correct; what it attributes is not signal.

So every importance measure here is paired with the same measure computed on
**shuffled labels**. The question is never "what is the top feature" but "is the
top feature further from the noise ceiling than noise usually gets", which is the
same standard `10_sanity_checks` applies to single-feature AUC and `09` applies
to model selection.

Two further checks that explainability write-ups usually skip:

- **Rank stability.** An importance ordering that reshuffles under a bootstrap
  resample is not a finding, however tight the bars look. Measured against the
  stability a shuffled-label ranking achieves, because *some* stability comes
  free from the feature distribution alone.
- **ALE instead of PDP.** Partial dependence marginalises by holding one feature
  fixed and averaging over the observed joint distribution of the rest, which
  evaluates the model on combinations that do not exist. With features
  correlated as strongly as these (`recency_ratio_90d` against
  `usage_last_30d`), that is extrapolation dressed as a curve. Accumulated local
  effects use the conditional distribution and stay on the data.
"""
import numpy as np
import pandas as pd
import shap
from scipy import stats
from sklearn.inspection import permutation_importance
from sklearn.model_selection import train_test_split
from sklearn.utils import resample

from .model import SEED, feature_names, model_ladder

TOP_K = 10


def _tree_model():
    """Rung 6 — LightGBM behind the shared preprocessor.

    The one-hot path rather than rung 7's native categoricals, because
    TreeExplainer wants a plain numeric matrix and the point here is the
    attribution, not the last 0.001 of AUC.
    """
    return model_ladder()[6][1]


def _fit_transform(estimator, X, y):
    """Fit the pipeline and hand back the encoded matrix its booster sees."""
    estimator.fit(X, y)
    encoded = estimator.named_steps["pre"].transform(X)
    return estimator, pd.DataFrame(encoded, columns=feature_names(estimator, X),
                                   index=X.index)


def fit_encoded(X, y, estimator=None):
    """Public form of the above: (fitted_booster, encoded_matrix).

    Callers that want to probe the model directly — an ALE curve, a
    counterfactual — need the bare estimator and the matrix in its encoded
    coordinates, since `ale` below perturbs encoded columns rather than raw ones.
    """
    fitted, encoded = _fit_transform(estimator or _tree_model(), X, y)
    return fitted.named_steps["clf"], encoded


def shap_importance(X, y, estimator=None):
    """Mean |SHAP| per encoded feature, plus the raw values for plotting."""
    estimator = estimator or _tree_model()
    fitted, encoded = _fit_transform(estimator, X, y)

    explainer = shap.TreeExplainer(fitted.named_steps["clf"])
    values = explainer.shap_values(encoded)
    # Binary LightGBM returns one matrix; some versions return one per class.
    if isinstance(values, list):
        values = values[1]
    if values.ndim == 3:
        values = values[:, :, 1]

    importance = pd.Series(np.abs(values).mean(axis=0), index=encoded.columns)
    return importance.sort_values(ascending=False), values, encoded


def shap_null(X, y, n_null=20, estimator=None, seed=SEED):
    """The same SHAP ranking, computed `n_null` times on shuffled labels.

    Returns the observed top importance alongside the null distribution of the
    top importance. A model fitted to noise still concentrates attribution on
    *some* feature; this measures how much.
    """
    observed, _, _ = shap_importance(X, y, estimator)
    rng = np.random.default_rng(seed)

    tops, rankings, ginis, overlaps = [], [], [], []
    observed_top_k = list(observed.head(TOP_K).index)
    for _ in range(n_null):
        shuffled = pd.Series(rng.permutation(y.values), index=y.index)
        null_importance, _, _ = shap_importance(X, shuffled, estimator)
        tops.append(null_importance.iloc[0])
        rankings.append(list(null_importance.head(TOP_K).index))
        ginis.append(concentration(null_importance))
        overlaps.append(_jaccard(observed_top_k, rankings[-1]))

    tops, ginis = np.array(tops), np.array(ginis)
    return {"observed_top_feature": observed.index[0],
            "observed_top_shap": round(float(observed.iloc[0]), 5),
            "null_top_mean": round(float(tops.mean()), 5),
            "null_top_p95": round(float(np.percentile(tops, 95)), 5),
            "p_value": round(float((tops >= observed.iloc[0]).mean()), 4),
            "observed_gini": round(concentration(observed), 4),
            "null_gini_mean": round(float(ginis.mean()), 4),
            "gini_p_value": round(float((ginis >= concentration(observed)).mean()), 4),
            # How much of the "real" top-10 a noise model reproduces by accident.
            "null_topk_overlap": round(float(np.mean(overlaps)), 4),
            "n_null": n_null,
            "null_rankings": rankings,
            "observed_ranking": observed_top_k}


def _jaccard(a, b):
    a, b = set(a), set(b)
    return len(a & b) / len(a | b) if a | b else 0.0


def rank_stability(X, y, n_boot=30, estimator=None, seed=SEED, shuffle=False):
    """How reproducible is the importance ranking under resampling?

    Bootstrap the rows, refit, re-rank, and measure agreement between every pair
    of runs — Spearman over the full ranking and Jaccard over the top-K.

    `shuffle=True` runs the identical procedure on permuted labels. That is the
    comparison that matters: a ranking driven by real structure should be
    *more* reproducible than one driven by noise, and if it is not, the apparent
    stability is coming from the feature distribution rather than the target.
    """
    estimator = estimator or _tree_model()
    rng = np.random.default_rng(seed)
    target = pd.Series(rng.permutation(y.values), index=y.index) if shuffle else y

    rankings, full = [], []
    for i in range(n_boot):
        idx = resample(np.arange(len(X)), random_state=seed + i)
        # A resample can drop one class entirely at 30% prevalence; skip rather
        # than fit a single-class model, and record how often it happened.
        if target.iloc[idx].nunique() < 2:
            continue
        importance, _, _ = shap_importance(X.iloc[idx], target.iloc[idx], estimator)
        rankings.append(list(importance.head(TOP_K).index))
        full.append(importance)

    jaccards = [_jaccard(rankings[i], rankings[j])
                for i in range(len(rankings)) for j in range(i + 1, len(rankings))]
    aligned = pd.concat(full, axis=1).fillna(0)
    spearmans = [stats.spearmanr(aligned.iloc[:, i], aligned.iloc[:, j]).statistic
                 for i in range(aligned.shape[1]) for j in range(i + 1, aligned.shape[1])]

    return {"labels": "shuffled" if shuffle else "observed",
            "n_fits": len(rankings),
            f"top{TOP_K}_jaccard": round(float(np.mean(jaccards)), 4),
            "spearman": round(float(np.mean(spearmans)), 4),
            "most_frequent_top1": pd.Series([r[0] for r in rankings]).value_counts().index[0],
            "distinct_top1": pd.Series([r[0] for r in rankings]).nunique()}


def permutation_null(X, y, n_repeats=20, n_null=10, estimator=None, seed=SEED):
    """Permutation importance on a held-out split, against a shuffled-label null.

    Model-agnostic and measured on data the model did not see, so unlike SHAP it
    reports importance *for generalisation* rather than importance *for the fit*.
    On a memorising model those are very different quantities — the boosters here
    reach train AUC 1.000 (notebook 08).
    """
    estimator = estimator or _tree_model()

    def run(target):
        X_tr, X_te, y_tr, y_te = train_test_split(
            X, target, test_size=0.3, stratify=target, random_state=seed)
        fitted = estimator.fit(X_tr, y_tr)
        result = permutation_importance(fitted, X_te, y_te, scoring="roc_auc",
                                        n_repeats=n_repeats, random_state=seed)
        return pd.Series(result.importances_mean, index=X.columns).sort_values(
            ascending=False)

    observed = run(y)
    rng = np.random.default_rng(seed)
    nulls = [run(pd.Series(rng.permutation(y.values), index=y.index)).iloc[0]
             for _ in range(n_null)]

    nulls = np.array(nulls)
    return observed, {"observed_top": round(float(observed.iloc[0]), 5),
                      "observed_top_feature": observed.index[0],
                      "null_top_mean": round(float(nulls.mean()), 5),
                      "null_top_p95": round(float(np.percentile(nulls, 95)), 5),
                      "p_value": round(float((nulls >= observed.iloc[0]).mean()), 4)}


def ale(estimator, X_encoded, feature, n_bins=10):
    """First-order accumulated local effects for one encoded feature.

    Within each quantile bin, replace the feature with the bin's two edges and
    take the *difference* in prediction for the points actually in that bin. That
    keeps every evaluation on a combination the data contains, which is precisely
    what partial dependence fails to do when features are correlated.

    Differences are accumulated across bins and centred, so the curve reads as
    "effect on the log-odds relative to the average account".
    """
    values = X_encoded[feature].values
    edges = np.unique(np.quantile(values, np.linspace(0, 1, n_bins + 1)))
    if len(edges) < 2:
        return pd.DataFrame(columns=["x", "ale", "n"])

    bins = np.clip(np.searchsorted(edges, values, side="left") - 1, 0, len(edges) - 2)
    effects = np.zeros(len(edges) - 1)
    counts = np.zeros(len(edges) - 1, dtype=int)

    for b in range(len(edges) - 1):
        mask = bins == b
        counts[b] = mask.sum()
        if not counts[b]:
            continue
        lo = X_encoded[mask].copy(); lo[feature] = edges[b]
        hi = X_encoded[mask].copy(); hi[feature] = edges[b + 1]
        # Log-odds rather than probability: differences are then additive, which
        # is what accumulating them across bins assumes.
        delta = (_logit(estimator.predict_proba(hi)[:, 1])
                 - _logit(estimator.predict_proba(lo)[:, 1]))
        effects[b] = delta.mean()

    curve = np.concatenate([[0], np.cumsum(effects)])
    weights = np.concatenate([[0], counts])
    centred = curve - np.average(curve, weights=weights if weights.sum() else None)
    return pd.DataFrame({"x": edges, "ale": centred, "n": weights})


def _logit(p, eps=1e-6):
    p = np.clip(p, eps, 1 - eps)
    return np.log(p / (1 - p))


def concentration(importance):
    """Gini coefficient of the importance vector.

    A real driver structure concentrates attribution on a few features; noise
    spreads it. One number to compare observed against null, complementing the
    top-feature test which only looks at the maximum.
    """
    values = np.sort(np.abs(importance.values))
    n = len(values)
    if n == 0 or values.sum() == 0:
        return np.nan
    index = np.arange(1, n + 1)
    return float((2 * (index * values).sum()) / (n * values.sum()) - (n + 1) / n)


__all__ = ["shap_importance", "shap_null", "rank_stability", "permutation_null",
           "ale", "concentration", "fit_encoded", "TOP_K"]
