import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import seaborn as sns
from pathlib import Path

from sklearn.metrics import (
    roc_curve, auc, precision_recall_curve,
    average_precision_score, confusion_matrix,
    classification_report, roc_auc_score,
)
import shap

FIGS_DIR = Path(__file__).parents[1] / "outputs" / "figures"
FIGS_DIR.mkdir(parents=True, exist_ok=True)

plt.rcParams.update({"figure.dpi": 130, "font.size": 11})


def find_best_threshold(y_true, y_proba, metric="f1"):
    thresholds = np.linspace(0.1, 0.9, 80)
    scores = []
    for t in thresholds:
        pred = (y_proba >= t).astype(int)
        from sklearn.metrics import f1_score, recall_score, precision_score
        if metric == "f1":
            s = f1_score(y_true, pred, zero_division=0)
        elif metric == "recall":
            s = recall_score(y_true, pred, zero_division=0)
        scores.append(s)
    best_t = thresholds[np.argmax(scores)]
    return round(float(best_t), 3), round(float(max(scores)), 4)


def plot_roc_pr(y_true, probas_dict, save_name="roc_pr"):
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))

    for label, proba in probas_dict.items():
        fpr, tpr, _ = roc_curve(y_true, proba)
        roc_auc = auc(fpr, tpr)
        axes[0].plot(fpr, tpr, lw=2, label=f"{label}  (AUC = {roc_auc:.3f})")

        prec, rec, _ = precision_recall_curve(y_true, proba)
        ap = average_precision_score(y_true, proba)
        axes[1].plot(rec, prec, lw=2, label=f"{label}  (AP = {ap:.3f})")

    axes[0].plot([0, 1], [0, 1], "k--", alpha=0.4)
    axes[0].set_xlabel("False Positive Rate")
    axes[0].set_ylabel("True Positive Rate")
    axes[0].set_title("ROC Curve")
    axes[0].legend(loc="lower right")

    baseline = y_true.mean()
    axes[1].axhline(baseline, color="k", linestyle="--", alpha=0.4,
                    label=f"Baseline ({baseline:.2f})")
    axes[1].set_xlabel("Recall")
    axes[1].set_ylabel("Precision")
    axes[1].set_title("Precision-Recall Curve")
    axes[1].legend(loc="upper right")

    plt.tight_layout()
    fig.savefig(FIGS_DIR / f"{save_name}.png", bbox_inches="tight")
    plt.show()
    return fig


def plot_confusion(y_true, y_pred, label="model"):
    cm = confusion_matrix(y_true, y_pred)
    tn, fp, fn, tp = cm.ravel()

    fig, ax = plt.subplots(figsize=(5, 4))
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", ax=ax,
                xticklabels=["Retained", "Churned"],
                yticklabels=["Retained", "Churned"])
    ax.set_xlabel("Predicted")
    ax.set_ylabel("Actual")
    ax.set_title(f"Confusion Matrix — {label}")
    plt.tight_layout()
    fig.savefig(FIGS_DIR / f"confusion_{label.lower().replace(' ', '_')}.png", bbox_inches="tight")
    plt.show()

    print(f"\n  True positives  (caught churners): {tp}")
    print(f"  False negatives (missed churners): {fn}")
    print(f"  False positives (false alarms):    {fp}")
    print(f"  True negatives  (correct retains): {tn}")
    return fig


def shap_summary(model, X, model_name="lgb", max_display=20, save=True):
    explainer = shap.TreeExplainer(model)
    shap_vals = explainer.shap_values(X)
    # lgb binary returns list[neg, pos], xgb returns 2D array
    sv = shap_vals[1] if isinstance(shap_vals, list) else shap_vals

    plt.figure(figsize=(9, 7))
    shap.summary_plot(sv, X, max_display=max_display, show=False)
    plt.tight_layout()
    if save:
        plt.savefig(FIGS_DIR / f"shap_{model_name}.png", bbox_inches="tight")
    plt.show()
    return sv


def shap_waterfall(model, X, idx=0, model_name="lgb", save=True):
    explainer = shap.TreeExplainer(model)
    explanation = explainer(X)
    # pick the positive-class slice for lgb
    if len(explanation.shape) == 3:
        explanation = explanation[:, :, 1]
    shap.waterfall_plot(explanation[idx], show=False)
    if save:
        plt.savefig(FIGS_DIR / f"shap_waterfall_{model_name}_{idx}.png", bbox_inches="tight")
    plt.show()


def cv_summary(cv_df):
    summary = cv_df[["roc_auc", "avg_precision", "f1"]].agg(["mean", "std"])
    print(summary.round(4).to_string())
    return summary
