# %% [markdown]
# # 05 — Results & Validation
#
# Deeper look at model performance, SHAP interpretability, and business recommendations.
#
# - SHAP global + local explanations
# - Segment-level performance (do we predict worse for certain cohorts?)
# - Business impact estimation
# - Strategic recommendations

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

from sklearn.model_selection import train_test_split
from sklearn.metrics import roc_auc_score, average_precision_score
import shap

from src.model import prep_xy, load_model
from src.evaluate import shap_summary, plot_roc_pr, plot_confusion

df = pd.read_csv("../data/processed/features.csv")
X, y = prep_xy(df)
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, stratify=y, random_state=42
)

lgb_model = load_model("lgb_churn")
xgb_model = load_model("xgb_churn")

with open("../outputs/models/config.json") as f:
    config = json.load(f)
threshold = config["lgb_threshold"]

lgb_proba = lgb_model.predict_proba(X_test)[:, 1]
lgb_pred = (lgb_proba >= threshold).astype(int)

print(f"Test set: {X_test.shape}  |  Churn rate: {y_test.mean():.3f}")
print(f"LightGBM AUC: {roc_auc_score(y_test, lgb_proba):.4f}")

# %% [markdown]
# ## 1. SHAP Analysis

# %%
# Global SHAP — what does the model actually rely on?
sv = shap_summary(lgb_model, X_test, model_name="lgb")

# %%
# SHAP dependence: feature_breadth vs churn (our strongest engagement signal)
shap_vals_arr = sv  # already positive class values

fig, ax = plt.subplots(figsize=(7, 5))
shap.dependence_plot(
    "feature_breadth", shap_vals_arr, X_test,
    interaction_index="avg_satisfaction", ax=ax, show=False
)
ax.set_title("SHAP Dependence: Feature Breadth\n(color = avg satisfaction score)")
plt.tight_layout()
plt.savefig("../outputs/figures/05_shap_dependence_breadth.png", bbox_inches="tight")
plt.show()

# %%
# example: waterfall for a high-risk account
test_df = X_test.copy()
test_df["churn_proba"] = lgb_proba
test_df["actual"] = y_test.values

high_risk = test_df[test_df["churn_proba"] > 0.7]
print(f"High-risk accounts (p > 0.7): {len(high_risk)}")

# show the top-risk one
risk_idx = test_df["churn_proba"].idxmax()
local_idx = test_df.index.get_loc(risk_idx)
print(f"\nHighest risk account: churn_proba = {test_df.loc[risk_idx, 'churn_proba']:.3f}, actual = {test_df.loc[risk_idx, 'actual']}")

explainer = shap.TreeExplainer(lgb_model)
explanation = explainer(X_test)
# pick positive class
if len(explanation.shape) == 3:
    explanation = explanation[:, :, 1]
shap.waterfall_plot(explanation[local_idx], show=False)
plt.tight_layout()
plt.savefig("../outputs/figures/05_shap_waterfall_highrisk.png", bbox_inches="tight")
plt.show()

# %% [markdown]
# ## 2. Segment-level Performance
#
# Does the model work equally well across plan tiers and industries?

# %%
test_full = df.iloc[X_test.index].copy()
test_full["churn_proba"] = lgb_proba
test_full["predicted"] = lgb_pred
test_full["correct"] = (lgb_pred == y_test.values).astype(int)

# %%
# AUC by plan tier
for tier in test_full["plan_tier"].dropna().unique():
    mask = test_full["plan_tier"] == tier
    if mask.sum() < 5:
        continue
    auc = roc_auc_score(test_full.loc[mask, "churn_flag"], test_full.loc[mask, "churn_proba"])
    n = mask.sum()
    n_churn = test_full.loc[mask, "churn_flag"].sum()
    print(f"  {tier:12s}  AUC={auc:.3f}  n={n}  churned={n_churn}")

# %%
# calibration check
from sklearn.calibration import calibration_curve

prob_true, prob_pred = calibration_curve(y_test, lgb_proba, n_bins=8)
fig, ax = plt.subplots(figsize=(6, 5))
ax.plot(prob_pred, prob_true, marker="o", label="LightGBM")
ax.plot([0, 1], [0, 1], "k--", alpha=0.5, label="Perfect calibration")
ax.set_xlabel("Mean Predicted Probability")
ax.set_ylabel("Fraction of Positives")
ax.set_title("Calibration Curve")
ax.legend()
plt.tight_layout()
plt.savefig("../outputs/figures/05_calibration.png", bbox_inches="tight")
plt.show()

# %% [markdown]
# ## 3. Business Impact Estimation
#
# Framing: what's the value of catching churners early?
#
# Assumptions (to discuss with stakeholders):
# - Average account MRR: ~$2,300
# - Intervention cost per at-risk account: $50 (outreach, discount, CSM time)
# - Intervention success rate: 30% (i.e., we retain 30% of accounts we reach out to)

# %%
# test set results
total_test = len(y_test)
actual_churners = y_test.sum()
caught_churners = (lgb_pred & y_test).sum()   # true positives
missed_churners = ((1 - lgb_pred) & y_test).sum()  # false negatives
false_alarms = (lgb_pred & (1 - y_test)).sum()  # false positives

print(f"Test set: {total_test} accounts, {actual_churners} actual churners")
print(f"  Caught:       {caught_churners}  (TP)")
print(f"  Missed:       {missed_churners}  (FN)")
print(f"  False alarms: {false_alarms}  (FP)")

# %%
avg_mrr = 2300
intervention_cost = 50
retention_rate = 0.30

mrr_saved = caught_churners * avg_mrr * retention_rate
intervention_spend = (caught_churners + false_alarms) * intervention_cost
net_value = mrr_saved - intervention_spend

print(f"\n--- Business Impact (Test Set) ---")
print(f"  MRR saved via interventions:  ${mrr_saved:,.0f}")
print(f"  Intervention spend:           ${intervention_spend:,.0f}")
print(f"  Net monthly value:            ${net_value:,.0f}")
print(f"\nAt 500 accounts (full), scale by {500/total_test:.1f}x -> est. net value: ${net_value*(500/total_test):,.0f}/mo")

# %% [markdown]
# ## 4. Strategic Recommendations
#
# Based on EDA + SHAP analysis, three actionable levers:

# %% [markdown]
# ### Recommendation 1: Feature Adoption Program
#
# SHAP shows `feature_breadth` is the single strongest predictor.
# Accounts using fewer than 35% of available features (14/40) are ~2.3x more likely to churn.
#
# **Action**: Trigger a proactive in-app onboarding sequence when an account's 90-day breadth
# falls below the 30th percentile. Pair with a CSM outreach for Enterprise accounts.
#
# **Testing approach**: A/B test — 50% of flagged accounts get the nudge campaign vs. control.
# Primary metric: 90-day retention. Secondary: feature breadth increase.

# %%
# support the recommendation with data
breadth_col = "feature_breadth" if "feature_breadth" in df.columns else None
if breadth_col:
    low_breadth = df["feature_breadth"] < 0.35
    print("Churn rate — low breadth (<35%): ", df.loc[low_breadth, "churn_flag"].mean().round(3))
    print("Churn rate — high breadth (>=35%):", df.loc[~low_breadth, "churn_flag"].mean().round(3))

# %% [markdown]
# ### Recommendation 2: Support Experience as Early Warning
#
# `escalation_rate` and `avg_resolution_hours` consistently appear in top SHAP features.
# Accounts with high escalation rates churn at nearly 2x the base rate.
#
# **Action**: Flag accounts where escalation_rate > 0.4 for priority CSM review.
# Also: satisfaction score below 2.5 after ticket resolution should auto-trigger a follow-up.
#
# **Testing approach**: Monitor 60-day churn rate for accounts that receive the follow-up
# vs. those that don't (propensity-score match to avoid selection bias).

# %%
if "escalation_rate" in df.columns:
    esc_high = df["escalation_rate"] > 0.4
    print("Churn rate — high escalation: ", df.loc[esc_high, "churn_flag"].mean().round(3))
    print("Churn rate — low escalation:  ", df.loc[~esc_high, "churn_flag"].mean().round(3))

# %% [markdown]
# ### Recommendation 3: Pricing / Downgrade Signal
#
# `n_downgrades` and `sub_churn_rate` are strong predictors.
# A downgrade event is often a precursor to full churn — but the gap varies.
#
# **Action**: When a downgrade is detected, trigger a product-value review call within 7 days.
# For Basic-tier accounts on month-to-month billing, offer a discounted annual commitment.
#
# **Testing approach**: Run holdout experiment — intervene with 50% of downgrade events.
# Measure 6-month retention and net revenue change.

# %%
if "n_downgrades" in df.columns:
    has_downgrade = df["n_downgrades"] > 0
    print("Churn rate — had a downgrade:    ", df.loc[has_downgrade, "churn_flag"].mean().round(3))
    print("Churn rate — no downgrade:       ", df.loc[~has_downgrade, "churn_flag"].mean().round(3))

# %% [markdown]
# ## 5. Deployment Architecture (High-Level)

# %% [markdown]
# ```
# ┌─────────────────────────────────────────────────────────┐
# │  Data Layer                                             │
# │  - CRM / billing → daily account snapshot              │
# │  - Product events → feature usage aggregations         │
# │  - Support system → ticket metrics                      │
# └────────────────┬────────────────────────────────────────┘
#                  │ nightly ETL
# ┌────────────────▼────────────────────────────────────────┐
# │  Feature Store (e.g. Feast / Tecton)                    │
# │  - Pre-computed account-level features                  │
# │  - 30d, 60d, 90d rolling windows                        │
# └────────────────┬────────────────────────────────────────┘
#                  │
# ┌────────────────▼────────────────────────────────────────┐
# │  Scoring Service (FastAPI or batch)                     │
# │  - Load lgb_churn.joblib                                │
# │  - Predict daily for all active accounts                │
# │  - Write risk scores to internal DB                     │
# └────────────────┬────────────────────────────────────────┘
#                  │
# ┌────────────────▼────────────────────────────────────────┐
# │  Action Layer                                           │
# │  - Push risk scores to CRM (Salesforce)                 │
# │  - Trigger Intercom campaigns for medium-risk           │
# │  - CSM queue for high-risk Enterprise accounts          │
# └─────────────────────────────────────────────────────────┘
# ```
#
# **Monitoring**:
# - Weekly: feature distribution drift (PSI per feature)
# - Monthly: AUC on closed accounts (delayed labels)
# - Retrain trigger: AUC drops >0.03 from baseline or significant drift in top-5 features
# - Log predictions + actuals → build labeled dataset for retraining

# %% [markdown]
# ## 6. Summary

# %%
print("=== Model Performance Summary ===")
print(f"LightGBM AUC (CV mean): see notebook 04")
print(f"LightGBM AUC (test):    {roc_auc_score(y_test, lgb_proba):.4f}")
print(f"AP (test):               {average_precision_score(y_test, lgb_proba):.4f}")
print(f"Threshold used:          {threshold}")

from sklearn.metrics import recall_score, precision_score, f1_score
print(f"\nAt threshold={threshold}:")
print(f"  Recall:    {recall_score(y_test, lgb_pred):.3f}")
print(f"  Precision: {precision_score(y_test, lgb_pred, zero_division=0):.3f}")
print(f"  F1:        {f1_score(y_test, lgb_pred, zero_division=0):.3f}")

# %%
# save final metrics
metrics = {
    "roc_auc_test": round(roc_auc_score(y_test, lgb_proba), 4),
    "avg_precision_test": round(average_precision_score(y_test, lgb_proba), 4),
    "recall_test": round(recall_score(y_test, lgb_pred), 4),
    "precision_test": round(precision_score(y_test, lgb_pred, zero_division=0), 4),
    "f1_test": round(f1_score(y_test, lgb_pred, zero_division=0), 4),
    "threshold": threshold,
}
pd.DataFrame([metrics]).to_csv("../outputs/reports/final_metrics.csv", index=False)
print("\nMetrics saved to outputs/reports/final_metrics.csv")
