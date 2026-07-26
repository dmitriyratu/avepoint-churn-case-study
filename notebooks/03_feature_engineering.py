# %% [markdown]
# # 03 — Feature Engineering
#
# Build a single flat feature table at the account level for modeling.
# All tables aggregate up to `account_id`.
#
# Feature groups:
# - **Subscription signals**: MRR, tenure, upgrade/downgrade history
# - **Engagement signals**: feature breadth, usage volume, error rate
# - **Support signals**: ticket volume, satisfaction, escalations
# - **Account metadata**: plan tier, industry, country, referral source
# - **Derived cross-table signals**: usage per seat, MRR per seat

# %%
import sys
sys.path.append("..")

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

from src.load_data import load_all
from src.clean import clean_all
from src.features import build_model_dataset, subscription_features, feature_usage_features, support_features

tables = clean_all(load_all())

# %%
# build each feature group independently first so we can inspect them
sub_feats = subscription_features(tables["subscriptions"])
print("Subscription features:", sub_feats.shape)
sub_feats.describe().T

# %%
usage_feats = feature_usage_features(tables["feature_usage"], tables["subscriptions"])
print("Usage features:", usage_feats.shape)
usage_feats.describe().T

# %%
support_feats = support_features(tables["support_tickets"])
print("Support features:", support_feats.shape)
support_feats.describe().T

# %% [markdown]
# ### Coverage check — not every account has data in every table

# %%
acc_ids = set(tables["accounts"]["account_id"])
print("Accounts in subscriptions:", sub_feats["account_id"].isin(acc_ids).sum(), "/ 500")
print("Accounts in usage:        ", usage_feats["account_id"].isin(acc_ids).sum(), "/ 500")
print("Accounts in tickets:      ", support_feats["account_id"].isin(acc_ids).sum(), "/ 500")

# Some accounts may have no ticket history — that's fine, they'll just get 0s after the merge.

# %%
# build the full dataset
df = build_model_dataset(tables)
print("Final dataset shape:", df.shape)
df.head(3)

# %%
# how many columns in each group
feature_groups = {
    "account_meta": ["seats", "is_trial", "days_since_signup"],
    "subscription": [c for c in df.columns if any(x in c for x in ["n_sub", "mrr", "upgrade", "downgrade", "auto_renew", "tenure", "sub_churn", "upgrade_net"])],
    "usage": [c for c in df.columns if any(x in c for x in ["usage", "feature", "error", "beta", "breadth"])],
    "support": [c for c in df.columns if any(x in c for x in ["ticket", "resolution", "response", "satisfaction", "escalat", "urgent"])],
    "one_hot": [c for c in df.columns if any(c.startswith(p) for p in ["industry_", "country_", "referral_", "plan_tier_", "latest_plan_", "billing_"])],
}
for g, cols in feature_groups.items():
    print(f"{g:15s}: {len(cols)} features")

# %% [markdown]
# ## Feature distributions and target correlations

# %%
target = "churn_flag"
y = df[target].astype(int)

# numeric features only for correlation
num_cols = df.select_dtypes(include=[np.number]).columns.drop([target])
corr = df[num_cols].corrwith(y.astype(float)).sort_values(key=abs, ascending=False)

fig, ax = plt.subplots(figsize=(8, 8))
corr.head(25).plot(kind="barh", ax=ax, color=["salmon" if v > 0 else "steelblue" for v in corr.head(25)])
ax.axvline(0, color="black", linewidth=0.8)
ax.set_title("Top 25 Features by Correlation with Churn")
ax.set_xlabel("Pearson r with churn_flag")
plt.tight_layout()
plt.savefig("../outputs/figures/03_feature_correlations.png", bbox_inches="tight")
plt.show()

# %%
# look at the strongest signals more closely
top_feats = corr.abs().head(8).index.tolist()

fig, axes = plt.subplots(2, 4, figsize=(16, 7))
for ax, col in zip(axes.flat, top_feats):
    df.boxplot(column=col, by=target, ax=ax)
    ax.set_title(col, fontsize=9)
    ax.set_xlabel("Churned")
plt.suptitle("Top Features vs Churn Flag", y=1.02)
plt.tight_layout()
plt.savefig("../outputs/figures/03_top_features_boxplot.png", bbox_inches="tight")
plt.show()

# %% [markdown]
# ## Missing value check on the final feature table

# %%
missing = df.isna().sum()
missing_pct = (missing / len(df) * 100).round(1)
missing_df = pd.DataFrame({"missing_n": missing, "missing_pct": missing_pct})
print(missing_df[missing_df["missing_n"] > 0].sort_values("missing_n", ascending=False))

# %%
# fill remaining nulls (accounts with no tickets or no usage get 0s)
fill_zero = ["n_tickets", "avg_resolution_hours", "avg_first_response_mins",
             "avg_satisfaction", "n_escalations", "urgent_pct", "escalation_rate",
             "total_usage_events", "unique_features_used", "total_usage_duration_mins",
             "total_errors", "beta_feature_pct", "avg_usage_count", "error_rate",
             "feature_breadth", "usage_per_seat", "tickets_per_seat", "mrr_per_seat"]

for col in fill_zero:
    if col in df.columns:
        df[col] = df[col].fillna(0)

print("Remaining nulls:", df.isna().sum().sum())

# %% [markdown]
# ## Save feature table

# %%
df.to_csv("../data/processed/features.csv", index=False)
print("Saved features.csv —", df.shape)
print("\nTarget distribution:")
print(df["churn_flag"].value_counts(normalize=True).round(3))
