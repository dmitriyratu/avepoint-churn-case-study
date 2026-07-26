# %% [markdown]
# # 01 — Exploratory Data Analysis
#
# Five-table relational dataset from Kaggle (Rivalytics / RavenStack).
# Goal: understand churn drivers before touching the model.
#
# Tables:
# - `accounts` — one row per customer, has `churn_flag` (target)
# - `subscriptions` — subscription history (many per account)
# - `feature_usage` — per-feature usage logs (via subscription_id)
# - `support_tickets` — support interactions
# - `churn_events` — logged churn reasons

# %%
import sys
sys.path.append("..")

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import seaborn as sns

from src.load_data import load_all
from src.clean import clean_all

sns.set_theme(style="whitegrid", palette="muted")
pd.set_option("display.max_columns", 40)

tables = clean_all(load_all())
acc = tables["accounts"]
subs = tables["subscriptions"]
usage = tables["feature_usage"]
tickets = tables["support_tickets"]
churn_ev = tables["churn_events"]

# %% [markdown]
# ## 1. Accounts overview

# %%
print(acc.shape)
acc.head()

# %%
print(f"Churn rate: {acc['churn_flag'].mean():.1%}")
print(f"Trial accounts: {acc['is_trial'].mean():.1%}")
acc["churn_flag"].value_counts()

# %%
fig, axes = plt.subplots(1, 3, figsize=(14, 4))

acc["plan_tier"].value_counts().plot(kind="bar", ax=axes[0], color=sns.color_palette("muted"))
axes[0].set_title("Accounts by Plan")
axes[0].tick_params(axis="x", rotation=0)

acc["industry"].value_counts().plot(kind="barh", ax=axes[1])
axes[1].set_title("Accounts by Industry")

acc["country"].value_counts().head(7).plot(kind="bar", ax=axes[2])
axes[2].set_title("Top Countries")
axes[2].tick_params(axis="x", rotation=45)

plt.tight_layout()
plt.savefig("../outputs/figures/01_account_distributions.png", bbox_inches="tight")
plt.show()

# %%
# churn rate by plan tier and industry
fig, axes = plt.subplots(1, 2, figsize=(12, 4))

(acc.groupby("plan_tier")["churn_flag"].mean().sort_values()
 .plot(kind="barh", ax=axes[0], color="steelblue"))
axes[0].axvline(acc["churn_flag"].mean(), color="red", linestyle="--", alpha=0.7, label="Overall")
axes[0].set_title("Churn Rate by Plan Tier")
axes[0].set_xlabel("Churn Rate")
axes[0].legend()

(acc.groupby("industry")["churn_flag"].mean().sort_values()
 .plot(kind="barh", ax=axes[1], color="coral"))
axes[1].axvline(acc["churn_flag"].mean(), color="red", linestyle="--", alpha=0.7)
axes[1].set_title("Churn Rate by Industry")
axes[1].set_xlabel("Churn Rate")

plt.tight_layout()
plt.savefig("../outputs/figures/01_churn_by_segment.png", bbox_inches="tight")
plt.show()

# %%
# signup cohorts
acc["signup_month"] = acc["signup_date"].dt.to_period("M")
cohort_churn = acc.groupby("signup_month")["churn_flag"].agg(["mean", "count"])
cohort_churn.columns = ["churn_rate", "n_accounts"]

fig, ax1 = plt.subplots(figsize=(12, 4))
ax2 = ax1.twinx()
cohort_churn["n_accounts"].plot(kind="bar", ax=ax1, alpha=0.4, color="gray", label="# Accounts")
cohort_churn["churn_rate"].plot(ax=ax2, color="red", marker="o", markersize=4, label="Churn Rate")
ax1.set_title("Signup Cohorts — Volume & Churn Rate")
ax1.set_ylabel("# Accounts")
ax2.set_ylabel("Churn Rate")
ax1.tick_params(axis="x", rotation=45)
plt.tight_layout()
plt.savefig("../outputs/figures/01_cohort_churn.png", bbox_inches="tight")
plt.show()

# %% [markdown]
# ## 2. Subscriptions

# %%
print(subs.shape)
subs.describe(include="all").T.head(20)

# %%
# active vs. ended subs
print("Active subscriptions (no end_date):", subs["end_date"].isna().sum())
print("MRR = 0:", (subs["mrr_amount"] == 0).sum())

# %%
fig, axes = plt.subplots(1, 2, figsize=(12, 4))

subs["mrr_amount"].clip(upper=10000).hist(bins=40, ax=axes[0], edgecolor="white")
axes[0].set_title("MRR Distribution (clipped at $10k)")
axes[0].set_xlabel("MRR ($)")

(subs.groupby("account_id").size()
 .hist(bins=20, ax=axes[1], edgecolor="white", color="coral"))
axes[1].set_title("Subscriptions per Account")
axes[1].set_xlabel("# Subscriptions")

plt.tight_layout()
plt.savefig("../outputs/figures/01_subscription_dist.png", bbox_inches="tight")
plt.show()

# %%
# churn rate by plan tier in subscriptions
subs_churn = (subs.groupby("plan_tier")["churn_flag"].agg(["mean", "count"])
              .rename(columns={"mean": "churn_rate", "count": "n_subs"})
              .sort_values("churn_rate"))
print(subs_churn)

# %%
# upgrade/downgrade dynamics
print("Accounts with at least one upgrade:",
      subs.groupby("account_id")["upgrade_flag"].any().sum())
print("Accounts with at least one downgrade:",
      subs.groupby("account_id")["downgrade_flag"].any().sum())

# %% [markdown]
# ## 3. Feature Usage

# %%
print(usage.shape)
usage.head()

# %%
# usage breadth per account (via subscription bridge)
bridge = subs[["subscription_id", "account_id"]].drop_duplicates()
u = usage.merge(bridge, on="subscription_id", how="left").dropna(subset=["account_id"])

feature_freq = (u.groupby("feature_name").size()
                 .sort_values(ascending=False)
                 .reset_index(name="usage_events"))

fig, ax = plt.subplots(figsize=(14, 4))
sns.barplot(data=feature_freq.head(20), x="feature_name", y="usage_events", ax=ax)
ax.set_title("Top 20 Features by Usage Volume")
ax.tick_params(axis="x", rotation=45)
plt.tight_layout()
plt.savefig("../outputs/figures/01_feature_usage_top20.png", bbox_inches="tight")
plt.show()

# %%
# usage breadth vs churn
breadth = (u.groupby("account_id")["feature_name"].nunique()
            .reset_index(name="unique_features"))
breadth = breadth.merge(acc[["account_id", "churn_flag"]], on="account_id")

fig, ax = plt.subplots(figsize=(7, 4))
breadth.groupby("churn_flag")["unique_features"].plot(
    kind="hist", bins=20, alpha=0.6, ax=ax, legend=True
)
ax.set_title("Feature Breadth: Churned vs Retained")
ax.set_xlabel("Unique Features Used")
ax.legend(["Retained (0)", "Churned (1)"])
plt.tight_layout()
plt.savefig("../outputs/figures/01_breadth_vs_churn.png", bbox_inches="tight")
plt.show()

print("Mean unique features — retained:",
      breadth.loc[breadth["churn_flag"] == False, "unique_features"].mean().round(1))
print("Mean unique features — churned:",
      breadth.loc[breadth["churn_flag"] == True, "unique_features"].mean().round(1))

# %%
# error rate analysis
err = (u.groupby("account_id")
       .agg(total_errors=("error_count", "sum"), total_events=("usage_id", "count"))
       .assign(error_rate=lambda x: x["total_errors"] / x["total_events"])
       .reset_index())
err = err.merge(acc[["account_id", "churn_flag"]], on="account_id")
print(err.groupby("churn_flag")["error_rate"].describe().round(4))

# %% [markdown]
# ## 4. Support Tickets

# %%
print(tickets.shape)
tickets.describe()

# %%
print("Missing satisfaction_score:", tickets["satisfaction_score"].isna().sum(),
      f"({tickets['satisfaction_score'].isna().mean():.1%})")
print("\nPriority distribution:")
print(tickets["priority"].value_counts())

# %%
fig, axes = plt.subplots(1, 2, figsize=(12, 4))

tix_by_acc = tickets.groupby("account_id").size().reset_index(name="n_tickets")
tix_by_acc = tix_by_acc.merge(acc[["account_id", "churn_flag"]], on="account_id")

tix_by_acc.groupby("churn_flag")["n_tickets"].plot(
    kind="hist", bins=15, alpha=0.6, ax=axes[0], legend=True
)
axes[0].set_title("Ticket Volume: Churned vs Retained")
axes[0].legend(["Retained", "Churned"])

sat_by_acc = tickets.groupby("account_id")["satisfaction_score"].mean().reset_index()
sat_by_acc = sat_by_acc.merge(acc[["account_id", "churn_flag"]], on="account_id")
sat_by_acc.boxplot(column="satisfaction_score", by="churn_flag", ax=axes[1])
axes[1].set_title("Avg Satisfaction: Churned vs Retained")
axes[1].set_xlabel("Churned (1 = Yes)")
plt.suptitle("")

plt.tight_layout()
plt.savefig("../outputs/figures/01_support_vs_churn.png", bbox_inches="tight")
plt.show()

# %% [markdown]
# ## 5. Churn Events

# %%
print(churn_ev.shape)
churn_ev.head()

# %%
print("Reason code distribution:")
print(churn_ev["reason_code"].value_counts())

# %%
fig, ax = plt.subplots(figsize=(8, 4))
churn_ev["reason_code"].value_counts().plot(kind="barh", ax=ax, color="salmon")
ax.set_title("Churn Reason Codes")
ax.set_xlabel("# Events")
plt.tight_layout()
plt.savefig("../outputs/figures/01_churn_reasons.png", bbox_inches="tight")
plt.show()

# %%
# refund amounts by reason
print(churn_ev.groupby("reason_code")["refund_amount_usd"].describe().round(2))

# %%
# accounts with multiple churn events (reactivated then churned again)
multi_churn = (churn_ev.groupby("account_id").size()
               .reset_index(name="n_churn_events")
               .query("n_churn_events > 1"))
print(f"Accounts with >1 churn events: {len(multi_churn)}")
print(multi_churn["n_churn_events"].value_counts())

# %% [markdown]
# ## 6. Key Takeaways
#
# - **22% churn rate** — moderate imbalance, manageable with class weighting
# - **Feature breadth** is a strong early signal: churned accounts use fewer features (~14 vs ~18 unique)
# - **Error rate** is elevated for churned accounts
# - **Pricing and support** are the top stated churn reasons
# - **Basic and Pro** plans churn at similar rates; Enterprise is notably lower
# - Some accounts have churned and reactivated multiple times — worth a separate reactivation model later
# - Satisfaction score is missing for 41% of tickets — imputed by priority median, which is defensible
