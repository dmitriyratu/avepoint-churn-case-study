# %% [markdown]
# # 02 — Data Cleaning
#
# Mostly light work here since the dataset is synthetic and pre-structured.
# Main issues to handle:
# 1. Date columns stored as strings
# 2. Missing `satisfaction_score` (~41% of support tickets)
# 3. Missing `feedback_text` in churn events (~25%)
# 4. Trial subscriptions with MRR = 0

# %%
import sys
sys.path.append("..")

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

from src.load_data import load_all
from src.clean import clean_all

tables_raw = load_all()
tables = clean_all(tables_raw)

# %% [markdown]
# ## 1. Date parsing

# %%
# before
print("Raw types:")
print(tables_raw["accounts"].dtypes[["signup_date"]])
print(tables_raw["subscriptions"].dtypes[["start_date", "end_date"]])
print()
# after
print("Cleaned types:")
print(tables["accounts"].dtypes[["signup_date"]])
print(tables["subscriptions"].dtypes[["start_date", "end_date"]])

# %% [markdown]
# ## 2. Missing values

# %%
for name, df in tables.items():
    missing = df.isna().sum()
    if missing.any():
        print(f"\n--- {name} ---")
        print(missing[missing > 0])

# %%
# satisfaction_score: fill with per-priority median
# Rationale: response rates differ by ticket type; low-priority tickets are more likely skipped
raw_sat = tables_raw["support_tickets"]["satisfaction_score"]
clean_sat = tables["support_tickets"]["satisfaction_score"]

fig, axes = plt.subplots(1, 2, figsize=(10, 3))
raw_sat.hist(bins=20, ax=axes[0])
axes[0].set_title("Satisfaction Score — Raw (with NaN)")
clean_sat.hist(bins=20, ax=axes[1], color="coral")
axes[1].set_title("Satisfaction Score — After Imputation")
plt.tight_layout()
plt.savefig("../outputs/figures/02_satisfaction_imputation.png", bbox_inches="tight")
plt.show()

# check imputed vs original means stay close
print("Original mean (non-null):", raw_sat.mean().round(3))
print("Imputed mean:            ", clean_sat.mean().round(3))

# %%
# per-priority medians (what we actually filled with)
(tables_raw["support_tickets"]
 .groupby("priority")["satisfaction_score"]
 .agg(["median", "count", lambda x: x.isna().mean()])
 .rename(columns={"<lambda_0>": "missing_pct"})
 .round(3))

# %% [markdown]
# ## 3. Subscriptions — trial rows with MRR = 0

# %%
subs = tables["subscriptions"]
zero_mrr = subs[subs["mrr_amount"] == 0]
print(f"Zero-MRR rows: {len(zero_mrr)} ({len(zero_mrr)/len(subs):.1%})")
print(zero_mrr[["is_trial", "plan_tier", "churn_flag"]].value_counts())

# These are legitimate trial subscriptions that either:
# (a) never converted — captured in churn_flag
# (b) were $0 enterprise pilots
# Decision: keep as-is. We'll engineer a trial_conversion feature downstream.

# %% [markdown]
# ## 4. Churn events — `end_date` nulls in subscriptions

# %%
print("Subscriptions with null end_date (active):",
      subs["end_date"].isna().sum(), "/", len(subs))

# Null end_date = subscription still active as of dataset creation.
# We'll treat these as "active" and use REFERENCE_DATE as the de-facto end for tenure calcs.

# %% [markdown]
# ## 5. Duplicate / consistency checks

# %%
# Each account should appear once in accounts table
assert tables["accounts"]["account_id"].nunique() == len(tables["accounts"]), \
    "Duplicate account_ids found!"

# Subscription IDs are unique
assert tables["subscriptions"]["subscription_id"].nunique() == len(tables["subscriptions"]), \
    "Duplicate subscription_ids!"

# All subscriptions belong to known accounts
subs_no_match = ~tables["subscriptions"]["account_id"].isin(tables["accounts"]["account_id"])
print("Subscriptions with no matching account:", subs_no_match.sum())

print("\nAll checks passed.")

# %% [markdown]
# ## 6. Save cleaned tables

# %%
import os
os.makedirs("../data/processed", exist_ok=True)

for name, df in tables.items():
    df.to_csv(f"../data/processed/{name}_clean.csv", index=False)
    print(f"Saved {name}_clean.csv — {df.shape}")
