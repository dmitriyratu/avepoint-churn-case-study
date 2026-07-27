# %% [markdown]
# # 02 — Data Cleaning
#
# Cleaning here is deliberately thin, and the reason is the organising principle
# in `docs/CLEANING_CHECKLIST.md`: a step that **learns a parameter from the
# data** — a median to fill with, a mean to scale by, a set of category levels —
# belongs inside the CV pipeline, not here. This module only does things whose
# answer is the same row by row.
#
# So this notebook covers:
#
# 1. Date columns stored as strings — stateless, done here
# 2. Redundant and duplicate rows — stateless, done here
# 3. Missing `satisfaction_score` (~41%) — **recorded, not filled**; the
#    imputation happens per fold in `model._pipe`
# 4. Trial subscriptions with MRR = 0 — inspected, kept

# %%
import sys
sys.path.insert(0, "..")

import matplotlib.pyplot as plt

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

# %% [markdown]
# ### `satisfaction_score` is left as `NaN` on purpose
#
# An earlier version filled it with a per-priority median right here, justified
# by "response rates differ by ticket severity." Two things were wrong with that.
#
# The justification is false — missing rates are flat across all four
# priorities. And the fill itself is a leak: a median computed over all 2,000
# tickets lets validation rows influence the value applied to training rows.
# `clean_all` now records the missingness and leaves the value alone.

# %%
tix = tables_raw["support_tickets"]
print("missing rate by priority (the claim that justified filling):")
print(tix.groupby("priority")["satisfaction_score"].apply(lambda s: s.isna().mean())
      .round(3).to_string())
print("\nFlat — there is no per-priority pattern to fill against.")
print(f"\nsatisfaction_score after clean_all: "
      f"{tables['support_tickets']['satisfaction_score'].isna().sum()} still NaN")
print("satisfaction_missing indicator added:",
      "satisfaction_missing" in tables["support_tickets"].columns)

# %% [markdown]
# ### What the column actually contains
#
# Worth plotting before trusting it as a feature. The schema documents a 1–5
# scale; the data uses **three** of those five values.

# %%
counts = tix["satisfaction_score"].value_counts(dropna=False).sort_index()
print(counts.to_string())

fig, ax = plt.subplots(figsize=(7, 3.5))
observed = tix["satisfaction_score"].dropna()
ax.hist(observed, bins=[0.5 + i for i in range(6)], edgecolor="white", color="steelblue")
ax.set_xticks(range(1, 6))
ax.set_xlabel("satisfaction score (schema says 1–5)")
ax.set_ylabel("tickets")
ax.set_title("Scores 1 and 2 never occur — the bottom of the scale is unused")
plt.tight_layout()
plt.savefig("../outputs/figures/02_satisfaction_distribution.png", bbox_inches="tight")
plt.show()

# %% [markdown]
# A dissatisfaction signal that cannot express dissatisfaction. Near-uniform over
# {3, 4, 5}, which is what an independent random draw looks like — and it is a
# small, early piece of evidence for the conclusion notebook 10 reaches the
# expensive way: this generator did not build a relationship between behaviour
# and churn.

# %% [markdown]
# ## 3. Subscriptions — trial rows with MRR = 0

# %%
subs = tables["subscriptions"]
zero_mrr = subs[subs["mrr_amount"] == 0]
print(f"Zero-MRR rows: {len(zero_mrr)} ({len(zero_mrr)/len(subs):.1%})")
print(zero_mrr["is_trial"].value_counts().to_string())

# %% [markdown]
# Every zero-MRR row is a trial. Kept as-is: a $0 trial is a real state, not a
# missing price. It reaches the model as `n_trial_subs` and `latest_is_trial`,
# both built from the truncated subscription history rather than from
# `accounts.is_trial`, which is current-as-of-extraction.

# %% [markdown]
# ## 4. `end_date` nulls are structural

# %%
print("Subscriptions with null end_date:",
      f"{subs['end_date'].isna().sum()} / {len(subs)} "
      f"({subs['end_date'].isna().mean():.1%})")

# %% [markdown]
# A null `end_date` means the subscription is still open — that is information,
# not absence. A ">60% missing, drop the column" rule would discard one of the
# most informative fields in the table. It is encoded as `n_ended_subs` /
# `pct_subs_ended` instead, and `truncate_tables` re-nulls any end date that
# falls after the cutoff, so a subscription that ends later still reads as open
# at prediction time.

# %% [markdown]
# ## 5. Duplicate / consistency checks
#
# On *every* key, not just the obvious ones. The first version checked
# `account_id` and `subscription_id` and never looked at `usage_id` — which is
# where the only duplicates actually are.

# %%
for name, key in [("accounts", "account_id"), ("subscriptions", "subscription_id"),
                  ("feature_usage", "usage_id"), ("support_tickets", "ticket_id"),
                  ("churn_events", "churn_event_id")]:
    dupes = int(tables_raw[name][key].duplicated().sum())
    note = "  <- dropped in clean.py" if dupes else ""
    print(f"  {name:16s} {key:18s} {dupes}{note}")

subs_no_match = ~tables["subscriptions"]["account_id"].isin(tables["accounts"]["account_id"])
print("\nSubscriptions with no matching account:", int(subs_no_match.sum()))
print(f"feature_usage rows after dedup: {len(tables['feature_usage'])} "
      f"(from {len(tables_raw['feature_usage'])})")

# %% [markdown]
# ## 6. Save cleaned tables

# %%
import os
os.makedirs("../data/processed", exist_ok=True)

for name, df in tables.items():
    df.to_csv(f"../data/processed/{name}_clean.csv", index=False)
    print(f"Saved {name}_clean.csv — {df.shape}")
