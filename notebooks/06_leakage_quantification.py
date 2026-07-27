# %% [markdown]
# # 06 — Quantifying Leakage
#
# Notebook 07 gates leakage with pass/fail checks. This one measures what each
# form of leakage is *worth*, by scoring the same model on the same folds and
# changing only what the features are allowed to see.
#
# Three designs:
#
# | | Features may see |
# |---|---|
# | **A** | the observation window only — correct |
# | **B** | plus rows dated after the cutoff |
# | **C** | plus columns derived from `churn_events` |

# %%
import sys
sys.path.append("..")

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import warnings
warnings.filterwarnings("ignore")

from sklearn.model_selection import cross_val_score

from src import pipeline
from src.features import build_model_dataset
from src.model import prep_xy, model_ladder, CV
from src.config import CUTOFF_DATE, POST_OUTCOME_COLS
import src.model as model_module

sns.set_theme(style="whitegrid")

data = pipeline.build()
tables, observed, cohort = data.tables, data.observed, data.cohort
_, estimator = model_ladder()[4]


def score(df):
    X, y = prep_xy(df)
    return cross_val_score(estimator, X, y, cv=CV, scoring="roc_auc")


# %%
design_a = score(build_model_dataset(observed, cohort, CUTOFF_DATE))
design_b = score(build_model_dataset(tables, cohort, CUTOFF_DATE))

# %%
# C: restore the post-outcome columns the pipeline normally refuses.
churn_derived = tables["churn_events"].groupby("account_id").agg(
    n_churn_events=("churn_event_id", "count"),
    total_refund_usd=("refund_amount_usd", "sum"),
    had_reactivation=("is_reactivation", "any"),
)
leaky = build_model_dataset(tables, cohort, CUTOFF_DATE).join(
    churn_derived, on="account_id").fillna({"n_churn_events": 0, "total_refund_usd": 0,
                                            "had_reactivation": False})

model_module.POST_OUTCOME_COLS = []          # temporarily disable the exclusion
design_c = score(leaky)
model_module.POST_OUTCOME_COLS = POST_OUTCOME_COLS

# %%
results = pd.DataFrame({
    "design": ["A. observation window only", "B. + post-cutoff rows",
               "C. + churn_events columns"],
    "auc": [design_a.mean(), design_b.mean(), design_c.mean()],
    "sd": [design_a.std(), design_b.std(), design_c.std()],
}).round(4)
print(results.to_string(index=False))
results.to_csv("../outputs/reports/leakage_comparison.csv", index=False)

# %%
fig, ax = plt.subplots(figsize=(8, 4.5))
sns.boxplot(data=pd.DataFrame({"A. correct": design_a, "B. sees future": design_b,
                               "C. sees outcome": design_c}),
            ax=ax, palette=["#2a9d8f", "#e9c46a", "#e76f51"])
ax.axhline(0.5, ls="--", c="k", alpha=.5, label="chance")
ax.set_ylabel("ROC-AUC (repeated 5x10 CV)")
ax.set_title("What each form of leakage is worth")
ax.legend()
plt.tight_layout()
plt.savefig("../outputs/figures/06_leakage_effect.png", bbox_inches="tight")
plt.show()

# %% [markdown]
# **Design C is the one that matters.** `n_churn_events`, `total_refund_usd` and
# `had_reactivation` reconstruct the label almost exactly — a refund is issued
# *because* the customer left — and the model goes to near-perfect.
#
# That is the signature worth recognising: an AUC near 1.0 on a churn problem is
# a bug report, not a result. `config.POST_OUTCOME_COLS` excludes these columns
# and `model.prep_xy` drops them unconditionally.
#
# Design B barely moves. Seeing post-cutoff rows adds mostly noise here, which is
# a useful reminder that leakage is not always *helpful* to the model — it is
# simply invalid.

# %% [markdown]
# ## Why the label had to be redefined
#
# `accounts.churn_flag` carries no date, so it cannot be placed relative to any
# cutoff. It also disagrees with the event log it should summarise.

# %%
accounts, events = tables["accounts"], tables["churn_events"]
has_event = accounts["account_id"].isin(events["account_id"])

print(f"accounts                        : {len(accounts)}")
print(f"churn_flag = True               : {int(accounts['churn_flag'].sum())}")
print(f"appearing in churn_events       : {int(has_event.sum())}")
print(f"agreement between definitions   : {(has_event == accounts['churn_flag']).mean():.1%}")
print()
print(pd.crosstab(accounts["churn_flag"], has_event.rename("has_churn_event")))

# %% [markdown]
# 37.6% agreement is worse than a coin flip. The event log is used as ground
# truth because it carries dates; an undated flag cannot support a forward-looking
# target at all.
#
# This is the kind of thing that caps achievable performance no matter how good
# the features are, and it is found by counting rather than by modelling.

# %% [markdown]
# ## Where this leaves the project
#
# - Post-outcome leakage is worth **+0.45 AUC** here, and is excluded.
# - The label is internally inconsistent for 62% of accounts.
# - Once a realistic intervention buffer is required, no configuration beats
#   chance — see the sensitivity sweep in `03_feature_engineering.py`.
#
# Full audit gates run in `07_leakage_audit.py`.
