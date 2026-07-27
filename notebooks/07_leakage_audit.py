# %% [markdown]
# # 07 — Leakage Audit & Cleaning Gates
#
# A high AUC on a churn problem is a claim that needs defending. It usually means
# one of three things: the generator built the signal in deliberately, something
# leaked, or the evaluation is optimistic. Only the first is legitimate, and it
# is the rarest.
#
# Notebook 06 removed the leakage I found by *reasoning* about the schema. This
# notebook turns that reasoning into tests that run over the built matrix — and
# those tests immediately found something the reading had missed.
#
# See `docs/DATA_DICTIONARY.md` for the field-by-field verdicts.

# %%
import sys
sys.path.append("..")

import pandas as pd
import numpy as np
import warnings
warnings.filterwarnings("ignore")

from src import pipeline
from src.clean import integrity_report
from src.config import CUTOFF_DATE, POST_OUTCOME_COLS
import src.audit as audit

data = pipeline.build()
raw, tables, obs = data.raw, data.tables, data.observed
cohort, df, X, y = data.cohort, data.frame, data.X, data.y
print(f"cohort {X.shape}, positives {int(y.sum())}, cutoff {CUTOFF_DATE.date()}")

# %% [markdown]
# ## Gate 1 — Temporal provenance
#
# The invariant: **no datetime value anywhere in the truncated tables may be at
# or after the cutoff.**
#
# Checking every datetime column matters. Filtering `support_tickets` on
# `submitted_at` leaves `closed_at` free to point into the future — which is
# exactly what happened: 5 tickets were opened in June and closed in July, so
# their `resolution_time_hours`, `satisfaction_score` and
# `first_response_time_minutes` were all unknowable at the cutoff. One of those
# columns was in the final model.

# %%
tp = audit.temporal_provenance(obs, CUTOFF_DATE)
print(tp.to_string(index=False))
print(f"\nviolations: {int((~tp['pass']).sum())}")
assert tp["pass"].all(), "temporal provenance violated"

# %% [markdown]
# `labeling.truncate_tables` now censors those fields rather than dropping the
# rows — the ticket *existing* before the cutoff is legitimate information; only
# its resolution is not.
#
# Measured when it was fixed, censoring moved CV AUC from 0.635 to 0.611. A leak
# that only costs 0.024 is still a leak, and it is the kind that scales badly: on
# real data where support outcomes are more predictive, the same bug would inflate
# the score much more. (The current headline figure is 0.618 — a later change moved
# one-hot encoding in-fold, which is independent of this fix.)

# %% [markdown]
# ## Gate 2 — Single-feature AUC
#
# The cheapest leak detector there is. A lone column that separates the classes
# at 0.80+ is almost never a discovery; it is usually the label wearing a
# different name.

# %%
sf = audit.single_feature_auc(X, y)
print(sf.head(12).to_string(index=False))
print(f"\nmax {sf['auc'].max():.4f} | warn {(sf['verdict'].str.startswith('WARN')).sum()}"
      f" | fail {(sf['verdict'].str.startswith('FAIL')).sum()}")

# %% [markdown]
# Nothing exceeds 0.65. Compare against the excluded post-outcome columns:

# %%
ce = tables["churn_events"].groupby("account_id").agg(
    n_churn_events=("churn_event_id", "count"),
    total_refund_usd=("refund_amount_usd", "sum")).reset_index()
probe = cohort[["account_id", "churned_next_180d"]].merge(ce, on="account_id", how="left").fillna(0)
for c in ["n_churn_events", "total_refund_usd"]:
    from sklearn.metrics import roc_auc_score
    a = roc_auc_score(probe["churned_next_180d"], probe[c])
    print(f"  {c:20s} single-feature AUC = {max(a, 1-a):.4f}   <- would trip the FAIL gate")
print(f"\nExcluded via config.POST_OUTCOME_COLS = {POST_OUTCOME_COLS}")

# %% [markdown]
# ## Gates 3-6 — separation, identifiers, duplicates, constants

# %%
print("perfect separation:", len(audit.perfect_separation(X, y)), "columns")
print()
print(audit.identifier_leakage(df, y).to_string(index=False))
print()
print(audit.duplicate_rows(X, df).to_string(index=False))
print()
print("constant columns:", len(audit.constant_columns(X)))

# %% [markdown]
# ## Cleaning gates
#
# ### Missingness: disposition depends on *why* a value is absent

# %%
print(audit.missingness_report(raw).to_string(index=False))

# %% [markdown]
# Three different situations, three different treatments:
#
# - **`end_date`, 90.3% null** — this is *structural*. The subscription is still
#   open. A naive ">60% missing, drop it" rule would throw away the single most
#   informative column in the table. Encoded as `n_open_subs` / `pct_subs_ended`.
# - **`satisfaction_score`, 41.2%** — genuinely absent (customer did not respond).
#   Kept as `NaN` through cleaning and imputed **inside the CV fold**, with a
#   `satisfaction_missing` indicator retained.
# - **`feedback_text`, 24.7%** — free text, not used as a feature.
#
# The reason imputation moved into the pipeline: filling with a median computed
# over the whole table lets validation rows influence the statistic applied to
# training rows. It is a small leak, but it is free to avoid.

# %%
from src.model import model_ladder
print("Every rung imputes inside the fold:")
print(" ", model_ladder()[4][1].named_steps)

# %% [markdown]
# ### NaN does not mean zero
#
# Filling every missing value with 0 — which the first version did — conflates
# three distinct meanings:
#
# | Feature type | Missing means | Fill |
# |---|---|---|
# | counts (`n_tickets`, `total_usage_events`) | genuinely zero activity | `0` |
# | recency (`days_since_last_usage`) | never happened — maximally stale, not "today" | observation-window length |
# | rates/means (`avg_satisfaction`, `error_rate`) | unknown, not zero | `NaN`, imputed in-fold |
#
# An account with no support tickets has an *undefined* average satisfaction, not
# a satisfaction of 0. Filling it with 0 invents a maximally unhappy customer.

# %%
print(f"NaNs deliberately retained for in-fold imputation: {int(X.isna().sum().sum())}")
print(X.isna().sum()[X.isna().sum() > 0].to_string())

# %% [markdown]
# ### Collinearity

# %%
print("dropped at |r| > 0.98:", df.attrs.get("dropped_collinear"))
print()
print(audit.collinear_pairs(X, threshold=0.95).to_string(index=False))

# %% [markdown]
# `feature_breadth` was `unique_features_used / 40` — correlated at exactly 1.000.
# Keeping both splits one effect across two coefficients and makes the linear
# model harder to read.

# %% [markdown]
# ### Source-data integrity
#
# Documented rather than silently repaired — these are upstream bugs, and in a
# real engagement they would go back to data engineering before modelling.

# %%
print(integrity_report(tables).to_string(index=False))

# %% [markdown]
# ## Full suite

# %%
res, passed = audit.run_all(X, y, df, obs, CUTOFF_DATE, raw_tables=raw)
for name, frame in res.items():
    print(f"  {name:22s} {len(frame):>3} rows")
print(f"\nSUITE: {'PASS' if passed else 'FAIL'}")
assert passed, "leakage audit failed"

# %% [markdown]
# ## Where this leaves the result
#
# After censoring, the honest numbers are:
#
# | | |
# |---|---|
# | CV ROC-AUC (L1 logistic) | **0.618** [0.50, 0.74] |
# | Permutation test | p = **0.013** |
# | Out-of-fold recall @ t=0.40 | 0.943 |
# | Out-of-fold precision | 0.494 |
# | Features retained by L1 | 5 |
#
# This is a genuinely marginal model: it beats chance, but the lower end of the
# interval sits right on 0.50. Censoring the leak cost real signal at the time it
# was applied; the figure recovered later for an unrelated reason (in-fold
# encoding), not because the leak came back.
#
# **What I would actually recommend**: this is a triage ranker for CSM outreach,
# not an automated action trigger. At 94% recall and 49% precision it is useful
# for ordering a call list where the cost of a wasted call is low. It is not
# usable for anything with a real cost attached to a false positive, and I would
# say so before anyone asked.
