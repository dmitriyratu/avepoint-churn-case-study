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
sys.path.insert(0, "..")

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
# its resolution is not. `ticket_open_at_cutoff` replaces the lost signal with
# something genuinely observable.
#
# The leak was small here — only 5 tickets — but it is the kind that scales
# badly. On real data where support outcomes are more predictive, the same
# pipeline bug would inflate the score much more, and nothing about the code
# would look different.

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
from sklearn.metrics import roc_auc_score
from src.config import TARGET

ce = tables["churn_events"].groupby("account_id").agg(
    n_churn_events=("churn_event_id", "count"),
    total_refund_usd=("refund_amount_usd", "sum")).reset_index()
probe = cohort[["account_id", TARGET]].merge(ce, on="account_id", how="left").fillna(0)
for c in ["n_churn_events", "total_refund_usd"]:
    a = roc_auc_score(probe[TARGET], probe[c])
    auc = max(a, 1 - a)
    verdict = ("would trip the FAIL gate" if auc >= audit.SINGLE_FEATURE_AUC_FAIL
               else "below the FAIL gate — excluded by name, not by statistic")
    print(f"  {c:20s} single-feature AUC = {auc:.4f}   <- {verdict}")
print(f"\nExcluded via config.POST_OUTCOME_COLS = {POST_OUTCOME_COLS}")

# %% [markdown]
# Worth noting which of those two the statistical gate would actually have
# caught. `n_churn_events` is far over the threshold and obvious.
# `total_refund_usd` is not — it sits well below 0.80 and would have sailed
# through, despite being a column that only exists *because* the customer left.
#
# That is the argument for `audit.forbidden_columns`: a leak is defined by
# provenance, not by how strongly it happens to predict in this sample.

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
print(" ", model_ladder()[2][1].named_steps)

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
# The suite passes, which is a statement about *validity*, not about quality. It
# says the number in notebook 04 is not inflated by anything this project knows
# how to detect — nothing more.
#
# The headline figures live in `outputs/models/config.json` so they cannot drift
# out of step with the run that produced them:

# %%
import json
config = json.load(open("../outputs/models/config.json"))
for key in ["model", "cohort_n", "positives", "cv_auc", "nested_cv_auc",
            "permutation_p", "oof_recall", "oof_precision"]:
    print(f"  {key:16s} {config[key]}")

# %% [markdown]
# **What I would actually recommend**: not to deploy this. The audit clearing is
# necessary but nowhere near sufficient — the ladder maximum does not separate
# from chance at the lower bound of its interval, and the nested estimate that
# accounts for having chosen it sits at chance.
#
# The audit's real value here is different and worth stating plainly: it is what
# lets me say the near-chance result is a fact about the data rather than a bug
# in the pipeline. A negative result you cannot trust is worthless; this one is
# gated.
