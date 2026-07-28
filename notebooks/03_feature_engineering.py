# %% [markdown]
# # 03 — Feature Engineering
#
# Builds the account-level feature matrix as of the prediction cutoff.
#
# Two rules govern everything here:
#
# 1. **Nothing dated at or after the cutoff may reach a feature.** Tables are
#    truncated first (`labeling.truncate_tables`), and fields that *resolve*
#    after the cutoff are censored even when the row itself predates it.
# 2. **Nothing derived from `churn_events` becomes a feature.** Those columns
#    describe the outcome. See `docs/DATA_DICTIONARY.md`.
#
# The primary framing is a 90-day horizon with **no buffer** — score today, act
# today, which is the standard default. A buffer demands lead time instead, and
# both dials are swept at the end of this notebook rather than assumed.
#
# Feature families follow the standard churn taxonomy (`docs/FEATURE_ENGINEERING.md`):
#
# | Family | Purpose |
# |---|---|
# | **Recency** | how long since the last meaningful action |
# | **Frequency** | activity counts over a 30/60/90/180-day ladder |
# | **Monetary** | MRR level, growth, volatility |
# | **Acceleration** | short window vs long window — the direction of travel |
# | **Trend** | fitted slope over weekly activity |
# | **Regularity** | gaps between active days — rhythm, not just volume |
# | **Support** | ticket load and its trend |
# | **Account** | industry, country, referral, plan tier, `latest_seats` |
#
# The level tells you how big an account is. The differences between windows tell
# you where it is heading, which is what a churn model needs.

# %%
import sys
sys.path.insert(0, "..")

import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
import warnings
warnings.filterwarnings("ignore")

from src import pipeline, robustness
from src.features import subscription_features, support_features, usage_features
from src.config import CUTOFF_DATE, EXTRACT_DATE, HORIZON_DAYS

sns.set_theme(style="whitegrid", palette="muted")

data = pipeline.build()
tables, obs, cohort = data.tables, data.observed, data.cohort

print(data.summary.to_string())

# %% [markdown]
# ## How much data survives truncation
#
# Roughly two thirds of the event rows are dated after the cutoff and are
# correctly unavailable. Seeing this number is a useful sanity check — if it were
# near zero, the truncation would not be doing anything.

# %%
for k in ["subscriptions", "feature_usage", "support_tickets", "churn_events"]:
    print(f"  {k:16s} {len(obs[k]):>6} / {len(tables[k]):>6}  "
          f"({len(obs[k])/len(tables[k]):.0%} retained)")

# %% [markdown]
# ## Block 1 — Subscription features
#
# Level tells you how big an account is; **direction** tells you where it is
# going. `seat_growth`, `mrr_growth_pct` and `upgrade_net` capture the second.

# %%
sub_feats = subscription_features(obs["subscriptions"], CUTOFF_DATE)
print(sub_feats.shape)
sub_feats.describe().T.round(2)

# %% [markdown]
# `tenure_days` is measured signup-to-cutoff. An earlier version measured it to
# `end_date.max()`, which silently stops the clock at whichever subscription
# closed first — wrong for the 62% of accounts holding both open and closed
# subscriptions.

# %% [markdown]
# ## Block 2 — Engagement features
#
# Recency and momentum matter more than lifetime totals: an account that used
# the product heavily last year and nothing this quarter looks healthy on
# volume alone.

# %%
usage_feats = usage_features(obs["feature_usage"], obs["subscriptions"], CUTOFF_DATE)
print(usage_feats.shape)
usage_feats[["total_usage_events", "unique_features_used", "days_since_last_usage",
             "usage_last_30d", "usage_last_90d", "usage_momentum", "error_rate"]].describe().T.round(2)

# %% [markdown]
# The windowed columns are anchored to the cutoff. In the first version they were
# anchored to a hardcoded date seven months past the end of the data, which made
# `usage_last_30d` and `usage_last_90d` identically zero for every account.

# %%
print("non-zero windowed activity (a zero-variance column would mean a bug):")
for c in ["usage_last_30d", "usage_last_90d", "usage_last_180d"]:
    if c in usage_feats.columns:
        print(f"  {c:18s} nunique={usage_feats[c].nunique():>4}  mean={usage_feats[c].mean():.1f}")

# %% [markdown]
# ## Block 3 — Support features
#
# `resolution_time_hours` and `satisfaction_score` are censored for tickets still
# open at the cutoff, so these aggregates skip them rather than counting a
# resolution that has not happened. `n_open_tickets` replaces that lost signal
# with something genuinely observable.

# %%
support_feats = support_features(obs["support_tickets"], CUTOFF_DATE)
print(support_feats.shape)
support_feats.describe().T.round(2)

# %% [markdown]
# ## Assemble

# %%
df, X, y = data.frame, data.X, data.y
print(f"feature matrix : {X.shape}  (categoricals still raw; encoded in-fold)")
from src.model import categorical_columns
print(f"categorical cols: {categorical_columns(X)}")
print(f"positives      : {int(y.sum())} ({y.mean():.1%})")
print(f"pruned: {data.dropped}")
print(f"events per variable: {y.sum()/X.shape[1]:.2f}   (want >= 10)")

# %% [markdown]
# Well under one event per variable — severely under-powered, and the number
# that predicts the modelling result in notebook 04. With this many columns and
# this few positives, regularisation matters more than capacity.

# %% [markdown]
# ### The contract, drawn
#
# Two things about this feature set are easy to state and hard to picture: where
# the 73 columns come from, and what "point-in-time" actually excludes. The
# figure below is the deck's version of both.
#
# Provenance is taken from block membership rather than from name patterns, so
# the counts are exact rather than a guess at what `usage_` means. The row
# counts on the right are what `truncate_tables` actually kept.

# %%
blocks = {"subscription": set(sub_feats.columns),
          "usage": set(usage_feats.columns),
          "support": set(support_feats.columns)}
families = {name: sum(c in cols for c in X.columns) for name, cols in blocks.items()}
# Per-seat ratios and the account attributes are built in `assemble`, so they
# belong to no upstream block.
families["account / ratios"] = X.shape[1] - sum(families.values())
families = pd.Series(families).sort_values(ascending=False)
print(families.to_string())
print(f"total: {families.sum()}  (matches X: {X.shape[1] == families.sum()})")

# %%
from matplotlib.patches import FancyArrowPatch

# Two validated categorical hues (dataviz slots 1 and 2) plus a neutral. One hue
# carries "the model may see this", the other "this is the outcome", and
# everything discarded is grey. The story is the cutoff, not the four tables, so
# the tables are deliberately not hued apart.
SEEN, LABEL, GONE = "#2a78d6", "#eb6834", "#c9ccd1"
INK, MUTED, RULE, ALERT = "#1A1A1A", "#5A6270", "#D8DCE0", "#B02E2E"

START = pd.Timestamp("2023-01-01")
PRED_END = CUTOFF_DATE + pd.Timedelta(days=HORIZON_DAYS)
DAY = pd.Timedelta(days=1)

fig, (ax_a, ax_b) = plt.subplots(
    1, 2, figsize=(13, 3.9), gridspec_kw={"width_ratios": [1, 2.6]})

ax_a.barh(range(len(families)), families.values, color=SEEN, height=0.42)
for i, n in enumerate(families.values):
    ax_a.text(n + 0.8, i, str(n), va="center", fontsize=12.5, color=INK,
              fontweight="bold")
ax_a.set_yticks(range(len(families)))
ax_a.set_yticklabels(families.index, fontsize=12)
ax_a.invert_yaxis()
ax_a.set_xlim(0, families.max() + 8)
ax_a.set_xticks([])
ax_a.grid(False)
for side in ax_a.spines:
    ax_a.spines[side].set_visible(False)
ax_a.set_title(f"{families.sum()} features, four families", loc="left",
               fontsize=13.5, fontweight="bold", color=INK, pad=12)

ax_b.axvspan(START, CUTOFF_DATE, color=SEEN, alpha=0.07, zorder=0)
for ypos, (key, label) in zip([3.0, 2.2, 1.4],
                              [("subscriptions", "subscriptions"),
                               ("feature_usage", "product usage"),
                               ("support_tickets", "support tickets")]):
    ax_b.barh(ypos, CUTOFF_DATE - START, left=START, height=0.30, color=SEEN, zorder=3)
    ax_b.barh(ypos, EXTRACT_DATE - CUTOFF_DATE - DAY, left=CUTOFF_DATE + DAY,
              height=0.30, color=GONE, zorder=3)
    ax_b.text(START - pd.Timedelta(days=16), ypos, label, ha="right", va="center",
              fontsize=11.5, color=INK)
    ax_b.text(EXTRACT_DATE + pd.Timedelta(days=18), ypos,
              f"{len(obs[key]):,} of {len(tables[key]):,} rows kept",
              ha="left", va="center", fontsize=10.5, color=MUTED)

ax_b.barh(0.6, PRED_END - CUTOFF_DATE, left=CUTOFF_DATE, height=0.30,
          color=LABEL, zorder=3)
ax_b.text(START - pd.Timedelta(days=16), 0.6, "churn events", ha="right",
          va="center", fontsize=11.5, color=INK)
ax_b.text(EXTRACT_DATE + pd.Timedelta(days=18), 0.6,
          f"{HORIZON_DAYS} days — the label, never a feature", ha="left",
          va="center", fontsize=10.5, color=MUTED)

# The zone labels sit on either side of the rule so neither can collide with it:
# what the model may see to the left, the rule's own name to the right.
ax_b.axvline(CUTOFF_DATE, color=ALERT, lw=2.2, zorder=5)
ax_b.text(START + pd.Timedelta(days=24), 3.62, "everything the model may see",
          fontsize=11, color=MUTED, va="center")
ax_b.text(CUTOFF_DATE + pd.Timedelta(days=22), 3.62,
          f"cutoff · {CUTOFF_DATE.day} {CUTOFF_DATE:%b %Y}", fontsize=11.5,
          fontweight="bold", color=ALERT, va="center", ha="left")

n_censored = int(obs["support_tickets"]["ticket_open_at_cutoff"].sum())
ax_b.add_patch(FancyArrowPatch(
    (CUTOFF_DATE + pd.Timedelta(days=215), 0.16),
    (CUTOFF_DATE - pd.Timedelta(days=6), 1.26),
    arrowstyle="-|>", mutation_scale=11, lw=1.2, color=ALERT,
    connectionstyle="arc3,rad=0.18", zorder=6))
ax_b.text(CUTOFF_DATE + pd.Timedelta(days=232), 0.14,
          f"{n_censored} tickets are still open on this line. Their fix time and\n"
          "satisfaction are blanked — nobody knew them on 30 June.",
          fontsize=10, color=ALERT, va="top")

ax_b.set_ylim(-0.30, 3.95)
ax_b.set_xlim(START - pd.Timedelta(days=215), EXTRACT_DATE + pd.Timedelta(days=430))
ax_b.set_yticks([])
ax_b.grid(False)
for side in ("top", "right", "left"):
    ax_b.spines[side].set_visible(False)
ax_b.spines["bottom"].set_color(RULE)
ticks = pd.date_range("2023-01-01", "2024-12-01", freq="6MS")
ax_b.set_xticks(ticks)
ax_b.set_xticklabels([d.strftime("%b %Y") for d in ticks], fontsize=10, color=MUTED)
ax_b.set_title("Every feature is built from one side of a line", loc="left",
               fontsize=13.5, fontweight="bold", color=INK, pad=12)

plt.tight_layout(w_pad=3)
plt.savefig("../outputs/figures/03_point_in_time.png", bbox_inches="tight", dpi=150)
plt.show()

# %% [markdown]
# The right-hand panel is the one worth dwelling on. Filtering rows on their own
# timestamp is the easy half; the hard half is the arrow. A ticket opened on 28
# June and closed on 1 July is a legitimate pre-cutoff row, but its
# `resolution_time_hours` and `satisfaction_score` did not exist on 30 June.
# Those two fields are blanked while the row itself is kept — and it is
# `audit.temporal_provenance`, not code review, that caught them.

# %% [markdown]
# ## Missing values: three meanings, three treatments
#
# The first version filled everything with 0, which conflates them.

# %%
na = X.isna().sum()
na = na[na > 0]
print("columns left as NaN for in-fold imputation:")
print(na.to_string() if len(na) else "  (none)")
print(f"\ntotal NaNs retained: {int(X.isna().sum().sum())}")
print("\ncounts -> 0 (no activity is genuinely zero)")
print("recency -> observation-window length (never used != used today)")
print("rates   -> NaN, imputed inside the CV fold by model._pipe")

# %% [markdown]
# ## Association with the target
#
# Reported here for orientation only — nothing is selected off this list.
# Regularisation inside cross-validation decides what the model leans on, and
# the selected rung is L2, which shrinks rather than drops.

# %%
from src.audit import encode_for_audit
Xe = encode_for_audit(X)   # diagnostic view only; the model encodes in-fold
corr = Xe.corrwith(y.astype(float)).sort_values(key=abs, ascending=False)
print(corr.head(15).round(4).to_string())
print(f"\nmax |r| = {corr.abs().max():.4f}")

# %%
top = corr.abs().head(15).index
fig, ax = plt.subplots(figsize=(8, 6))
vals = corr[top]
vals.plot(kind="barh", ax=ax, color=["salmon" if v > 0 else "steelblue" for v in vals])
ax.axvline(0, color="black", lw=.8)
ax.set_title(f"Top 15 features by |correlation| with {HORIZON_DAYS}-day churn")
ax.set_xlabel("Pearson r")
plt.tight_layout()
plt.savefig("../outputs/figures/03_feature_correlations.png", bbox_inches="tight")
plt.show()

# %% [markdown]
# ## Leakage gate before anything is modelled

# %%
res, passed = data.audit()
print(f"max single-feature AUC : {res['single_feature_auc']['auc'].max():.4f}")
print(f"temporal provenance    : {'PASS' if res['temporal_provenance']['pass'].all() else 'FAIL'}")
print(f"\nSUITE: {'PASS' if passed else 'FAIL'}")
assert passed, "leakage audit failed — do not proceed to modelling"

# %%
df.to_csv("../data/processed/features_temporal.csv", index=False)
print(f"saved features_temporal.csv  {df.shape}")

# %% [markdown]
# ## The two design dials
#
# Two choices decide what question is being asked, and both have to be swept
# rather than assumed:
#
# - **Horizon** — how far forward the label looks. "Churn in the next N days."
# - **Buffer** — how much lead time the model must give. Zero is the standard
#   default (score today, act today). A non-zero buffer pulls the feature cutoff
#   back, forcing the model to warn *before* the customer is visibly leaving.
#   Accounts that churn during the buffer drop out, which is the point: at
#   scoring time nobody could have acted on them.
#
# Computed here rather than read from a file. An earlier version of this
# notebook loaded a committed CSV; when the cohort definition changed, the CSV
# went on reporting the old population and nothing caught it.
#
# Each cell carries an interval but **no permutation test**. Twelve p-values,
# with the smallest one highlighted, is the same selection error notebook 09 is
# about: under a true null the minimum of twelve is small by construction. The
# significance test is run once, below, on the pre-specified primary cell.

# %%
sweep = robustness.horizon_buffer_sweep()
print(sweep.to_string(index=False))
sweep.to_csv("../outputs/reports/horizon_buffer_sweep.csv", index=False)

# %%
fig, ax = plt.subplots(figsize=(8.5, 4.5))
for buffer, group in sweep.groupby("buffer"):
    ax.errorbar(group["horizon"], group["cv_auc"],
                yerr=[group["cv_auc"] - group["ci_lo"],
                      group["ci_hi"] - group["cv_auc"]],
                marker="o", capsize=3, label=f"buffer {buffer}d")
ax.axhline(0.5, ls="--", c="r", alpha=.6, label="chance")
ax.set(xlabel="horizon (days)", ylabel="CV ROC-AUC",
       title="Every interval spans chance, at every horizon and lead time")
ax.legend(fontsize=8)
plt.tight_layout()
plt.savefig("../outputs/figures/03_horizon_buffer_sweep.png", bbox_inches="tight")
plt.show()

# %%
clears = sweep[sweep["ci_lo"] > 0.5]
print(f"cells whose interval clears chance: {len(clears)} of {len(sweep)}")
print(f"AUC range across the grid          : {sweep['cv_auc'].min():.3f} "
      f"to {sweep['cv_auc'].max():.3f}")
print(f"typical half-width of one interval : "
      f"{((sweep['ci_hi'] - sweep['ci_lo']) / 2).mean():.3f}")
print("\n-> The spread across twelve design choices is smaller than the "
      "uncertainty\n   in measuring any one of them.")

# %% [markdown]
# ### The one significance test, on the pre-specified cell
#
# 90-day horizon, no buffer — the project's primary framing, chosen because it
# is the standard SaaS formulation, not because it scored well.

# %%
primary = robustness.primary_significance(X, y)
for k, v in primary.items():
    print(f"  {k:14s} {v}")

# %% [markdown]
# Two caveats that have to travel with that number. It holds the **model** fixed,
# and the model was chosen as the top of a ten-rung ladder — notebook 04's nested
# CV is the estimate that accounts for that. And the short-horizon cells above are
# **underpowered rather than proven null**: at 30 days there are too few positives
# to detect a modest effect, so "no signal" is not what that row says.

# %%
short = sweep.query("horizon == 30 and buffer == 0").iloc[0]
print(f"30-day horizon, no buffer: {int(short['positives'])} positives "
      f"in {int(short['n'])} accounts ({short['rate']:.1%})")
print("Too few events to distinguish a modest effect from nothing.")

# %% [markdown]
# ## Did the richer feature families help?
#
# The window ladder, acceleration ratios, trend slope, gap statistics and MRR
# volatility added roughly twenty features. Measured against the set without
# them, on identical folds.

# %%
comparison, enriched_cols = robustness.feature_set_comparison(X, y)
print(comparison.to_string(index=False))
print(f"\nenriched families contribute {len(enriched_cols)} of {X.shape[1]} columns")
comparison.to_csv("../outputs/reports/feature_set_comparison.csv", index=False)

# %% [markdown]
# Reporting this rather than quietly keeping the richer set is the point. At this
# many positives, extra columns cost about as much in variance as they return in
# signal — so the binding constraint is **data, not feature engineering**. The
# usage logs have 19,128 of 24,979 rows predating their own subscription, and
# the three recorded churn signals are mutually unrelated, so the choice of
# ground truth is an unverifiable assumption. No feature work fixes either.
