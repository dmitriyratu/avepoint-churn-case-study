# %% [markdown]
# # 13 — Driver analysis: SHAP, and whether to believe it
#
# "Why are users leaving" has a machine-learning answer as well as the
# descriptive one in notebook 11: fit a model, explain it, read off the drivers.
# SHAP is the standard tool and it is genuinely good — exact additive
# attribution, sound game-theoretic footing, a plot everyone can read.
#
# It is also the single most dangerous thing you can run on this dataset, for a
# reason worth stating precisely:
#
# > SHAP explains **the model**. It does not check whether the model learned
# > anything. A model fitted to pure noise still has an exact Shapley
# > decomposition, and it still renders as a tidy ranked bar chart with a
# > clear winner.
#
# Notebooks 04 and 12 established that every model here sits at chance. So this
# notebook runs the full modern explainability stack — SHAP, permutation
# importance, ALE — and pairs each measure with the same measure computed on
# **shuffled labels**. The question is never "what is the top driver" but
# "is the top driver further from the noise ceiling than noise usually gets".

# %%
import sys
sys.path.insert(0, "..")

import warnings
warnings.filterwarnings("ignore")

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import shap

from src import drivers, pipeline
from src.model import load_model, feature_names

sns.set_theme(style="whitegrid", palette="muted")
pd.set_option("display.width", 130)

data = pipeline.build()
X, y = data.X, data.y
print(f"{X.shape[0]} accounts, {X.shape[1]} raw features, {int(y.sum())} positives")

# %% [markdown]
# ## 1. What SHAP says
#
# TreeSHAP on the gradient booster, over the encoded matrix its trees actually
# see.

# %%
importance, shap_values, encoded = drivers.shap_importance(X, y)
print(f"encoded features: {encoded.shape[1]}\n")
print(importance.head(12).round(5).to_string())

# %%
fig, ax = plt.subplots(figsize=(9, 6))
top = importance.head(15).iloc[::-1]
ax.barh(top.index, top.values, color="steelblue")
ax.set_xlabel("mean |SHAP value|")
ax.set_title("The chart that would go in the deck")
plt.tight_layout()
plt.savefig("../outputs/figures/13_shap_importance.png", bbox_inches="tight")
plt.show()

# %%
shap.summary_plot(shap_values, encoded, max_display=12, show=False)
plt.title("SHAP beeswarm — direction and magnitude per account")
plt.tight_layout()
plt.savefig("../outputs/figures/13_shap_beeswarm.png", bbox_inches="tight")
plt.show()

# %% [markdown]
# That is a perfectly respectable driver chart. `tickets_per_seat` leads,
# followed by `latest_mrr` and `days_since_signup`; the beeswarm shows sensible
# directional structure. Read without a null it says: **support load per seat
# drives churn, then contract size, then tenure.** Every one of those is
# plausible, which is exactly the problem.

# %% [markdown]
# ### The same chart, from labels that mean nothing
#
# The fastest way to see what that chart is worth is to build it twice: once on
# the real labels, once on labels shuffled at random. If the second is also
# confident, ordered and plausible-looking, then confidence, order and
# plausibility are not evidence of anything.

# %%
shuffled_y = pd.Series(
    np.random.default_rng(0).permutation(y.values), index=y.index, name=y.name)
shuffled_importance, _, _ = drivers.shap_importance(X, shuffled_y)


def _tidy(name):
    """Encoded columns carry a num__/cat__ prefix the audience does not need."""
    return name.split("__", 1)[-1].replace("_", " ")


fig, axes = plt.subplots(1, 2, figsize=(13, 4.4), sharex=True)
panels = [(axes[0], importance, "Real labels", "#2a78d6"),
          (axes[1], shuffled_importance, "Labels shuffled at random", "#eb6834")]
for ax, series, title, colour in panels:
    top = series.head(12).iloc[::-1]
    ax.barh([_tidy(i) for i in top.index], top.values, color=colour, height=0.62)
    ax.set_xlabel("mean |SHAP value|", fontsize=10.5)
    ax.set_title(title, loc="left", fontsize=13, fontweight="bold")
    ax.tick_params(labelsize=10.5)
    ax.grid(False)
    for side in ("top", "right", "left"):
        ax.spines[side].set_visible(False)
plt.tight_layout(w_pad=3)
plt.savefig("../outputs/figures/13_shap_real_vs_shuffled.png",
            bbox_inches="tight", dpi=150)
plt.show()

# %% [markdown]
# **They are the same chart.** Same shape, same decay, comparable magnitudes,
# and the noise model is every bit as decisive about its winner. Nothing on the
# left is available to a reader that is not also on the right, which is why a
# driver chart without a null beside it is not a result — it is a rendering.

# %% [markdown]
# ## 2. What noise says
#
# Identical procedure, labels shuffled. If the ranking above reflects structure,
# the top feature should stand well clear of what shuffling produces.

# %%
# 50 rather than 20: at 20 the smallest reportable p-value is 0.05, which is
# exactly the region the concentration test lands in.
null = drivers.shap_null(X, y, n_null=50)
for key, value in null.items():
    if key not in ("null_rankings", "observed_ranking"):
        print(f"  {key:22s} {value}")

# %% [markdown]
# **The top feature is inside the noise distribution.** Observed mean |SHAP| of
# 0.318 against a shuffled-label mean of 0.278 and a p95 of 0.399 — **p = 0.24**,
# so one run in four on random labels produces a stronger apparent driver than
# the real data does.
#
# The concentration test looks at the shape of the whole ranking rather than its
# maximum: is attribution piled onto a few features, as a real driver structure
# would pile it? Observed Gini 0.678 against a null mean of 0.635, **p = 0.08**.
# Marginally more concentrated than noise, not clearing 0.05, and worth reporting
# as the one measure in this notebook that leans at all in the positive
# direction. It is also the weakest kind of evidence available — an effect that
# needs a 50-run null to see at p = 0.08 is not a driver analysis.
#
# The overlap line needs its own comparison to mean anything. A shuffled-label
# model reproduces 15% of the "real" top-10 (Jaccard). That sounds low until you
# note that two *bootstrap resamples of the real data* only reach 30%, and two
# shuffled runs reach 20% — the observed-versus-null overlap sits in the same
# band as null-versus-null. Section 3 makes this concrete.

# %%
fig, ax = plt.subplots(figsize=(8, 4))
ax.axvline(null["observed_top_shap"], color="crimson", lw=2,
           label=f"observed {null['observed_top_shap']:.3f}")
ax.axvline(null["null_top_mean"], color="grey", ls="--",
           label=f"null mean {null['null_top_mean']:.3f}")
ax.axvspan(null["null_top_mean"], null["null_top_p95"], alpha=0.2, color="grey",
           label="null mean to p95")
ax.set_xlabel("mean |SHAP| of the top feature"); ax.set_yticks([])
ax.set_title(f"Top SHAP driver vs shuffled labels (p = {null['p_value']})")
ax.legend(); plt.tight_layout()
plt.savefig("../outputs/figures/13_shap_null.png", bbox_inches="tight")
plt.show()

# %% [markdown]
# ## 3. Is the ranking even reproducible?
#
# Separate from whether it is real: does it survive resampling the same rows?
# Bootstrap, refit, re-rank, and measure agreement between every pair of runs.
# Run on shuffled labels too, because some reproducibility comes free from the
# feature distribution and only the difference is attributable to the target.

# %%
observed_stability = drivers.rank_stability(X, y, n_boot=25)
shuffled_stability = drivers.rank_stability(X, y, n_boot=25, shuffle=True)
print(pd.DataFrame([observed_stability, shuffled_stability]).to_string(index=False))

# %% [markdown]
# **Twelve different features take the top spot across 25 bootstrap resamples**
# of the same 177 accounts — and twelve do so under shuffled labels as well. The
# top-10 sets overlap 30% between runs, against 20% for noise.
#
# So the ranking is a little more stable than noise and nowhere near stable
# enough to name a driver from. "`tickets_per_seat` is the top churn driver" is a
# statement about one bootstrap sample, and the next resample of the same
# customers would have named something else.

# %% [markdown]
# ## 4. Importance for *generalisation*, not for the fit
#
# SHAP attributes what the model did on data it was fitted to. Permutation
# importance measures how much a feature is worth on **held-out** rows. On a
# memorising model those are very different quantities, and notebook 08
# established that every model here memorises — the boosters reach train AUC
# 1.000 against validation 0.54.

# %%
perm_importance, perm_null = drivers.permutation_null(X, y, n_repeats=10, n_null=8)
print(perm_importance.head(8).round(5).to_string())
print()
for key, value in perm_null.items():
    print(f"  {key:22s} {value}")

# %% [markdown]
# **The observed top permutation importance sits *below* the shuffled-label
# mean** — 0.011 against 0.046, p = 1.0. Not "indistinguishable from noise":
# comprehensively beaten by it.
#
# The gap between this and the SHAP result is the actual diagnosis, and it is
# more informative than either number alone:
#
# | measure | what it scores | result |
# |---|---|---|
# | SHAP | contribution to the **fitted** predictions | clean ranking, at the noise mean |
# | Permutation | contribution to **held-out** AUC | below the noise mean |
#
# A feature that helps in-sample and hurts out-of-sample is memorisation. SHAP
# is faithfully reporting how the model uses `tickets_per_seat` to fit 177 rows,
# and permutation importance is reporting that this use does not transfer to a
# 54-row holdout. Both are correct; only one of them answers the product team's
# question.

# %% [markdown]
# ## 5. ALE curves for the nominal top drivers
#
# Accumulated local effects rather than partial dependence. PDP holds one feature
# fixed and averages over the observed joint distribution of the rest, which
# evaluates the model on feature combinations that do not exist — with
# `usage_last_30d` and `recency_ratio_90d` correlated as strongly as they are
# here, that is extrapolation drawn as a curve. ALE uses the conditional
# distribution and stays on the data.

# %%
booster, encoded_full = drivers.fit_encoded(X, y)
top_features = importance.head(4).index

fig, axes = plt.subplots(2, 2, figsize=(12, 8))
for ax, feature in zip(axes.ravel(), top_features):
    curve = drivers.ale(booster, encoded_full, feature, n_bins=10)
    ax.plot(curve["x"], curve["ale"], "o-", color="steelblue")
    ax.axhline(0, color="grey", ls=":", lw=1)
    ax.set_title(feature, fontsize=10)
    ax.set_xlabel("feature value"); ax.set_ylabel("ALE (log-odds)")
plt.suptitle("Accumulated local effects for the four nominal top drivers", y=1.01)
plt.tight_layout()
plt.savefig("../outputs/figures/13_ale_curves.png", bbox_inches="tight")
plt.show()

# %% [markdown]
# Non-monotone, sign-changing, and with total swings that are small on the
# log-odds scale. None of these is a curve you could brief a CSM on ("risk rises
# above N tickets per seat") because none of them has a stable direction.

# %% [markdown]
# ## 6. Do two models tell the same story?
#
# A last, cheap check that needs no null at all. The shipped model is L2
# logistic; the explanations above are TreeSHAP on a gradient booster. Both were
# fitted to the same 177 accounts. If there is a real driver structure, two
# reasonable models should find overlapping versions of it.

# %%
linear = load_model("churn_model")
coefficients = pd.Series(linear.named_steps["clf"].coef_[0],
                         index=feature_names(linear, X)).abs().sort_values(ascending=False)

shap_top = list(importance.head(10).index)
linear_top = list(coefficients.head(10).index)
overlap = set(shap_top) & set(linear_top)

print("TreeSHAP top 10           :", shap_top)
print("\nLogistic |coef| top 10  :", linear_top)
print(f"\noverlap: {len(overlap)} of 10 -> {sorted(overlap) if overlap else 'none'}")
print("\nfor scale, top-10 Jaccard overlaps measured elsewhere in this notebook:")
print(f"  observed vs itself, bootstrap resampled : {observed_stability['top10_jaccard']}")
print(f"  shuffled vs itself, bootstrap resampled : {shuffled_stability['top10_jaccard']}")
print(f"  observed vs shuffled-label models       : {null['null_topk_overlap']}")

# %% [markdown]
# Two competent models on identical data, and their top-10 driver lists barely
# intersect. There is no shared driver structure to find, because there is no
# driver structure.

# %%
verdict = pd.DataFrame([
    {"check": "SHAP top feature vs shuffled labels",
     "observed": null["observed_top_shap"], "null": null["null_top_mean"],
     "p": null["p_value"], "verdict": "at the noise mean"},
    {"check": "Attribution concentration (Gini)",
     "observed": null["observed_gini"], "null": null["null_gini_mean"],
     "p": null["gini_p_value"], "verdict": "marginally above noise, p > 0.05"},
    {"check": "Top-10 stability under bootstrap (Jaccard)",
     "observed": observed_stability["top10_jaccard"],
     "null": shuffled_stability["top10_jaccard"], "p": np.nan,
     "verdict": f"{observed_stability['distinct_top1']} distinct winners in 25 fits"},
    {"check": "Permutation importance vs shuffled labels",
     "observed": perm_null["observed_top"], "null": perm_null["null_top_mean"],
     "p": perm_null["p_value"], "verdict": "below the noise mean"},
    {"check": "TreeSHAP vs logistic top-10 agreement",
     "observed": len(overlap) / 10, "null": np.nan, "p": np.nan,
     "verdict": "models disagree on the drivers"},
])
verdict.to_csv("../outputs/reports/driver_analysis.csv", index=False)
print(verdict.to_string(index=False))

# %% [markdown]
# ## Takeaway
#
# The full modern explainability stack runs cleanly on this data and produces a
# confident, plausible, well-formatted answer that is **entirely an artefact**.
# Five independent checks point the same way, and each one closes off a slide
# someone would otherwise ship:
#
# 1. the top SHAP driver sits inside the shuffled-label distribution (p = 0.24)
# 2. attribution is only marginally more concentrated than noise (p = 0.08) —
#    the single measure here that leans positive, and it does not clear 0.05
# 3. twelve different features win across 25 resamples of the same rows
# 4. permutation importance on held-out data is *beaten* by noise (p = 1.0)
# 5. two models fitted to the same data disagree about the drivers (2 of 10)
#
# The methodological point generalises past this dataset, and it is the reason
# this notebook exists rather than a single line saying "the model is at
# chance": **explainability tooling has no failure mode.** SHAP does not decline
# to produce a ranking when the model is worthless, no library warns you, and the
# output is more persuasive than a p-value precisely because it is visual and
# specific. The null comparison is not optional rigour on top of a SHAP analysis
# — without it a SHAP analysis cannot distinguish a driver from a coincidence.
#
# For the product team's "why", the answer from this notebook is the same as
# from 11 and 12: **not who, but when.** Notebook 12's calendar-time effect is
# the only surviving explanation, and it is invisible to every method here
# because a factor that moves all accounts equally at a given moment has no
# cross-sectional variance for a feature-attribution method to attribute.
