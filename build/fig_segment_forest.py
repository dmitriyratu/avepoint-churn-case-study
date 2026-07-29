"""No segment predicts churn — the figure for the exploration slide.

Bars of churn rate invite the comparison the slide is arguing against: a viewer
sees Cybersecurity at 0.09 against DevTools at 0.41 and reads a threefold
effect, when it is two churners out of twenty-three. Intervals fix that. Every
group's interval crosses the base rate, so the null is something you see rather
than something the caption asserts.

Run from the repo root:  python build/fig_segment_forest.py
"""
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from scipy import stats

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src import pipeline  # noqa: E402
from src.config import TARGET  # noqa: E402

sns.set_theme(style="whitegrid", palette="muted")

INK, MUTED, RULE, ALERT = "#1A1A1A", "#5A6270", "#D8DCE0", "#B02E2E"
DOT = "#2a78d6"

VARIABLES = [
    ("industry", "Industry"),
    ("referral_source", "How they found us"),
    ("plan_tier", "Plan"),
    ("country", "Country"),
    ("is_trial", "Started on a trial"),
]

cohort = pipeline.build().cohort
base = cohort[TARGET].mean()

# Each variable gets a header row of its own, so nothing has to share a line.
items, p_values = [], {}
for column, label in VARIABLES:
    table = pd.crosstab(cohort[column], cohort[TARGET])
    p_values[label] = stats.chi2_contingency(table)[1]
    grouped = cohort.groupby(column)[TARGET].agg(["mean", "size"])
    grouped = grouped.sort_values("mean", ascending=False)
    items.append({"kind": "header", "label": label})
    for name, (rate, n) in grouped.iterrows():
        # Wilson interval: the small groups here are exactly where the normal
        # approximation misbehaves, and some of them contain zero churners.
        lo, hi = stats.binomtest(int(round(rate * n)), int(n)).proportion_ci(
            confidence_level=0.95, method="wilson")
        items.append({"kind": "row", "name": str(name), "rate": rate,
                      "n": int(n), "lo": lo, "hi": hi})

XMAX, LABEL_X, N_X = 0.78, -0.030, 0.795
height = 0.255 * len(items) + 1.5
fig, ax = plt.subplots(figsize=(9.8, height))
fig.patch.set_facecolor("white")
ax.set_facecolor("white")

# Labels here stay to the minimum a reader needs to decode an axis. Everything
# that explains or argues belongs in the slide beside it.
top = len(items) - 1
ax.axvline(base, color=ALERT, lw=1.6, zorder=2)
ax.text(base, top + 0.5, f"base rate {base:.0%}", color=ALERT, fontsize=10.5,
        ha="center", va="bottom")

for i, item in enumerate(items):
    y = top - i
    if item["kind"] == "header":
        ax.text(-0.30, y, f"{item['label']}   ·   p = {p_values[item['label']]:.2f}",
                ha="left", va="center", fontsize=11, fontweight="bold", color=INK)
        if i:
            ax.axhline(y + 0.58, color=RULE, lw=1, zorder=1)
        continue
    ax.plot([item["lo"], item["hi"]], [y, y], color=DOT, lw=1.7, alpha=0.55,
            solid_capstyle="round", zorder=3)
    ax.scatter(item["rate"], y, s=32, color=DOT, zorder=4, linewidths=0)
    ax.text(LABEL_X, y, item["name"], ha="right", va="center", fontsize=10.5,
            color=INK)
    ax.text(N_X, y, f"n = {item['n']}", ha="left", va="center", fontsize=9.5,
            color=MUTED)

ax.set_xlim(-0.02, XMAX)
ax.set_ylim(-0.8, len(items) + 0.6)
ax.set_yticks([])
ax.grid(False)
for side in ("top", "right", "left"):
    ax.spines[side].set_visible(False)
ax.spines["bottom"].set_color(RULE)
ax.set_xticks([0, 0.2, 0.4, 0.6])
ax.set_xticklabels(["0%", "20%", "40%", "60%"], fontsize=10, color=MUTED)
ax.set_xlabel("share of the group that left within 90 days",
              fontsize=10.5, color=MUTED, labelpad=8)
ax.tick_params(axis="x", length=0)

out = ROOT / "outputs" / "figures" / "05_segment_forest.png"
plt.savefig(out, bbox_inches="tight", dpi=200)
print("wrote", out)
print(" ".join(f"{k}={v:.3f}" for k, v in p_values.items()))
