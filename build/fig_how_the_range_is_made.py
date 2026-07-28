"""Where the error bar comes from — the figure for slide 3.

Two steps that people conflate. Asking the question at four dates decides how
much data there is. Splitting those rows and re-scoring decides how sure we are.
The picture separates them, because "0.56 give or take 0.05" is meaningless
until you can see that the 0.05 is a spread of real measurements.

Run from the repo root:  python build/fig_how_the_range_is_made.py
"""
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from matplotlib.gridspec import GridSpec

from sklearn.model_selection import StratifiedGroupKFold, cross_val_score

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src import pipeline, robustness  # noqa: E402
from src.config import TARGET  # noqa: E402
from src.labeling import build_cohort  # noqa: E402

sns.set_theme(style="whitegrid", palette="muted")

INK, MUTED, RULE, ALERT = "#1A1A1A", "#5A6270", "#D8DCE0", "#B02E2E"
TRAIN, TEST = "#2a78d6", "#eb6834"
N_FOLDS, N_ROUNDS = 5, 5

cutoffs = robustness.rolling_origin_cutoffs(n=4)
tables = pipeline.clean_all(pipeline.load_all())
counts = [len(build_cohort(tables, cutoff=c, prediction_start=c)) for c in cutoffs]

X, y, groups = robustness.pooled_dataset(cutoffs)
scores = []
for seed in range(N_ROUNDS):
    cv = StratifiedGroupKFold(N_FOLDS, shuffle=True, random_state=seed)
    scores += list(cross_val_score(robustness._selected(), X, y,
                                   scoring="roc_auc", cv=cv.split(X, y, groups)))
scores = np.array(scores)

fig = plt.figure(figsize=(13.0, 3.5))
gs = GridSpec(1, 3, width_ratios=[1.02, 0.92, 1.16], wspace=0.42, figure=fig)
fig.patch.set_facecolor("white")


def frame(ax, title, step):
    ax.set_facecolor("white")
    ax.grid(False)
    for side in ax.spines:
        ax.spines[side].set_visible(False)
    ax.set_title(f"{step}   {title}", loc="left", fontsize=12,
                 fontweight="bold", color=INK, pad=12)


# ---------------------------------------------------------------- step 1
ax = fig.add_subplot(gs[0])
frame(ax, "How much data", "STEP 1")
for i, (cut, n) in enumerate(zip(cutoffs, counts)):
    ypos = len(cutoffs) - i
    ax.barh(ypos, n, height=0.46, color=TRAIN, zorder=3)
    ax.text(-6, ypos, f"{cut:%b %Y}", ha="right", va="center", fontsize=10,
            color=INK)
    ax.text(n + 6, ypos, f"{n}", ha="left", va="center", fontsize=10,
            color=MUTED)
ax.plot([0, 200], [0.42, 0.42], color=RULE, lw=1.1)
ax.text(200, -0.12, f"{sum(counts)} rows", ha="right", va="center",
        fontsize=12.5, fontweight="bold", color=INK)
ax.text(-6, -0.12, "one question, four dates", ha="right", va="center",
        fontsize=10, color=MUTED, style="italic")
ax.set_xlim(-95, 250)
ax.set_ylim(-0.75, 4.75)
ax.set_xticks([])
ax.set_yticks([])

# ---------------------------------------------------------------- step 2
ax = fig.add_subplot(gs[1])
frame(ax, "How sure we are", "STEP 2")
for row in range(N_FOLDS):
    for col in range(N_FOLDS):
        held_out = col == row
        ax.barh(N_FOLDS - row, 0.86, left=col, height=0.62, align="center",
                color=TEST if held_out else TRAIN, zorder=3)
    ax.text(-0.25, N_FOLDS - row, f"split {row + 1}", ha="right", va="center",
            fontsize=9.5, color=MUTED)
    ax.text(N_FOLDS + 0.15, N_FOLDS - row, "→", ha="left", va="center",
            fontsize=10, color=MUTED)
ax.text(N_FOLDS / 2 - 0.07, 0.30,
        f"then reshuffle and repeat, {N_ROUNDS} rounds in all",
        ha="center", va="center", fontsize=10, color=MUTED, style="italic")
ax.barh(-0.52, 0.86, left=0, height=0.34, color=TEST, zorder=3)
ax.text(1.02, -0.52, "scored on", va="center", fontsize=9.5, color=MUTED)
ax.barh(-0.52, 0.86, left=3.6, height=0.34, color=TRAIN, zorder=3)
ax.text(4.62, -0.52, "trained on", va="center", fontsize=9.5, color=MUTED)
ax.set_xlim(-1.9, N_FOLDS + 2.1)
ax.set_ylim(-1.05, N_FOLDS + 0.75)
ax.set_xticks([])
ax.set_yticks([])

# ---------------------------------------------------------------- the range
ax = fig.add_subplot(gs[2])
frame(ax, f"{len(scores)} measurements, not one", "THE RANGE")
lo, hi = scores.mean() - scores.std(), scores.mean() + scores.std()
ax.axvspan(lo, hi, color=TRAIN, alpha=0.11, zorder=1)
rng = np.random.default_rng(0)
ax.scatter(scores, rng.uniform(0.30, 1.32, len(scores)), s=34, color=TRAIN,
           alpha=0.78, linewidths=0, zorder=4)
ax.axvline(scores.mean(), color=INK, lw=1.8, zorder=5)
ax.axvline(0.50, color=ALERT, lw=1.5, ls=(0, (4, 3)), zorder=5)
ax.text(0.497, 1.70, "guessing", ha="right", fontsize=10, color=ALERT)
ax.text(scores.mean(), 1.70, f"{scores.mean():.3f} ± {scores.std():.3f}",
        ha="center", fontsize=11.5, fontweight="bold", color=INK)
ax.text(0.5, -0.17, "every dot is the same model, scored on customers it never saw",
        transform=ax.transAxes, ha="center", va="center", fontsize=9.5,
        color=MUTED, style="italic")
ax.set_xlim(0.415, 0.715)
ax.set_ylim(-0.10, 2.05)
ax.set_yticks([])
ax.set_xticks([0.45, 0.50, 0.55, 0.60, 0.65, 0.70])
ax.set_xticklabels([f"{t:.2f}" for t in [0.45, 0.5, 0.55, 0.6, 0.65, 0.7]],
                   fontsize=9.5, color=MUTED)
ax.tick_params(axis="x", length=0, pad=3)

out = ROOT / "outputs" / "figures" / "03_how_the_range_is_made.png"
plt.savefig(out, bbox_inches="tight", dpi=200)
print("wrote", out)
print(f"{len(scores)} scores  mean {scores.mean():.4f}  sd {scores.std():.4f}  "
      f"min {scores.min():.3f}  max {scores.max():.3f}")
