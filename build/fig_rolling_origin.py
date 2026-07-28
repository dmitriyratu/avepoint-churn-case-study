"""The rolling-origin timeline used on slide 2 of the main deck.

One question — "will this customer leave in the next 90 days?" — asked from four
quarter-end vantage points. The picture has to carry three facts at once: that
features never cross the cutoff, that the answer window always closes inside the
extract, and that this is one question repeated rather than four questions.

Run from the repo root:  python build/fig_rolling_origin.py
"""
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src import pipeline, robustness  # noqa: E402
from src.config import HORIZON_DAYS, TARGET  # noqa: E402
from src.labeling import build_cohort  # noqa: E402

sns.set_theme(style="whitegrid", palette="muted")

INK, MUTED, RULE, ALERT = "#1A1A1A", "#5A6270", "#D8DCE0", "#B02E2E"
SEEN, LABEL = "#2a78d6", "#eb6834"

START = pd.Timestamp("2023-01-01")
EXTRACT_END = pd.Timestamp("2024-12-31")
DAY = pd.Timedelta(days=1)
HORIZON = pd.Timedelta(days=HORIZON_DAYS)

tables = pipeline.clean_all(pipeline.load_all())
cutoffs = robustness.rolling_origin_cutoffs(n=4)
rows = []
for c in cutoffs:
    cohort = build_cohort(tables, cutoff=c, prediction_start=c)
    rows.append((c, len(cohort), int(cohort[TARGET].sum())))

fig, ax = plt.subplots(figsize=(12.5, 2.35))
ax.set_facecolor("white")
fig.patch.set_facecolor("white")

# Earliest cutoff on top, so the eye reads the timeline downwards as it moves
# forwards — the same direction the sentence "asked again each quarter" runs.
for i, (cut, n, pos) in enumerate(rows):
    y = len(rows) - 1 - i
    ax.barh(y, cut - START, left=START, height=0.40, color=SEEN, zorder=3)
    ax.barh(y, HORIZON, left=cut + DAY, height=0.40, color=LABEL, zorder=3)
    ax.plot([cut, cut], [y - 0.28, y + 0.28], color=ALERT, lw=1.8, zorder=5)
    ax.text(START - pd.Timedelta(days=20), y, f"{cut:%d %b %Y}", ha="right",
            va="center", fontsize=10.5, color=INK, fontweight="bold")
    ax.text(EXTRACT_END + pd.Timedelta(days=95), y,
            f"{n} could still leave · {pos} did", ha="left", va="center",
            fontsize=10.5, color=MUTED)

ax.axvline(EXTRACT_END, color=RULE, lw=1.4, ls=(0, (3, 3)), zorder=2)
ax.text(EXTRACT_END, len(rows) - 0.42, "the data stops here", fontsize=9.5,
        color=MUTED, ha="right", va="bottom", style="italic")

legend = [(SEEN, "what we knew on that date"),
          (LABEL, f"the next {HORIZON_DAYS} days — the answer, never a feature")]
for j, (colour, text) in enumerate(legend):
    x = START + pd.Timedelta(days=j * 480)
    ax.barh(-1.05, pd.Timedelta(days=42), left=x, height=0.26, color=colour,
            zorder=3)
    ax.text(x + pd.Timedelta(days=56), -1.05, text, va="center", fontsize=10,
            color=MUTED)

ax.set_xlim(START - pd.Timedelta(days=230), EXTRACT_END + pd.Timedelta(days=560))
ax.set_ylim(-1.55, len(rows) - 0.15)
ax.set_yticks([])
ax.grid(False)
for side in ("top", "right", "left", "bottom"):
    ax.spines[side].set_visible(False)
ticks = pd.date_range("2023-01-01", "2024-12-01", freq="3MS")
ax.set_xticks(ticks)
ax.set_xticklabels([d.strftime("%b\n%Y") if d.month == 1 else d.strftime("%b")
                    for d in ticks], fontsize=9, color=MUTED)
ax.tick_params(axis="x", length=0, pad=2)
ax.set_title("One question, asked from four dates", loc="left", fontsize=13.5,
             fontweight="bold", color=INK, pad=10)

out = ROOT / "outputs" / "figures" / "02_rolling_origin.png"
plt.savefig(out, bbox_inches="tight", dpi=200)
print("wrote", out)
