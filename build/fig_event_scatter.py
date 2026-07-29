"""Why the inputs hold nothing — the figure for the risks slide.

A customer's activity should happen after they sign up and cluster around their
tenure, so a plot of signup date against event date should be a band hugging the
diagonal. It is a uniform cloud instead: the events were generated independently
of the customers they are attached to. Every usage and support feature is built
from these rows, which is why no amount of feature work moves the score.

Run from the repo root:  python build/fig_event_scatter.py
"""
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

sns.set_theme(style="whitegrid", palette="muted")

INK, MUTED, RULE, ALERT = "#1A1A1A", "#5A6270", "#D8DCE0", "#B02E2E"
DOT = "#2a78d6"

RAW = ROOT / "data" / "raw"
accounts = pd.read_csv(RAW / "ravenstack_accounts.csv",
                       parse_dates=["signup_date"])
signup = accounts.set_index("account_id")["signup_date"]

# Usage is keyed by subscription, so it reaches its customer through one hop.
subs = pd.read_csv(RAW / "ravenstack_subscriptions.csv")
sub_to_account = subs.set_index("subscription_id")["account_id"]
usage = pd.read_csv(RAW / "ravenstack_feature_usage.csv",
                    parse_dates=["usage_date"])
usage["account_id"] = usage["subscription_id"].map(sub_to_account)

panels = [
    ("Product usage", usage, "usage_date"),
    ("Support tickets", pd.read_csv(RAW / "ravenstack_support_tickets.csv",
                                    parse_dates=["submitted_at"]), "submitted_at"),
]

fig, axes = plt.subplots(1, 2, figsize=(11.6, 4.6))
fig.patch.set_facecolor("white")
lo, hi = pd.Timestamp("2023-01-01"), pd.Timestamp("2024-12-31")
rng = np.random.default_rng(0)

for ax, (title, frame, date_col) in zip(axes, panels):
    frame = frame.assign(signup=frame["account_id"].map(signup)).dropna(
        subset=["signup", date_col])
    r = frame["signup"].astype("int64").corr(frame[date_col].astype("int64"))
    before = (frame[date_col] < frame["signup"]).mean()

    # Thin the cloud so the density is readable; the shape is what matters.
    shown = frame.sample(min(len(frame), 4000), random_state=0)

    ax.set_facecolor("white")
    ax.fill_between([lo, hi], [lo, hi], [lo, lo], color=ALERT, alpha=0.055,
                    zorder=1, linewidth=0)
    ax.scatter(shown["signup"], shown[date_col], s=5, color=DOT, alpha=0.20,
               linewidths=0, zorder=3)
    ax.plot([lo, hi], [lo, hi], color=ALERT, lw=1.6, zorder=4)

    ax.text(pd.Timestamp("2024-11-20"), pd.Timestamp("2023-02-20"),
            f"{before:.0%} of rows dated\nbefore the customer existed",
            ha="right", va="bottom", fontsize=10.5, color=ALERT)
    ax.text(pd.Timestamp("2023-02-10"), pd.Timestamp("2024-11-10"),
            f"r = {r:.3f}", fontsize=13, fontweight="bold", color=INK,
            va="top")
    ax.text(pd.Timestamp("2023-02-10"), pd.Timestamp("2024-09-20"),
            "0 means no connection at all", fontsize=10, color=MUTED, va="top")

    ax.set_title(title, loc="left", fontsize=13.5, fontweight="bold",
                 color=INK, pad=10)
    ax.set_xlabel("when the customer signed up", fontsize=11, color=MUTED,
                  labelpad=7)
    ax.set_ylabel("when the event happened", fontsize=11, color=MUTED,
                  labelpad=7)
    ax.set_xlim(lo, hi)
    ax.set_ylim(lo, hi)
    ax.grid(False)
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)
    for side in ("left", "bottom"):
        ax.spines[side].set_color(RULE)
    ticks = pd.date_range("2023-01-01", "2025-01-01", freq="6MS")
    ax.set_xticks(ticks)
    ax.set_yticks(ticks)
    ax.set_xticklabels([f"{d:%b %y}" for d in ticks], fontsize=9.5, color=MUTED)
    ax.set_yticklabels([f"{d:%b %y}" for d in ticks], fontsize=9.5, color=MUTED)
    ax.tick_params(length=0)

fig.text(0.5, -0.035,
         "Real data would sit as a band above the red line — activity follows "
         "signup. This is a square cloud, so the events belong to nobody in "
         "particular.",
         ha="center", fontsize=11, color=MUTED, style="italic")

plt.tight_layout(w_pad=3.4)
out = ROOT / "outputs" / "figures" / "04_event_dates_vs_signup.png"
plt.savefig(out, bbox_inches="tight", dpi=200)
print("wrote", out)
