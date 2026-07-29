"""The conventional univariate pass: distributions, categories, missingness.

Every panel names the table and column it came from, because "seats" and
"priority" mean nothing without knowing which of the five files they live in.

Every panel is one series, so no categorical palette is involved — one hue
throughout, with red reserved as a flag and always carrying a text label, never
colour alone. Dashed reference lines mark what an even split would look like,
because several of these categoricals turn out to sit on it.

Run from the repo root:  python build/fig_eda_profile.py
"""
import sys
import warnings
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd

warnings.filterwarnings("ignore")
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

RAW = ROOT / "data" / "raw"
OUT = ROOT / "outputs" / "figures" / "21_eda_profile.png"

INK, MUTED, RULE = "#1A1A1A", "#5A6270", "#C9CFD6"
SOURCE = "#98A0AA"
HUE, FLAG = "#2E5F8A", "#B02E2E"

acc = pd.read_csv(RAW / "ravenstack_accounts.csv", parse_dates=["signup_date"])
sub = pd.read_csv(RAW / "ravenstack_subscriptions.csv",
                  parse_dates=["start_date", "end_date"])
tix = pd.read_csv(RAW / "ravenstack_support_tickets.csv")
use = pd.read_csv(RAW / "ravenstack_feature_usage.csv")
evt = pd.read_csv(RAW / "ravenstack_churn_events.csv")
owner = sub.set_index("subscription_id")["account_id"]

END = pd.Timestamp("2024-12-31")

NUMERIC = [
    ("Monthly spend per contract", "subscriptions.mrr_amount",
     sub.mrr_amount, "$"),
    ("Seats per customer", "accounts.seats", acc.seats, ""),
    ("Support tickets per customer", "support_tickets, counted per account",
     tix.groupby("account_id").size(), ""),
    ("Usage rows per customer", "feature_usage, counted per account",
     use.assign(a=use.subscription_id.map(owner)).groupby("a").size(), ""),
    ("Days since signup", "accounts.signup_date",
     (END - acc.signup_date).dt.days, ""),
]

CATEGORICAL = [
    ("Plan tier", "accounts.plan_tier", acc.plan_tier, None),
    ("Industry", "accounts.industry", acc.industry, None),
    ("How they found us", "accounts.referral_source", acc.referral_source, None),
    ("Ticket priority", "support_tickets.priority", tix.priority,
     "four levels, drawn evenly"),
    ("Churn reason", "churn_events.reason_code", evt.reason_code,
     "six reasons, drawn evenly"),
]

MISSING = [
    ("subscriptions.end_date", 90.3, "a contract that never ends"),
    ("support_tickets.satisfaction_score", 41.2, "and only ever 3, 4, 5 when set"),
    ("churn_events.feedback_text", 24.7, "free text, never used"),
    ("every other column, all five tables", 0.0, "nothing missing at all"),
]


def tidy(ax, title, source, flag=None):
    """Title, then the table and column it came from, then any red flag."""
    for s in ("top", "right"):
        ax.spines[s].set_visible(False)
    for s in ("left", "bottom"):
        ax.spines[s].set_color(RULE)
    ax.tick_params(colors=MUTED, labelsize=9, length=3)
    ax.set_title(title, fontsize=11.5, color=INK, loc="left",
                 pad=38 if flag else 18)
    ax.text(0, 1.115 if flag else 1.03, source, transform=ax.transAxes,
            fontsize=9, color=SOURCE, family="monospace", va="bottom")
    if flag:
        ax.text(0, 1.01, flag, transform=ax.transAxes, fontsize=9.5,
                color=FLAG, style="italic", va="bottom")


def histogram(ax, title, source, series, prefix):
    series = pd.Series(series).dropna()
    ax.hist(series, bins=26, color=HUE, alpha=.85, lw=0)
    med = series.median()
    ax.axvline(med, color=INK, lw=1.3, ls="--", zorder=3)
    ax.text(med, ax.get_ylim()[1] * .96, f" median {prefix}{med:,.0f}",
            fontsize=9.5, color=INK, va="top")
    tidy(ax, title, source)
    ax.set_ylabel("count", fontsize=9, color=MUTED)


def category(ax, title, source, series, flag):
    share = series.value_counts().sort_values() / len(series) * 100
    ax.barh(range(len(share)), share.values, color=FLAG if flag else HUE,
            alpha=.85, height=.68)
    ax.set_yticks(range(len(share)))
    ax.set_yticklabels(share.index, fontsize=9.5, color=INK)
    ax.axvline(100 / len(share), color=INK, lw=1.2, ls="--", zorder=3)
    for i, v in enumerate(share.values):
        ax.text(v + 1.1, i, f"{v:.0f}%", fontsize=9, color=MUTED, va="center")
    ax.set_xlim(0, max(share.max() * 1.42, 100 / len(share) * 1.6))
    tidy(ax, title, source, flag)
    ax.set_xlabel("share of rows", fontsize=9, color=MUTED)


def missingness(ax):
    """Laid out by hand in axes coordinates: name, bar, value, note.

    Nothing here is a real axis, so no tick label can be clipped by a margin
    and the columns line up whatever the figure width.
    """
    ax.axis("off")
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.text(0, .97, "Missing values — only three columns have any",
            fontsize=11.5, color=INK, va="top")

    x_name, x_bar, bar_w, x_pct, x_note = 0.0, 0.235, 0.30, 0.565, 0.615
    for i, (name, value, note) in enumerate(MISSING):
        y = .70 - i * .215
        colour = FLAG if value else HUE
        ax.text(x_name, y, name, fontsize=10.5, color=INK, va="center",
                family="monospace")
        ax.add_patch(plt.Rectangle((x_bar, y - .052), bar_w, .104,
                                   facecolor="#F2F4F6", lw=0))
        if value:
            ax.add_patch(plt.Rectangle((x_bar, y - .052),
                                       bar_w * value / 100, .104,
                                       facecolor=colour, alpha=.85, lw=0))
        ax.text(x_pct, y, f"{value:.0f}%", fontsize=11, color=colour,
                va="center", ha="right", weight="bold")
        ax.text(x_note, y, note, fontsize=10.5, color=MUTED, va="center")

    ax.text(x_bar, .70 - 3 * .215 - .105, "0%", fontsize=8.5, color=SOURCE,
            va="top", ha="center")
    ax.text(x_bar + bar_w, .70 - 3 * .215 - .105, "100% of rows", fontsize=8.5,
            color=SOURCE, va="top", ha="center")


fig = plt.figure(figsize=(16.4, 7.6))
grid = fig.add_gridspec(3, 5, height_ratios=[1, 1, .52], hspace=.74, wspace=.34,
                        left=.045, right=.99, top=.90, bottom=.075)

for i, (title, source, series, prefix) in enumerate(NUMERIC):
    histogram(fig.add_subplot(grid[0, i]), title, source, series, prefix)

for i, (title, source, series, flag) in enumerate(CATEGORICAL):
    category(fig.add_subplot(grid[1, i]), title, source, series, flag)

missingness(fig.add_subplot(grid[2, :]))

fig.savefig(OUT, dpi=190, facecolor="white")
print("wrote", OUT)
for title, source, series, _ in CATEGORICAL:
    share = series.value_counts(normalize=True).mul(100).round(0)
    print(f"  {source:38s} {dict(share)}")
