"""Event dates against the signup date of the customer they belong to.

Every behavioural feature — usage in the last 30 days, tickets in the last 90 —
assumes an event happened to the customer it is filed under, after they existed.
Plotting the two dates against each other tests that assumption directly, and
the third panel shows what the same plot looks like when it holds.
"""
import sys
import warnings
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.dates as mdates
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

warnings.filterwarnings("ignore")
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

OUT = ROOT / "outputs" / "figures" / "18_events_not_joined.png"
RAW = ROOT / "data" / "raw"

INK = "#1A1A1A"
MUTED = "#5A6270"
BLUE = "#2E5F8A"
RED = "#B02E2E"
GREEN = "#1E7A4B"
RULE = "#C9CFD6"

SEED = 0
START, END = pd.Timestamp("2023-01-01"), pd.Timestamp("2024-12-31")


def load():
    """Both event tables, carrying the signup date of the account they belong to.

    Usage is filed against a subscription rather than an account, so it reaches
    its customer through the subscriptions table.
    """
    acc = pd.read_csv(RAW / "ravenstack_accounts.csv", parse_dates=["signup_date"])
    subs = pd.read_csv(RAW / "ravenstack_subscriptions.csv")
    use = pd.read_csv(RAW / "ravenstack_feature_usage.csv", parse_dates=["usage_date"])
    tix = pd.read_csv(RAW / "ravenstack_support_tickets.csv", parse_dates=["submitted_at"])

    signup = acc.set_index("account_id")["signup_date"]
    owner = subs.set_index("subscription_id")["account_id"]
    use = use.assign(signup=use["subscription_id"].map(owner).map(signup))
    tix = tix.assign(signup=tix["account_id"].map(signup))
    return (acc,
            use.dropna(subset=["signup"]).rename(columns={"usage_date": "event"}),
            tix.dropna(subset=["signup"]).rename(columns={"submitted_at": "event"}))


def realistic(acc, rng, per_account=50):
    """The same plot for a file where the join was recorded properly.

    Activity starts when the customer does and thins out with tenure — the
    shape every behavioural feature is written to expect.
    """
    signup = acc["signup_date"].repeat(per_account).reset_index(drop=True)
    room = (END - signup).dt.days.clip(lower=1).to_numpy()
    offset = rng.exponential(120, len(signup)).round()
    keep = offset < room
    signup = signup[keep]
    event = signup + pd.to_timedelta(offset[keep], unit="D")
    return pd.DataFrame({"signup": signup.to_numpy(), "event": event.to_numpy()})


def panel(ax, df, title, colour, note):
    x = mdates.date2num(df["signup"])
    y = mdates.date2num(df["event"])
    lo, hi = mdates.date2num(START), mdates.date2num(END)

    ax.fill_between([lo, hi], [lo, hi], lo, color=RED, alpha=0.055, lw=0,
                    zorder=0)
    ax.plot([lo, hi], [lo, hi], color=MUTED, lw=1.3, ls="--", zorder=2)
    ax.scatter(x, y, s=2.2, color=colour, alpha=0.16, lw=0, zorder=1,
               rasterized=True)

    before = (df["event"] < df["signup"]).mean()
    r = np.corrcoef(x, y)[0, 1]

    ax.set_xlim(lo, hi)
    ax.set_ylim(lo, hi)
    ax.set_aspect("equal")
    for s in ("top", "right"):
        ax.spines[s].set_visible(False)
    for s in ("left", "bottom"):
        ax.spines[s].set_color(RULE)
    ax.tick_params(colors=MUTED, labelsize=9, length=3)
    ax.xaxis.set_major_locator(mdates.YearLocator())
    ax.yaxis.set_major_locator(mdates.YearLocator())
    ax.xaxis.set_major_formatter(mdates.DateFormatter("%Y"))
    ax.yaxis.set_major_formatter(mdates.DateFormatter("%Y"))
    ax.set_xlabel("the day this customer signed up", fontsize=10, color=MUTED)
    ax.set_ylabel("the day the event happened", fontsize=10, color=MUTED)

    ax.set_title(title, fontsize=13, weight="bold", color=colour, loc="left",
                 pad=26)
    ax.text(0, 1.045, f"r = {r:.3f}    ·    {before:.0%} land before signup",
            transform=ax.transAxes, fontsize=10.5, color=MUTED, va="bottom")
    ax.text(0.035, 0.955, note, transform=ax.transAxes, fontsize=9.5,
            color=MUTED, va="top", style="italic")


def main():
    rng = np.random.default_rng(SEED)
    acc, use, tix = load()

    fig, axes = plt.subplots(1, 3, figsize=(13.2, 4.35))
    fig.subplots_adjust(left=0.055, right=0.99, top=0.83, bottom=0.13,
                        wspace=0.30)

    panel(axes[0], use, "Feature usage, as delivered", RED,
          "shaded: impossible\nactivity before signup")
    panel(axes[1], tix, "Support tickets, as delivered", RED,
          "shaded: impossible\nactivity before signup")
    panel(axes[2], realistic(acc, rng), "What a correct join looks like", GREEN,
          "every event above the line,\nclustered after signup")

    fig.savefig(OUT, dpi=200, facecolor="white")
    print("wrote", OUT)
    for name, df in (("usage", use), ("tickets", tix)):
        bad = (df["event"] < df["signup"]).sum()
        r = np.corrcoef(mdates.date2num(df["signup"]),
                        mdates.date2num(df["event"]))[0, 1]
        print(f"  {name:8s} n={len(df):6d}  before signup={bad:6d} "
              f"({bad / len(df):.1%})  r={r:+.4f}")


if __name__ == "__main__":
    main()
