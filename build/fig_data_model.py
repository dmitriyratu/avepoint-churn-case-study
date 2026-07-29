"""The five tables, how they join, and which joins can be trusted.

Every table resolves by ID. Two of them do not resolve in time, and that
distinction is the whole story of this project, so the diagram carries it on
the arrows rather than in a caption.
"""
import sys
import tempfile
import warnings
from pathlib import Path

import graphviz
from PIL import Image

warnings.filterwarnings("ignore")
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

OUT = ROOT / "outputs" / "figures" / "19_data_model.png"

INK = "#1A1A1A"
MUTED = "#5A6270"
BLUE = "#2E5F8A"
RED = "#B02E2E"
GREEN = "#1E7A4B"
RULE = "#C9CFD6"
DPI = 200

FONT = "Segoe UI"

TABLES = [
    ("accounts", "accounts", "500 rows · one per customer", GREEN,
     [("account_id", "the customer"),
      ("signup_date", "the day they joined"),
      ("industry, country, plan_tier", "who they are"),
      ("churn_flag", "unusable — no date")]),
    ("subscriptions", "subscriptions", "5,000 rows · 10 per customer", GREEN,
     [("subscription_id", "the contract"),
      ("account_id", "joins accounts"),
      ("start_date, end_date", "only 486 have ended"),
      ("mrr_amount, seats, plan_tier", "what they pay")]),
    ("usage", "feature_usage", "25,000 rows", RED,
     [("subscription_id", "joins subscriptions"),
      ("usage_date", "unrelated to the customer"),
      ("usage_count, error_count", "what they did")]),
    ("tickets", "support_tickets", "2,000 rows · 492 customers", RED,
     [("account_id", "joins accounts"),
      ("submitted_at", "unrelated to the customer"),
      ("priority, satisfaction_score", "only ever 3, 4 or 5")]),
    ("churn", "churn_events", "600 rows · 352 customers", RED,
     [("account_id", "joins accounts"),
      ("churn_date", "the target is built here"),
      ("reason_code", "175 customers leave twice")]),
]

EDGES = [
    ("accounts", "subscriptions", "1 to 10", GREEN,
     "sound: no contract starts\nbefore its customer did"),
    ("subscriptions", "usage", "1 to 5", RED,
     "broken in time: r = 0.002\n53% dated before signup"),
    ("accounts", "tickets", "1 to 4", RED,
     "broken in time: r = 0.016\n54% dated before signup"),
    ("accounts", "churn", "1 to 1.7", RED,
     "dates are a coin toss\nbetween signup and file end"),
]


def cell(text, colour, size, bold=False, pad=3):
    weight = ' face="Segoe UI Semibold"' if bold else ""
    return (f'<td align="left" cellpadding="{pad}">'
            f'<font point-size="{size}" color="{colour}"{weight}>{text}</font></td>')


def table_node(key, name, subtitle, accent, columns):
    head = (f'<tr><td colspan="2" align="left" cellpadding="6" bgcolor="{accent}18">'
            f'<font point-size="15" color="{accent}" face="Segoe UI Semibold">'
            f'{name}</font><br align="left"/>'
            f'<font point-size="11" color="{MUTED}">{subtitle}</font></td></tr>')
    rows = "".join(f"<tr>{cell(col, INK, 12)}{cell(note, MUTED, 11)}</tr>"
                   for col, note in columns)
    label = (f'<<table border="0" cellborder="0" cellspacing="0">'
             f"{head}{rows}</table>>")
    return key, label, accent


def build():
    g = graphviz.Digraph("data_model")
    g.attr(rankdir="LR", bgcolor="white", ranksep="1.5", nodesep="0.35",
           splines="spline", pad="0.15")
    g.attr("node", shape="box", style="rounded", fontname=FONT,
           color=RULE, penwidth="1.4", margin="0.02")
    g.attr("edge", fontname=FONT, fontsize="10", penwidth="1.6",
           arrowsize="0.7", fontcolor=MUTED)

    for spec in TABLES:
        key, label, accent = table_node(*spec)
        g.node(key, label=label, color=accent + "66")

    for src, dst, card, colour, note in EDGES:
        g.edge(src, dst, label=f"  {card}\\n{note}  ", color=colour,
               fontcolor=colour if colour == RED else MUTED,
               style="solid" if colour == GREEN else "dashed")
    return g


def main():
    with tempfile.TemporaryDirectory() as d:
        path = build().render(Path(d) / "er", format="png", cleanup=True)
        Image.open(path).convert("RGB").save(OUT, dpi=(DPI, DPI))
    im = Image.open(OUT)
    print("wrote", OUT, im.size)


if __name__ == "__main__":
    main()
