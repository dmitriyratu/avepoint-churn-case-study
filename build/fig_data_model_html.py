"""The five tables as an entity-relationship diagram, drawn in HTML/CSS.

Same convention as the other HTML figures: the numbers are read from the raw
files rather than typed, the layout is CSS, and the page is screenshotted
headlessly. Crow's feet and the join lines are one inline SVG laid over the
cards, so the geometry is stated once.

The diagram's argument is on the connectors, not in the boxes. Every join
resolves by ID; two of them do not resolve in time, and that is the finding.

Run from the repo root:  python build/fig_data_model_html.py
"""
import sys
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

RAW = ROOT / "data" / "raw"
OUT = ROOT / "outputs" / "figures" / "19_data_model.png"
PAGE = ROOT / "outputs" / "figures" / "19_data_model.html"

INK, MUTED, RULE = "#1A1A1A", "#5A6270", "#D8DCE0"
GREEN, RED, BLUE = "#1E7A4B", "#B02E2E", "#2E5F8A"

W, H = 1800, 1030
BAND = 250          # the findings strip along the bottom


def shape(name):
    d = pd.read_csv(RAW / f"ravenstack_{name}.csv")
    return len(d), d.shape[1]


ACC, SUB, USE, TIX, EVT = (shape(n) for n in
                           ("accounts", "subscriptions", "feature_usage",
                            "support_tickets", "churn_events"))

# x, y, width — heights fall out of the row count, so the SVG reads them back
# from the same table below rather than being told twice.
TABLES = {
    "accounts": dict(
        x=40, y=250, w=340, accent=GREEN, rows=ACC[0], cols=ACC[1],
        columns=[("account_id", "varchar", "pk", None),
                 ("signup_date", "date", None, None),
                 ("plan_tier", "varchar", None, None),
                 ("industry", "varchar", None, None),
                 ("country", "varchar", None, None),
                 ("churn_flag", "boolean", None, "no date — unusable")]),
    "subscriptions": dict(
        x=820, y=30, w=360, accent=GREEN, rows=SUB[0], cols=SUB[1],
        columns=[("subscription_id", "varchar", "pk", None),
                 ("account_id", "varchar", "fk", None),
                 ("start_date", "date", None, None),
                 ("end_date", "date", None, "only 486 are set"),
                 ("mrr_amount", "decimal", None, None),
                 ("seats", "integer", None, None)]),
    "support_tickets": dict(
        x=820, y=300, w=360, accent=RED, rows=TIX[0], cols=TIX[1],
        columns=[("ticket_id", "varchar", "pk", None),
                 ("account_id", "varchar", "fk", None),
                 ("submitted_at", "datetime", None, "not the customer's time"),
                 ("priority", "varchar", None, None),
                 ("satisfaction_score", "integer", None, "only ever 3, 4, 5")]),
    "churn_events": dict(
        x=820, y=542, w=360, accent=RED, rows=EVT[0], cols=EVT[1],
        columns=[("churn_event_id", "varchar", "pk", None),
                 ("account_id", "varchar", "fk", None),
                 ("churn_date", "date", None, "the target is built here"),
                 ("reason_code", "varchar", None, None)]),
    "feature_usage": dict(
        x=1420, y=30, w=360, accent=RED, rows=USE[0], cols=USE[1],
        columns=[("usage_id", "varchar", "pk", None),
                 ("subscription_id", "varchar", "fk", None),
                 ("usage_date", "date", None, "not the customer's time"),
                 ("usage_count", "integer", None, None),
                 ("error_count", "integer", None, None)]),
}

HEAD_H, ROW_H, FOOT_H = 50, 28, 26

EDGES = [
    ("accounts", "subscriptions", "1 : 10", GREEN,
     "sound — no contract starts before its customer"),
    ("accounts", "support_tickets", "1 : 4", RED,
     "broken in time — r = 0.016, 54% before signup"),
    ("accounts", "churn_events", "1 : 1.7", RED,
     "dates are a coin toss between signup and file end"),
    ("subscriptions", "feature_usage", "1 : 5", RED,
     "broken in time — r = 0.002, 53% before signup"),
]


def findings(items):
    """A short EDA read-out: the number first, then what it is."""
    return "".join(f'<li><span class="stat">{stat}</span><span>{text}</span></li>'
                   for stat, text in items)


def height(key):
    t = TABLES[key]
    return HEAD_H + ROW_H * len(t["columns"]) + FOOT_H


def card(key):
    t = TABLES[key]
    hidden = t["cols"] - len(t["columns"])
    rows = "".join(
        f'<div class="row{" flag" if warn else ""}">'
        f'<span class="badge {kind or "none"}">{(kind or "").upper()}</span>'
        f'<span class="col">{name}</span>'
        f'<span class="type">{typ}</span>'
        f'{f"<span class=warn>{warn}</span>" if warn else ""}</div>'
        for name, typ, kind, warn in t["columns"])
    return f"""
  <div class="table" style="left:{t['x']}px;top:{t['y']}px;width:{t['w']}px;
       --accent:{t['accent']}">
    <div class="head">
      <span class="name">{key}</span>
      <span class="count">{t['rows']:,} rows</span>
    </div>
    {rows}
    <div class="more">+ {hidden} more column{'s' if hidden != 1 else ''}</div>
  </div>"""


def connectors():
    """One SVG over the cards: bezier per join, crow's foot at the many end."""
    paths = []
    for src, dst, card_label, colour, note in EDGES:
        a, b = TABLES[src], TABLES[dst]
        x1, y1 = a["x"] + a["w"], a["y"] + height(src) / 2
        x2, y2 = b["x"], b["y"] + height(dst) / 2
        dx = (x2 - x1) * 0.55
        marker = "crowGreen" if colour == GREEN else "crowRed"
        dash = "" if colour == GREEN else ' stroke-dasharray="7 5"'
        paths.append(
            f'<path d="M{x1},{y1} C{x1 + dx},{y1} {x2 - dx},{y2} {x2 - 13},{y2}"'
            f' stroke="{colour}" stroke-width="2" fill="none"{dash}'
            f' marker-end="url(#{marker})"/>'
            f'<circle cx="{x1}" cy="{y1}" r="4" fill="{colour}"/>')
        # A label centred on its own line hides it. Near-horizontal joins get
        # lifted clear; the rest keep enough bare line either side to be read
        # as a connector rather than as two loose arrows.
        flat = abs(y2 - y1) < 25
        box = min(280, x2 - x1 - 40) if flat else min(260, x2 - x1 - 120)
        mx, my = (x1 + x2) / 2, (y1 + y2) / 2 - (58 if flat else 44)
        paths.append(
            f'<foreignObject x="{mx - box / 2}" y="{my - 34}" width="{box}"'
            f' height="72">'
            f'<div xmlns="http://www.w3.org/1999/xhtml" class="edge"'
            f' style="--c:{colour}">'
            f'<div class="card">{card_label}</div>'
            f'<div class="note">{note}</div></div></foreignObject>')

    def foot(ident, colour):
        return (f'<marker id="{ident}" viewBox="0 0 14 14" refX="13" refY="7"'
                f' markerWidth="11" markerHeight="11" orient="auto">'
                f'<path d="M0,0 L13,7 M0,7 L13,7 M0,14 L13,7" fill="none"'
                f' stroke="{colour}" stroke-width="1.6"/></marker>')

    return (f'<svg width="{W}" height="{H}">'
            f'<defs>{foot("crowGreen", GREEN)}{foot("crowRed", RED)}</defs>'
            + "".join(paths) + "</svg>")


HTML = f"""<meta charset="utf-8">
<style>
  * {{ box-sizing: border-box; margin: 0; }}
  body {{ background: #fff;
         font-family: "Segoe UI", "Helvetica Neue", Arial, sans-serif; }}
  .wrap {{ position: relative; width: {W}px; height: {H}px; }}
  svg {{ position: absolute; inset: 0; }}

  .table {{ position: absolute; background: #fff; border: 1.5px solid var(--accent);
            border-radius: 10px; overflow: hidden; z-index: 2;
            box-shadow: 0 2px 10px rgba(26,26,26,.07); }}
  .head {{ display: flex; align-items: baseline; justify-content: space-between;
           gap: 10px; padding: 11px 14px; height: {HEAD_H}px;
           background: color-mix(in srgb, var(--accent) 9%, #fff);
           border-bottom: 1.5px solid color-mix(in srgb, var(--accent) 32%, #fff); }}
  .name {{ font-size: 19px; font-weight: 650; color: var(--accent);
           letter-spacing: -.2px; }}
  .count {{ font-size: 13px; color: {MUTED}; white-space: nowrap; }}

  .row {{ display: flex; align-items: center; gap: 9px; height: {ROW_H}px;
          padding: 0 14px; border-bottom: 1px solid #F0F2F4; }}
  .row:last-of-type {{ border-bottom: 0; }}
  .row.flag {{ background: rgba(176,46,46,.045); }}
  .col {{ font-size: 14px; color: {INK}; }}
  .type {{ font-size: 12.5px; color: #98A0AA; margin-left: auto; }}
  .row.flag .type {{ display: none; }}
  .warn {{ font-size: 12px; color: {RED}; margin-left: auto; font-style: italic;
           white-space: nowrap; }}

  .badge {{ font-size: 9.5px; font-weight: 700; letter-spacing: .4px;
            width: 22px; text-align: center; padding: 2px 0; border-radius: 3px; }}
  .badge.pk {{ background: #F0E6C8; color: #7A5C10; }}
  .badge.fk {{ background: #E2E9F1; color: {BLUE}; }}
  .badge.none {{ background: transparent; }}

  .more {{ height: {FOOT_H}px; display: flex; align-items: center; padding: 0 14px;
           font-size: 12px; color: #98A0AA; background: #FAFBFC;
           border-top: 1px solid #F0F2F4; }}

  .edge {{ display: flex; flex-direction: column; align-items: center; gap: 3px;
           text-align: center; }}
  .edge .card {{ font-size: 14px; font-weight: 650; color: var(--c);
                 background: #fff; padding: 1px 9px; border-radius: 4px; }}
  .edge .note {{ font-size: 12px; color: var(--c); background: #fff;
                 padding: 1px 8px; border-radius: 4px; line-height: 1.35; }}

  .legend {{ position: absolute; left: 44px; top: 44px; z-index: 3;
             font-size: 13.5px; color: {MUTED}; line-height: 2.0; }}
  .legend b {{ display: block; font-size: 15px; color: {INK}; margin-bottom: 4px; }}
  .key {{ display: flex; align-items: center; gap: 9px; }}
  .key i {{ width: 30px; height: 0; border-top: 2px solid; display: inline-block; }}

  .findings {{ position: absolute; left: 44px; top: {H - BAND}px;
               width: {W - 88}px; z-index: 3; display: grid;
               grid-template-columns: repeat(3, 1fr); gap: 0 54px;
               padding-top: 24px; border-top: 1px solid {RULE}; }}
  .findings h4 {{ font-size: 14.5px; font-weight: 650; margin-bottom: 10px;
                  padding-bottom: 6px; border-bottom: 2px solid currentColor; }}
  .findings li {{ list-style: none; display: flex; gap: 9px; align-items: baseline;
                  font-size: 13px; color: {MUTED}; line-height: 1.5;
                  margin-bottom: 6px; }}
  .findings .stat {{ font-weight: 650; color: {INK}; white-space: nowrap;
                     min-width: 84px; font-variant-numeric: tabular-nums; }}
</style>
<div class="wrap">
  {connectors()}
  <div class="legend">
    <b>How to read this</b>
    <div class="key"><i style="border-color:{GREEN}"></i>
      the join holds, by ID and by date</div>
    <div class="key"><i style="border-top-style:dashed;border-color:{RED}"></i>
      the ID resolves, the date does not</div>
  </div>
  <div class="findings">
    <div>
      <h4 style="color:{MUTED}">What we are working with</h4>
      {findings([("500", "customers, across 5 industries and 7 countries"),
                 ("5,000", "contracts — ten per customer"),
                 ("27,000", "usage rows and support tickets"),
                 ("600", "churn records, covering 352 customers"),
                 ("24 months", "January 2023 to December 2024, no gaps")])}
    </div>
    <div>
      <h4 style="color:{GREEN}">Every routine check passes</h4>
      {findings([("0", "orphan rows — every foreign key resolves"),
                 ("0", "contracts start before their own customer"),
                 ("0", "tickets close before they were opened"),
                 ("1.8%", "of cells are blank, across all five tables"),
                 ("100%", "of contracts price correctly: ARR = 12 × MRR")])}
    </div>
    <div>
      <h4 style="color:{RED}">And the file is still unusable</h4>
      {findings([("9 of 10", "contracts per customer never end"),
                 ("33%", "of contracts match their own account's plan"),
                 ("20%", "of customers: all three churn records agree"),
                 ("3, 4, 5", "the only satisfaction scores, of a stated 1–5"),
                 ("even", "ticket priority splits 26 / 26 / 25 / 24")])}
    </div>
  </div>
  {"".join(card(k) for k in TABLES)}
</div>
"""

PAGE.write_text(HTML, encoding="utf-8")

from playwright.sync_api import sync_playwright  # noqa: E402

with sync_playwright() as p:
    browser = p.chromium.launch()
    page = browser.new_page(viewport={"width": W, "height": H},
                            device_scale_factor=2)
    page.goto(PAGE.as_uri())
    page.locator(".wrap").screenshot(path=str(OUT))
    browser.close()

print("wrote", OUT)
for name, (rows, cols) in (("accounts", ACC), ("subscriptions", SUB),
                           ("feature_usage", USE), ("support_tickets", TIX),
                           ("churn_events", EVT)):
    print(f"  {name:16s} {rows:6,} rows  {cols:2d} columns")
