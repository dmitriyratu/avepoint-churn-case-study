"""What each mistake is worth — the leakage figure for the class-imbalance slide.

The earlier version labelled the three builds "correct", "sees future" and
"sees outcome", which read as three separate variants. They are not: each one
adds a mistake to the one before it, and the third is the second plus outcome
columns. That is why the outcome columns are worth 0.37 rather than 0.21, and
the picture has to say so.

Reads outputs/reports/leakage_comparison.csv, written by notebook 06.

Run from the repo root:  python build/fig_leakage_html.py
"""
import sys
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

XLO, XHI = 0.35, 0.85
CHANCE = 0.50

comparison = pd.read_csv(ROOT / "outputs" / "reports" / "leakage_comparison.csv")
score = dict(zip(comparison["design"], comparison["auc"]))
spread = dict(zip(comparison["design"], comparison["sd"]))

STEPS = [
    ("A. observation window only",
     "Built properly",
     "Only what we knew on 30 June",
     "ok"),
    ("B. + post-cutoff rows",
     "Then we let in activity from after 30 June",
     "Usage and tickets we would not have had yet",
     "bad"),
    ("C. + churn_events columns",
     "Then we also let in the churn record",
     "Refund amount and reason code, written the day they left",
     "bad"),
]


def pos(value):
    return (min(max(value, XLO), XHI) - XLO) / (XHI - XLO) * 100


rows = ""
previous = None
for key, headline, detail, kind in STEPS:
    auc, sd = score[key], spread[key]
    move = "" if previous is None else f"{auc - previous:+.2f}"
    rows += f"""
    <div class="step">
      <span class="label">
        <b>{headline}</b>
        <em>{detail}</em>
      </span>
      <span class="track">
        <i class="chance" style="left:{pos(CHANCE):.2f}%"></i>
        <i class="bar {kind}" style="left:{pos(min(auc, CHANCE)):.2f}%;
           width:{abs(pos(auc) - pos(CHANCE)):.2f}%"></i>
        <i class="dot {kind}" style="left:{pos(auc):.2f}%"></i>
        <em class="score {kind}" style="left:{pos(auc):.2f}%">{auc:.2f}</em>
      </span>
      <span class="move">{move}</span>
    </div>"""
    previous = auc

ticks = "".join(f'<span style="left:{pos(t):.2f}%">{t:.1f}</span>'
                for t in (0.4, 0.5, 0.6, 0.7, 0.8))

HTML = f"""<meta charset="utf-8">
<style>
  * {{ box-sizing: border-box; margin: 0; padding: 0; }}
  body {{
    width: 1000px; background: #fff; color: #1A1A1A;
    font-family: Calibri, "Segoe UI", system-ui, sans-serif;
    -webkit-font-smoothing: antialiased;
  }}
  .wrap {{ padding: 22px 26px 24px; }}

  .step {{ display: flex; align-items: center; height: 62px; }}
  .label {{ flex: 0 0 330px; padding-right: 20px; }}
  .label b {{ display: block; font-size: 14.5px; }}
  .label em {{ display: block; font-style: normal; font-size: 12.5px;
               color: #5A6270; margin-top: 2px; }}

  .track {{ flex: 1 1 auto; min-width: 0; position: relative; height: 62px; }}
  .track i, .track em {{ position: absolute; display: block; }}
  .chance {{ top: 6px; bottom: 6px; width: 0; border-left: 1px dashed #9AA3AF; }}
  .bar {{ top: 50%; height: 12px; margin-top: -6px; border-radius: 2px; }}
  .bar.ok {{ background: #2a78d6; opacity: .35; }}
  .bar.bad {{ background: #eb6834; opacity: .35; }}
  .dot {{ top: 50%; width: 12px; height: 12px; margin: -6px 0 0 -6px;
          border-radius: 50%; }}
  .dot.ok {{ background: #2a78d6; }}
  .dot.bad {{ background: #eb6834; }}
  .score {{ top: 9px; transform: translateX(-50%); font-size: 13.5px;
            font-weight: 700; font-style: normal; }}
  .score.ok {{ color: #2a78d6; }}
  .score.bad {{ color: #eb6834; }}

  .move {{ flex: 0 0 62px; padding-left: 18px; font-size: 14px; font-weight: 700;
           color: #5A6270; }}

  .axis {{ position: relative; height: 20px; margin: 2px 62px 0 330px; }}
  .axis span {{ position: absolute; transform: translateX(-50%); font-size: 12px;
                color: #5A6270; }}
  .cap {{ margin: 6px 62px 0 330px; font-size: 12.5px; color: #5A6270; }}
</style>
<div class="wrap">
  {rows}
  <div class="axis">{ticks}</div>
  <div class="cap">ROC-AUC &nbsp;·&nbsp; dashed line = 0.50, a coin flip
    &nbsp;·&nbsp; each row keeps the mistake above it</div>
</div>
"""

page_path = ROOT / "outputs" / "figures" / "06_leakage_steps.html"
page_path.write_text(HTML, encoding="utf-8")

from playwright.sync_api import sync_playwright  # noqa: E402

out = ROOT / "outputs" / "figures" / "06_leakage_steps.png"
with sync_playwright() as p:
    browser = p.chromium.launch()
    page = browser.new_page(viewport={"width": 1000, "height": 320},
                            device_scale_factor=3)
    page.goto(page_path.as_uri())
    page.locator(".wrap").screenshot(path=str(out))
    browser.close()

print("wrote", out)
print(comparison.to_string(index=False))
