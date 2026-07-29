"""Why the ranking is worth nothing — the economics figure for recommendation 3.

The heatmap this replaces swept call cost against customer value, which asks the
reader to hold four numbers in their head to reach a conclusion that needs two.
A call pays for itself above a 10.3% chance of leaving; everyone in the cohort
is at 30.5%. The decision is already right for every customer, so no ordering
can improve it. The sweep is reduced to the two points where the conclusion
would actually flip.

Run from the repo root:  python build/fig_breakeven_html.py
"""
import sys
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.economics import DEFAULT_EFFECTIVENESS, DEFAULT_INTERVENTION_COST  # noqa: E402

XMAX = 0.40
CLV = 7310.0
BASE = 0.305
TREAT_ALL = 52_400
WITH_MODEL = 53_000

breakeven = DEFAULT_INTERVENTION_COST / (CLV * DEFAULT_EFFECTIVENESS)
# The two points where "call everyone" stops being the right decision.
cost_flip = BASE * CLV * DEFAULT_EFFECTIVENESS
value_flip = DEFAULT_INTERVENTION_COST / (BASE * DEFAULT_EFFECTIVENESS)
gain = WITH_MODEL - TREAT_ALL


def pos(value):
    return value / XMAX * 100


HTML = f"""<meta charset="utf-8">
<style>
  * {{ box-sizing: border-box; margin: 0; padding: 0; }}
  body {{
    width: 1040px; background: #fff; color: #1A1A1A;
    font-family: Calibri, "Segoe UI", system-ui, sans-serif;
    -webkit-font-smoothing: antialiased;
  }}
  .wrap {{ padding: 24px 28px 26px; }}
  h3 {{ font-size: 14px; font-weight: 700; margin-bottom: 14px; }}
  .muted {{ color: #5A6270; }}

  .scale {{ position: relative; height: 128px; margin: 0 30px; }}
  .rail {{ position: absolute; left: 0; right: 0; top: 52px; height: 10px;
           background: #EDF0F3; border-radius: 5px; }}
  .worth {{ position: absolute; top: 52px; height: 10px; background: #1e7a4b;
            opacity: .85; border-radius: 0 5px 5px 0; }}
  .mark {{ position: absolute; top: 38px; height: 38px; width: 2px;
           transform: translateX(-1px); }}
  .mark.a {{ background: #1e7a4b; }}
  .mark.b {{ background: #1A1A1A; }}
  .tag {{ position: absolute; transform: translateX(-50%); text-align: center;
          white-space: nowrap; }}
  .tag b {{ display: block; font-size: 19px; }}
  .tag em {{ display: block; font-style: normal; font-size: 12.5px;
             color: #5A6270; margin-top: 1px; }}
  .tag.a {{ top: 0; }}
  .tag.a b {{ color: #1e7a4b; }}
  .tag.b {{ top: 84px; }}
  .ticks {{ position: absolute; left: 0; right: 0; top: 66px; }}
  .ticks span.hide {{ display: none; }}
  .ticks span {{ position: absolute; transform: translateX(-50%); font-size: 12px;
                 color: #9AA3AF; }}

  .verdict {{ margin: 22px 30px 0; padding: 12px 16px; background: #F2F8F4;
              border-left: 3px solid #1e7a4b; font-size: 14px; }}
  .verdict b {{ font-size: 15px; }}

  .flip {{ display: flex; gap: 40px; margin: 20px 30px 0; }}
  .flip div {{ flex: 1 1 0; font-size: 13px; color: #5A6270; }}
  .flip b {{ display: block; font-size: 17px; color: #1A1A1A; margin-bottom: 2px; }}

  .money {{ margin: 24px 30px 0; padding-top: 18px; border-top: 1px solid #E4E7EB; }}
  .mrow {{ display: flex; align-items: center; height: 34px; }}
  .mname {{ flex: 0 0 200px; font-size: 13.5px; }}
  .mtrack {{ flex: 1 1 auto; position: relative; height: 34px; }}
  .mbar {{ position: absolute; top: 50%; height: 18px; margin-top: -9px;
           border-radius: 2px; }}
  .mval {{ flex: 0 0 190px; padding-left: 16px; font-size: 13.5px; }}
  .mval b {{ font-size: 15px; }}
</style>
<div class="wrap">
  <h3>A call is worth making above a 10% chance of leaving. Everyone here is at 31%.</h3>
  <div class="scale">
    <div class="rail"></div>
    <div class="worth" style="left:{pos(breakeven):.2f}%;right:0"></div>
    <div class="mark a" style="left:{pos(breakeven):.2f}%"></div>
    <div class="mark b" style="left:{pos(BASE):.2f}%"></div>
    <div class="tag a" style="left:{pos(breakeven):.2f}%">
      <b>{breakeven:.1%}</b><em>a call pays for itself above here</em></div>
    <div class="tag b" style="left:{pos(BASE):.2f}%">
      <b>{BASE:.1%}</b><em>everyone in this group</em></div>
    <div class="ticks">
      <span style="left:0%">0%</span>
      <span style="left:{pos(0.10):.2f}%">10%</span>
      <span style="left:{pos(0.20):.2f}%">20%</span>
      <span class="hide" style="left:{pos(0.30):.2f}%">30%</span>
      <span style="left:100%">40%</span>
    </div>
  </div>

  <div class="verdict"><b>So call all of them.</b> The green stretch is where a
    call makes money, and every customer in the cohort sits inside it. Ranking
    only helps when you have to leave someone out.</div>

  <div class="flip">
    <div><b>${cost_flip:,.0f}</b>what a call would have to cost before targeting
      beats calling everyone. It costs ${DEFAULT_INTERVENTION_COST:,.0f}.</div>
    <div><b>${value_flip:,.0f}</b>what a customer would have to be worth, at most,
      for the same. They are worth ${CLV:,.0f}.</div>
  </div>

  <div class="money">
    <div class="mrow">
      <span class="mname">Call everyone</span>
      <span class="mtrack"><i class="mbar" style="left:0;width:{TREAT_ALL / WITH_MODEL * 100:.1f}%;background:#1e7a4b"></i></span>
      <span class="mval"><b>${TREAT_ALL:,}</b></span>
    </div>
    <div class="mrow">
      <span class="mname">Use the model to rank</span>
      <span class="mtrack"><i class="mbar" style="left:0;width:100%;background:#2a78d6"></i></span>
      <span class="mval"><b>${WITH_MODEL:,}</b>
        <span class="muted">&nbsp;+${gain} &middot; {gain / TREAT_ALL:.0%}</span></span>
    </div>
  </div>
</div>
"""

page_path = ROOT / "outputs" / "figures" / "15_breakeven.html"
page_path.write_text(HTML, encoding="utf-8")

from playwright.sync_api import sync_playwright  # noqa: E402

out = ROOT / "outputs" / "figures" / "15_breakeven_simple.png"
with sync_playwright() as p:
    browser = p.chromium.launch()
    page = browser.new_page(viewport={"width": 1040, "height": 520},
                            device_scale_factor=3)
    page.goto(page_path.as_uri())
    page.locator(".wrap").screenshot(path=str(out))
    browser.close()

print("wrote", out)
print(f"break-even {breakeven:.3f}  base {BASE}  cost flip ${cost_flip:,.0f}  "
      f"value flip ${value_flip:,.0f}  gain ${gain}")
