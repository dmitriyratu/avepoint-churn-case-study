"""The rolling-origin timeline for slide 2, rendered through HTML/CSS.

One question — "will this customer leave in the next 90 days?" — asked from four
quarter-end vantage points. The picture has to carry three facts at once: that
features never cross the cutoff, that the answer window always closes inside the
extract, and that this is one question repeated rather than four questions.

Run from the repo root:  python build/fig_timeline_html.py
"""
import sys
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src import pipeline, robustness  # noqa: E402
from src.config import HORIZON_DAYS, TARGET  # noqa: E402
from src.labeling import build_cohort  # noqa: E402

START = pd.Timestamp("2023-01-01")
EXTRACT_END = pd.Timestamp("2024-12-31")
SPAN = (EXTRACT_END - START).days

cutoffs = robustness.rolling_origin_cutoffs(n=4)
tables = pipeline.clean_all(pipeline.load_all())
rows = []
for c in cutoffs:
    cohort = build_cohort(tables, cutoff=c, prediction_start=c)
    rows.append((c, len(cohort), int(cohort[TARGET].sum())))


def pct(date):
    """Position on the two-year timeline, as a CSS percentage."""
    return (date - START).days / SPAN * 100


bars = ""
for cut, n, pos in rows:
    known, answer = pct(cut), pct(cut + pd.Timedelta(days=HORIZON_DAYS)) - pct(cut)
    bars += f"""
    <div class="lane">
      <span class="when">{cut:%d %b %Y}</span>
      <span class="track">
        <i class="known" style="width:{known:.2f}%"></i>
        <i class="answer" style="width:{answer:.2f}%"></i>
        <i class="cut" style="left:{known:.2f}%"></i>
      </span>
      <span class="tally"><b>{n}</b> could still leave &nbsp;&middot;&nbsp;
        <b>{pos}</b> did</span>
    </div>"""

months = pd.date_range("2023-01-01", "2024-12-01", freq="3MS")
ticks = "".join(
    f'<span style="left:{pct(m):.2f}%">{m:%b}'
    + (f'<em>{m:%Y}</em>' if m.month == 1 else "") + "</span>"
    for m in months)

HTML = f"""<meta charset="utf-8">
<style>
  * {{ box-sizing: border-box; margin: 0; padding: 0; }}
  body {{
    width: 1240px; background: #fff; color: #1A1A1A;
    font-family: Calibri, "Segoe UI", system-ui, sans-serif;
    -webkit-font-smoothing: antialiased;
  }}
  .wrap {{ padding: 24px 28px 26px; }}
  h2 {{ font-size: 17px; font-weight: 700; margin-bottom: 20px; }}
  .note {{ color: #5A6270; font-style: italic; font-size: 13.5px; }}

  .lane {{ display: flex; align-items: center; height: 40px; }}
  .when {{ flex: 0 0 104px; font-size: 14px; font-weight: 700; text-align: right;
           padding-right: 14px; }}
  .track {{ flex: 1 1 auto; min-width: 0; position: relative; display: flex;
            align-items: center; height: 100%; }}
  .known, .answer {{ height: 19px; display: block; }}
  .known {{ background: #2a78d6; }}
  .answer {{ background: #eb6834; }}
  .cut {{ position: absolute; top: 7px; bottom: 7px; width: 2px;
          background: #B02E2E; transform: translateX(-1px); }}
  .tally {{ flex: 0 0 196px; padding-left: 20px; font-size: 13.5px; color: #5A6270; }}
  .tally b {{ color: #1A1A1A; font-weight: 700; }}

  /* the extract edge runs the height of the lanes, behind everything */
  .field {{ position: relative; }}
  .edge {{ position: absolute; top: -4px; bottom: 30px; width: 0;
           border-left: 1px dashed #C2C8D0; }}
  .edge span {{ position: absolute; top: -18px; right: 7px; white-space: nowrap;
                font-size: 12.5px; color: #5A6270; font-style: italic; }}

  .axis {{ position: relative; height: 34px; margin: 8px 104px 0 118px; }}
  .axis span {{ position: absolute; transform: translateX(-50%); font-size: 12px;
                color: #5A6270; text-align: center; }}
  .axis em {{ display: block; font-style: normal; font-size: 12px; }}

  .key {{ display: flex; align-items: center; gap: 9px; margin: 6px 0 0 118px;
          font-size: 13.5px; color: #5A6270; }}
  .key i {{ width: 26px; height: 13px; display: block; }}
  .key .gap {{ width: 22px; }}
</style>
<div class="wrap">
  <h2>One question, asked from four dates</h2>
  <div class="field">
    <div class="edge" style="left:calc(118px + (100% - 314px) * {pct(EXTRACT_END) / 100:.4f})">
      <span>the data stops here</span></div>
    {bars}
    <div class="axis">{ticks}</div>
  </div>
  <div class="key">
    <i style="background:#2a78d6"></i><span>what we knew on that date</span>
    <span class="gap"></span>
    <i style="background:#eb6834"></i><span>the next {HORIZON_DAYS} days &mdash;
      the answer, never a feature</span>
  </div>
</div>
"""

page_path = ROOT / "outputs" / "figures" / "02_rolling_origin.html"
page_path.write_text(HTML, encoding="utf-8")

from playwright.sync_api import sync_playwright  # noqa: E402

out = ROOT / "outputs" / "figures" / "02_rolling_origin_html.png"
with sync_playwright() as p:
    browser = p.chromium.launch()
    page = browser.new_page(viewport={"width": 1240, "height": 420},
                            device_scale_factor=3)
    page.goto(page_path.as_uri())
    page.locator(".wrap").screenshot(path=str(out))
    browser.close()

print("wrote", out)
