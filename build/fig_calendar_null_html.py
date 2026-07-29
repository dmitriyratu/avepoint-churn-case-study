"""Churn is not accelerating — the figure for recommendation 1.

The slide it replaces led with the mechanism and buried the test, which is the
wrong order: the mechanism is subtle and the test is obvious. So this shows one
thing large. The observed monthly churn rate, and the band produced by a copy of
the same file with the churn dates replaced by random draws and nothing else
touched. The two are the same shape, so the rise belongs to the file.

The uniformity check that motivates the null is a strip along the bottom rather
than a panel of its own. It is supporting evidence, not the argument.

Run from the repo root:  python build/fig_calendar_null_html.py
"""
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import stats

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src import generator, pipeline  # noqa: E402
from src.config import EXTRACT_DATE  # noqa: E402

N_SIMS = 200
W, H = 900, 300          # plotting area in SVG units

tables = pipeline.clean_all(pipeline.load_all())
band, summary = generator.calendar_hazard_null(tables, n_sims=N_SIMS, seed=0)

# Where each churn date sits inside its own signup-to-extract window.
events = tables["churn_events"].merge(
    tables["accounts"][["account_id", "signup_date"]], on="account_id", how="left")
span = (pd.Timestamp(EXTRACT_DATE) - events["signup_date"]).dt.days
position = ((events["churn_date"] - events["signup_date"]).dt.days / span).clip(0, 1)
position = position.dropna()
ks_p = stats.kstest(position, "uniform").pvalue

ymax = max(band["hazard"].max(), band["null_hazard_hi"].max()) * 1.08
n = len(band)


def xy(i, value):
    return f"{i / (n - 1) * W:.2f},{H - value / ymax * H:.2f}"


observed = " ".join(xy(i, v) for i, v in enumerate(band["hazard"]))
null_mid = " ".join(xy(i, v) for i, v in enumerate(band["null_hazard"]))
band_area = (" ".join(xy(i, v) for i, v in enumerate(band["null_hazard_hi"])) + " " +
             " ".join(xy(i, v) for i, v in reversed(list(enumerate(band["null_hazard_lo"])))))

xticks = "".join(
    f'<span style="left:{i / (n - 1) * 100:.2f}%">{pd.Period(p).strftime("%b %y")}</span>'
    for i, p in enumerate(band["period"]) if i % 3 == 0)
yticks = "".join(
    f'<span style="bottom:{v / ymax * 100:.1f}%">{v:.0%}</span>'
    for v in (0, 0.05, 0.10, 0.15, 0.20, 0.25) if v <= ymax)

# The uniformity strip: ten equal slices of the signup-to-extract window.
counts, _ = np.histogram(position, bins=np.linspace(0, 1, 11))
share = counts / counts.sum()
bars = "".join(
    f'<i style="height:{s / share.max() * 100:.1f}%"><b>{s:.0%}</b></i>'
    for s in share)

HTML = f"""<meta charset="utf-8">
<style>
  * {{ box-sizing: border-box; margin: 0; padding: 0; }}
  body {{
    width: 1120px; background: #fff; color: #1A1A1A;
    font-family: Calibri, "Segoe UI", system-ui, sans-serif;
    -webkit-font-smoothing: antialiased;
  }}
  .page {{ padding: 18px 26px 20px; }}
  .muted {{ color: #5A6270; }}

  .key {{ display: flex; gap: 30px; align-items: center; font-size: 13px;
          margin-bottom: 12px; }}
  .key b {{ font-weight: 700; }}
  .key .sw {{ display: inline-block; width: 26px; height: 11px; margin-right: 8px;
              vertical-align: 0px; border-radius: 2px; }}
  .key .ln {{ display: inline-block; width: 26px; border-top: 2.5px solid #1A1A1A;
              margin-right: 8px; vertical-align: 4px; }}

  .chart {{ position: relative; padding: 0 0 0 44px; }}
  .chart svg {{ width: 100%; height: 300px; display: block;
                border-bottom: 1px solid #C9CFD6; }}
  .yaxis {{ position: absolute; left: 0; top: 0; bottom: 0; width: 40px; }}
  .yaxis span {{ position: absolute; right: 8px; transform: translateY(50%);
                 font-size: 11.5px; color: #5A6270; }}
  .xaxis {{ position: relative; height: 18px; margin: 6px 0 0 44px; }}
  .xaxis span {{ position: absolute; transform: translateX(-50%); font-size: 11.5px;
                 color: #5A6270; }}

  .fill {{ fill: #B02E2E; opacity: .13; }}
  .nullline {{ fill: none; stroke: #B02E2E; stroke-width: 2;
               stroke-dasharray: 7 5; vector-effect: non-scaling-stroke; }}
  .obs {{ fill: none; stroke: #1A1A1A; stroke-width: 2.6;
          vector-effect: non-scaling-stroke; }}

  .verdict {{ display: flex; gap: 0; margin-top: 16px; }}
  .v {{ flex: 1 1 0; padding: 0 18px; border-left: 1px solid #E4E7EB; }}
  .v:first-child {{ border-left: 0; padding-left: 0; }}
  .v b {{ display: block; font-size: 19px; }}
  .v span {{ display: block; font-size: 12.5px; color: #5A6270; margin-top: 3px;
             line-height: 1.4; }}

  .strip {{ margin-top: 16px; padding-top: 14px; border-top: 1px solid #E4E7EB;
            display: flex; gap: 20px; align-items: flex-end; }}
  .stripnote {{ flex: 0 0 340px; font-size: 12.5px; color: #5A6270; line-height: 1.5; }}
  .stripnote b {{ color: #1A1A1A; font-size: 13px; display: block;
                  margin-bottom: 3px; }}
  .hist {{ flex: 1 1 auto; display: flex; align-items: flex-end; gap: 4px;
           height: 74px; }}
  .hist i {{ flex: 1 1 0; background: #2a78d6; position: relative; border-radius: 2px; }}
  .hist i b {{ position: absolute; top: -17px; left: 0; right: 0; text-align: center;
               font-size: 11px; font-weight: 400; color: #5A6270; }}
  .histax {{ display: flex; justify-content: space-between; font-size: 12px;
             color: #5A6270; margin-top: 5px; }}
</style>
<div class="page">
  <div class="key">
    <span><span class="ln"></span><b>what we actually see</b></span>
    <span><span class="sw" style="background:#B02E2E;opacity:.2"></span>
      what a file with <b>random churn dates</b> produces &mdash; {N_SIMS} rebuilds,
      95% range</span>
    <span class="muted">share of at-risk customers leaving that month</span>
  </div>

  <div class="chart">
    <div class="yaxis">{yticks}</div>
    <svg viewBox="0 0 {W} {H}" preserveAspectRatio="none">
      <polygon class="fill" points="{band_area}"/>
      <polyline class="nullline" points="{null_mid}"/>
      <polyline class="obs" points="{observed}"/>
    </svg>
  </div>
  <div class="xaxis">{xticks}</div>

  <div class="verdict">
    <div class="v"><b>&times;{summary['observed_annual']:.2f} a year</b>
      <span>the rise we observe</span></div>
    <div class="v"><b>&times;{summary['null_annual']:.2f} a year</b>
      <span>the rise random dates produce on their own</span></div>
    <div class="v"><b>{summary['months_inside_band']} of
      {summary['months_total']}</b>
      <span>months where the black line sits inside the red band</span></div>
    <div class="v"><b>{summary['observed_percentile_in_null']:.0f}th percentile</b>
      <span>where our result falls among pure noise &mdash; the middle</span></div>
  </div>

  <div class="strip">
    <div class="stripnote">
      <b>Why random dates were the null</b>
      Take each churn date and ask where it falls between that customer's signup
      and the last day of the file. Real churn would cluster somewhere. This is
      flat &mdash; indistinguishable from a random draw, KS p = {ks_p:.2f}.
    </div>
    <div>
      <div class="hist">{bars}</div>
      <div class="histax"><span>their signup day</span>
        <span>the last day of the file</span></div>
    </div>
  </div>
</div>
"""

page_path = ROOT / "outputs" / "figures" / "12_calendar_null.html"
page_path.write_text(HTML, encoding="utf-8")

from playwright.sync_api import sync_playwright  # noqa: E402

out = ROOT / "outputs" / "figures" / "12_calendar_null.png"
with sync_playwright() as p:
    browser = p.chromium.launch()
    page = browser.new_page(viewport={"width": 1120, "height": 700},
                            device_scale_factor=3)
    page.goto(page_path.as_uri())
    page.locator(".page").screenshot(path=str(out))
    browser.close()

print("wrote", out)
print(f"observed x{summary['observed_annual']:.2f}/yr  null x{summary['null_annual']:.2f}/yr  "
      f"{summary['months_inside_band']}/{summary['months_total']} inside  "
      f"pct {summary['observed_percentile_in_null']:.1f}  KS p {ks_p:.3f}")
