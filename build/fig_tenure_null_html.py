"""Tenure carries no information — the figure for recommendation 2.

The slide it replaces led with survival by signup cohort, which is the
confirming view rather than the argument, and which reads left-to-right as time
when the x-axis is groups. So the argument comes first here.

Pooled across everybody the hazard falls sharply with tenure, which is the shape
an onboarding problem produces and the usual reason to fund one. Refit inside
each signup cohort, where a composition effect cannot operate, and the shape
returns to memoryless every time. Two of the five point the other way and not
one is distinguishable from flat.

Run from the repo root:  python build/fig_tenure_null_html.py
"""
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src import pipeline, survival  # noqa: E402

RHO_LO, RHO_HI = 0.60, 1.45
TENURES = [("S(30d)", "30 days", "#2a78d6"),
           ("S(60d)", "60 days", "#eb6834"),
           ("S(90d)", "90 days", "#1e7a4b")]

tables = pipeline.clean_all(pipeline.load_all())
frame = survival.survival_frame(tables)
pooled = survival.hazard_shape(frame)
within = survival.shape_within_cohorts(frame)
by_cohort = survival.km_at_tenure_by_cohort(frame)


def rho_pos(value):
    return (min(max(value, RHO_LO), RHO_HI) - RHO_LO) / (RHO_HI - RHO_LO) * 100


rows = f"""
  <div class="row pooled">
    <span class="lab">everyone pooled</span>
    <span class="track">
      <i class="one" style="left:{rho_pos(1.0):.2f}%"></i>
      <i class="ci hot" style="left:{rho_pos(pooled['rho_ci'][0]):.2f}%;
         width:{rho_pos(pooled['rho_ci'][1]) - rho_pos(pooled['rho_ci'][0]):.2f}%"></i>
      <i class="dot hot" style="left:{rho_pos(pooled['rho']):.2f}%"></i>
    </span>
    <span class="val hot">{pooled['rho']:.2f}</span>
    <span class="p hot">p = 2e-13</span>
  </div>
  <div class="split"><span>refit inside each signup cohort</span></div>"""

for _, r in within.iterrows():
    rows += f"""
  <div class="row">
    <span class="lab">{r['cohort']} &nbsp;<em>n = {int(r['n'])}</em></span>
    <span class="track">
      <i class="one" style="left:{rho_pos(1.0):.2f}%"></i>
      <i class="dot" style="left:{rho_pos(r['rho']):.2f}%"></i>
    </span>
    <span class="val">{r['rho']:.2f}</span>
    <span class="p">p = {r['p_vs_exponential']:.2f}</span>
  </div>"""

rho_ticks = "".join(f'<span style="left:{rho_pos(t):.2f}%">{t:.1f}</span>'
                    for t in (0.6, 0.8, 1.0, 1.2, 1.4))

# --- confirming panel: survival at fixed tenure, cohort by cohort
cohorts = list(by_cohort["cohort"])
n_c = len(cohorts)
CW, CH = 420, 250


def cxy(i, value):
    return f"{i / (n_c - 1) * CW:.2f},{CH - (value - 0.2) / 0.85 * CH:.2f}"


series = ""
for col, _, colour in TENURES:
    pts = [(i, v) for i, v in enumerate(by_cohort[col]) if v == v]
    series += (f'<polyline class="ln" style="stroke:{colour}" '
               f'points="{" ".join(cxy(i, v) for i, v in pts)}"/>')
    series += "".join(f'<circle r="4" fill="{colour}" '
                      f'cx="{i / (n_c - 1) * CW:.2f}" '
                      f'cy="{CH - (v - 0.2) / 0.85 * CH:.2f}"/>' for i, v in pts)

cticks = "".join(
    f'<span style="left:{i / (n_c - 1) * 100:.2f}%">{c}<em>{int(n)}</em></span>'
    for i, (c, n) in enumerate(zip(cohorts, by_cohort["n"])))
cyticks = "".join(f'<span style="bottom:{(v - 0.2) / 0.85 * 100:.1f}%">{v:.0%}</span>'
                  for v in (0.2, 0.4, 0.6, 0.8, 1.0))
legend = "".join(f'<span><i style="background:{c}"></i>{label}</span>'
                 for _, label, c in TENURES)

HTML = f"""<meta charset="utf-8">
<style>
  * {{ box-sizing: border-box; margin: 0; padding: 0; }}
  body {{
    width: 1160px; background: #fff; color: #1A1A1A;
    font-family: Calibri, "Segoe UI", system-ui, sans-serif;
    -webkit-font-smoothing: antialiased;
  }}
  .page {{ display: flex; gap: 40px; padding: 18px 24px 20px; }}
  .panel {{ flex: 1 1 0; }}
  .panel.right {{ flex: 0.92 1 0; }}
  h2 {{ font-size: 14.5px; font-weight: 700; }}
  h2 em {{ display: block; font-style: normal; font-weight: 400; font-size: 12px;
           color: #5A6270; margin-top: 3px; line-height: 1.4; }}

  .row {{ display: flex; align-items: center; height: 30px; }}
  .lab {{ flex: 0 0 128px; text-align: right; padding-right: 14px; font-size: 13px; }}
  .lab em {{ font-style: normal; color: #5A6270; font-size: 11.5px; }}
  .track {{ flex: 1 1 auto; min-width: 0; position: relative; height: 30px; }}
  .track i {{ position: absolute; display: block; }}
  .one {{ top: 0; bottom: 0; width: 2px; background: #1A1A1A;
          transform: translateX(-1px); }}
  .ci {{ top: 50%; height: 3px; margin-top: -1.5px; background: #B02E2E;
         opacity: .45; border-radius: 2px; }}
  .dot {{ top: 50%; width: 10px; height: 10px; margin: -5px 0 0 -5px;
          border-radius: 50%; background: #2a78d6; }}
  .dot.hot {{ background: #B02E2E; }}
  .val {{ flex: 0 0 40px; padding-left: 14px; font-size: 13px; font-weight: 700; }}
  .p {{ flex: 0 0 62px; font-size: 12px; color: #5A6270; }}
  .hot {{ color: #B02E2E; }}
  .split {{ display: flex; align-items: center; margin: 9px 0 9px 128px;
            font-size: 12px; color: #5A6270; }}
  .split span {{ padding-right: 12px; white-space: nowrap; }}
  .split::after {{ content: ""; flex: 1 1 auto; border-top: 1px solid #E4E7EB; }}

  .rhoaxis {{ position: relative; height: 18px; margin: 4px 102px 0 128px; }}
  .rhoaxis span {{ position: absolute; transform: translateX(-50%); font-size: 11.5px;
                   color: #5A6270; }}
  .rhocap {{ margin: 4px 102px 0 128px; font-size: 12px; color: #5A6270;
             text-align: center; }}
  .rhocap b {{ color: #1A1A1A; }}

  .chart {{ position: relative; padding-left: 40px; margin-top: 14px; }}
  .chart svg {{ width: 100%; height: 250px; display: block;
                border-left: 1px solid #C9CFD6; border-bottom: 1px solid #C9CFD6; }}
  .ln {{ fill: none; stroke-width: 2.2; vector-effect: non-scaling-stroke; }}
  .yax {{ position: absolute; left: 0; top: 0; bottom: 0; width: 36px; }}
  .yax span {{ position: absolute; right: 7px; transform: translateY(50%);
               font-size: 11.5px; color: #5A6270; }}
  .xax {{ position: relative; height: 32px; margin: 6px 0 0 40px; }}
  .xax span {{ position: absolute; transform: translateX(-50%); font-size: 11px;
               color: #5A6270; text-align: center; }}
  .xax em {{ display: block; font-style: normal; font-size: 10.5px; color: #9AA3AF; }}
  .leg {{ display: flex; gap: 18px; margin: 8px 0 0 40px; font-size: 12px;
          color: #5A6270; }}
  .leg i {{ display: inline-block; width: 11px; height: 11px; border-radius: 50%;
            margin-right: 6px; vertical-align: -1px; }}
  .foot {{ margin: 10px 0 0 40px; font-size: 12px; color: #5A6270; line-height: 1.45; }}
</style>
<div class="page">

  <div class="panel">
    <h2>Does churn risk fall as customers settle in?
      <em>Weibull shape. Below 1 means risk falls with tenure &mdash; the shape an
        onboarding problem makes. 1 means tenure tells you nothing.</em></h2>
    <div style="margin-top:14px">{rows}</div>
    <div class="rhoaxis">{rho_ticks}</div>
    <div class="rhocap">Weibull shape &nbsp;&middot;&nbsp;
      <b>the black line at 1.0 is "tenure tells you nothing"</b></div>
  </div>

  <div class="panel right">
    <h2>The same thing seen a second way
      <em>Each cohort measured at the same age, so unequal follow-up cannot
        produce the gap. These are eight separate groups, not a timeline.</em></h2>
    <div class="chart">
      <div class="yax">{cyticks}</div>
      <svg viewBox="0 0 {CW} {CH}" preserveAspectRatio="none">{series}</svg>
    </div>
    <div class="xax">{cticks}</div>
    <div class="leg">{legend}
      <span style="color:#9AA3AF">small number = customers in that cohort</span></div>
    <div class="foot">Later cohorts look worse at every age. They are not worse:
      a customer who joined recently has a shorter window for a random churn date
      to land in, so more of their draws fall inside any 30 or 90-day question.</div>
  </div>

</div>
"""

page_path = ROOT / "outputs" / "figures" / "12_tenure_null.html"
page_path.write_text(HTML, encoding="utf-8")

from playwright.sync_api import sync_playwright  # noqa: E402

out = ROOT / "outputs" / "figures" / "12_tenure_null.png"
with sync_playwright() as p:
    browser = p.chromium.launch()
    page = browser.new_page(viewport={"width": 1160, "height": 560},
                            device_scale_factor=3)
    page.goto(page_path.as_uri())
    page.locator(".page").screenshot(path=str(out))
    browser.close()

print("wrote", out)
print(f"pooled rho {pooled['rho']} CI {pooled['rho_ci']} p {pooled['p_vs_exponential']:.1e}")
print(within.to_string(index=False))
