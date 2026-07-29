"""Why ranking cannot beat calling everyone — recommendation 3.

You can price every threshold from its confusion matrix:

    net = TP × e × V − (TP + FP) × C

Raising the threshold means skipping people. Each skip is worth +C if they
would have stayed (TN) and −(e×V − C) if they would have left (FN). Under these
economics that ratio is about 9 to 1, which a near-chance ranking cannot deliver.
Sweeping every threshold confirms it: the best find is +$600 on $52,400.

Run from the repo root:  python build/fig_breakeven_html.py
"""
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.economics import DEFAULT_EFFECTIVENESS, DEFAULT_INTERVENTION_COST  # noqa: E402

CLV = 7310.0
BASE = 0.305
N_COHORT = 177
TREAT_ALL = 52_400
WITH_MODEL = 53_000

COST = DEFAULT_INTERVENTION_COST
EFF = DEFAULT_EFFECTIVENESS
SAVE = EFF * CLV                    # expected value of catching one true leaver
FN_LOSS = SAVE - COST               # net loss vs treat-all for skipping a leaver
TN_GAIN = COST                      # net gain vs treat-all for skipping a stayer
RATIO = FN_LOSS / TN_GAIN           # TNs needed per FN to break even vs treat-all
N_NEED = int(round(RATIO))
gain = WITH_MODEL - TREAT_ALL

# Nine small markers for the ratio visual.
dots_ok = "".join('<i class="dot ok"></i>' for _ in range(N_NEED))
dot_bad = '<i class="dot bad"></i>'

HTML = f"""<meta charset="utf-8">
<style>
  * {{ box-sizing: border-box; margin: 0; padding: 0; }}
  body {{
    width: 1120px; background: #fff; color: #1A1A1A;
    font-family: Calibri, "Segoe UI", system-ui, sans-serif;
    -webkit-font-smoothing: antialiased;
  }}
  .wrap {{ padding: 18px 26px 20px; }}
  .muted {{ color: #5A6270; }}

  .head {{
    display: flex; align-items: baseline; gap: 14px;
    padding-bottom: 12px; border-bottom: 1px solid #E4E7EB;
  }}
  .head b {{ font-size: 15px; }}
  .head span {{ font-size: 13px; color: #5A6270; }}
  .head span i {{ font-style: normal; color: #1A1A1A; font-weight: 700; }}

  .row {{ display: flex; gap: 16px; margin-top: 14px; }}
  .card {{
    flex: 1 1 0; padding: 14px 16px 15px;
    border: 1px solid #E4E7EB; border-radius: 4px;
  }}
  .card.bad {{ border-color: #e0b4b4; background: #FAF3F3; }}
  .card.ok {{ border-color: #b7d4c4; background: #F2F8F4; }}
  .card .lbl {{ font-size: 12.5px; color: #5A6270; margin-bottom: 4px; }}
  .card .lbl em {{
    float: right; font-style: normal; font-size: 11.5px; font-weight: 700;
    letter-spacing: .04em; opacity: .85;
  }}
  .card .amt {{ font-size: 30px; font-weight: 700; line-height: 1; }}
  .card.ok .amt {{ color: #1e7a4b; }}
  .card.bad .amt {{ color: #B02E2E; }}
  .card .why {{ font-size: 13px; color: #5A6270; margin-top: 7px; line-height: 1.4; }}
  .card .why b {{ color: #1A1A1A; }}

  .rule {{
    margin-top: 14px; padding: 12px 16px; background: #F4F6F9;
    border-left: 3px solid #1A1A1A; display: flex; align-items: center; gap: 22px;
  }}
  .rule .text {{ flex: 1 1 auto; font-size: 14px; line-height: 1.45; }}
  .rule .text b {{ font-size: 15px; }}
  .rule .text span {{ color: #5A6270; }}
  .ratio {{ flex: 0 0 auto; text-align: center; }}
  .ratio .dots {{ display: flex; gap: 5px; align-items: center; justify-content: center; }}
  .dot {{
    display: inline-block; width: 14px; height: 14px; border-radius: 3px;
  }}
  .dot.ok {{ background: #1e7a4b; }}
  .dot.bad {{ background: #B02E2E; width: 18px; height: 18px; margin-left: 6px; }}
  .ratio em {{
    display: block; font-style: normal; font-size: 12px; color: #5A6270;
    margin-top: 5px;
  }}
  .ratio em b {{ color: #1A1A1A; }}

  .foot {{
    display: flex; gap: 28px; align-items: center;
    margin-top: 14px; padding-top: 14px; border-top: 1px solid #E4E7EB;
  }}
  .bars {{ flex: 1 1 auto; }}
  .mrow {{ display: flex; align-items: center; height: 30px; }}
  .mname {{ flex: 0 0 175px; font-size: 13.5px; }}
  .mtrack {{ flex: 1 1 auto; position: relative; height: 30px; }}
  .mbar {{
    position: absolute; top: 50%; height: 15px; margin-top: -7.5px;
    border-radius: 2px;
  }}
  .mval {{ flex: 0 0 155px; padding-left: 12px; font-size: 13.5px; }}
  .mval b {{ font-size: 15px; }}
  .aside {{ flex: 0 0 300px; font-size: 12.5px; color: #5A6270; line-height: 1.45; }}
  .aside b {{ color: #1A1A1A; }}
</style>
<div class="wrap">
  <div class="head">
    <b>Net at any threshold = TP &times; ${SAVE:,.0f} &minus; (TP + FP) &times; ${COST:,.0f}</b>
    <span>priced from the confusion matrix. Raising the threshold means
      <i>skipping</i> people &mdash; price each skip against calling everyone.</span>
  </div>

  <div class="row">
    <div class="card ok">
      <div class="lbl">Skip a customer who stays<em>TN</em></div>
      <div class="amt">+${TN_GAIN:,.0f}</div>
      <div class="why">You save the call cost. That is the only thing ranking
        can earn by leaving someone out.</div>
    </div>
    <div class="card bad">
      <div class="lbl">Skip a customer who leaves<em>FN</em></div>
      <div class="amt">&minus;${FN_LOSS:,.0f}</div>
      <div class="why">You give up the expected save
        (<b>{EFF:.0%} &times; ${CLV:,.0f} = ${SAVE:,.0f}</b>) and only keep the
        ${COST:,.0f} you did not spend.</div>
    </div>
  </div>

  <div class="rule">
    <div class="text">
      <b>To beat calling everyone, skip about {N_NEED} true stayers for every
        true leaver you miss.</b><br>
      <span>That is the bar the confusion matrix has to clear at every
        threshold. A model near chance does not separate that cleanly &mdash;
        lift the cut and the TPs fall almost as fast as the FPs.</span>
    </div>
    <div class="ratio">
      <div class="dots">{dots_ok}{dot_bad}</div>
      <em><b>{N_NEED} TN</b> per <b>1 FN</b></em>
    </div>
  </div>

  <div class="foot">
    <div class="bars">
      <div class="mrow">
        <span class="mname">Call everyone ({N_COHORT})</span>
        <span class="mtrack"><i class="mbar" style="left:0;width:{TREAT_ALL / WITH_MODEL * 100:.1f}%;background:#1e7a4b"></i></span>
        <span class="mval"><b>${TREAT_ALL:,}</b></span>
      </div>
      <div class="mrow">
        <span class="mname">Best model threshold</span>
        <span class="mtrack"><i class="mbar" style="left:0;width:100%;background:#2a78d6"></i></span>
        <span class="mval"><b>${WITH_MODEL:,}</b>
          <span class="muted">&nbsp;+${gain} · {gain / TREAT_ALL:.0%}</span></span>
      </div>
    </div>
    <div class="aside">Every threshold swept on out-of-fold scores. The
      <b>+${gain}</b> is itself generous: the cut was chosen on the same
      customers it is scored on. Cohort base rate <b>{BASE:.0%}</b> against a
      break-even of <b>{COST / SAVE:.0%}</b>.</div>
  </div>
</div>
"""

page_path = ROOT / "outputs" / "figures" / "15_breakeven.html"
page_path.write_text(HTML, encoding="utf-8")

from playwright.sync_api import sync_playwright  # noqa: E402

out = ROOT / "outputs" / "figures" / "15_breakeven_simple.png"
with sync_playwright() as p:
    browser = p.chromium.launch()
    page = browser.new_page(viewport={"width": 1120, "height": 520},
                            device_scale_factor=3)
    page.goto(page_path.as_uri())
    page.locator(".wrap").screenshot(path=str(out))
    browser.close()

print("wrote", out)
print(f"TN +${TN_GAIN:,.0f}  FN -${FN_LOSS:,.0f}  need {RATIO:.1f}:1  "
      f"treat-all ${TREAT_ALL:,}  best ${WITH_MODEL:,}")
