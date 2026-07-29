"""What to do about churn — the summary figure.

Constructive framing: lead with what pays and what to start Monday, then the
traps the file invents, then a scoped data ask (billing is usable; usage and
tickets poison the model). Interpretability belongs under "do not" — SHAP still
ranks when the model learned nothing.

Run from the repo root:  python build/fig_what_to_do_html.py
"""
import sys
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src import pipeline  # noqa: E402
from src.config import TARGET  # noqa: E402
from src.economics import DEFAULT_EFFECTIVENESS, DEFAULT_INTERVENTION_COST  # noqa: E402

RAW = ROOT / "data" / "raw"

cohort = pipeline.build().cohort
base_rate = cohort[TARGET].mean()
n_cohort, n_left = len(cohort), int(cohort[TARGET].sum())

accounts = pd.read_csv(RAW / "ravenstack_accounts.csv", parse_dates=["signup_date"])
signup = accounts.set_index("account_id")["signup_date"]
subs = pd.read_csv(RAW / "ravenstack_subscriptions.csv")

usage = pd.read_csv(RAW / "ravenstack_feature_usage.csv", parse_dates=["usage_date"])
usage["account_id"] = usage["subscription_id"].map(
    subs.set_index("subscription_id")["account_id"])
usage_early = (usage["usage_date"] <
               usage["account_id"].map(signup)).mean()

tickets = pd.read_csv(RAW / "ravenstack_support_tickets.csv",
                      parse_dates=["submitted_at"])
tickets_early = (tickets["submitted_at"] <
                 tickets["account_id"].map(signup)).mean()

flag_rate = accounts["churn_flag"].astype(bool).mean()
events = pd.read_csv(RAW / "ravenstack_churn_events.csv")
log_rate = accounts["account_id"].isin(events["account_id"]).mean()

CLV = 7310.0
TREAT_ALL = 52_400
save = DEFAULT_EFFECTIVENESS * CLV
tn_per_fn = int(round((save - DEFAULT_INTERVENTION_COST) / DEFAULT_INTERVENTION_COST))
breakeven = DEFAULT_INTERVENTION_COST / save

HTML = f"""<meta charset="utf-8">
<style>
  * {{ box-sizing: border-box; margin: 0; padding: 0; }}
  body {{
    width: 1180px; background: #fff; color: #1A1A1A;
    font-family: Calibri, "Segoe UI", system-ui, sans-serif;
    -webkit-font-smoothing: antialiased;
  }}
  .page {{ display: flex; gap: 0; padding: 16px 20px 18px; align-items: stretch; }}
  .col {{ flex: 1 1 0; padding: 0 16px; border-left: 1px solid #E4E7EB; }}
  .col:first-child {{ border-left: 0; padding-left: 0; }}

  h2 {{ font-size: 14.5px; font-weight: 700; padding-bottom: 8px;
        border-bottom: 3px solid; margin-bottom: 11px; }}
  .go h2 {{ color: #1E7A4B; border-color: #1E7A4B; }}
  .stop h2 {{ color: #B02E2E; border-color: #B02E2E; }}
  .ask h2 {{ color: #2a78d6; border-color: #2a78d6; }}

  .item {{ margin-bottom: 11px; }}
  .item b {{ display: block; font-size: 13px; margin-bottom: 3px; }}
  .item span {{ display: block; font-size: 12px; color: #5A6270; line-height: 1.42; }}
  .item span i {{ font-style: normal; font-weight: 700; color: #1A1A1A; }}

  .big {{ display: flex; gap: 12px; margin-bottom: 11px; }}
  .big div b {{ display: block; font-size: 22px; line-height: 1; }}
  .big div span {{ display: block; font-size: 11px; color: #5A6270; margin-top: 3px;
                   line-height: 1.25; }}
  .go .big b {{ color: #1E7A4B; }}
  .stop .big b {{ color: #B02E2E; }}
  .ask .big b {{ color: #2a78d6; }}
</style>
<div class="page">

  <div class="col go">
    <h2>Do this now</h2>
    <div class="big">
      <div><b>${TREAT_ALL // 1000}k</b><span>net value of calling<br>everyone at risk</span></div>
      <div><b>{tn_per_fn}:1</b><span>why ranking cannot<br>beat that policy</span></div>
    </div>
    <div class="item">
      <b>Call every at-risk customer</b>
      <span>Base rate <i>{base_rate:.0%}</i> against a <i>{breakeven:.0%}</i>
        break-even. Skipping a leaver costs about <i>{tn_per_fn}&times;</i> what
        skipping a stayer saves, so the model sweep adds only ~1% over calling
        everyone — and that is the positive finding, not a consolation.</span>
    </div>
    <div class="item">
      <b>Hold half back, and log every call</b>
      <span>Randomise who gets the call; measure retention at 180 days. Open a
        CRM field for the touch on Monday so &ldquo;what works&rdquo; stops being
        unanswerable.</span>
    </div>
    <div class="item">
      <b>Only pilot effects that can halve churn</b>
      <span>At this company&rsquo;s signup rate, a 15-point drop is readable in
        ~15 months. A 5-point programme needs a decade — kill it at the gate.</span>
    </div>
  </div>

  <div class="col stop">
    <h2>Do not spend on these</h2>
    <div class="big">
      <div><b>2.8&times;</b><span>calendar rise the<br>file invents alone</span></div>
      <div><b>p=0.24</b><span>top SHAP driver<br>vs shuffled labels</span></div>
    </div>
    <div class="item">
      <b>Do not investigate &ldquo;what changed in 2024&rdquo;</b>
      <span>Random churn dates reproduce the rise (<i>2.76&times;</i>). The trend
        is a property of the extract, not the business.</span>
    </div>
    <div class="item">
      <b>Do not act on the SHAP ranking</b>
      <span>It looks clean and is led by tickets-per-seat — built from tables
        that are not joined in time. Against shuffled labels the top score is
        noise; explainability still ranks when the model learned nothing.</span>
    </div>
    <div class="item">
      <b>Do not fund tenure-targeted onboarding</b>
      <span>Inside any signup cohort, day-300 leaves as often as day-10. That
        rejects &ldquo;new logos are fragile&rdquo; as the reason to spend — not
        every activation idea, only this one.</span>
    </div>
  </div>

  <div class="col ask">
    <h2>Fix this, then ask again</h2>
    <div class="big">
      <div><b>0.41</b><span>AUC from usage &amp;<br>tickets alone</span></div>
      <div><b>0.63</b><span>AUC once those<br>tables are dropped</span></div>
    </div>
    <div class="item">
      <b>Rebuild usage and support timestamps — keep billing</b>
      <span><i>{usage_early:.0%}</i> of usage and <i>{tickets_early:.0%}</i> of
        tickets predate the customer. Those features predict <i>worse than
        chance</i> and drag the rest under. Subscriptions are the usable spine
        (start date tracks signup; plan &times; seats pricing holds).</span>
    </div>
    <div class="item">
      <b>Pick one churn definition</b>
      <span>Flag says <i>{flag_rate:.0%}</i>, event log <i>{log_rate:.0%}</i>,
        and they agree at chance. Until one is ground truth, every score is
        provisional.</span>
    </div>
    <div class="item">
      <b>Re-run the pipeline on the new export</b>
      <span>Same code, new date. Why / who / what-works become answerable, and
        a feature ranking would finally be worth reading.</span>
    </div>
  </div>

</div>
"""

page_path = ROOT / "outputs" / "figures" / "15_what_to_do.html"
page_path.write_text(HTML, encoding="utf-8")

from playwright.sync_api import sync_playwright  # noqa: E402

out = ROOT / "outputs" / "figures" / "15_what_to_do.png"
with sync_playwright() as p:
    browser = p.chromium.launch()
    page = browser.new_page(viewport={"width": 1180, "height": 560},
                            device_scale_factor=3)
    page.goto(page_path.as_uri())
    page.locator(".page").screenshot(path=str(out))
    browser.close()

print("wrote", out)
print(f"cohort {n_cohort}, left {n_left} ({base_rate:.1%})  "
      f"tn_per_fn {tn_per_fn}  treat_all ${TREAT_ALL:,}  "
      f"usage early {usage_early:.1%}  tickets early {tickets_early:.1%}")
