"""What to do about churn — the summary figure.

The rest of the deck earns its conclusions one at a time and each argument takes
a slide. This is the version to present: three columns, no statistics, and every
number a count taken straight from the tables.

Run from the repo root:  python build/fig_what_to_do_html.py
"""
import sys
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src import pipeline  # noqa: E402
from src.config import TARGET  # noqa: E402

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

HTML = f"""<meta charset="utf-8">
<style>
  * {{ box-sizing: border-box; margin: 0; padding: 0; }}
  body {{
    width: 1180px; background: #fff; color: #1A1A1A;
    font-family: Calibri, "Segoe UI", system-ui, sans-serif;
    -webkit-font-smoothing: antialiased;
  }}
  .page {{ display: flex; gap: 0; padding: 20px 24px 22px; align-items: stretch; }}
  .col {{ flex: 1 1 0; padding: 0 22px; border-left: 1px solid #E4E7EB; }}
  .col:first-child {{ border-left: 0; padding-left: 0; }}

  h2 {{ font-size: 15px; font-weight: 700; padding-bottom: 9px;
        border-bottom: 3px solid; margin-bottom: 14px; }}
  .go h2 {{ color: #1E7A4B; border-color: #1E7A4B; }}
  .stop h2 {{ color: #B02E2E; border-color: #B02E2E; }}
  .ask h2 {{ color: #2a78d6; border-color: #2a78d6; }}

  .item {{ margin-bottom: 15px; }}
  .item b {{ display: block; font-size: 13.5px; margin-bottom: 4px; }}
  .item span {{ display: block; font-size: 12.5px; color: #5A6270; line-height: 1.5; }}
  .item span i {{ font-style: normal; font-weight: 700; color: #1A1A1A; }}

  .big {{ display: flex; gap: 20px; margin-bottom: 16px; }}
  .big div b {{ display: block; font-size: 25px; line-height: 1; }}
  .big div span {{ display: block; font-size: 11.5px; color: #5A6270; margin-top: 4px;
                   line-height: 1.3; }}
  .go .big b {{ color: #1E7A4B; }}
</style>
<div class="page">

  <div class="col go">
    <h2>Do this now</h2>
    <div class="big">
      <div><b>{base_rate:.0%}</b><span>of at-risk customers<br>left within 90 days</span></div>
      <div><b>1%</b><span>what the model adds<br>over calling everyone</span></div>
    </div>
    <div class="item">
      <b>Call every at-risk customer</b>
      <span>A call is worth making whenever it costs less than the customer's
        churn risk, as a share of what a save is worth. At <i>{base_rate:.0%}</i>
        that is true of everyone in the cohort, so there is nothing for a ranking
        to improve.</span>
    </div>
    <div class="item">
      <b>Split the list in half and hold one half back</b>
      <span>Measure retention after 180 days against the half nobody called.
        Without a control group there is no way to know whether the calls did
        anything.</span>
    </div>
  </div>

  <div class="col stop">
    <h2>Do not do this</h2>
    <div class="item">
      <b>Do not investigate what changed in 2024</b>
      <span>Churn appears to rise <i>2.8&times; a year</i>. Rebuild the file with
        random churn dates and it rises <i>2.76&times;</i>. The trend belongs to
        the file.</span>
    </div>
    <div class="item">
      <b>Do not fund the onboarding programme</b>
      <span>New customers look fragile for the same reason. Inside any single
        signup cohort, a day-300 customer leaves as often as a day-10 one.</span>
    </div>
    <div class="item">
      <b>Do not target a segment</b>
      <span>Ten ways of slicing the customer base, none of them separates leavers
        from stayers. The best looked real at p = 0.03 until you account for
        having asked ten questions.</span>
    </div>
  </div>

  <div class="col ask">
    <h2>Collect this, then ask again</h2>
    <div class="item">
      <b>Log every call, discount and campaign, with a timestamp</b>
      <span>Not one appears anywhere in these five tables. With no record of what
        was tried, no question about what works can be answered.</span>
    </div>
    <div class="item">
      <b>Join usage and support to the right customer</b>
      <span><i>{usage_early:.0%}</i> of usage rows and <i>{tickets_early:.0%}</i>
        of support tickets are dated before the customer existed. Every feature
        built from them is noise.</span>
    </div>
    <div class="item">
      <b>Agree one definition of churn</b>
      <span>Three columns claim to record it. One says <i>{flag_rate:.0%}</i> of
        customers left, another says <i>{log_rate:.0%}</i>, and they agree no more
        often than chance.</span>
    </div>
    <div class="item">
      <span>The pipeline takes a date, so re-running all of this on a corrected
        export is one command.</span>
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
      f"usage early {usage_early:.1%}  tickets early {tickets_early:.1%}  "
      f"flag {flag_rate:.1%}  log {log_rate:.1%}")
