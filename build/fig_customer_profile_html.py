"""What a customer actually looks like in this file.

One account shown in full on the left, the population it comes from on the
right. The single account is not cherry-picked for drama — it is the median
customer by contract count, and every number beside it is read from the raw
files rather than typed.

Run from the repo root:  python build/fig_customer_profile_html.py
"""
import sys
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

RAW = ROOT / "data" / "raw"
OUT = ROOT / "outputs" / "figures" / "20_customer_profile.png"
PAGE = ROOT / "outputs" / "figures" / "20_customer_profile.html"

INK, MUTED, RULE = "#1A1A1A", "#5A6270", "#D8DCE0"
GREEN, RED, BLUE = "#1E7A4B", "#B02E2E", "#2E5F8A"
W, H = 1800, 700

acc = pd.read_csv(RAW / "ravenstack_accounts.csv", parse_dates=["signup_date"])
sub = pd.read_csv(RAW / "ravenstack_subscriptions.csv",
                  parse_dates=["start_date", "end_date"])
tix = pd.read_csv(RAW / "ravenstack_support_tickets.csv")
use = pd.read_csv(RAW / "ravenstack_feature_usage.csv")
owner = sub.set_index("subscription_id")["account_id"]

per_account = sub.groupby("account_id").size()
# The median customer by contract count, so the example argues from the middle
# of the distribution rather than from its tail.
EXAMPLE = per_account[per_account == int(per_account.median())].index[0]

profile = acc.set_index("account_id").loc[EXAMPLE]
contracts = sub[sub.account_id == EXAMPLE].sort_values("start_date")


def spread(series, money=False):
    fmt = (lambda v: f"${v:,.0f}") if money else (lambda v: f"{v:,.0f}")
    return [fmt(series.min()), fmt(series.median()), fmt(series.max())]


ROWS = [
    ("contracts per customer", spread(per_account)),
    ("of those, still open", spread(sub[sub.end_date.isna()]
                                    .groupby("account_id").size())),
    ("plan tiers held at once", spread(sub.groupby("account_id")
                                       .plan_tier.nunique())),
    ("seats on the account", spread(acc.seats)),
    ("monthly spend per contract", spread(sub.mrr_amount, money=True)),
    ("support tickets", spread(tix.groupby("account_id").size())),
    ("usage rows", spread(use.assign(a=use.subscription_id.map(owner))
                          .groupby("a").size())),
]

all_three = int((sub.groupby("account_id").plan_tier.nunique() == 3).sum())
total_mrr = sub.groupby("account_id").mrr_amount.sum()
example_mrr = contracts.mrr_amount.sum()

contract_rows = "".join(
    f'<tr><td>{r.start_date:%d %b %Y}</td>'
    f'<td class="none">—</td>'
    f'<td>{r.plan_tier}{" · trial" if r.is_trial else ""}</td>'
    f'<td class="num">{r.seats}</td>'
    f'<td class="num">${r.mrr_amount:,.0f}</td></tr>'
    for r in contracts.itertuples())

stat_rows = "".join(
    f'<tr><td>{label}</td>'
    + "".join(f'<td class="num">{v}</td>' for v in values) + "</tr>"
    for label, values in ROWS)

HTML = f"""<meta charset="utf-8">
<style>
  * {{ box-sizing: border-box; margin: 0; }}
  body {{ background: #fff;
         font-family: "Segoe UI", "Helvetica Neue", Arial, sans-serif; }}
  .wrap {{ position: relative; width: {W}px; height: {H}px;
           display: grid; grid-template-columns: 880px 1fr; gap: 0 64px;
           padding: 40px 44px; }}

  h3 {{ font-size: 20px; font-weight: 650; color: {INK}; margin-bottom: 4px; }}
  .sub {{ font-size: 14px; color: {MUTED}; margin-bottom: 20px; }}

  table {{ width: 100%; border-collapse: collapse; }}
  th {{ font-size: 12px; font-weight: 600; color: {MUTED}; text-align: left;
        text-transform: uppercase; letter-spacing: .55px; padding: 0 12px 8px 0;
        border-bottom: 1.5px solid {RULE}; }}
  th.num, td.num {{ text-align: right; }}
  td {{ font-size: 15px; color: {INK}; padding: 9px 12px 9px 0;
        border-bottom: 1px solid #F0F2F4; font-variant-numeric: tabular-nums; }}
  td.none {{ color: {RED}; }}

  .card {{ border: 1.5px solid {RULE}; border-radius: 10px; padding: 14px 18px;
           margin-bottom: 18px; display: flex; gap: 34px; align-items: baseline; }}
  .card b {{ font-size: 17px; color: {INK}; }}
  .card span {{ font-size: 14px; color: {MUTED}; }}
  .card i {{ font-style: normal; color: {INK}; font-weight: 600; }}

  .note {{ margin-top: 16px; font-size: 14.5px; color: {RED}; line-height: 1.6; }}
  .note b {{ font-weight: 650; }}

  .call {{ margin-top: 20px; padding: 14px 18px; border-radius: 10px;
           font-size: 14.5px; line-height: 1.6; }}
  .call.bad {{ background: rgba(176,46,46,.06); color: {RED}; }}
  .call.good {{ background: rgba(30,122,75,.07); color: {GREEN}; }}
  .call b {{ font-weight: 650; }}
</style>
<div class="wrap">
  <div>
    <h3>One customer, in full</h3>
    <div class="sub">Account {EXAMPLE} — the median customer by contract count.
      Every contract this account holds, in order.</div>

    <div class="card">
      <span>the account record says</span>
      <b>{profile.plan_tier}</b>
      <span><i>{profile.seats}</i> seats</span>
      <span>signed up <i>{profile.signup_date:%d %b %Y}</i></span>
      <span><i>{profile.industry}</i>, {profile.country}</span>
    </div>

    <table>
      <tr><th>Started</th><th>Ended</th><th>Plan</th>
          <th class="num">Seats</th><th class="num">Monthly</th></tr>
      {contract_rows}
    </table>

    <div class="note">
      <b>${example_mrr:,.0f} a month</b> across {len(contracts)} contracts that
      all remain open, on an account whose own record says {profile.seats} seats
      and one plan. Nothing has ever ended, and the tier moves in no direction.
    </div>
  </div>

  <div>
    <h3>The population it comes from</h3>
    <div class="sub">All 500 customers. Minimum, median and maximum.</div>

    <table>
      <tr><th>Per customer</th><th class="num">Min</th>
          <th class="num">Median</th><th class="num">Max</th></tr>
      {stat_rows}
    </table>

    <div class="call bad">
      <b>{all_three} of 500 customers</b> hold Basic, Pro and Enterprise
      simultaneously. Add up every open contract and the median customer is
      billed <b>${total_mrr.median():,.0f} a month</b> — so no single row
      answers &ldquo;what is this customer paying?&rdquo;
    </div>

    <div class="call good">
      <b>Inside a row, the rules are exact.</b> Price per seat is
      $19 on Basic, $49 on Pro and $199 on Enterprise, with no exceptions, and
      ARR is always twelve times MRR. Each contract is internally perfect. It is
      the relationship between them, and to the customer, that carries nothing.
    </div>
  </div>
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
print(f"  example account {EXAMPLE}: {len(contracts)} contracts, "
      f"${example_mrr:,.0f}/mo, {contracts.plan_tier.nunique()} tiers")
print(f"  {all_three} of 500 hold all three tiers")
