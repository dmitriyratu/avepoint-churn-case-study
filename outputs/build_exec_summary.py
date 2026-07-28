"""Executive summary: 6 slides, structured around the three questions asked.

Slides 2, 3 and 4 answer the three questions in the order they were asked, each
with a plain verdict at the top and its limits stated on the same slide rather
than collected in a footnote at the end.
"""
from pathlib import Path

from deck_style import (BLUE, GREEN, H, INK, M, MUTED, RED, W, blank, bullets,
                        footnote, header, new_deck, note, picture, run, stat, tb)
from pptx.util import Inches, Pt

OUT = Path(__file__).with_name("AvePoint_Executive_Summary.pptx")

prs = new_deck()


def add(kicker, title, colour=INK):
    s = blank(prs)
    header(s, kicker, title, colour)
    return s


def verdict(slide, text, colour, top=Inches(1.80)):
    """The one-line answer, sitting directly under the question."""
    tf = tb(slide, M, top, W - 2 * M, Inches(0.75))
    run(tf.paragraphs[0], text, 21, bold=True, color=colour)
    return tf


# ============================================================ 1 · BOTTOM LINE
s = blank(prs)
tf = tb(s, M, Inches(1.5), W - 2 * M, Inches(2.2))
run(tf.paragraphs[0], "Customer churn: what we found", 40, bold=True)
p = tf.add_paragraph(); p.space_before = Pt(12)
run(p, "Executive summary. You asked three questions. Here are the three "
       "answers.", 17, color=MUTED)

bullets(s, [
    ("Why are customers leaving?", INK, True),
    ("  Not because of who they are. Because of when. Churn has roughly "
     "quadrupled over two years, and it affects every kind of customer equally.",
     INK),
    ("Can we predict who will leave?", INK, True),
    ("  No, and not with more modelling either. Two data problems have to be "
     "fixed first.", RED),
    ("What actions will improve retention?", INK, True),
    ("  Calling every at-risk customer already pays for itself today. Beyond "
     "that we cannot yet prove what works, because we have never recorded taking "
     "an action.", GREEN),
], top=Inches(3.5), size=16, space=7)
note(s, "One minute. Give all three answers immediately, then spend one slide on "
        "each. Two of the three answers are negative and I say so on slide one "
        "rather than building up to it.")

# ============================================================ 2 · QUESTION 1
s = add("Question 1", "Why are customers leaving?")
verdict(s, "Not because of who they are. Because of when they were with us.", BLUE)
picture(s, "12_calendar_hazard.png", height=Inches(3.0), top=Inches(2.55))
bullets(s, [
    ("In 2023 about 5 in every 100 customers left each month. By December 2024 it "
     "was 22 in every 100. The customer base stayed the same size, so this is not "
     "a side effect of growth.", INK, True),
    ("We tested every customer characteristic we hold: industry, country, plan, "
     "company size, how they found us, product usage and support history. None of "
     "them separates the customers who leave from the ones who stay.", INK),
    ("What we cannot yet say is why the rate rose. The data holds nothing that "
     "changes over time, so we can prove the increase is real but not name the "
     "cause. Slide 6 says what would fix that.", RED),
], top=Inches(5.5), size=13.5, space=6)

# ============================================================ 3 · QUESTION 2
s = add("Question 2", "Can we predict churn before it happens?")
verdict(s, "No. Not from this data, and not with a better algorithm.", RED)
bullets(s, [
    ("We tried three genuinely different approaches. All three performed about as "
     "well as flipping a coin.", INK, True),
    "",
    ("There is a second, harder problem underneath that.", INK, True),
    "  Even the faint signal that does exist only appears in the days just before "
    "a customer leaves. If we ask the model for 30 days of warning, so somebody "
    "could actually act on it, performance drops to no better than guessing. A "
    "prediction that arrives too late to use is not useful.",
    "",
    ("The reasons are about data, not method.", INK, True),
    "  Our records disagree about who has even churned. Three separate sources "
    "agree for only one customer in five.",
    "  Most product usage records are dated before the subscription they belong "
    "to, so anything built on recent activity is built on nonsense.",
    "  We only have 54 customers who left. That is far too few to learn from.",
    "",
    ("What this means practically: do not buy or build a churn scoring product "
     "yet. Fix the two data problems, then measure again. That is one command "
     "once the data is clean.", BLUE, True),
])

# ============================================================ 4 · QUESTION 3
s = add("Question 3", "What actions will improve retention?")
verdict(s, "One action already pays for itself. Beyond that, we cannot yet prove "
           "what works.", GREEN)
bullets(s, [
    ("Do this now: call every at-risk customer, and do not rank them.", GREEN, True),
    "  A call costs around $150. A customer is worth around $7,300. If one call "
    "in five succeeds, it pays for itself on any customer with more than a 10% "
    "chance of leaving. In this group 31% leave, so the maths is comfortable.",
    "  Calling everyone is profitable today, with no model needed. That is why "
    "the failed prediction work does not hold up the decision.",
    "",
    ("Do not do this: the onboarding programme.", RED, True),
    "  It looks like new customers are more fragile, which is the usual reason to "
    "invest here. The pattern is a trick of the numbers. Corrected, a customer of "
    "ten months is just as likely to leave as one of ten days.",
    "",
    ("The limit on everything else we might try.", INK, True),
    "  Nowhere in our data is there a record of a customer being called, offered "
    "a discount, or put into a campaign. With no record of what we did, there is "
    "no way to measure what worked. Every other retention idea is currently an "
    "opinion, including good ones.",
])
footnote(s, "The first action rests on cost and value, which we know well. It "
            "does not rest on the model. To go further we need to log what we do "
            "and run one deliberate trial. Slide 6 covers both.")

# ============================================================ 5 · NUMBERS
s = add("The economics", "Why calling everyone is the right call today")
stat(s, M, Inches(2.1), "$7,300", "What an average customer is worth to us")
stat(s, Inches(4.3), Inches(2.1), "31%", "Of at-risk customers leave within 90 days")
stat(s, Inches(7.9), Inches(2.1), "10%", "The break-even point for making a call")
stat(s, M, Inches(4.1), "$52,400", "Value of calling this group of 177 customers", GREEN)
stat(s, Inches(4.3), Inches(4.1), "1%", "Extra value a perfect ranking would add", MUTED)
stat(s, Inches(7.9), Inches(4.1), "15 months", "To prove a retention programme works", MUTED)
bullets(s, [
    ("Customers leave three times more often than the break-even point, so "
     "contacting all of them makes money without any model involved. A perfect "
     "ranking would add about 1% on top. That is why the failed model does not "
     "block the decision.", INK, True),
], top=Inches(6.0), size=15)
footnote(s, "Cost and success rates are estimates and should be replaced with "
            "finance's figures. The conclusion holds across a wide range of them.")

# ============================================================ 6 · ASK
s = add("What we need", "Three data requests, and what each one unlocks")
bullets(s, [
    ("To explain why churn is rising  (Question 1)", BLUE, True),
    "  Give us anything that changes over time and could affect customers: price "
    "changes, release dates, outages, competitor moves. We can already prove the "
    "rise is real. This would let us name the cause.",
    "",
    ("To make prediction possible  (Question 2)", INK, True),
    "  Agree one definition of churn and apply it everywhere. Our three current "
    "sources agree for one customer in five.",
    "  Fix the product usage timestamps. Most records are dated before the "
    "subscription they belong to.",
    "  This is the highest-leverage item on the page, because it also unblocks "
    "the other two.",
    "",
    ("To learn which actions work  (Question 3)", GREEN, True),
    "  Start logging every call, discount and campaign with a date and a customer "
    "attached. Then run one deliberate trial where half the at-risk customers are "
    "contacted and half are not.",
    "  Be aware we can only detect a large effect at our size. Roughly halving "
    "churn would show up in about 15 months. A smaller improvement would take "
    "years to prove.",
], size=15, space=7)
footnote(s, "None of these is a modelling problem. All three are data collection, "
            "and none of them is expensive.", INK)

prs.save(OUT)
print(f"{OUT.name}  ({OUT.stat().st_size/1e6:.2f} MB, {len(prs.slides._sldIdLst)} slides)")
