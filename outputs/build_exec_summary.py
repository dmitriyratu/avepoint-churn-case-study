"""Executive summary: 6 slides, structured around the three questions asked.

Slides 2, 3 and 4 answer the three questions in the order they were asked, each
with a plain verdict at the top and its limits stated on the same slide rather
than collected in a footnote at the end.

Evidence convention. Every claim strong enough to change a decision carries its
number in the same sentence, and each slide ends with the honest limit on that
claim in the footnote. Executives do not need the method, but they do need to
see that a number exists and where it stops holding. Anything a sceptical reader
would ask next goes in the speaker notes, not on the slide.

Claims are stated at the strength the evidence supports. "We could not find X"
is not "X does not exist", and slide 3 says so explicitly, because the planted
positive control scores the same as the real one.
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


# Bullets on the three question slides clear the verdict line, which sits at
# 1.80 and is one line of 21pt. Do not let these default back to 1.88.
BODY_TOP = Inches(2.34)

# ============================================================ 1 · BOTTOM LINE
s = blank(prs)
tf = tb(s, M, Inches(1.5), W - 2 * M, Inches(2.2))
run(tf.paragraphs[0], "Customer churn: what we found", 40, bold=True)
p = tf.add_paragraph(); p.space_before = Pt(12)
run(p, "Executive summary. You asked three questions. Here are the three "
       "answers, each with the evidence behind it.", 17, color=MUTED)

bullets(s, [
    ("Why are customers leaving?", INK, True),
    ("  The pattern is about when, not who. Churn roughly quadrupled over two "
     "years, and no customer characteristic we hold separates those who left "
     "from those who stayed.", INK),
    ("Can we predict who will leave?", INK, True),
    ("  Not from this data as it stands. Two data faults have to be fixed "
     "before the question can be answered properly. A better algorithm is not "
     "the missing piece.", RED),
    ("What actions will improve retention?", INK, True),
    ("  Contacting every at-risk customer already pays for itself on the cost "
     "arithmetic alone. Beyond that we cannot yet prove what works, because no "
     "retention action has ever been recorded.", GREEN),
], top=Inches(3.5), size=16, space=7)
footnote(s, "Each answer carries its supporting numbers on its own slide, "
            "including where the evidence is thin. Two of the three are limited "
            "by the data we hold, not by the analysis.")
note(s, "One minute. Give all three answers immediately, then spend one slide on "
        "each. Two of the three answers are negative and I say so on slide one "
        "rather than building up to it. If pressed on how firm they are: answer "
        "one is the strongest result in the study, answer two is a 'cannot "
        "tell' rather than a 'no', answer three rests on cost arithmetic we can "
        "check rather than on any model.")

# ============================================================ 2 · QUESTION 1
s = add("Question 1", "Why are customers leaving?")
verdict(s, "The clearest signal is when they were with us, not who they are.", BLUE)
picture(s, "12_calendar_hazard.png", height=Inches(2.50), top=Inches(2.45))
bullets(s, [
    ("Churn rose from about 5 in every 100 customers a month in 2023 to 22 in "
     "100 by December 2024, on a customer base that did not grow. A 2.8x "
     "increase per year, and the most tightly estimated effect in the data.",
     INK, True),
    ("We tested every characteristic we hold: industry, country, plan, size, "
     "how they found us, usage, support history. None separated leavers from "
     "stayers by more than chance produces. The best reaches AUC 0.62, against "
     "AUC 0.61 from deliberately scrambled labels.", INK),
    ("What we cannot yet say is why the rate rose. The data holds nothing that "
     "varies over time, so we can establish the increase but not attribute it. "
     "Slide 6 says what would fix that.", RED),
], top=Inches(5.02), size=12.5, space=4)
footnote(s, "AUC is ranking accuracy: 0.50 is a coin flip, 1.00 is perfect. "
            "With 54 churners we could only have detected a large difference "
            "between customer types, so \"none found\" is not proof that none "
            "exists. The rise over calendar time is the opposite case: large, "
            "and precisely estimated.")
note(s, "If asked how solid each half is. The rise: Poisson trend on calendar "
        "month with accounts-at-risk as offset, rate ratio 1.089 per month, "
        "p = 2e-16. The flat cross-section: Cox over 21 characteristics returns "
        "global p = 0.57 and concordance 0.571; log-rank across 7 segments has "
        "a smallest adjusted p of 0.39. The two facts are one fact. Something "
        "that moves every account at once leaves no differences between "
        "accounts for a model to find.")

# ============================================================ 3 · QUESTION 2
s = add("Question 2", "Can we predict churn before it happens?")
verdict(s, "Not from this data as it stands, and a better algorithm is not the "
           "missing piece.", RED)
bullets(s, [
    ("We tried three genuinely different approaches. None beat chance once we "
     "account for having tried many.", INK, True),
    "  Best model: AUC 0.58, on a scale where 0.50 is a coin flip and 1.00 is "
    "perfect. Its uncertainty spans AUC 0.37 to 0.75, the same search on "
    "deliberately scrambled data reached AUC 0.58 or better in 7 of 20 "
    "attempts, and priced for the fact that we picked the winner it is AUC 0.53.",
    "",
    ("The careful version: we cannot tell, rather than there is nothing there.",
     INK, True),
    "  A planted relationship that is real but weak also reaches AUC 0.58 here. "
    "At 54 churners the two are indistinguishable. Hence re-measure, rather "
    "than close the question.",
    "",
    ("The warning time we would need makes it harder, not easier.", INK, True),
    "  Asking for 30 days of notice, so somebody could act, moves every horizon "
    "into the AUC 0.42 to 0.52 band. A prediction that arrives too late to use "
    "is not useful.",
    "",
    ("Two data faults are doing most of the damage.", INK, True),
    "  Three sources disagree on who churned: they agree on about one account "
    "in five, which is what two unrelated columns would do.",
    "  Roughly three-quarters of usage rows are dated before the subscription "
    "they belong to.",
    "",
    ("Practically: do not buy or build churn scoring yet. Fix those two faults "
     "and re-measure, which is one command once the data is clean.", BLUE, True),
], top=BODY_TOP, size=14, space=3)
footnote(s, "AUC is ranking accuracy: 0.50 is a coin flip, 1.00 is perfect. The "
            "metric is not doing the work — precision-recall and F1 agree, and "
            "on F1 the model beats a no-model policy by less than the optimism "
            "in its own tuned threshold. Ladder and null tests in the technical "
            "deck.")
note(s, "Two challenges to expect on this slide.\n\n"
        "1. 'But AUC 0.58 is above 0.50, so there is something there.' The "
        "answer is that 0.583 is the maximum of a 15-model search, not a "
        "measurement of one model. Nested cross-validation, which prices in "
        "that search, gives AUC 0.534 +/- 0.016. The same search on shuffled "
        "labels averages AUC 0.566 and clears 0.583 in 7 of 20 trials. "
        "Permutation p = 0.076.\n\n"
        "2. 'Is AUC the right metric at 31% positives, or should this be F1 or "
        "precision-recall?' AUC is right here because its null is 0.50 at any "
        "base rate, which is what makes the twelve horizon/buffer cells "
        "comparable when their positive rates run 11% to 45%. But it does not "
        "matter which we use: average precision is 0.378 against a base-rate "
        "null of 0.305, permutation p = 0.070, against p = 0.076 on AUC. And F1 "
        "is 0.497 against 0.467 for contacting everyone -- a lift of 0.029, "
        "which is smaller than the 0.049 optimism in its own tuned threshold. "
        "F1 is the weakest of the three for this claim precisely because it is "
        "threshold-dependent and has no natural null.\n\n"
        "The three approaches are the cross-sectional classifier, a Cox "
        "survival model (concordance 0.509) and the horizon sweep. Do not "
        "overclaim the negative: the planted weak-signal control scores AUC "
        "0.584 against our 0.583, so 'underpowered' is the defensible claim and "
        "'no signal exists' is not.")

# ============================================================ 4 · QUESTION 3
s = add("Question 3", "What actions will improve retention?")
verdict(s, "One action pays for itself today. Beyond that, we cannot yet prove "
           "what works.", GREEN)
bullets(s, [
    ("Do this now: contact every at-risk customer, and do not rank them.",
     GREEN, True),
    "  A call costs around $150 and an average customer is worth around $7,300. "
    "If one call in five succeeds, it pays for itself on anyone with more than a "
    "10% chance of leaving. In this group 31% left within 90 days, so the margin "
    "is wide.",
    "  This rests on the cost arithmetic rather than on the model, which is why "
    "the prediction result does not hold up the decision.",
    "",
    ("Hold off on this: the structured onboarding programme.", RED, True),
    "  New customers look more fragile, which is the usual reason to invest here. "
    "The pattern appears to be composition rather than tenure: pooled together "
    "the risk looks front-loaded, but within each signup cohort it is flat. On "
    "that reading a ten-month customer is about as likely to leave as a ten-day one.",
    "",
    ("Everything else we might try is currently untestable, not disproven.",
     INK, True),
    "  No call, discount or campaign is recorded anywhere in the data. Without a "
    "record of what we did there is no way to measure what worked, so the other "
    "retention ideas are opinions for now, including the good ones.",
], top=Inches(2.45), size=15, space=5)
footnote(s, "Costs and save rates are our estimates and should be replaced with "
            "finance's figures. The onboarding finding is a within-cohort "
            "result on 54 churners and would be worth revisiting if richer "
            "cohort data arrives.")
note(s, "The first action is the one thing on this deck that does not depend on "
        "the modelling. Break-even is 10.3% against a base rate of 30.5%, a "
        "factor of three, so it survives large errors in the cost assumptions. "
        "On onboarding: the pooled hazard genuinely does fall, rho = 0.737 with "
        "p = 1.7e-13, which is why the obvious analysis recommends the "
        "programme. Within each signup cohort rho comes back to about 1 (range "
        "0.87 to 1.25, none significant), so the pooled shape is which cohorts "
        "are present at each tenure, not tenure itself. This reverses a "
        "recommendation an earlier pass of this work made.")

# ============================================================ 5 · NUMBERS
s = add("The economics", "Why calling everyone is the right call today")
stat(s, M, Inches(2.1), "$7,300", "Average customer value, 3 years, discounted")
stat(s, Inches(4.3), Inches(2.1), "31%", "Of this at-risk group left within 90 days (54 of 177)")
stat(s, Inches(7.9), Inches(2.1), "10%", "Break-even chance of leaving for one call")
stat(s, M, Inches(4.1), "$52,400", "Value of calling all 177, at a 1-in-5 save rate", GREEN)
stat(s, Inches(4.3), Inches(4.1), "1%", "What a perfect ranking would add on top", MUTED)
stat(s, Inches(7.9), Inches(4.1), "15 months", "To prove a halving of churn; smaller gains take years", MUTED)
bullets(s, [
    ("Customers here leave about three times as often as the break-even point, "
     "so contacting all of them makes money without any model involved. A "
     "perfect ranking would add roughly 1% on top. That is why the prediction "
     "result does not block this decision.", INK, True),
], top=Inches(5.60), size=14)
footnote(s, "The call cost and the 1-in-5 save rate are our assumptions rather "
            "than finance's, and should be replaced. We varied both widely and "
            "the treat-everyone conclusion held: break-even would have to rise "
            "past 31% to reverse it.")
note(s, "Every figure here is arithmetic on quantities we can check, not model "
        "output. Net value at the model's best threshold is $53,000 against "
        "$52,400 for treating everyone, hence the 1%, and that 1% is itself "
        "optimistic because the threshold was chosen on the same data. The 15 "
        "months comes from a minimum detectable effect of 17.2pp at 88 per arm; "
        "a 5pp improvement would need about ten years.")

# ============================================================ 6 · ASK
s = add("What we need", "Three data requests, and what each one unlocks")
bullets(s, [
    ("To explain why churn is rising  (Question 1)", BLUE, True),
    "  Give us anything that varies over time and could plausibly affect "
    "customers: price changes, release dates, outages, competitor moves. We can "
    "already establish the rise. This is what would let us attribute it.",
    "",
    ("To make prediction answerable  (Question 2)", INK, True),
    "  Agree one definition of churn and apply it everywhere. The three "
    "definitions we currently have agree on about one account in five.",
    "  Fix the product usage timestamps. Roughly three-quarters of rows are "
    "dated before the subscription they belong to.",
    "  We would put this first, because it also unblocks the other two.",
    "",
    ("To learn which actions work  (Question 3)", GREEN, True),
    "  Start logging every call, discount and campaign with a date and an "
    "account attached. Then run one deliberate trial where half the at-risk "
    "customers are contacted and half are not.",
    "  Worth setting expectations on what a trial of our size can settle: "
    "roughly halving churn would show up in about 15 months, while a 5-point "
    "improvement would take years.",
], size=15, space=7)
footnote(s, "None of these is a modelling problem. All three are data "
            "collection, none of them is expensive, and none asks for a "
            "decision on tooling yet.", INK)
note(s, "Close on the ask rather than on the negative results. The second "
        "request is the one to push, because the label and the timestamps block "
        "any re-measurement of question two and also limit what we can say "
        "about the other two.")

prs.save(OUT)
print(f"{OUT.name}  ({OUT.stat().st_size/1e6:.2f} MB, {len(prs.slides._sldIdLst)} slides)")
