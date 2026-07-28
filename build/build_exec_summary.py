"""Executive summary: 6 slides, structured around the three questions asked.

Slides 2, 3 and 4 answer the three questions in the order they were asked, each
with a plain verdict at the top and its limits stated on the same slide rather
than collected in a footnote at the end.

Evidence convention. Every claim strong enough to change a decision carries its
number in the same sentence, and each slide ends with the honest limit on that
claim in the footnote. Executives do not need the method, but they do need to
see that a number exists and where it stops holding. Anything a sceptical reader
would ask next goes in the speaker notes, not on the slide.

Claims are stated at the strength the evidence supports, and that strength is
not uniform. For product usage and support the negative is flat — those records
were never linked to the customers they describe, so there was nothing to find.
For customer characteristics it is an underpowered null: "we could not detect X"
rather than "X does not exist", because the planted weak-signal control scores
the same as the real one. Slide 3 separates the two on the slide itself.

Slide 2 states a withdrawn finding rather than hiding it. The rise in churn over
2024 was the strongest result in the study and it is reproduced exactly by a
random-date simulation (src/generator.py, notebook 16). The deck presents the
test once, forwards, as the answer to Question 1 — not as a correction appended
to a claim made earlier in the same deck.
"""
from pathlib import Path

from deck_style import (BLUE, GREEN, H, INK, M, MUTED, RED, W, blank, bullets,
                        footnote, header, new_deck, note, picture, run, stat, tb)
from pptx.util import Inches, Pt

OUT = Path(__file__).resolve().parents[1] / "outputs" / "decks" / "AvePoint_Executive_Summary.pptx"
OUT.parent.mkdir(parents=True, exist_ok=True)

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
    ("  This data cannot tell us. The one pattern that looked like an answer — "
     "churn climbing through 2024 — is produced by the way the file was built, "
     "and we can show that directly.", RED),
    ("Can we predict who will leave?", INK, True),
    ("  No, and a better algorithm is not the missing piece. Most of what we "
     "hold about customer behaviour is not connected to the customers it is "
     "supposed to describe.", RED),
    ("What actions will improve retention?", INK, True),
    ("  Contacting every at-risk customer already pays for itself on the cost "
     "arithmetic alone. Beyond that we cannot yet prove what works, because no "
     "retention action has ever been recorded.", GREEN),
], top=Inches(3.5), size=16, space=7)
footnote(s, "Each answer carries its numbers on its own slide. The first two are "
            "limited by the data we hold, not by the analysis — and knowing that "
            "is what stops us spending money on either of them.")
note(s, "One minute. Give all three answers immediately, then one slide each. "
        "Two of the three are negative and I say so here rather than building "
        "up to it.\n\n"
        "The first answer changed late in the work, and that is the part worth "
        "being open about if asked. We had a strong result — churn accelerating "
        "2.8x a year, p = 2e-16 — and it is reproduced exactly by a simulation "
        "that contains nothing but a random number generator. Slide 2 shows the "
        "overlay. Catching it is worth more than the finding was, because acting "
        "on it would have cost a quarter of somebody's time.\n\n"
        "The third answer is untouched by any of this. It rests on cost "
        "arithmetic we can check, not on the model.")

# ============================================================ 2 · QUESTION 1
s = add("Question 1", "Why are customers leaving?", RED)
verdict(s, "This data cannot tell us — and the pattern that looked like an "
           "answer is not one.", RED)
picture(s, "16_artefact_exec.png", height=Inches(2.55), top=Inches(2.42))
bullets(s, [
    ("Churn looks like it climbed from 5 in every 100 customers a month to 22 in "
     "100 by December 2024. That was our strongest number by a distance.",
     INK, True),
    ("It is manufactured. Every churn date in the file is a random date between "
     "the day the customer joined and the last day of the file. Random dates "
     "crowd towards the end of a file, which makes churn look like it is "
     "speeding up when nothing has happened. Rebuilding the data from that one "
     "rule reproduces the black line exactly — that is the red band.", RED, True),
    ("We also tested every customer characteristic we hold: industry, country, "
     "plan, size, how they found us, usage and support history. None separates "
     "the customers who left from those who stayed.", INK),
], top=Inches(5.10), size=12, space=4)
footnote(s, "What this is worth: it stops us spending a quarter hunting for a "
            "2024 price change, outage or competitor move that did not happen. "
            "Slide 6 says what would actually let us answer the question.", INK)
note(s, "Expect: 'you showed us a rising chart last time — what changed?'\n\n"
        "What changed is the test. The original one asked whether the rise was "
        "bigger than random noise. It was, comfortably: rate ratio 1.089 a "
        "month, p = 2e-16. The right question on a data extract is different — "
        "would the file produce this on its own. Nothing can be recorded after "
        "the last day of the file, so if churn dates are assigned at random they "
        "pile up at the end and the rate appears to climb.\n\n"
        "We tested that by rebuilding the data: same customers, same joining "
        "dates, same number of departures each, only the dates redrawn at "
        "random. That gives a 2.78x annual rise against the 2.79x we observe, "
        "with all 24 months inside the band. Our result sits at the 52nd "
        "percentile of pure chance.\n\n"
        "If pressed on the flat cross-section: Cox over 21 characteristics "
        "returns global p = 0.57 and concordance 0.571; log-rank across 7 "
        "segments has a smallest adjusted p of 0.39.\n\n"
        "The honest summary is that we now have a proven negative rather than an "
        "unproven positive, and the proven negative saves money.")

# ============================================================ 3 · QUESTION 2
s = add("Question 2", "Can we predict churn before it happens?")
verdict(s, "Not from this data as it stands, and a better algorithm is not the "
           "missing piece.", RED)
picture(s, "05_all_framings.png", height=Inches(1.92), top=Inches(2.30))
bullets(s, [
    ("The reason is the data, and it is now specific.", INK, True),
    "  Product usage and support tickets are not tied to the customers they "
    "describe. Their dates are scattered at random across the whole two years, "
    "which is why three-quarters of usage rows are dated before the subscription "
    "they belong to and half the tickets before the customer existed.",
    "  The thing we are predicting is a random date too, as slide 2 shows. And "
    "three sources disagree on who churned: they agree on about one account in "
    "five, which is what two unrelated columns would do.",
    "",
    ("How firm is this? Firmer for some parts than others.", INK, True),
    "  For usage and support we can say there is nothing there: those records "
    "were never linked to the customer in the first place. For customer "
    "characteristics — industry, plan, size — the honest claim is that we could "
    "not detect anything with 54 churners, not that nothing exists.",
    "",
    ("Practically: do not buy or build churn scoring on this. It needs a "
     "different export, not a better model — see slide 6.", BLUE, True),
], top=Inches(4.34), size=12.5, space=3)
footnote(s, "The scale: pick one customer who left and one who stayed — the "
            "number is the chance the model scored the leaver as riskier. 0.50 "
            "is a coin toss, 1.00 is always right. The faint bars are how far "
            "the score moves between test splits. The bottom row is the one "
            "that matters operationally: ask for 30 days of warning, so "
            "somebody can act, and it lands below a coin toss. "
            "Precision-recall and F1 agree, and "
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
        "survival model (concordance 0.509) and the horizon sweep.\n\n"
        "Be precise about how strong the negative is, because it differs by "
        "source. For usage and support it is flat: those timestamps correlate "
        "with their own account's signup date at r = 0.002 and r = 0.014, so "
        "there was never anything to find. For customer characteristics it is "
        "an underpowered null rather than an absence — the planted weak-signal "
        "control scores AUC 0.584 against our 0.583, so at 54 churners the two "
        "are indistinguishable. The slide separates them deliberately.")

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
    "It is the same random date as slide 2: a customer who joined recently has a "
    "shorter window for that date to land in, so they look worse at every age "
    "without anything about them being worse. Within a single joining month, a "
    "ten-month customer is as likely to leave as a ten-day one.",
    "",
    ("Everything else we might try is currently untestable, not disproven.",
     INK, True),
    "  No call, discount or campaign is recorded anywhere in the data. Without a "
    "record of what we did there is no way to measure what worked, so the other "
    "retention ideas are opinions for now, including the good ones.",
], top=Inches(2.45), size=15, space=5)
footnote(s, "Costs and save rates are our estimates and should be replaced with "
            "finance's figures. The conclusion holds across a wide range of "
            "them: break-even would have to rise past 31% to reverse it.")
note(s, "The first action is the one thing on this deck that does not depend on "
        "the modelling, or on slide 2. Break-even is 10.3% against a base rate "
        "of 30.5%, a factor of three, so it survives large errors in the cost "
        "assumptions.\n\n"
        "On onboarding: the pooled hazard genuinely does fall, rho = 0.737 with "
        "p = 1.7e-13, which is why the obvious analysis funds the programme. "
        "Simulated data with no tenure effect in it at all clears the "
        "significance bar in 93% of runs, and more strongly than the real data "
        "does — p = 0.002 against 0.006. Within each signup cohort rho comes "
        "back to about 1 (range 0.87 to 1.25, none significant). This reverses "
        "a recommendation an earlier pass of this work made.")

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
    ("An export we can trust  (unlocks Questions 1 and 2 together)", BLUE, True),
    "  Every usage record and support ticket carried with the customer and the "
    "date it actually happened. Today those dates are unrelated to the customer "
    "they are filed against, which is the single reason both questions are "
    "unanswerable.",
    "  One definition of churn, applied everywhere, with a date on it. The three "
    "we currently have agree on about one account in five.",
    "  This is the request to push. Nothing else on this page matters without it, "
    "and re-running our analysis afterwards is one command.",
    "",
    ("Then, and only then, the business timeline  (Question 1)", INK, True),
    "  Price changes, release dates, outages, competitor moves. This is the right "
    "request, but it is worth nothing against the current export — we would be "
    "matching real events to random dates.",
    "",
    ("To learn which actions work  (Question 3)", GREEN, True),
    "  Start logging every call, discount and campaign with a date and an "
    "account attached. Then run one deliberate trial where half the at-risk "
    "customers are contacted and half are not.",
    "  Worth setting expectations on what a trial of our size can settle: "
    "roughly halving churn would show up in about 15 months, while a 5-point "
    "improvement would take years.",
], size=14, space=6)
footnote(s, "None of these is a modelling problem. All three are data "
            "collection, none of them is expensive, and none asks for a "
            "decision on tooling yet.", INK)
note(s, "Close on the ask rather than on the negative results.\n\n"
        "The ordering is the message. The obvious request after slide 2 is "
        "'give us the 2024 event timeline', and it is the wrong one to lead "
        "with: joining real business events onto timestamps that were assigned "
        "at random produces confident nonsense. Fix the export first.\n\n"
        "Request three is independent of the other two and can start on Monday. "
        "It costs a field in the CRM.")

prs.save(OUT)
print(f"{OUT.name}  ({OUT.stat().st_size/1e6:.2f} MB, {len(prs.slides._sldIdLst)} slides)")
