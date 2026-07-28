"""Build the 15-slide case-study deck, following the assignment brief.

Structure follows the five parts the brief asks for:
  1 Problem framing   2 EDA + feature engineering   3 Modelling
  4 Strategic recommendations                       5 Mentorship + scalability

Layout and the plain-English rule live in deck_style.py, shared with the
executive summary so the two cannot drift apart.
"""
from pathlib import Path

from deck_style import (BLUE, GREEN, H, INK, M, MUTED, RED, W, blank, bullets,
                        footnote, header, new_deck, note, picture, run, tb)
from pptx.util import Inches, Pt

OUT = Path(__file__).with_name("AvePoint_Case_Study.pptx")

prs = new_deck()
_tb, _run = tb, run


def add(kicker, title, colour=INK):
    s = blank(prs)
    header(s, kicker, title, colour)
    return s


# ============================================================ 1 · TITLE
s = blank(prs)
tf = _tb(s, M, Inches(2.4), W - 2 * M, Inches(2.8))
_run(tf.paragraphs[0], "Why customers leave, and what to do about it", 40, bold=True)
p = tf.add_paragraph(); p.space_before = Pt(16)
_run(p, "Senior ML Engineer case study. RavenStack subscription data: "
        "500 customers across 5 tables.", 18, color=MUTED)
p = tf.add_paragraph(); p.space_before = Pt(28)
_run(p, "The short version: no model here beats a coin flip, the one pattern "
        "that looked like a cause comes from how the file was written, and "
        "three actions still follow.", 19, bold=True, color=BLUE)
note(s, "Thirty to forty minutes, five parts, following the brief. I say the "
        "headline up front because the value of this work is in proving the "
        "negatives properly and in what we do anyway.\n\n"
        "The second clause is the part to sit on if there is only time for one "
        "thing. The strongest result I had — churn accelerating through 2024, "
        "p = 2e-16 — is reproduced exactly by a simulation containing nothing "
        "but a random number generator. Slide 10 is that test. Finding it is "
        "the piece of work I would most want to be judged on.")

# ============================================================ 2 · PROBLEM
s = add("Part 1 · Problem framing", "The business problem, and a target I had to rebuild")
bullets(s, [
    ("This work supports two decisions. Who should the success team call this "
     "quarter? What should the product team fix?", INK, True),
    "By the churn event log, seven out of ten customers leave in the end. The "
    "typical customer pays $931 a month and is worth about $7,300 in total. "
    "Churn is the largest cost we can actually control.",
    "",
    ("The churn flag in the data cannot be used.", RED, True),
    "  It has no date on it. So we cannot tell whether a customer had already "
    "left on any given day.",
    "  It agrees with the churn event log for 188 of the 500 customers. Two "
    "unrelated columns would agree for about 193. So the flag is not a rough "
    "version of the truth. It tells us nothing at all about who left.",
    "  The other two records do not line up either. A churn date and a "
    "subscription ending fall on the same day 2% of the time. The usual gap "
    "is two months.",
    "",
    ("So I rebuilt the target as a question with a date in it:", INK, True),
    "  On 30 June 2024, using only what we knew that day, will this customer "
    "leave within the next 90 days?",
    "  That gives 177 customers who could still leave. 54 of them did, which is 31%.",
    "  Features use only data dated before that day. Anything still unresolved on "
    "that day is blanked out.",
], size=14, space=6)
footnote(s, "I found this by counting, not by modelling. No algorithm recovers a "
            "fact the source data never recorded the same way twice.")
note(s, "Lead with the target problem. Along with slide 10 it is the most "
        "valuable thing here, and it shows the instinct to check the label "
        "before fitting anything.\n\n"
        "If challenged on the 188: the point is not that agreement is low, it is "
        "that it is exactly what chance produces. The flag fires for 22% of "
        "customers and the event log for 70%, so two unrelated columns agree "
        "38.6% of the time. We observe 37.6%. Cohen's kappa is -0.02, where 0 "
        "means unrelated, and chi-square gives p = 0.56.\n\n"
        "Pre-empt the obvious follow-up: 'so invert the flag and you get 62%'. "
        "No. Inverted it scores 62.4% against 61.4% expected by chance, kappa "
        "+0.03. There is no information in it either way round.\n\n"
        "The subscription line is the strongest part because it never touches "
        "the flag: 386 churn events can be compared to a subscription ending, "
        "and only 6 land on the same day. Median gap 62 days. All three "
        "recordings of 'this customer left' are mutually unrelated.\n\n"
        "Computed in src/audit.py by label_source_agreement and "
        "churn_date_coherence, and shown in notebooks 01 and 10.")

# ============================================================ 3 · METRICS
s = add("Part 1 · Problem framing", "How we would judge success")
bullets(s, [
    ("For the model", INK, True),
    "  ROC-AUC measures how well the model sorts customers by risk. 0.5 means it "
    "is guessing. I always report a range around it, never a single number.",
    "  I also report precision and recall, because they say what the CS team "
    "actually experiences when they work the list.",
    "  The headline number comes from nested cross-validation. That is the only "
    "version that accounts for the fact that I also chose the model.",
    "",
    ("For the business", INK, True),
    "  Retention 180 days after any change we make.",
    "  Money kept, so that a saved customer who needed a large discount is not "
    "counted as a win.",
    "  Value of the campaign compared with calling everyone. Zero is the wrong "
    "thing to compare against.",
    "",
    ("Limits I set before starting", BLUE, True),
    "  The model has to beat guessing at the bottom of its range, not just on "
    "average. It also has to beat simply calling every customer.",
    "  Scores order a call list. They never trigger an action on their own.",
])

# ============================================================ 4 · RISKS
s = add("Part 1 · Problem framing", "Risks and assumptions, said out loud")
bullets(s, [
    ("Two of the five tables are not joined to their customers in time.",
     RED, True),
    "  Usage and support ticket dates are spread at random across the whole two "
    "years. They line up with their own customer's signup date at r = 0.002 and "
    "r = 0.014, where 0 means no connection at all.",
    "  That is what produces the numbers people quote: 19,128 of 24,979 usage "
    "rows are dated before the subscription they belong to, and 1,077 of 2,000 "
    "tickets before the customer signed up.",
    "  Nothing links inside those tables either. Ticket priority does not predict "
    "how fast we answered. Plan tier does not predict how much people used.",
    "  Satisfaction is documented as a 1 to 5 score, but only ever takes the "
    "values 3, 4 and 5.",
    "  There are three ways to say a customer churned. All three agree for only "
    "20% of customers.",
    "",
    ("Subscriptions is the one table built with rules, and the only one I lean "
     "on. Price follows the plan and the seat count, ARR is exactly 12 times "
     "MRR, and no subscription starts before its customer did.", INK, True),
    "",
    ("There is very little to learn from: 54 customers who left, against 73 "
     "features. Well under one example per feature, so every result carries a "
     "wide range.", INK, True),
    "",
    ("Assumptions I am making", INK, True),
    "  A 90 day window and a 30 June 2024 cutoff. I picked both before testing "
    "anything, so they are not chosen to flatter the result.",
    "  Customers still active when the data ends are treated as unknown, not as "
    "successes.",
    "  This is a generated dataset, so every finding is tested against a null "
    "that imitates the generator — including the findings I liked. Slide 10 is "
    "that test, and it is the one that changed the answer.",
], size=14, space=5)

# ============================================================ 5 · EDA
s = add("Part 2 · Data exploration", "What the data does and does not contain")
picture(s, "01_churn_by_segment.png", height=Inches(3.4), top=Inches(1.8))
bullets(s, [
    ("No group of customers stands out. I checked industry, country, plan, how "
     "they found us, and whether they started on a trial. None of them separates "
     "the customers who leave from the ones who stay.", INK, True),
    ("I explored the relationship with churn on a held out slice of the data. "
     "That way the data I modelled on did not influence what I chose to look at.",
     MUTED),
], top=Inches(5.3), size=14)

# ============================================================ 6 · FEATURES
s = add("Part 2 · Feature engineering", "73 features, and the reason for each one")
picture(s, "03_point_in_time.png", height=Inches(3.05), top=Inches(1.68))
bullets(s, [
    ("What each family measures.", INK, True),
    "  Subscription: what they pay, how steady it is, upgrades, downgrades, how "
    "long they have been a customer.  Usage: activity over 30 to 180 days, and "
    "whether it is speeding up or slowing down.  Support: ticket load, how fast "
    "we answered, how long fixes took, satisfaction, escalations.  Account and "
    "ratios: industry, country, how they found us, plan, and per-seat versions of "
    "usage and tickets so a 10 seat and a 500 seat customer sit on the same scale.",
    "",
    ("The same code builds these in training and in production.", BLUE, True),
    "  Training passes 30 June 2024, production passes today, so the two cannot "
    "drift apart. The blanking on the right is the part that reading the code "
    "misses: an automatic check found those 5 tickets, I did not.",
    "",
    ("Being honest about what worked: the 21 extra features I engineered made the "
     "score slightly worse, not better. I kept them anyway, because swapping to "
     "the smaller set just because it scored higher is the exact mistake I warn "
     "about later. None of them could have helped, for the reason slide 8 gives.",
     RED),
], top=Inches(4.90), size=11.5, space=3)

# ============================================================ 7 · LADDER
s = add("Part 3 · Modelling", "Choosing an algorithm: make it earn the complexity")
picture(s, "04_model_ladder.png", height=Inches(3.5), top=Inches(1.8))
bullets(s, [
    ("I start with a model that uses no features at all. It has to score 0.5. "
     "Without that floor you cannot tell whether a real model is doing anything. "
     "Then I add complexity one step at a time: a single rule, then logistic "
     "regression, then a random forest, then three gradient boosting models.", INK, True),
    ("Every model is tested on exactly the same splits, so the comparison is "
     "fair. The boosting models get their preferred treatment of missing values "
     "and categories. With only 177 customers, the simple regularised model still "
     "wins.", MUTED),
], top=Inches(5.3), size=13.5)

# ============================================================ 8 · PERFORMANCE
s = add("Part 3 · Modelling", "How it performs, and why the number is low", RED)
bullets(s, [
    ("Best single model, taken at face value:  AUC 0.583, range 0.37 to 0.75",
     MUTED),
    ("The honest score:  AUC 0.534, give or take 0.016", INK, True),
    ("Guessing:  AUC 0.500", MUTED),
    "",
    ("The gap between those two numbers is the point.", RED, True),
    "  Picking the best of ten models is itself a form of fitting. Once you "
    "measure that properly, 0.049 of AUC disappears. That is most of what "
    "looked like signal.",
    "  Across 25 test rounds, eight of the ten models won at least once. A "
    "genuinely better model wins nearly every time. Eight different winners is "
    "what noise looks like.",
    "",
    ("Why the score is low. Each cause is measured, not guessed at.", INK, True),
    "  The inputs hold nothing. Usage and ticket dates are scattered at random "
    "across the two years and match their own customer's signup date at r = 0.002 "
    "and r = 0.014. Inside those tables, priority does not predict how fast we "
    "answered and plan does not predict how much people used.",
    "  The target is a random date. Every churn date is a coin toss between the "
    "day the customer joined and the last day of the file. Slide 10 is that test.",
    "  Subscriptions is the one table with real rules, and none of it relates to "
    "churn.",
    "  Sample size is real but secondary: 54 churners against 73 features.",
    "  It is not leakage, not the feature build, and not tuning. I checked all "
    "three.",
], size=13.5, space=6)
footnote(s, "Scores are ROC-AUC: 0.50 is guessing, 1.00 is perfect. I confirmed "
            "the pipeline works by planting targets of known strength — a strong "
            "one scores AUC 0.965, a deliberately weak one 0.584, a meaningless "
            "one 0.494. The method works. AUC 0.534 is the right answer to the "
            "question this data can be asked.", INK)

# ============================================================ 9 · REQUIRED THREE
s = add("Part 3 · Modelling", "Imbalance, leakage and interpretability")
bullets(s, [
    ("Class imbalance", INK, True),
    "  31% of customers left, so the classes are uneven but not badly so. I "
    "weight the rare class rather than inventing synthetic customers. With only "
    "54 real examples, synthetic ones are copies of copies and they flatter the "
    "score.",
    "  I set the cut-off to favour catching leavers, because missing one costs "
    "far more than a wasted phone call. That gives 75% of leavers caught, and "
    "1 in 3 of the flagged customers actually leaving.",
    "  I report ROC-AUC because its baseline stays 0.50 at any class balance, "
    "which keeps the twelve horizon cells comparable when their churn rates run "
    "11% to 45%. Precision-recall and F1 agree, so the metric is not choosing "
    "the answer.",
    "",
    ("Data leakage", RED, True),
    "  I do not rely on reading the code. A set of automatic checks runs before "
    "any result is reported, and it blocks the result if anything fails.",
    "  The checks look for dates after the cutoff, for banned columns by name, "
    "and for any single feature that predicts too well.",
    "  Columns that only exist after a customer leaves are worth 0.37 of extra "
    "AUC. They push the model to AUC 0.79, which looks like a good model rather "
    "than a broken one. That is exactly why they are dangerous.",
    "",
    ("Interpretability, and how it can mislead you", BLUE, True),
    "  SHAP gives a clean, confident chart of what drives the prediction. I "
    "checked it against shuffled data, where by definition there is nothing to "
    "find. The chart looked just as convincing.",
    "  The top feature beats random labels only 3 times in 4. Twelve different "
    "features take first place across 25 reruns. These tools never tell you when "
    "they have nothing to say, so I pair every one with a random baseline.",
], size=14, space=5)

# ============================================================ 10 · INSIGHT 1
s = add("Part 4 · Recommendation 1",
        "Do not go looking for what changed in 2024", RED)
picture(s, "16_generator_artefact.png", height=Inches(3.05), top=Inches(1.70))
bullets(s, [
    ("Churn looks like it is accelerating. About 5 in every 100 customers left "
     "each month in 2023, rising to 22 in 100 by December 2024 — a 2.8x rise per "
     "year, p = 2e-16. It was by far the strongest number in this study.", INK, True),
    ("It is not a fact about customers. Every churn date in this file is a "
     "random date between the day the customer joined and the last day of the "
     "extract. A random date has nowhere to go but the end of the file, so the "
     "rate climbs on its own, with nothing happening in the business.", RED, True),
    ("Rebuilding the data from that one rule gives the same answer: a 2.78x rise "
     "per year, all 24 months inside the range chance produces, and our result "
     "sitting at the 52nd percentile of pure noise. So there is nothing here to "
     "investigate, and that is the recommendation.", INK, True),
], top=Inches(5.00), size=12.5, space=4)
footnote(s, "The test that mattered was not \"is this bigger than noise\" — it "
            "was, comfortably. It was \"would the file produce this on its own\". "
            "On generated data that is the only null worth testing, and it is "
            "the one I had not run.", INK)
note(s, "This is the slide I would want to be asked about.\n\n"
        "How the artefact works: no churn date can land after 2024-12-31. If "
        "each is drawn uniformly between signup and that boundary, the hazard "
        "is 1/(END - t), which rises without limit as t approaches the end. "
        "Pooled across accounts that signed up over two years, that is "
        "indistinguishable by eye from a business problem.\n\n"
        "The evidence, in order. Rescale each churn date to its position in the "
        "account's own window: uniform, KS p = 0.92 on 600 events, mean position "
        "0.503, and uniform inside every signup quarter. Then the simulation — "
        "keep each account's real signup date and real number of churn events, "
        "redraw only the dates, run the same survival.calendar_hazard used to "
        "produce the original claim, 400 times. Observed rate ratio 1.0893, null "
        "1.0885 with a 95% band of [1.072, 1.106].\n\n"
        "The p-value is the part worth dwelling on. p = 2e-16 felt like the most "
        "secure number I had. The null rule produces a median p of 2.8e-16 — "
        "very slightly more significant than the real data. A p-value tells you "
        "nothing about whether the null it tests is the one that matters.\n\n"
        "If asked what I would have done differently: run this before writing "
        "the recommendation, not after. On any generated or vendor-supplied "
        "extract, simulate the file before trusting a time trend in it.\n\n"
        "src/generator.py, notebook 16.")

# ============================================================ 11 · INSIGHT 2
s = add("Part 4 · Recommendation 2", "Do not build the onboarding programme", RED)
picture(s, "12_cohort_gradient.png", height=Inches(3.0), top=Inches(1.78))
bullets(s, [
    ("New customers look fragile, and risk looks like it falls with age. That is "
     "the usual reason to fund onboarding, and the evidence for it looks strong: "
     "30-day survival drops from 0.98 for the 2023 intake to 0.59 for the 2024 "
     "one, measured at the same age.", INK, True),
    ("It is the same random date. A customer who joined recently has a shorter "
     "window for that date to land in, so a larger share of their draws fall "
     "inside any 90-day question we ask. They look worse at every age without "
     "anything about them being worse.", RED, True),
    ("Simulated data with no tenure effect in it at all clears the significance "
     "bar in 93% of runs, and does so more strongly than the real data does "
     "(p = 0.002 against 0.006). Within one joining month there is nothing left: "
     "a day-300 customer leaves as often as a day-10 one.", INK, True),
], top=Inches(4.95), size=12.5, space=4)
footnote(s, "The recommendation does not change; the reason for it gets one "
            "layer deeper. This is the same finding as slide 10, applied to a "
            "spending decision instead of an investigation.")

# ============================================================ 12 · INSIGHT 3
s = add("Part 4 · Recommendation 3", "Call every at-risk customer, do not rank them", GREEN)
picture(s, "15_breakeven_grid.png", height=Inches(3.3), top=Inches(1.8))
bullets(s, [
    ("A call costs about $150. A customer is worth about $7,300. If the call "
     "works one time in five, it pays for itself on any customer with more than "
     "a 10% chance of leaving. In this group, 31% leave.", INK, True),
    ("So calling everyone already makes money, and no ranking can improve on a "
     "decision that is right for every customer. Using the model adds $600 on "
     "top of $52,400. That is a 1% gain, and even that is generous.", GREEN, True),
], top=Inches(5.3), size=13)
footnote(s, "This is one division. Had I run it first, it would have shown that "
            "a working model was never the thing standing between us and the "
            "decision.")

# ============================================================ 13 · TESTING
s = add("Part 4 · Testing approach", "How I would test this, and what is worth testing")
picture(s, "15_experiment_power.png", height=Inches(3.2), top=Inches(1.9), left=Inches(7.1))
bullets(s, [
    ("Recommendation 1 is not an experiment. It is a "
     "data request.", INK, True),
    "  Nothing in this extract can say why customers "
    "leave, because the timestamps were never recorded "
    "against the customers they belong to. Ask for an "
    "export where they are, then rerun this pipeline "
    "and compare against the score written down today.",
    "",
    ("Recommendation 3 is a proper experiment.", INK, True),
    "  Split the at-risk customers in half at random. "
    "One half gets a call, one half does not. Measure "
    "retention after 180 days, and measure money kept "
    "as well, so a costly save is not mistaken for a win.",
    "",
    ("But be realistic about what we can detect.", RED, True),
    "  At about 21 new customers a month, only a very "
    "large effect can be proved:",
    "  Halving churn takes about 15 months to show.",
    "  A 5 point improvement would take over 10 years.",
], top=Inches(1.88), width=Inches(6.0), size=14)
footnote(s, "Two separate methods agree on this. A standard power calculation "
            "and a simulation both say we cannot detect anything smaller than "
            "about a 15 point change.")

# ============================================================ 14 · MENTORSHIP
s = add("Part 5 · Mentorship", "What I would teach a junior engineer")
bullets(s, [
    ("I would hand them my own worst mistake on this project, because that is "
     "where the judgement that transfers to other work lives.", INK, True),
    "",
    ("1. Be hardest on the finding you like.", RED, True),
    "  I had one positive result and a page of nulls, and I checked the nulls. "
    "The positive one was manufactured by the file. Ask of any finding: what "
    "else could have produced this, and can I generate it from nothing?",
    ("2. A small p-value is not evidence. It is evidence against one null.",
     RED, True),
    "  p = 2e-16 felt unarguable. Random dates gave p = 3e-16. If the null is "
    "the wrong one, no amount of significance rescues it.",
    ("3. Start with the target, not the model.", INK, True),
    "  My first version modelled a flag unrelated to the event log. Counting "
    "rows found it. Then check your own check: I first called 38% agreement "
    "‘worse than a coin flip’, and it is not — for two columns this different "
    "in size, chance agreement is 39%, not 50%.",
    ("4. Ask of every column: would I really have this on the day I predict?",
     INK, True),
    "  Then write the check down so a machine runs it. Reading caught the "
    "obvious problem; the automatic check caught the one I missed.",
    ("5. Find the floor, and report the range.", INK, True),
    "  Always run the model with no features first. A range that includes "
    "guessing says something the average hides.",
    ("6. Choosing a model is part of fitting it.", BLUE, True),
    "  Here the gap between the best model and the honest score was the entire "
    "apparent signal.",
], size=13, space=4)
footnote(s, "I would teach all of it as experiments rather than lectures. Switch "
            "off the banned column list and watch AUC jump from 0.58 to 0.79. "
            "Then simulate a dataset with nothing in it and watch it produce a "
            "publishable-looking trend.")

# ============================================================ 15 · DEPLOY
s = add("Part 5 · Scalability", "How I would run this in production")
tf = _tb(s, M, Inches(1.88), Inches(6.1), Inches(4.6))
for line in [
    "Sources:  billing and CRM, product usage, support desk",
    "        |  runs once a night",
    "Feature build  (THE SAME CODE USED IN TRAINING)",
    "   · every calculation takes a date and looks only before it",
    "   · blanks out anything not yet known on that date",
    "   · the leakage checks run here and stop the job if they fail",
    "        |",
    "Scoring  (once a day is plenty for a 90 day question)",
    "   · writes the customer, the score, and the main reasons",
    "        |",
    "Use  (a call list for the success team)",
    "   · scores also written back to the CRM for context",
    "   · nothing is ever triggered automatically",
]:
    p = tf.paragraphs[0] if line.startswith("Sources") else tf.add_paragraph()
    p.space_after = Pt(3)
    bold = line.endswith("TRAINING)") or line.split("  ")[0] in (
        "Sources:", "Feature build", "Scoring", "Use")
    _run(p, line, 12.5, bold=bold, color=INK if bold else MUTED)

bullets(s, [
    ("What I would watch", INK, True),
    "  Whether the incoming data still looks like the data we trained on. "
    "Checked weekly.",
    "  Whether columns have gone missing or empty. Checked daily, and the job "
    "fails if the shape changes.",
    "  The leakage checks. Run every single time, and they stop the job.",
    "  Whether the scores themselves have drifted. Checked weekly.",
    "  Whether the model is still accurate. Checked every quarter, once we know "
    "who actually left.",
    "",
    ("We only learn if we were right 90 days later. So accuracy checks are "
     "always late. Watching the incoming data is the early warning, and accuracy "
     "confirms it afterwards.", MUTED),
    "",
    ("The part I would argue for hardest is running the leakage checks in "
     "production. The bug in this project was in the pipeline, and pipeline bugs "
     "come back every time someone adds a feature.", BLUE, True),
], top=Inches(1.88), left=Inches(7.1), width=Inches(5.5), size=13)
footnote(s, "This is what I would build if the model were worth shipping. On "
            "these results it is not, and this extract cannot be repaired into "
            "one — the timestamps have to be collected against the right "
            "customers, not corrected. Once they are, the pipeline takes a "
            "date, so re-measuring is one command.", INK)

prs.save(OUT)
print(f"{OUT.name}  ({OUT.stat().st_size/1e6:.2f} MB, {len(prs.slides._sldIdLst)} slides)")
