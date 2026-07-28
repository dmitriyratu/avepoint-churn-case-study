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
_run(p, "The short version: the model does not beat a coin flip. "
        "The analysis still gives three clear actions.", 19, bold=True, color=BLUE)
note(s, "Thirty to forty minutes, five parts, following the brief. I say the "
        "headline up front because the value of this work is in proving the "
        "negative properly and in what we do anyway.")

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
])
footnote(s, "I found this by counting, not by modelling. No algorithm recovers a "
            "fact the source data never recorded the same way twice.")
note(s, "Lead with the target problem. It is the highest-value finding and it "
        "shows the instinct to check the label before fitting anything.\n\n"
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
    ("The data has real quality problems. I report them rather than quietly "
     "patching them.", RED, True),
    "  19,128 of the 24,979 usage records are dated before the subscription they "
    "belong to had even started.",
    "  1,077 of the 2,000 support tickets are dated before the customer signed up.",
    "  Satisfaction is documented as a 1 to 5 score, but only ever takes the "
    "values 3, 4 and 5.",
    "  There are three ways to say a customer churned. All three agree for only "
    "20% of customers.",
    "",
    ("There is very little to learn from.", INK, True),
    "  We have 54 customers who left and 73 features. That is well under one "
    "example per feature, so every result comes with a wide range.",
    "",
    ("Assumptions I am making", INK, True),
    "  A 90 day window and a 30 June 2024 cutoff. I picked both before testing "
    "anything, so they are not chosen to flatter the result.",
    "  Customers still active when the data ends are treated as unknown, not as "
    "successes.",
    "  This is a generated dataset. It may hold no real signal at all, so the "
    "work is built to tell that apart from a modelling mistake.",
])

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
bullets(s, [
    ("Four families of feature", INK, True),
    "  Subscription: how many they have had, what they pay, how steady that "
    "payment is, upgrades, downgrades, and how long they have been a customer.",
    "  Usage: how much they used the product in the last 30, 60, 90 and 180 days. "
    "Whether that is speeding up or slowing down. How long the quiet gaps are.",
    "  Support: how many tickets, how fast we answered, how long we took to fix "
    "things, their satisfaction scores, and how often things were escalated.",
    "  Ratios: usage and tickets per seat, so a 10 seat customer and a 500 seat "
    "customer can be compared fairly.",
    "",
    ("The important part is that features can only see the past.", BLUE, True),
    "  Every calculation takes a date and only looks before it. Training passes "
    "30 June 2024 and production passes today. It is the same code both times, so "
    "the two cannot drift apart.",
    "  A ticket opened in June and closed in July still carries a fix time that "
    "nobody knew in June. Those fields are blanked. An automatic check caught 5 "
    "such tickets that I had missed by reading the code.",
    "",
    ("Being honest about what worked: the 21 extra features I engineered made the "
     "score slightly worse, not better. I kept them anyway, because swapping to "
     "the smaller set just because it scored higher is the exact mistake I warn "
     "about later.", RED),
])

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
    ("Best single model, taken at face value:  0.583, with a range of 0.37 to 0.75",
     MUTED),
    ("The honest score:  0.534, give or take 0.016", INK, True),
    ("Guessing:  0.500", MUTED),
    "",
    ("The gap between those two numbers is the point.", RED, True),
    "  Picking the best of ten models is itself a form of fitting. Once you "
    "measure that properly, 0.049 of the score disappears. That is most of what "
    "looked like signal.",
    "  Across 25 test rounds, eight of the ten models won at least once. A "
    "genuinely better model wins nearly every time. Eight different winners is "
    "what noise looks like.",
    "",
    ("Why the score is low. I tested each possible cause rather than guessing.",
     INK, True),
    "  Not enough data. This is the main one. Adding rows still improves the "
    "score steadily, so we have not hit the ceiling of the method.",
    "  Target risk, which I cannot put a number on. The target comes from the "
    "churn event log, and the other two records of leaving are unrelated to it. "
    "At least one of the three is noise and nothing says which.",
    "  A natural limit in the data. Two customers who look almost identical still "
    "end up differently 41% of the time. Two random customers differ 42% of the "
    "time. There is almost nothing to separate them on.",
    "  It is not leakage, not the features, and not tuning. I checked all three.",
], size=14)
footnote(s, "I confirmed the pipeline works by planting targets of known "
            "strength. A strong one scores 0.965. A deliberately weak one scores "
            "0.584. A meaningless one scores 0.494. The method is fine. The data "
            "does not support more.", INK)

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
    "",
    ("Data leakage", RED, True),
    "  I do not rely on reading the code. A set of automatic checks runs before "
    "any result is reported, and it blocks the result if anything fails.",
    "  The checks look for dates after the cutoff, for banned columns by name, "
    "and for any single feature that predicts too well.",
    "  Columns that only exist after a customer leaves are worth 0.37 of extra "
    "score. They push the model to 0.79, which looks like a good model rather "
    "than a broken one. That is exactly why they are dangerous.",
    "",
    ("Interpretability, and how it can mislead you", BLUE, True),
    "  SHAP gives a clean, confident chart of what drives the prediction. I "
    "checked it against shuffled data, where by definition there is nothing to "
    "find. The chart looked just as convincing.",
    "  The top feature beats random labels only 3 times in 4. Twelve different "
    "features take first place across 25 reruns. These tools never tell you when "
    "they have nothing to say, so I pair every one with a random baseline.",
])

# ============================================================ 10 · INSIGHT 1
s = add("Part 4 · Recommendation 1", "Find out what changed during 2024", BLUE)
picture(s, "12_calendar_hazard.png", height=Inches(3.7), top=Inches(1.8))
bullets(s, [
    ("Each month, more of our customers leave than the month before. In 2023 "
     "about 5 in every 100 left each month. By December 2024 it was 22 in every "
     "100. The customer base stayed the same size throughout, so this is not a "
     "growth effect. The rise is far too large and too steady to be luck.",
     INK, True),
    ("This also explains the model. The cause is hitting every customer at the "
     "same time, so comparing customers with each other cannot find it.", BLUE),
], top=Inches(5.65), size=13)
footnote(s, "What to do: join pricing changes, release dates and competitor "
            "events onto this timeline. The data we have holds nothing that "
            "changes over time, so this is a clear and answerable request.")

# ============================================================ 11 · INSIGHT 2
s = add("Part 4 · Recommendation 2", "Do not build the onboarding programme", RED)
picture(s, "12_cohort_gradient.png", height=Inches(3.5), top=Inches(1.8))
bullets(s, [
    ("At first it looks like new customers are fragile and risk drops with age. "
     "That is the standard reason to invest in onboarding, and the evidence for "
     "it looks very strong.", INK, True),
    ("It is not real. Customers who joined recently leave faster at every age. "
     "They are also the only ones young enough to appear in the early weeks. "
     "Mixing the two makes age look like the cause. Within a single joining "
     "month, a customer at day 300 is just as likely to leave as one at day 10.",
     RED, True),
], top=Inches(5.45), size=13)

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
    ("Recommendation 1 is not an A/B test.", INK, True),
    "  It is a data job. Add the sources that change "
    "over time, then rerun this pipeline and compare "
    "against the score recorded today.",
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
    ("I would hand them the leakage work, because that is where the judgement "
     "that transfers to other projects lives.", INK, True),
    "",
    ("1. Start with the target, not the model.", INK, True),
    "  My first version modelled a flag that turns out to be unrelated to the "
    "event log. Counting rows found it. Then check your own check: I first "
    "called 38% agreement ‘worse than a coin flip’, and it is not — for two "
    "columns this different in size, chance agreement is 39%, not 50%.",
    ("2. Ask of every column: would I really have this on the day I predict?",
     INK, True),
    "  Then write that check down so a machine runs it. Careful reading caught "
    "the obvious problem. The automatic check caught the one I missed.",
    ("3. A high score is a question, not an answer.", INK, True),
    "  The first time they see a great number, it should start a hunt for "
    "leakage, not a celebration.",
    ("4. Find the floor before you celebrate.", INK, True),
    "  Always run the model that uses no features first.",
    ("5. Report the range, not the single number.", INK, True),
    "  A range that includes guessing tells you something the average hides.",
    ("6. Choosing a model is part of fitting it.", BLUE, True),
    "  In this project, the gap between the best model and the honest score was "
    "the entire apparent signal. This is the lesson I would spend the most time on.",
], size=14)
footnote(s, "I would teach it as an experiment, not a lecture. Have them switch "
            "off the banned column list and watch the score jump from 0.58 to "
            "0.79 on its own.")

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
            "these results it is not. Fix the target and the timestamps first, "
            "then measure again. The pipeline takes a date, so that is one "
            "command.", INK)

prs.save(OUT)
print(f"{OUT.name}  ({OUT.stat().st_size/1e6:.2f} MB, {len(prs.slides._sldIdLst)} slides)")
