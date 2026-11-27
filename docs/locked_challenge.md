# How surprising is the locked result? A challenge on development only, 27 Nov 2026 (plan written before the run)

Question. The locked eight gave AUROC 0.645 and met 1 of 4 usefulness targets, against 0.762 and 4 of 4 at the point
on development. Is that what eight people can give by chance, or does it need another cause (a harder cohort,
selection on development)?

What is not done. The locked eight are not rescored and their outcomes are not read. No model, cut-off or rule
changes. Only the locked summary numbers already in the report are used as reference values.

Method (`scripts/locked_challenge.py`, output `outputs/locked_challenge/locked_challenge.json`). From the 23
development people, draw 8 without replacement, 5,000 times (seed 20260927). On each draw's committed served
answers (outputs/evaluation_v2/dev_answers.csv, before rule R22, as the locked run was) compute AUROC on the
answered meals and the four usefulness numbers with the declared costs and targets from configs/default.yaml.
Report the share of draws that (a) meet at most 1 of 4 targets, (b) have AUROC at or below 0.645, (c) clear at
most 6.5% of meals, (d) have a cost reduction at or below -5.7%; and the same four shares among the draws whose
spike rate is at least 40% (the locked cohort's was 42.0%).

Predictions, fixed now. Development answers are optimistic (cut-offs and intercept choices were tuned on these
people), so chance alone should rarely look like the locked result: (a) under 15% of draws, (b) under 10%,
(c) under 5%, (d) under 15%. Among harder draws the shares should be higher. If the predictions hold, sample size
alone does not explain the locked failure; if they fail, the locked result is within what eight development
people give and the development estimate was never as firm as its point values.

Limits. Subsets of the people the system was tuned on can only show sampling noise, not shift; the draws overlap,
so the shares are descriptive, not a test.
