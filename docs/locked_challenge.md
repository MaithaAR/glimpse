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

## Results (27 Nov, outputs/locked_challenge/locked_challenge.json)

| share of draws | all 5,000 | the 749 with spike rate of 40% or more | predicted |
|---|---|---|---|
| (a) at most 1 of 4 targets met | 16.8% | 48.9% | under 15%: not met |
| (b) AUROC at or below 0.645 | 2.8% | 0.8% | under 10%: met; "higher among harder draws": not met |
| (c) cleared at most 6.5% | 1.1% | 7.1% | under 5%: met |
| (d) cost reduction at or below -5.7% | 8.8% | 34.6% | under 15%: met |

Mean targets met per draw: 2.9 of 4 (1.75 among the harder draws). AUROC over draws: median 0.753, 95% range 0.641 to 0.846.

Reading. The usefulness verdict "4 of 4 at the point" on development was never secure at eight people: one draw in
six meets at most one target, and half of the draws with a spike rate like the locked cohort's. The drop in AUROC
is a different matter: sampling eight development people almost never gives 0.645, and harder draws do not rank
worse. So chance and a harder cohort can account for much of the usefulness failure, but not for the loss of
ranking. The locked result stands as the test result.

## Added after the first run (post hoc, same day)

- Base rate alone (same script): re-weighting the 481 development meals to a 42% spike rate, with every decision
  unchanged, cuts the cost reduction from 14.4% to 2.8%, the cleared share from 22.7% to 20.3%, and raises misses
  among clears from 7.3% to 10.5%. It does not explain the locked cohort clearing only 6.5% of meals.
- Is there a cut-off rule that survives a harder cohort? (`scripts/locked_challenge_rules.py`,
  outputs/locked_challenge/rules.json; parameters chosen leave-one-fold-out.) No. The best candidate, clearing each
  person's lowest-risk 10% of meals so far, lowers the share of harder draws meeting at most 1 of 4 targets from
  48.9% to 39.7% but raises it from 16.8% to 19.1% over all draws. The people who spike often spike on about a
  quarter of their own lowest-risk meals, above the 16.7% break-even at 5:1, so the shortfall is in ranking meals
  within such people, which a threshold cannot repair. Nothing here is served or scored on the locked people.

