# Evaluation protocol (fixed before any model was trained)

Written on the date of the commit that adds it, before training, so that no choice below can depend on results.
Settings referred to here live in `configs/default.yaml` (`splits`, `cv`, `baselines`, `targets`).

## Which data makes which decision
| Decision | Data used | Never uses |
|---|---|---|
| Scaling statistics, missing-value fills | training people of that run (refitted inside every fold) | validation, test, locked |
| Number of training epochs (CNN) | 4 inner person folds inside the training people (median best epoch) | validation, test, locked |
| Logistic-regression strength (C) | the same 4 inner folds | validation, test, locked |
| Whether the photo cross-validation runs | logged-carbs check on the out-of-fold predictions (rule below) | validation, test, locked |
| Decision thresholds (low risk / caution / can't judge) | the 4 validation people, with the 28-person model | test, locked |
| Threshold stability check (reported, not used to choose) | pooled cross-validated predictions of the 23 development target people | locked |
| **Main result** | out-of-fold predictions over the 23 development target people (5 outer folds, 3 seeds) | locked |
| Held-out confirmation | the fixed test people (outer fold 1), scored once with the 28-person model | — |
| Final check (final submission only) | the 8 locked people, once | — |

## Metrics
- **Primary:** AUROC on the target cohort (healthy + prediabetes), cross-validated, with a 95% range from re-sampling **people**.
- **Also:** AUPRC (compared with the base rate), Brier score, calibration (reliability diagram and ECE, equal-width bins with counts).
- **Business:** cost reduction versus the best simple policy, share of meals cleared as low risk and spikes among them, abstention rate, latency.
- Type 2 diabetes people help training but are never scored.

## Comparison models (same folds, same normaliser, one setting tuned each)
| Model | Inputs | Tuned on inner folds |
|---|---|---|
| Prevalence | none (training spike rate) | — |
| Profile only (logistic regression) | the 8 profile inputs | C from `baselines.logreg_c_grid` |
| Engineered trends (logistic regression) | last glucose, slope (30 and 60 min), mean, spread, change over 2 h, mean heart rate + the 8 profile inputs | C from `baselines.logreg_c_grid` |
| **SensorCNN** (from scratch) | 3 × 24 window + 8 profile inputs | number of epochs |

## Decision rules (from the proposal)
- **RQ2:** the CNN beats engineered trends only if the lower end of the 95% range of the AUROC gain is above `targets.rq2_min_lower_bound` (0). Otherwise it is reported as a tie or a loss.
- **Calibration:** ECE ≤ `targets.ece_max` (0.05).
- **Status labels:** *met* only if the whole 95% range meets the target; *on track* if only the estimate does; otherwise *at risk*.

## Meal information (added in Step 4c, before running it)
- **C grid widened** to `baselines.logreg_c_grid` = 0.001 ... 10: the old minimum (0.01) was chosen in 9 of 10 tunings,
  so the best setting may have been below the grid. Heavy shrinkage could hide a useful carbs input.
- **Logged-carbs check (4c-1).** The same logistic regression (trend + profile inputs) with and without log(1 + logged
  carbs), same folds and inner folds. Logged carbs are the upper bound for photo-estimated carbs.
- **Rule:** skip the photo cross-validation only if the whole 95% person-bootstrap range of the AUROC gain is below
  `fusion.carbs_check_min_gain` (0.02). Any other result, including an unclear one, means proceed.
- **One main comparison** for the photo: AUROC gain of photo carbs over no meal information (paired person bootstrap).
  Logged carbs are a reference arm, not a competitor. No extra arms (interactions, other learners) are added after
  seeing results.
- **Known limit:** the bootstrap resamples people but not the inner-fold tuning, so the range is slightly too narrow.

## Photo cross-validation and fusion (fixed after 4c-1, before 4c-2 ran)
- **Photo estimates for the fusion test side:** out-of-fold, the 4b recipe re-chosen inside each outer fold's training
  people (same inner folds), one seed.
- **Photo estimates for the fusion training side:** nested. The chosen recipe is refitted per inner fold with fixed
  epochs (no early stopping) and used on that inner fold's held-out people, so training and test estimates are made
  the same way. They come from slightly fewer people, so they are a little noisier.
- **Carb baselines:** training median; frozen ConvNeXt + ridge (alpha from the inner folds). Reported on the scored
  meals that have a photo.
- **Fusion arms (4c-3):** no meal; + photo (log of estimated grams + bin probabilities); + logged carbs (reference);
  + shuffled photo estimates (control: photo estimates permuted among the held-out meals of each fold).
- **Primary result:** pooled AUROC gain of the photo arm over no meal, paired person bootstrap (23 people).
  Within-fold AUROC is reported as a check.
- **RQ1** (photo vs logged carbs, margin 0.03) is kept as written in the proposal. With 23 people the 95% range is
  expected to be about 0.06 wide, so "inconclusive" is a likely, honest outcome.
- Scored meals without a photo get the training median estimate and are reported with and without.

## Added after 4c-2 results, before 4c-3 ran (post-hoc, reported as secondary)
- 4c-2 showed frozen ConvNeXt + ridge with a lower carb error than the fine-tuned model (20.4 vs 22.5 g; paired person
  bootstrap of the difference 0.7 to 3.4 g), and ridge alpha at the top of its grid in every fold.
- **Ridge photo arm** (log of the ridge estimate) is added to the fusion with nested/OOF estimates made like the
  fine-tuned ones, alpha grid widened to `fusion.ridge_alpha_grid_posthoc`. It is labelled post-hoc and is never
  the main result, whatever it scores. The primary arm stays the fine-tuned photo model, as written above.
- **Shuffled control:** the fitted photo model scores the test meals with their photo inputs swapped among the
  test meals of the same fold.
- **Diagnostic:** calibration slope of the photo estimates (log scale), out-of-fold vs nested.

## Amendment to the decision cut-offs (after Step 5a, before the test was opened)
- Step 5a: cut-offs chosen on the 4 validation people (spike rate 19%) were 0.31 / 0.31. The pre-registered stability
  check failed: on the pooled cross-validation (spike rate 33%) they gave no cost saving (+2%).
- **Main cut-offs** now follow from the pre-registered costs alone, assuming honest probabilities: low risk below
  0.143, caution from 0.20, can't judge in between. They use no data, so no test label can influence them.
  Calibration of the cross-validated main model on folds 2-5 (test people excluded) supported this assumption.
- **Secondary rows (reported, not chosen between afterwards):** the pre-registered 0.31; cut-offs chosen on the
  cross-validation without the test people.
- **Sensitivity:** the main rule recomputed for every pre-registered cost setting (`costs.sweep_*`).
- **Success criteria** are unchanged (`targets.usefulness`), each with a 95% range by person.
- **Role of the test:** the 5 test people are also inside the cross-validation, so the test is a sanity check of the
  28-person pipeline; the cross-validation is the main estimate. Usefulness is also reported on the
  cross-validation with the main cut-offs (they use no data, so this is fair).
- The amendment file (`checkpoints/thresholds_amended.json`, with its hash) is committed before `glimpse evaluate`
  scores the test; the command refuses to run otherwise.

## Added after Step 5 (post-hoc, secondary)
- **Carb-bin head check:** a fusion arm with the fine-tuned photo estimate in grams only (no bin probabilities), same
  folds and estimates. The gain of the main arm over it shows what the carb-bin head adds. Never the main result.
