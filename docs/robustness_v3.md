# The course's remaining tests, and a more robust photo branch (v3 candidate); registered 26 Nov 2026, before any run

The author's decision after three rubric cold grades and a review of every lecture and lab: run the professor's
tests that the report lists as gaps, and fix what they reveal, without touching anything frozen. Written by
the assistant at her instruction. Development people only: the locked eight are spent (Rule Zero) and are not
read. v1 stays served whatever these runs show. A new model (v3) can only be a development-scored candidate,
reported as such, for the same reason v2 is.

Common harness (as S1/S2 in docs/final_switch.md): v1's 15 trend and profile inputs plus 4 photo inputs, the
logistic fusion with C fixed at 0.03, refitted per outer fold on the other folds' meals, scored out of fold, no
intercept; development meals that have a photo. The v1 photo inputs come from outputs/photo_cv/photo_oof.csv.
Photo work uses the frozen ConvNeXt-Tiny backbone's cached summaries (one pass per photo; the pretrained
weights never change); the last stage is not retrained here.

| | test (lecture) | what | prediction fixed now |
|---|---|---|---|
| R1 | LR range test (week 3) | photo heads on frozen summaries, and SensorCNN: learning rate raised from 1e-6 to 1 over 200 steps, smoothed loss; suggested rate = the rate at the minimum divided by 10 | the used rates (1e-3 for both) lie within 10x of the suggestion |
| R2 | overfit one batch (week 11, case c1) | the same two models on 32 training items, 300 steps; final training loss and gradient norm | both fit the batch (loss below 10% of its start); no vanishing or exploding gradient |
| R3 | label audit of the worst meals (week 11, case c3) | the 20 development meals with the highest log-loss under v1's served risk against all development meals: share with a rise within 5 mg/dL of the 50 mg/dL cut, post-meal coverage under 90% or a gap over 15 min, carbs eaten below carbs served, baseline at or above 140 mg/dL | the worst 20 are enriched for boundary rises and partial eating (at least twice the base rate on one of the four) |
| R4 | classical photo baseline (weeks 6-7, the five baselines) | a 24-bin colour histogram (HSV) + ridge, out of fold on the same folds; zero-shot (CLIP) is not possible offline here and is recorded as not run | the classical baseline is worse than frozen summaries + ridge on carb MAE |
| R5 | regularised photo heads, v3 (weeks 3 and 5; the 0.35 train-validation gap) | heads on frozen summaries; weight decay {1e-4, 1e-2, 1e-1} x dropout {0.2, 0.5} x augmented views {0, 2}, chosen per outer fold on the 4 inner folds by validation carb MAE; out-of-fold carb estimates | v3's train-validation gap is smaller than v1's 0.35; carb MAE within 2 g of v1's photo estimate; harness AUROC v3 minus v1 within ±0.02 (paired person bootstrap) |
| R6 | the repeated-meal fix (week 11, case c2; D8) | R5's chosen recipe retrained per outer fold without any training meal whose exact macro profile appears among the held-out people; then the harness with the same meals removed from the fusion's training side; photo gain = with photo minus without | the photo gain stays near zero (within ±0.02), as D8 found for v1: the photo does not read carbs beyond meal recognition |
| R7 | seed spread against the claimed gain (week 4) | R5 with 5 seeds: harness AUROC range against the photo's gain over the no-photo harness | the seed range is at least half the photo's gain |
| R8 | re-weighted test for a hidden subgroup (week 11, case c4) | v1's served development answers re-weighted to the locked eight's mix of sex and glycaemic status; AUROC and spikes among clears | re-weighting alone does not explain the 0.762 to 0.645 drop (re-weighted AUROC above 0.72) |

Decision rule for v3, fixed now: v3 is reported as a candidate whatever the result. It would become the
recommended next product only if R5 meets all three predictions and R6 shows a photo gain whose 95% interval
excludes zero; even then it is not served without a new clean test. Outputs: outputs/robustness_v3/.
