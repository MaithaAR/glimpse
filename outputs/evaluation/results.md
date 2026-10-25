## Cross-validation (main estimate)

| Model | AUROC (95% range) | within folds | AUPRC | Brier |
|---|---|---|---|---|
| Base rate (always the same guess) | 0.403 (0.309-0.518) | 0.500 | 0.285 | 0.231 |
| Profile only (logistic regression) | 0.672 (0.567-0.764) | 0.676 | 0.487 | 0.204 |
| Profile + trend (logistic regression) | 0.686 (0.604-0.772) | 0.699 | 0.547 | 0.199 |
| Profile + trend (gradient boosting) | 0.689 (0.610-0.778) | 0.714 | 0.531 | 0.204 |
| Profile + SensorCNN (from scratch) | 0.689 (0.603-0.773) | 0.702 | 0.551 | 0.201 |
| + photo carbs (MAIN) | 0.724 (0.659-0.796) | 0.734 | 0.585 | 0.191 |
| + logged carbs (reference) | 0.717 (0.648-0.796) | 0.729 | 0.587 | 0.192 |
| + shuffled photo (control) | 0.679 (0.590-0.766) | 0.674 | 0.502 | 0.204 |
| + ridge photo (post-hoc) | 0.703 (0.630-0.783) | 0.716 | 0.565 | 0.196 |
| + photo grams only, no bins (post-hoc) | 0.704 (0.635-0.782) | 0.712 | 0.584 | 0.195 |

**Paired differences (95% range by person):**

- photo minus profile + trend: +0.037 (+0.003 to +0.073)
- logged_carbs minus profile + trend: +0.031 (+0.003 to +0.063)
- photo_shuffled minus profile + trend: -0.008 (-0.046 to +0.025)
- ridge_photo minus profile + trend: +0.017 (-0.011 to +0.044)
- photo_grams_only minus profile + trend: +0.018 (-0.007 to +0.042)
- carb ranges added (main minus grams only): +0.020 (+0.005 to +0.035)
- RQ1 photo minus logged carbs: +0.007 (-0.020 to +0.032) -> non-inferior
- RQ2 SensorCNN minus profile + trend: +0.002 (-0.016 to +0.022) -> tie or worse

Base spike rate 33% (481 meals, 23 people).

## Targets status

| Target | Estimate | 95% range | Status |
|---|---|---|---|
| Photo adds to spike prediction (main result) | +0.037 | +0.003 to +0.073 | met |
| RQ1: photo not worse than logged carbs by > margin | +0.007 | -0.020 to +0.032 | met |
| RQ2: SensorCNN beats hand-made trends | +0.002 | -0.016 to +0.022 | at risk |
| Calibration: ECE <= 0.05 | +0.052 | +0.052 to +0.125 | at risk |
| Cost reduction >= 10% (CV, main cut-offs) | +0.090 | -0.045 to +0.225 | at risk |
| Meals cleared >= 20% (CV, main cut-offs) | +0.143 | +0.076 to +0.225 | at risk |
| Spikes among cleared <= 10% (CV, main cut-offs) | +0.116 | +0.030 to +0.208 | at risk |
| Can't judge <= 20% (CV, main cut-offs) | +0.106 | +0.068 to +0.146 | met |
| Response time p95 <= 300 ms (just demo) | 282 ms | - | met |

## Fixed test (sanity check): 5 people, 96 meals, spike rate 29%

| Model (fitted on the 28 training people) | AUROC (95% range) | AUPRC | Brier | ECE |
|---|---|---|---|---|
| Base rate (always the same guess) | 0.500 (0.500-0.500) | 0.292 | 0.236 | 0.171 |
| Profile only (logistic regression) | 0.765 (0.571-0.818) | 0.606 | 0.183 | 0.126 |
| Profile + trend (logistic regression) | 0.695 (0.602-0.816) | 0.500 | 0.192 | 0.085 |
| Profile + trend (gradient boosting) | 0.632 (0.512-0.744) | 0.419 | 0.203 | 0.088 |
| Profile + SensorCNN (from scratch) | 0.674 (0.563-0.815) | 0.466 | 0.208 | 0.173 |
| + photo carbs (MAIN) | 0.742 (0.670-0.900) | 0.618 | 0.182 | 0.129 |

| Cut-offs | low risk below | caution from | cleared | spikes among cleared | can't judge | cost saving |
|---|---|---|---|---|---|---|
| main (cost-only) | 0.143 | 0.200 | 3% | 0% | 9% | +7% |
| pre-registered (validation) | 0.310 | 0.310 | 42% | 15% | 0% | +6% |
| CV without test people | 0.180 | 0.180 | 9% | 11% | 0% | +4% |

## CV without the 5 test people (18 people, 385 meals; added after the results)

| profile_lr | no_meal | sensor_cnn | photo | logged_carbs |
|---|---|---|---|---|
| 0.659 | 0.686 | 0.691 | 0.719 | 0.715 |

Photo gain over profile + trend: +0.033 (95% range -0.007 to +0.077).

Main model minus profile-only on the test (paired, by person): -0.023 (95% range -0.114 to +0.159).

## Calibration of the main model

Equal-width bins (15 over 0-1); empty bins omitted. ECE = meal-weighted mean gap between the two columns.
The bootstrap range of ECE is biased upward (resampled data always look less calibrated).

**Cross-validation (481 meals): ECE 0.052**

| predicted (mean in bin) | observed spike rate | meals |
|---|---|---|
| 0.04 | 0.17 | 12 |
| 0.10 | 0.10 | 50 |
| 0.17 | 0.14 | 58 |
| 0.23 | 0.21 | 61 |
| 0.30 | 0.30 | 66 |
| 0.37 | 0.28 | 58 |
| 0.43 | 0.48 | 52 |
| 0.50 | 0.38 | 34 |
| 0.56 | 0.52 | 25 |
| 0.64 | 0.52 | 21 |
| 0.70 | 0.61 | 28 |
| 0.75 | 0.93 | 14 |
| 0.84 | 1.00 | 2 |

**Fixed test (96 meals): ECE 0.129**

| predicted (mean in bin) | observed spike rate | meals |
|---|---|---|
| 0.11 | 0.00 | 3 |
| 0.17 | 0.11 | 9 |
| 0.23 | 0.24 | 21 |
| 0.29 | 0.00 | 7 |
| 0.36 | 0.08 | 12 |
| 0.44 | 0.33 | 15 |
| 0.49 | 0.55 | 11 |
| 0.56 | 0.25 | 8 |
| 0.64 | 0.75 | 4 |
| 0.69 | 0.83 | 6 |

**Where the miscalibration comes from (no test labels used; added after the results)**

| Group | People | Mean predicted risk | Observed spike rate |
|---|---|---|---|
| CV without test people: rare spikers (below median) | 9 | 0.30 | 0.16 |
| CV without test people: frequent spikers | 9 | 0.42 | 0.54 |
| Validation people, 28-person model | 4 | 0.31 | 0.19 |

Correlation between a person's spike rate and (predicted - observed): -0.77.

