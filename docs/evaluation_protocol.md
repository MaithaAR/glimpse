# Evaluation protocol (fixed before any model was trained)

Written on the date of the commit that adds it, before training, so that no choice below can depend on results.
Settings referred to here live in `configs/default.yaml` (`splits`, `cv`, `baselines`, `targets`).

## Which data makes which decision
| Decision | Data used | Never uses |
|---|---|---|
| Scaling statistics, missing-value fills | training people of that run (refitted inside every fold) | validation, test, locked |
| Number of training epochs (CNN) | 4 inner person folds inside the training people (median best epoch) | validation, test, locked |
| Logistic-regression strength (C) | the same 4 inner folds | validation, test, locked |
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
