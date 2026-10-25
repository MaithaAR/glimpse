# Validation-chosen cut-offs on recalibrated risk (no test labels used; added after the results)

Fitted on: 4 validation people (99 meals), frozen 28-person model. Platt slope 1.40, intercept -0.47.

Cut-offs chosen on the recalibrated validation predictions: low risk below 0.17, caution from 0.17. Frozen in `checkpoints/thresholds_validation_recal.json` (sha256 0407245816f4). Stability check: failed (no cost saving or nothing cleared on the CV check).

| Split | Meals | ECE before | ECE after |
|---|---|---|---|
| Validation (fitted here) | 99 | 0.126 | 0.074 |
| CV without test and validation people | 286 (14 people) | 0.081 | 0.138 |

| CV check, by person | Predicted before | Observed | Predicted after |
|---|---|---|---|
| Rare spikers | 0.32 | 0.20 | 0.20 |
| Frequent spikers | 0.45 | 0.59 | 0.36 |

| Rule | Cleared | Spikes among cleared | Can't judge | Cost saving |
|---|---|---|---|---|
| Validation people: validation-chosen, recalibrated | 53% | 6% | 0% | +43% |
| Validation people: cost-only (deployed) | 13% | 0% | 9% | +22% |
| CV check: validation-chosen, recalibrated | 39% | 22% | 0% | -22% |
| CV check: cost-only (deployed) | 12% | 12% | 10% | +5% |
