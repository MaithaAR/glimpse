# Model card: GLIMPSE pre-meal spike warning (final, 26 Nov 2026)

The intermediate card is kept at docs/archive/model_card_intermediate.md. Every number here is in
docs/final_report.md with its evidence file.

**What it does.** Before a meal it estimates the chance that glucose rises by at least 50 mg/dL above the
30-minute pre-meal mean within 2 h, and answers *low risk* (risk below 0.15), *caution* (0.16 or above) or
*can't judge* (between the two, or when an input rule fails), always with a one-sentence reason.

**Intended use.** A wellness nudge for adults with prediabetes who already wear a CGM. Not a medical
device. Not for type 1 diabetes, insulin dosing or treatment decisions. No claim for type 2 diabetes:
those participants were used for training only.

**Parts (all frozen 6 Nov under a pre-registered procedure).**
- Input rules R1-R11 run first and refuse with a named reason (docs/rule_ledger.md).
- Photo: ConvNeXt-Tiny (ImageNet-22k pre-trained, 1k fine-tuned; pinned revision and SHA-256), last stage
  fine-tuned, two heads (grams, 4 carb bins); 27.8M parameters. Role: carb-entry assist. It did not pass
  its pre-registered risk gate (+0.025 AUROC, −0.010 to +0.061).
- Unfamiliar-photo check: kNN distance to the training photos, cut 0.614; no carb guess above it.
- Fusion: 20-parameter logistic regression over 8 profile inputs, 7 CGM/heart-rate trend features and 4
  photo inputs; Platt map (0.96, −0.20); per-person intercept fitted on the person's earlier labelled
  meals (shrinkage m = 1.0). The folder command and app have no history input yet, so they answer at the
  population calibration (Platt only); the evaluation replays history in deployment order.
- SensorCNN (5,217 parameters, from scratch) is evaluated, not deployed: it ties the trend features.

**Training data.** CGMacros v1.0.0 (PhysioNet, CC BY-NC-SA 4.0): 37 non-locked adults (23 healthy or
prediabetic development people, 14 type 2 diabetes), about 10 days each, USA, one study, repeated
standardised meals. 1,221 photos at most touched by fine-tuning (cap 10,000).

**Evaluation.**

| | development (23 people, 5-fold, 481 meals) | locked (8 people, scored once 25 Nov, 153 meals) |
|---|---|---|
| AUROC, shipped (95% person-clustered) | 0.762 (0.682–0.829) | 0.645 (0.540–0.736) |
| AUROC, raw fusion | 0.726 | 0.611 |
| AUPRC (base rate) | 0.591 (0.328) | 0.547 (0.420) |
| Brier (constant forecast) | 0.180 (0.221) | 0.230 (0.244) |
| ECE, registered 15 equal-width bins / equal-mass (target ≤ 0.05) | 0.052 / 0.049 | 0.061 / 0.098 |
| AUROC gain of the intercept over raw, paired | +0.037 (−0.004, +0.071), 5 of 5 folds | +0.033 (−0.054, +0.100) |
| low risk / caution / can't judge | 23% / 64% / 14% | 7% / 91% / 3% |
| spikes among meals cleared | 7.3% (2.9–14.2%) | 2 of 10 |
| cost reduction vs always caution (target ≥ 10%) | 14.4% | −5.7% |

The development estimate carries disclosed selection pressure (fold 1 was scored at the intermediate);
the locked column is the clean measurement and fails the usefulness targets.

**Known failures.** Flips between low risk and caution (either direction) stay below 1.8% under the
registered CGM and photo degradations, 3.7% with heart rate missing. Visually familiar but nutritionally unfamiliar dishes pass the photo check
with a confident wrong carb bin (R7). The label is baseline-relative, so a rising pre-meal trace lowers
the predicted risk (probe B). On a harder cohort the one-point abstention band (0.15, 0.16) flags almost
every meal.

**Monitoring and feedback.** docs/monitoring.md: photo drift (demonstrated), outcome, behaviour and
calibration signals. The CGM labels every meal at +2 h; replaying logged answers showed the outcome rule
is too noisy at 80-answer windows to serve as an alarm (outputs/monitoring/replay.json).

**Ethics.** De-identified public data under the source study's approval; no new human data; no end-user
or clinical validation.
