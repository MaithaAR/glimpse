# GLIMPSE: Pre-Meal Glucose-Spike Risk from a Meal Photo, CGM Trend and Profile

**Maitha Alhosani (solo)** · MAAI7103 Deep Learning · 27 September 2026 · Code and logs: `~/projects/glimpse` (git, commit `9f12e3c`)

## 1. Venture and user

**Problem.** Prediabetes programmes lend members a continuous glucose monitor (CGM) for a few weeks. Members log meals, and dietitians review traces afterwards, but the CGM shows a spike only *after* eating. GLIMPSE (*Glycemic-response Inference from Meal Photos and Sensor Evidence*) gives a calibrated spike risk *before* the first bite. The user is an adult with prediabetes; the programme provider pays per member for guidance that scales beyond dietitian time. January AI already predicts responses from food photos; GLIMPSE bets that live pre-meal glucose, calibration and abstention make this more trustworthy.

**Scenario.** Mariam photographs rice with chicken while her glucose is already rising. The card shows **caution**, a predicted-rise interval and its inputs (large carb estimate, rising trend); a slider shows how the *model's* risk changes with less carbohydrate, not dietary advice.

**Journey.** *Onboarding:* 72 hours of meals set a personal prior (population prior if fewer than 5 meals). *Each meal:* a cost model on the calibrated risk picks low risk, caution or **abstain** (out-of-distribution photo, CGM gap, or risk in the cost-optimal middle band). *Feedback:* each observed CGM rise becomes a label and updates the prior only after its 2-h window closes; carb corrections become photo labels. *Review and monitoring:* abstentions and a 5% audit go to a dietitian queue; calibration drift, abstention, OOD rate and latency are tracked.

## 2. System concept

| Component | Model | Params | Training |
|---|---|---|---|
| Photo → served carbs, protein, fat, kcal (+ fibre) | EfficientNet-B0 | 5.3M | **Fine-tuned**: Nutrition5k (≤5,006), then CGMacros (≤1,486; adds fibre), under 10k items; Huber on log macros |
| Pre-meal CGM and heart-rate encoder (24 × 5 min) | 1-D CNN | 0.1–1M | **From scratch** |
| Fusion: spike logit, 10/50/90% rise quantiles | MLP, non-negative carb weight | <1M | From scratch; class-weighted cross-entropy + pinball loss |
| Photo OOD (kNN distance to training embeddings) | DINOv2-S | 22M | Frozen |

**Budget:** about 29M parameters, offline PyTorch/Docker, p95 latency ≤300 ms on a laptop CPU. **Why deep learning:** essential for the photo (no tabular route from pixels to macros); for the sensor sequence it is tested (RQ2), and engineered trends are kept if the CNN loses.

## 3. Data and learning plan

**Data.** CGMacros (PhysioNet) follows 45 adults (15 healthy, 16 prediabetes, 14 type 2 diabetes) for about 10 days, with Dexcom CGM, Fitbit heart rate, before-meal photos, labs, consumed macros and share eaten. Nutrition5k (5,006 plates, no fibre labels) provides photo pre-training. **Typical example:** a before-photo, 24 Dexcom readings with heart rate, age/BMI/A1c and meal hour; label = spike if the Dexcom peak rises ≥50 mg/dL above the 30-min pre-meal mean within 2 h (30 and 70 as sensitivity checks). **Usable window:** ≥3 baseline readings, ≥70% post-meal coverage, no gap over 20 min. **Reference macros:** consumed ÷ share eaten gives noisy "served" macros; share below 10% is excluded.

**Counts.** 1,526 usable meals → minus 453 onboarding and 19 boundary → 1,054 operational (769 in the 31-person **target cohort** of healthy plus prediabetes, 34.2% spikes, 27 without a photo) → 896 isolated (no meal within the next 2 h), 634 of them target (35.2% spikes).

**Splits and leakage.**

- **Primary:** nested person-grouped cross-validation (CV) on the 37 development people (23 target, plus 14 type 2 diabetes used for training only).
- **Fit within folds:** every learned stage (photo fine-tuning, population prior, fusion, temperature, conformal quantile, thresholds) uses only each outer fold's training people; fusion trains on inner out-of-fold photo predictions.
- **Repeated meals:** 62% of target meals share an exact macro profile with another person's meal (standardised study meals), so the photo model is also evaluated with test-fold profiles excluded from its training.
- **Locked set:** 8 target people (seeded, SHA-256 hash in `splits/locked_final_set.json`) are run once in week 11, descriptively only.

**Research questions.** Three models share sensor, profile and prior inputs and differ only in meal information: *none*, *photo-predicted* or *reference* served macros. Intervals are paired person-bootstrap 95% CIs (2,000 resamples).

- **RQ1:** Δ = AUROC(photo) − AUROC(reference) is *non-inferior* if its lower bound > −0.03, *inferior* if its upper bound < −0.03, otherwise *inconclusive*. The photo *adds value* if the lower bound of G_photo = AUROC(photo) − AUROC(none) > 0.
- **RQ2:** CNN encoder vs engineered trends, all else identical. It *beats* them if the paired lower bound > 0; non-inferiority (−0.03) is reported separately (it would only justify dropping hand-crafted features).

**One executable evaluation.** `python3 scripts/03_photo_experiment.py` (flags `--profile-excluded`, `--noise-control`) builds out-of-fold photo predictions (5 person folds × 3 seeds), fits three gradient-boosted (GBM) classifiers per fold, and writes carb MAE and bias (vs a fold-fitted median constant), AUROCs, and Δ and G_photo with CI bounds and **interval width** to `results/*.json`. The final RQ1 reuses this protocol with the **same neural fusion** in all three meal-information conditions; matched GBMs remain baselines.

## 4. Scope, evidence and success

**Photo pilot (completed 27 September).**
Frozen ImageNet EfficientNet-B0 features feed a ridge head predicting log served macros (α = 300, the script default; α = 100 and 1,000 run afterwards), nested in every outer fold and never trained on the person it predicts. It is scored on 611 matched isolated target-cohort meals (31 people, 35.0% spikes, including the 8 locked people, so optimistic).

| Meal information | AUROC | Gain over none (95% CI) | Carb MAE (fold-fitted constant) |
|---|---|---|---|
| None | 0.677 | n/a | n/a |
| Reference macros | 0.754 | +0.077 (+0.043, +0.118) | n/a |
| **Photo, person-excluded** | **0.723** | **+0.046 (+0.018, +0.077)** | 26.2 g (27.4 g), bias −4.5 g |
| Photo, also profile-excluded | 0.710 | +0.033 (+0.003, +0.066) | 30.0 g (29.1 g), bias −11.9 g |
| Noise images (control) | 0.669 | −0.008 (−0.022, +0.006) | 29.0 g (27.4 g) |

**What this shows.** The photo adds spike signal under both splits, *consistent with* useful nutritional ranking (carb Spearman 0.37) despite weak gram accuracy, and with some reliance on recurring meal profiles; profile exclusion also shrinks training data, so neither mechanism is isolated. **RQ1 is inconclusive:** Δ = −0.031 (−0.067, +0.000), width 0.068; at this width non-inferiority needs Δ ≳ 0, and 23 development people are likely to reduce precision. The gain stays positive at α = 100 (+0.028, MAE 31.8 g) and 1,000 (+0.046, MAE 23.1 g); no clear gain was detected from noise images.

**Earlier pilots** (`pilot/test5.log`, `test3.log`): neural network vs GBM −0.013 (−0.051, +0.028); learned vs engineered sequence features −0.021 (−0.042, +0.005), so RQ2 is open; ECE 0.040, 0.037 after Platt scaling. All pilots include the 8 locked people, so they show internal feasibility only. The photo pilot fits the population-prior fallback (1 person, 24 meals) within folds; the earlier pilots used all people and are preliminary.

**Success criteria (target cohort).**

- **Discrimination:** RQ1 and RQ2 as above. Fusion vs a GBM on the same inputs uses the −0.03 rule; a CNN+GBM ensemble is a separate prespecified comparison, not a rescue.
- **Calibration:** ECE ≤0.05. Conformal rise intervals: 75–85% pooled coverage at 80% nominal (no guarantee), with mean width in mg/dL and median-rise MAE vs a GBM regressor.
- **Usefulness** (the 769 target-cohort operational meals, including those followed by more eating, whose response is not attributed to one meal; missing photos and pre-meal CGM gaps count as abstentions; assumed spike / no-spike costs: low risk 5/0, caution 0/1, abstain 2/0.5, swept): cost ≥10% below the best trivial policy, ≥20% cleared with ≤10% of those spiking, abstention ≤20%.

**Milestones.** **Week 4 gate:** fine-tuned carb MAE below 23.1 g (best frozen, person-excluded) and below the fold-fitted constant (29.1 g) under profile exclusion, with G_photo lower bound > 0 on both splits. Week 6: RQ2. Week 7: calibration, abstention. Week 10: app. Week 11: locked set.

**Minimum deliverable and fallback.** The minimum deliverable is the calibrated fused model with abstention and the Gradio app. If the week-4 gate fails, the app switches to **confirm-carbs mode** (the photo proposes, the user adjusts). This is operational; RQ1 is judged only by its rule. Confirm-carbs mode gets a *simulated correction-error sensitivity analysis* (reference carbs perturbed by 25% and 40%); exact carbs are an idealised reference scenario, and real user accuracy is not established.

**Product assumption, not yet validated:** dietitians accept "cannot judge this meal"; week 2 tests it with two dietitians and a 5-person think-aloud. **Limitations.** No dosing, hypoglycaemia or causal claims (CGMacros has no medication data). 45 people, one dataset; served macros assume proportional eating. Packaging and drink photos are hard (9-photo spot check); a planned 30-photo Emirati-dish OOD set is exploratory. Non-commercial licence; no overlap with MAAI7102.
