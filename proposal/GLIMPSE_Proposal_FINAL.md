# GLIMPSE: Pre-Meal Glucose-Spike Risk from a Meal Photo, CGM Trend and Profile

**Maitha Alhosani (solo)** · MAAI7103 Deep Learning · 27 September 2026 · Code and logs: [github.com/MaithaAR/glimpse](https://github.com/MaithaAR/glimpse) (pilot results: [commit `3609060`](https://github.com/MaithaAR/glimpse/tree/3609060/results))

## 1. Venture and user

**Problem.** Some prediabetes and nutrition programmes give members a continuous glucose monitor (CGM); members log meals by hand and dietitians review traces afterwards, but the CGM shows a spike only *after* eating. GLIMPSE (*Glycemic-response Inference from Meal Photos and Sensor Evidence*) gives a calibrated spike risk *before* the first bite. The user is an adult with prediabetes; the proposed business model is a per-member provider fee (willingness to pay unestablished). January AI predicts responses from food photos without a sensor; GLIMPSE bets that live pre-meal glucose, calibration and abstention add trust.

**Scenario and journey.** Mariam photographs rice with chicken while her glucose is rising; the card shows **caution**, a predicted-rise interval and its inputs (large carb estimate, rising trend). *Onboarding:* 72 hours of meals set a personal prior (population prior if fewer than 5 meals). *Each meal:* a cost model on the calibrated risk picks low risk, caution or **abstain** (out-of-distribution photo, CGM gap, or risk in the cost-optimal middle band). *Feedback:* each CGM rise becomes a label and updates the prior after its 2-h window closes; carb corrections become photo labels. *Review:* abstentions and a 5% audit go to a dietitian queue.

## 2. System concept

**Capabilities:** (1) photo-based macro estimation, (2) pre-meal spike risk with a rise interval, (3) a slider showing the *model's* carbohydrate sensitivity (not dietary advice), (4) abstention with dietitian review, (5) monitoring of drift, abstention, OOD rate and latency. Deterministic checks (CGM gaps, missing photo) run before the models; cost thresholds after.

| Component | Model | Params | Training |
|---|---|---|---|
| Photo → served carbs, protein, fat, kcal (+ fibre) | EfficientNet-B0 | 5.3M | **Fine-tuned**: Nutrition5k (≤5,006), then CGMacros (≤1,486, adds fibre); <10k items; Huber loss |
| Pre-meal CGM and heart-rate encoder (24 × 5 min) | 1-D CNN | 0.1–1M | **From scratch** |
| Fusion: spike logit, 10/50/90% rise quantiles | MLP, non-negative carb weight | <1M | From scratch; cross-entropy + pinball loss |
| Photo OOD (kNN distance to training embeddings) | DINOv2-S | 22M | Frozen |

**Budget:** about 29M parameters, offline PyTorch/Docker, p95 ≤300 ms on the author's MacBook Pro CPU. **Why deep learning:** macro estimation needs learned visual representations of foods and portions; for the sensor sequence it is tested (RQ2).

## 3. Data and learning plan

**Data.** [CGMacros](https://physionet.org/content/cgmacros/1.0.0/) (PhysioNet) follows 45 adults (15 healthy, 16 prediabetes, 14 type 2 diabetes) for about 10 days, with Dexcom CGM, Fitbit heart rate, meal photos, labs, consumed macros and share eaten; [Nutrition5k](https://github.com/google-research-datasets/Nutrition5k) (5,006 plates, no fibre) provides photo pre-training. **Typical example:** a before-photo, 24 Dexcom readings with heart rate, age/BMI/A1c and meal hour; label = spike if the peak rises ≥50 mg/dL above the 30-min pre-meal mean within 2 h (30 and 70 as checks). **Usable window:** ≥3 baseline readings, ≥70% post-meal coverage, no gap >20 min. **Reference:** consumed ÷ share eaten gives noisy "served" macros (share <10% excluded).

**Counts.** 1,526 usable − 453 onboarding − 19 boundary = 1,054 operational meals (769 in the 31-person **target cohort**, healthy plus prediabetes: 34.2% spikes, 27 without photo); 896 isolated (no meal in the next 2 h), 634 of them target (35.2% spikes).

**Splits and leakage.**

- **Primary:** nested person-grouped cross-validation (CV) on the 37 development people (23 target, plus 14 type 2 diabetes for training only); inner folds serve validation and tuning.
- **Fit within folds:** every learned stage (photo model, population prior, fusion, calibration, conformal, thresholds) uses only outer-fold training people; fusion uses out-of-fold photo predictions.
- **Repeated meals:** 62% of isolated target meals share an exact macro profile with another person's, so photo models are also tested with test-fold profiles excluded; near-duplicate photos (perceptual hash) will share a fold.
- **Locked set:** 8 target people frozen in `splits/` (SHA-256 check), run once in week 11, descriptively.
- **Imbalance** is mild (35% spikes): unweighted loss, cost-based thresholds, class weighting as an ablation.

**Research questions.** Three models differ only in meal information (*none*, *photo-predicted*, *reference* macros); intervals are paired person-bootstrap 95% CIs.

- **RQ1:** Δ = AUROC(photo) − AUROC(reference) is *non-inferior* if its lower bound > −0.03, *inferior* if its upper bound < −0.03, otherwise *inconclusive*. The photo *adds value* if the lower bound of G_photo = AUROC(photo) − AUROC(none) > 0. The 0.03 margin is a prespecified tolerance for removing manual logging, not a clinical one.
- **RQ2:** CNN encoder vs engineered trends, all else identical. It *beats* them if the lower bound > 0; non-inferiority (−0.03) is reported separately (preliminary pilot: −0.021, CI −0.042 to +0.005).

**One executable evaluation.** `scripts/03_photo_experiment.py` builds out-of-fold photo predictions (5 person folds × 3 seeds), fits three gradient-boosted (GBM) classifiers per fold and writes carb MAE, AUROCs, Δ and G_photo with CI bounds and **width**. Final RQ1 reuses it with the **same neural fusion** in all three conditions (GBMs as baselines).

## 4. Scope, evidence and success

**Photo pilot (completed 27 September).**
Frozen ImageNet EfficientNet-B0 features feed a ridge head (α = 300) predicting log served macros, nested in each outer fold; 611 isolated target meals (31 people, 35.0% spikes; locked people included, so internal evidence only); source `results/`.

| Meal information | AUROC | Gain over none (95% CI) | Carb MAE (fold-fitted constant) |
|---|---|---|---|
| None | 0.677 | n/a | n/a |
| Reference macros | 0.754 | +0.077 (+0.043, +0.118) | n/a |
| **Photo, person-excluded** | **0.723** | **+0.046 (+0.018, +0.077)** | 26.2 g (27.4 g), bias −4.5 g |
| Photo, also profile-excluded | 0.710 | +0.033 (+0.003, +0.066) | 30.0 g (29.1 g), bias −11.9 g |
| Noise images (control) | 0.669 | −0.008 (−0.022, +0.006) | 29.0 g (27.4 g) |

**What this shows.** The photo adds spike signal under both splits, *consistent with* nutritional ranking (carb Spearman 0.37) despite weak gram accuracy, and with some reliance on recurring meal profiles. **RQ1 pilot answers:** incremental value is **positive** (G_photo lower bound +0.018); non-inferiority is **inconclusive**, with Δ = −0.031 (−0.067, +0.000; width 0.068 before rounding), so at this width it needs Δ ≳ 0 (fewer development people likely widen it); gains hold at α = 100 and 1,000.

**Success criteria (target cohort).**

- **Discrimination:** RQ1, RQ2; fusion vs GBM on the same inputs (−0.03 rule); a CNN+GBM ensemble is compared separately.
- **Calibration:** ECE ≤0.05 with reliability diagrams. Conformal rise intervals: 75–85% pooled coverage at 80% nominal (no guarantee), with mean width (mg/dL) and median-rise MAE vs a GBM regressor.
- **Explainability:** photo swaps at fixed sensor state, flat vs rising traces at equal glucose, branch masking.
- **Robustness and latency:** blurred photos, missing labs or heart rate, simulated CGM gaps (ΔAUROC, ΔECE, abstention); p95 ≤300 ms.
- **Usefulness** (769 target operational meals incl. those followed by more eating, not attributed to one meal; missing photos and pre-meal CGM gaps count as abstentions; assumed costs spike/no spike: low risk 5/0, caution 0/1, abstain 2/0.5, swept): cost ≥10% below the best trivial policy, ≥20% cleared with ≤10% spiking, abstention ≤20%. *Pilot* (GBM, isolated meals, `results/usefulness_pilot.log`): photo 0.3% cost reduction, 10.5% cleared (14.1% spiking), ECE 0.071; reference macros 9.6%, 16.7% (11.8%), ECE 0.056; targets not yet met.

**Milestones.** **Week 4 gate:** fine-tuned carb MAE below 23.1 g (frozen, α = 1,000, person-excluded) and below the fold-fitted constant (29.1 g) under profile exclusion, with G_photo lower bound > 0 on both. Week 6: RQ2; 7: calibration; 10: app; 11: locked set.

**Minimum deliverable and fallback.** Minimum: calibrated fused model, abstention, Gradio app. If the week-4 gate fails, the app uses **confirm-carbs mode** (photo proposes, user adjusts), assessed with *simulated* correction errors (±25%, ±40%), not real users; RQ1 keeps its rule.

**Reproducibility:** incremental git history, W&B exports, and a package (Docker image, pinned requirements, locked split, scripts regenerating every table).

**Product assumption (unvalidated):** dietitians accept "cannot judge this meal"; planned for project week 2 (two dietitians, 5-person think-aloud). **Limitations.** No dosing, hypoglycaemia or causal claims; 45 people, one dataset; proportional eating assumed; packaging/drink photos are hard; the 30-photo Emirati OOD set (planned) is exploratory; non-commercial licence; no MAAI7102 overlap.
