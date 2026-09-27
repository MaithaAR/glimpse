# GLIMPSE: Pre-Meal Glucose-Spike Risk from a Meal Photo, CGM Trend and Profile

**Maitha Alhosani (solo)** · MAAI7103 Deep Learning · 27 September 2026 · Code and logs: git repository `glimpse`, commit `c1c8a74`

## 1. Venture and user

**Problem.** Prediabetes programmes lend members a continuous glucose monitor (CGM); members log meals and dietitians review traces, but the CGM shows a spike only *after* eating. GLIMPSE (*Glycemic-response Inference from Meal Photos and Sensor Evidence*) gives a calibrated spike risk *before* the first bite. The user is an adult with prediabetes; the proposed business model is a per-member provider fee (willingness to pay not yet established). January AI predicts responses from food photos; GLIMPSE bets that live pre-meal glucose, calibration and abstention add trust.

**Scenario and journey.** Mariam photographs rice with chicken while her glucose is rising; the card shows **caution**, a predicted-rise interval and its inputs (large carb estimate, rising trend), and a slider shows the *model's* carbohydrate sensitivity, not dietary advice. *Onboarding:* 72 hours of meals set a personal prior (population prior if fewer than 5 meals). *Each meal:* a cost model on the calibrated risk picks low risk, caution or **abstain** (out-of-distribution photo, CGM gap, or risk in the cost-optimal middle band). *Feedback:* each observed CGM rise becomes a label and updates the prior only after its 2-h window closes; carb corrections become photo labels. *Review:* abstentions and a 5% audit go to a dietitian queue; drift, abstention, OOD rate and latency are monitored.

## 2. System concept

| Component | Model | Params | Training |
|---|---|---|---|
| Photo → served carbs, protein, fat, kcal (+ fibre) | EfficientNet-B0 | 5.3M | **Fine-tuned**: Nutrition5k (≤5,006), then CGMacros (≤1,486, adds fibre); <10k items; Huber loss |
| Pre-meal CGM and heart-rate encoder (24 × 5 min) | 1-D CNN | 0.1–1M | **From scratch** |
| Fusion: spike logit, 10/50/90% rise quantiles | MLP, non-negative carb weight | <1M | From scratch; class-weighted cross-entropy + pinball loss |
| Photo OOD (kNN distance to training embeddings) | DINOv2-S | 22M | Frozen |

**Budget:** about 29M parameters, offline PyTorch/Docker, p95 latency ≤300 ms on a laptop CPU. **Why deep learning:** macro estimation needs learned visual representations of foods and portions; for the sensor sequence it is tested (RQ2), keeping engineered trends if the CNN loses.

## 3. Data and learning plan

**Data.** [CGMacros](https://physionet.org/content/cgmacros/1.0.0/) (PhysioNet) follows 45 adults (15 healthy, 16 prediabetes, 14 type 2 diabetes) for about 10 days, with Dexcom CGM, Fitbit heart rate, meal photos, labs, consumed macros and share eaten; [Nutrition5k](https://github.com/google-research-datasets/Nutrition5k) (5,006 plates, no fibre) provides photo pre-training. **Typical example:** a before-photo, 24 Dexcom readings with heart rate, age/BMI/A1c and meal hour; label = spike if the peak rises ≥50 mg/dL above the 30-min pre-meal mean within 2 h (30 and 70 as checks). **Usable window:** ≥3 baseline readings, ≥70% post-meal coverage, no gap over 20 min. **Reference:** consumed ÷ share eaten gives noisy "served" macros (share <10% excluded).

**Counts.** 1,526 usable − 453 onboarding − 19 boundary = 1,054 operational meals (769 in the 31-person **target cohort**, healthy plus prediabetes: 34.2% spikes, 27 without photo); 896 isolated (no meal in the next 2 h), 634 of them target (35.2% spikes).

**Splits and leakage.**

- **Primary:** nested person-grouped cross-validation (CV) on the 37 development people (23 target, plus 14 type 2 diabetes used for training only).
- **Fit within folds:** every learned stage (photo model, population prior, fusion, calibration, conformal, thresholds) uses only outer-fold training people; fusion uses inner out-of-fold photo predictions.
- **Repeated meals:** 62% of target meals share an exact macro profile with another person's (standardised meals), so photo models are also tested with test-fold profiles excluded from training.
- **Locked set:** 8 target people (hashed in `splits/`) run once in week 11, descriptively.

**Research questions.** Three models differ only in meal information (*none*, *photo-predicted*, *reference* macros); intervals are paired person-bootstrap 95% CIs.

- **RQ1:** Δ = AUROC(photo) − AUROC(reference) is *non-inferior* if its lower bound > −0.03, *inferior* if its upper bound < −0.03, otherwise *inconclusive*. The photo *adds value* if the lower bound of G_photo = AUROC(photo) − AUROC(none) > 0. The 0.03 margin is a prespecified research tolerance for removing manual logging, not a clinical margin.
- **RQ2:** CNN encoder vs engineered trends, all else identical. It *beats* them if the lower bound > 0; non-inferiority (−0.03) is reported separately (preliminary pilot: −0.021, CI −0.042 to +0.005).

**One executable evaluation.** `scripts/03_photo_experiment.py` (`--profile-excluded`, `--noise-control`) builds out-of-fold photo predictions (5 person folds × 3 seeds), fits three gradient-boosted (GBM) classifiers per fold, and writes carb MAE vs a fold-fitted median constant, AUROCs, and Δ and G_photo with CI bounds and **width**. The final RQ1 reuses it with the **same neural fusion** in all three conditions; GBMs remain baselines.

## 4. Scope, evidence and success

**Photo pilot (completed 27 September).**
Frozen ImageNet EfficientNet-B0 features feed a ridge head (α = 300, default) predicting log served macros, nested in each outer fold, never trained on the predicted person; 611 isolated target meals (31 people, 35.0% spikes; locked people included, so internal evidence only). Source: `results/photo_experiment{,_profile_excluded,_noise_control}.json`.

| Meal information | AUROC | Gain over none (95% CI) | Carb MAE (fold-fitted constant) |
|---|---|---|---|
| None | 0.677 | n/a | n/a |
| Reference macros | 0.754 | +0.077 (+0.043, +0.118) | n/a |
| **Photo, person-excluded** | **0.723** | **+0.046 (+0.018, +0.077)** | 26.2 g (27.4 g), bias −4.5 g |
| Photo, also profile-excluded | 0.710 | +0.033 (+0.003, +0.066) | 30.0 g (29.1 g), bias −11.9 g |
| Noise images (control) | 0.669 | −0.008 (−0.022, +0.006) | 29.0 g (27.4 g) |

**What this shows.** The photo adds spike signal under both splits, *consistent with* nutritional ranking (carb Spearman 0.37) despite weak gram accuracy, plus some reliance on recurring meal profiles (not isolated, since profile exclusion shrinks training data). **RQ1 pilot answers:** incremental value is **positive** (G_photo lower bound +0.018 > 0); non-inferiority is **inconclusive**, with Δ = −0.031 (−0.067, +0.000; unrounded −0.0673 to +0.0004, width 0.068), so it needs Δ ≳ 0, and 23 development people are likely to reduce precision. Gains stay positive at α = 100 and 1,000; noise images show no clear gain.

**Success criteria (target cohort).**

- **Discrimination:** RQ1, RQ2; fusion vs GBM on the same inputs (−0.03 rule); a CNN+GBM ensemble is a separate prespecified comparison.
- **Calibration:** ECE ≤0.05 with reliability diagrams. Conformal rise intervals: 75–85% pooled coverage at 80% nominal (no guarantee), with mean width in mg/dL and median-rise MAE vs a GBM regressor.
- **Explainability:** photo swaps at fixed sensor state, flat vs rising traces at equal glucose, branch masking.
- **Robustness:** blurred photos, missing labs or heart rate, simulated CGM gaps (ΔAUROC, ΔECE, abstention).
- **Usefulness** (769 target operational meals incl. those followed by more eating, not attributed to one meal; missing photos and pre-meal CGM gaps count as abstentions; assumed spike/no-spike costs low risk 5/0, caution 0/1, abstain 2/0.5, swept): cost ≥10% below the best trivial policy, ≥20% cleared with ≤10% spiking, abstention ≤20%.

**Milestones.** **Week 4 gate:** fine-tuned carb MAE below 23.1 g (best frozen, person-excluded) and below the fold-fitted constant (29.1 g) under profile exclusion, with G_photo lower bound > 0 on both. Week 6: RQ2; 7: calibration; 10: app; 11: locked set.

**Minimum deliverable and fallback.** Minimum: the calibrated fused model with abstention and the Gradio app. If the week-4 gate fails, the app uses **confirm-carbs mode** (photo proposes, user adjusts); RQ1 is still judged only by its rule. It gets a *simulated correction-error sensitivity analysis* (carbs ±25%, ±40%), not a real-user test.

**Reproducibility deliverables:** incremental git history, W&B exports (configs, metrics, fold predictions) and a package (Docker image, pinned requirements, locked split, scripts regenerating every table).

**Product assumption (unvalidated):** dietitians accept "cannot judge this meal"; tested in week 2 with two dietitians and a 5-person think-aloud. **Limitations.** No dosing, hypoglycaemia or causal claims; 45 people, one dataset; proportional-eating assumption; packaging and drink photos are hard; the planned 30-photo Emirati OOD set is exploratory; non-commercial licence. No overlap with MAAI7102.
