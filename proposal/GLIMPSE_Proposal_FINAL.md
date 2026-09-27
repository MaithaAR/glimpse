# GLIMPSE: Pre-Meal Glucose-Spike Risk from a Meal Photo, CGM Trend and Profile

**Maitha Alhosani (solo)** · MAAI7103 Deep Learning · 27 September 2026 · Code and logs: `~/projects/glimpse` (git, commit `373b0e3`)

## 1. Venture and user

**Problem.** Prediabetes programmes lend members a continuous glucose monitor (CGM) for a few weeks. Members log meals, and dietitians review traces afterwards, but the CGM shows a spike only *after* eating. GLIMPSE (*Glycemic-response Inference from Meal Photos and Sensor Evidence*) gives a calibrated spike risk *before* the first bite.
- **User and buyer:** an adult with prediabetes uses it; the programme provider pays per member for guidance that scales beyond dietitian time.
- **Differentiation:** January AI already predicts responses from food photos. GLIMPSE's untested bet is that live pre-meal glucose, calibration and abstention make that guidance more trustworthy.

**Scenario.** Mariam photographs rice with chicken at lunch while her glucose is already rising. The risk card shows **caution**, a predicted-rise interval and the inputs behind it (large carb estimate, rising trend). A slider shows how the *model's* risk changes with less carbohydrate; it is not dietary advice.

**Journey.**
- **Onboarding:** 72 hours of meals set a personal prior (population prior if fewer than 5 meals).
- **Each meal:** a cost model on the calibrated risk picks low risk, caution or **abstain**. It abstains for an out-of-distribution photo, a CGM gap, or risk in the cost-optimal middle band.
- **Feedback:** each observed CGM rise becomes a label and updates the prior; carbohydrate corrections become photo labels.
- **Review:** abstentions and a 5% audit go to a dietitian queue. Monitoring tracks calibration drift, abstention, OOD rate and latency.

## 2. System concept

| Component | Model | Params | Training |
|---|---|---|---|
| Photo → served carbs, protein, fat, kcal (+ fibre) | EfficientNet-B0 | 5.3M | **Fine-tuned**: Nutrition5k, then CGMacros (adds fibre); Huber loss on log macros |
| Pre-meal CGM and heart-rate encoder (24 × 5 min) | 1-D CNN | 0.1–1M | **From scratch** |
| Fusion: spike logit, 10/50/90% rise quantiles | MLP, non-negative carb weight | <1M | From scratch; class-weighted cross-entropy + pinball loss |
| Photo OOD (kNN distance to training embeddings) | DINOv2-S | 22M | Frozen |

- **Budget:** the total is about 29M parameters, run offline in PyTorch/Docker with a p95 latency target of ≤300 ms on a laptop CPU.
- **Why deep learning:** it is essential for the photo, since there is no tabular route from pixels to macros. For the sensor sequence it is tested, not assumed (RQ2).
- **Fallback for RQ2:** if the CNN encoder loses, the fusion keeps engineered trend features.

## 3. Data and learning plan

**Data.** CGMacros (PhysioNet) follows 45 adults (15 healthy, 16 prediabetes, 14 type 2 diabetes) for about 10 days. It records Dexcom CGM, Fitbit heart rate, before-meal photos, labs and consumed macros with the share eaten. Nutrition5k (5,006 plates, no fibre labels) provides photo pre-training.

- **Typical example:** one before-photo, 24 Dexcom readings with heart rate, age/BMI/A1c and the meal hour. It is labelled a spike if the Dexcom peak rises ≥50 mg/dL above the 30-min pre-meal mean within 2 h (30 and 70 are sensitivity thresholds).
- **Usable window:** ≥3 baseline readings, ≥70% post-meal coverage, no gap over 20 min.
- **Reference macros:** consumed ÷ share eaten gives "served" macros. These are noisy supervision, and meals with share below 10% are excluded.

**Counts.** 1,526 meals pass the window rules → minus 453 onboarding and 19 boundary meals → 1,054 operational → 896 isolated (no further meal within 2 h) → 634 in the 31-person **target cohort** (healthy plus prediabetes, 35.2% spikes). 1,486 meal rows have a before-photo.

**Splits and leakage.**
- **Primary:** nested person-grouped cross-validation (CV) on the 37 development people (23 target, plus 14 type 2 diabetes used for training only).
- **Fit within folds:** every learned stage (photo fine-tuning, fusion, temperature, conformal quantile, thresholds) is fit only on each outer fold's training people. Fusion trains on inner out-of-fold photo predictions.
- **Repeated meals:** 62% of target meals share an exact logged macro profile with another person's meal (standardised study meals). The photo model is therefore also evaluated with those profiles excluded from its training.
- **Locked set:** 8 target people (seeded, SHA-256 hash in `splits/locked_final_set.json`) are run once in week 11, descriptively only.

**Research questions.** Three models have identical sensor, profile and prior inputs and differ only in meal information: *none*, *photo-predicted* or *reference* served macros. All intervals are paired person-bootstrap 95% CIs (2,000 resamples of people).
- **RQ1:** Δ = AUROC(photo) − AUROC(reference) is *non-inferior* if its lower bound > −0.03, *inferior* if its upper bound < −0.03, otherwise *inconclusive*. The photo *adds value* if the lower bound of G_photo = AUROC(photo) − AUROC(none) > 0.
- **RQ2:** CNN encoder vs engineered trend features, everything else identical, same −0.03 rule.

**One executable evaluation.** `python3 scripts/03_photo_experiment.py` (flags `--profile-excluded`, `--noise-control`) builds out-of-fold photo predictions (5 person folds × 3 seeds) and fits the three gradient-boosted (GBM) classifiers per fold. It writes carb MAE and bias, AUROCs, and Δ and G_photo with CI bounds and **interval width** to `results/*.json`. The final evaluation reuses this protocol with the fine-tuned photo model and neural fusion.

## 4. Scope, evidence and success

**Photo pilot (completed 27 September).**
Frozen ImageNet EfficientNet-B0 features feed a ridge head predicting log served macros (α = 300, the script default; α = 100 and 1,000 were run afterwards). The head is nested in every outer fold and never trains on the person it predicts. It is scored on 611 matched isolated target-cohort meals (31 people, 35.0% spikes); logs are in `results/`.

| Meal information (sensor, profile and prior identical) | AUROC | Gain over none (95% CI) | Carb MAE (constant: 27.2 g) |
|---|---|---|---|
| None | 0.675 | n/a | n/a |
| Reference served macros | 0.752 | +0.077 (+0.044, +0.117) | n/a |
| **Photo, person-excluded** | **0.722** | **+0.047 (+0.019, +0.077)** | 26.2 g, bias −4.5 g |
| Photo, person- *and* meal-profile-excluded | 0.709 | +0.034 (+0.004, +0.065) | 30.0 g, bias −11.9 g |
| Random-noise images (negative control) | 0.665 | −0.010 (−0.023, +0.003) | 29.0 g |

**What this shows.**
- **The photo adds spike signal under both splits.** It comes from carb *ranking* (Spearman 0.37), not gram accuracy, and part of it came from recognising repeated standard meals.
- **RQ1 is inconclusive:** Δ = −0.030 (−0.064, +0.002), a width of 0.066. At about 0.077 width with 23 people, non-inferiority needs Δ ≳ +0.01.
- **Checks:** the gain stays positive at α = 100 (+0.030, MAE 31.8 g) and α = 1,000 (+0.050, MAE 23.1 g), and uninformative images add none.

**Earlier pilots** (`pilot/test5.log`, `pilot/test3.log`): neural network vs GBM −0.013 (−0.051, +0.028); learned vs engineered sequence features −0.021 (−0.042, +0.005), so RQ2 is open; expected calibration error (ECE) 0.040, and 0.037 after Platt scaling (all 45 people, earlier rules). All figures include the 8 locked people and a population prior computed on all people, so they are optimistic.

**Success criteria (target cohort).**
- **Discrimination:** RQ1 and RQ2 as above. Fusion must be non-inferior to a GBM on the same inputs (−0.03 rule); otherwise the CNN+GBM ensemble is reported.
- **Calibration:** ECE ≤0.05. Conformal rise intervals with 75–85% pooled coverage at 80% nominal (no guarantee claimed).
- **Usefulness** (operational cohort; assumed costs for spike / no spike: low risk 5/0, caution 0/1, abstain 2/0.5, swept in sensitivity analysis): cost ≥10% below the best trivial policy, ≥20% of meals cleared with ≤10% of those spiking, abstention ≤20%.

**Milestones.** **Week 4 gate:** the fine-tuned photo model reaches carb MAE below 23.1 g (best frozen result, person-excluded) and below the 27.2 g constant under profile exclusion, with a G_photo lower bound > 0 on both splits. Week 6: RQ2. Week 7: calibration and abstention. Week 10: app. Week 11: locked set.

**Minimum deliverable and fallback.** The minimum deliverable is the fused model with calibration, abstention and the Gradio app. If the week-4 gate fails, the app switches to **confirm-carbs mode**: the photo proposes an estimate and the user adjusts it. RQ1 is then reported as a negative result. The course requirements (three modalities, a fine-tuned and a from-scratch model) hold either way.

**Product assumption, not yet validated.** The assumption is that dietitians accept "cannot judge this meal" rather than a forced answer. It will be tested in week 2 with two dietitian conversations and a 5-person think-aloud (can users tell *abstain* from *low risk*?), and the result reported as found.

**Out of scope and limitations.** No dosing, hypoglycaemia or causal claims (CGMacros has no medication data). There are 45 people and one dataset, and served macros assume proportional eating. A 9-photo spot check flagged packaging and drinks as hard. A planned 30-photo Emirati-dish OOD set is exploratory. The licence is non-commercial, so a product would need programme-collected data. There is no overlap with MAAI7102.
