# GLIMPSE: Does a Meal Photo Preserve the Glucose-Spike Signal? A Pre-Meal Risk Card from Photo, Sensor and Profile Data

**Team:** Maitha Alhosani (solo) · **Course:** MAAI7103 Deep Learning · **Proposal date:** 27 September 2026

Before eating, a person with prediabetes photographs her plate. GLIMPSE (*Glycemic-response Inference from Meal Photos and Sensor Evidence*) combines the photo with her last two hours of continuous glucose monitor (CGM) and heart-rate data and her profile. It returns a calibrated risk of a large glucose spike, or abstains.

The project asks two questions:
- **RQ1:** Do photo-predicted macros preserve the predictive signal of reference macros reconstructed from the meal log?
- **RQ2:** Does a learned pre-meal sensor representation beat engineered trend features?

## 1. The venture and its user

Some prediabetes and wellness programmes give members a CGM for a few weeks, but it shows a spike only after eating, too late to change the decision.

- **User:** an adult with prediabetes, or a wellness member.
- **Buyer:** the programme provider, an employer or insurer wellness scheme.
- **Differentiation (an untested hypothesis):** January AI already predicts glucose responses from food photos without a sensor. GLIMPSE bets that live pre-meal glucose state, calibration and explicit abstention make that guidance more trustworthy.

**Scenario.** At lunch, Mariam photographs rice with chicken. Her glucose has been rising since a mid-morning coffee with dates. GLIMPSE shows *caution*: a high spike probability, a predicted-rise interval, and two reasons (a large carbohydrate estimate and a rising trend). A slider shows how the *model's* prediction changes with a lower carbohydrate estimate. This shows model sensitivity, not dietary advice.

**Journey.**
- **Onboarding:** the user enters a profile (labs optional). Meals whose 2-hour windows finish within the first 72 hours build a personal prior (spike rate and mean rise). With fewer than 5 such meals, a population prior from the training target cohort is used. The 72-hour sensor period is real adoption friction.
- **At each meal, one action:**
  - *low risk*: no warning;
  - *caution*: a warning with reasons;
  - *abstain*: "cannot judge this meal", plus generic guidance. Triggered by an unreadable or unfamiliar photo, a CGM gap, or low confidence.
- **Dietitian queue:** abstained meals and a 5% random audit are reviewed later, for learning only. The target is ≤0.5 reviews per user per day, about 1 minute of dietitian time at 2 minutes per review.
- **Feedback:** users can correct the carbohydrate estimate, and completed CGM windows label predictions.

**Boundaries.** No dosing, hypoglycaemia or causal claims. CGMacros has no medication data, so the intended exclusion of insulin and sulfonylurea users cannot be verified. The licence (CC BY-NC-SA) limits use to coursework.

## 2. The system concept

**Four capabilities:**
1. A **risk card** (calibrated probability, rise interval, main inputs).
2. A **carbohydrate-sensitivity slider**.
3. **Abstention** with a review queue.
4. **Feedback and monitoring** (confidence, abstention rate, calibration drift, out-of-distribution (OOD) rate, latency).

**Three modalities, each with its own encoder:**
- the **before-meal photo**;
- a **CGM and heart-rate time series** (24 five-minute steps);
- **static tabular context** (profile, labs, time of day).

| Component | Model | Params | Notes |
|---|---|---|---|
| Photo → served carbs, protein, fat (**fine-tuned**) | EfficientNet-B0 | ~5M | Nutrition5k overhead subset (<5,006) → CGMacros development before-photos (<1,640). Fibre added at the CGMacros stage. |
| Sensor encoder (**from scratch**) | 1-D CNN | 0.1–1M | The smallest size is a serious candidate for 24 steps. |
| Fusion (**from scratch**) | MLP | <1M | Spike logit, rise quantiles, meal-only and sensor-only auxiliary heads. |
| OOD embedding (**frozen**) | DINOv2-S | 22M | Photo distance to training data. |

**Training.**
- **Image model:** Huber loss on log-macros. The backbone is frozen first, then its last blocks are unfrozen.
- **Fusion model:** binary cross-entropy for the spike, a pinball loss for the 10/50/90% rise quantiles, and 0.3-weighted auxiliary-head losses, with weight decay and early stopping.
- **Model selection:** inner folds select sizes, loss weights and freezing depth.
- **Carbohydrate constraint:** carbohydrate enters the spike logit only through a non-negative-weighted term, so predicted risk cannot rise when the carbohydrate estimate alone falls.
- **Branch disagreement:** the gap between the auxiliary heads is kept as an abstention signal only if it improves the risk–coverage curve.

**Budget.** About 27M pretrained or fine-tuned parameters (limit 1B) and under 2M trained from scratch. The latency target is a 95th percentile (p95) of ≤300 ms, offline, in PyTorch, in arm64 Docker on the author's MacBook Pro CPU (4 threads, chip logged).

## 3. The data and learning plan

**CGMacros** (PhysioNet) covers 45 adults (15 healthy, 16 prediabetes, 14 type 2 diabetes), each for about 10 days. It includes:
- Dexcom and Libre CGM and Fitbit heart rate;
- before-meal and after-meal photos, and labs;
- macros estimated for the *consumed* meal, plus a percentage consumed.

**Nutrition5k** has 5,006 cafeteria plates, with overhead images for a subset and carbohydrate, protein and fat labels but no fibre (to be confirmed on download). Transfer from these photos to phone photos is evaluated.

**Reference macros.** The before-photo shows the served plate, so the reference is consumed macros ÷ share eaten, after normalising the inconsistently coded share field.
- This assumes **proportional consumption**, which is false for mixed meals: uneaten rice and eaten chicken differ.
- Shares below 10% are excluded, and field semantics are audited on photo pairs.
- Only 8.8% of meals are partly eaten, so a **fully-consumed-only sensitivity analysis** tests the assumption.
- A consumed-macro model is a *retrospective comparator*, not a guaranteed upper bound.

**Inputs and label.** Inputs are only the before-photo, CGM and heart rate up to the meal, and profile, labs and time. Share eaten, after-photos and consumed macros are never inputs.

The label is a Dexcom peak rise of at least 50 mg/dL over the 30-minute pre-meal mean within 2 hours. This is an operational definition, so 30 and 70 mg/dL are also reported. A window is **usable** when:
- the baseline has at least 3 readings;
- at least 70% of post-meal readings are present;
- no gap exceeds 20 minutes, since a missed peak can hide a spike.

A stricter rule (≥90% present, no gap over 10 minutes) is also reported.

**Pilot accounting (final rules):**
1. 1,640 meals had ≥70% of post-meal readings; 1,526 also passed the baseline, gap and share rules.
2. Removing 453 onboarding meals and 19 whose window crossed the 72-hour boundary left 1,054.
3. Removing 158 meals followed by another meal within 2 hours left **896** (634 from the 31-person **target cohort**, healthy plus prediabetes).

Spike base rates are 42.0% overall and 35.2% in the target cohort. Overlapping meals are also reported separately.

| Set | People | Use |
|---|---|---|
| Development | 23 target + 14 type 2 diabetes (training only) | Nested person-grouped cross-validation (CV). Each outer fold fits model, calibrator, conformal correction and thresholds on **its own training participants only**. Outer predictions are pooled only for reporting. |
| **Locked final set** | 8 target (4 healthy, 4 prediabetes) | IDs frozen on 27 September (seeded draw, SHA-256 hash). The final pipeline uses development people, with a fixed calibration subset. |
| Emirati stress set | 30 dishes (to be collected) + 30 in-distribution photos | Exploratory OOD check. Threshold set on development data at 95% in-distribution acceptance. |

**About the locked set.** It was in the exploratory pilot, so its results are *prespecified final analyses with prior exposure disclosed*. It is internal evidence only, and 8 people cannot establish broad generalisation. It evaluates new users after onboarding, and only their personal prior updates.

**Leakage and duplicates.**
- Every learned stage shares the same person exclusions.
- Fusion trains on **out-of-fold photo predictions**.
- Near-duplicate photos will be removed.
- For the unseen-meal-type test, meals with identical logged macros form one type and are held out together.
- A 9-photo spot check found 3–4 packaging or drink photos; these will form a named hard-example set.

## 4. Scope, evidence, and success

**Deliverables:**
- an offline containerised Gradio app (risk card and slider, abstention, review queue, monitoring);
- W&B logs, incremental git history and a reproducibility package;
- the pilot scripts, logs and locked-set file, which are saved and will be committed first.

**Out of scope:** dosing, hypoglycaemia, causal claims, substitution advice and live CGM integration.

**Stretch goals:** a text-note branch, a pretrained time-series encoder, self-supervised pretraining and int8 quantisation.

**Exploratory pilot.** A gradient-boosted tree (GBM) on macros, pre-meal glucose, engineered trends, heart rate, labs and the prior. AUROC on held-out people (5-fold × 3 seeds). These results include the locked set, so they are optimistic.

| Meal information | All 45 | Target cohort (31) |
|---|---|---|
| Consumed macros (retrospective comparator) | 0.786 | 0.725 |
| Served macros (reconstructed reference) | 0.783 | 0.727 |
| Served macros, fully consumed meals only | — | 0.721 |

Under the same rules, a small from-scratch network (sequence, trends and a monotone carbohydrate head) scored 0.722 on target-cohort meals, against 0.734 for the GBM (paired difference −0.013, 95% CI −0.051 to +0.028). Their ensemble scored 0.740. A learned-only sequence encoder trailed engineered trends (−0.021, CI −0.042 to +0.005), so RQ2 is genuinely open.

**Prespecified analyses** use paired person-level bootstrap intervals. A pilot precision simulation, adding 25% error to macros, gave Δ ≈ 0.00 with a 95% CI about 0.04 wide across target people. For 8-person sets the CI was about 0.06 wide and inconclusive in ~40% of draws. So the **primary** analysis for both research questions uses nested CV on the development target people, and the locked set is a secondary check.

**RQ1.** Δ = AUROC(photo-predicted served macros) − AUROC(reconstructed served macros). Sensor, profile and prior inputs are identical, and there is no image embedding.
- **Non-inferior** if the lower 95% bound exceeds −0.03.
- **Inferior** if the upper bound is below −0.03.
- **Inconclusive** otherwise.

Carbohydrate mean absolute error (MAE) and bias are reported alongside. The image embedding is a separate experiment.

**RQ2.** Paired ΔAUROC of the learned sensor encoder vs engineered trends, with other inputs identical.

**Exploratory targets** (target cohort; counts and denominators reported):
- **Discrimination:** fusion AUROC ≥ a *fair GBM* given the same photo-predicted macros, engineered sensor features, profile and prior.
- **Rise prediction:** MAE below a GBM rise regressor.
- **Calibration:** expected calibration error (ECE) ≤ 0.05 with 15 equal-mass bins, plus reliability diagrams.
- **Rise intervals:** *participant-weighted split-conformal quantile regression*, where each calibration person's scores are weighted 1/(their meal count). No coverage guarantee is claimed under within-person dependence. Pooled coverage targets 75–85% at 80% nominal, and person-averaged coverage and width are reported.
- **Usefulness:** low-risk, caution and abstain proportions are reported separately. The system is useful only if:
  - expected cost is ≥10% below the better of *always caution* and *always low risk*;
  - ≥20% of meals are cleared as low risk, with ≤10% of cleared meals spiking;
  - abstention is ≤20%.

  Zero cleared meals counts as failure.

**Cost assumptions** (spike / no spike): low risk 5 / 0, caution 0 / 1, abstain 2 / 0.5. The sweeps vary the missed-spike cost from 3 to 10 and scale both abstain costs by a common 0.5–4× multiplier. Thresholds minimise expected cost on inner folds.

**Formative checks:**
- A 5-person think-aloud test checks that users can tell abstain from low risk. It is formative, not a validation.
- Latency is measured on the stated benchmark.

**Explainability and robustness:**
- **Probes:** swap the photo with the sensor state fixed; compare flat vs rising traces at equal current glucose; mask one branch at a time.
- **Failure taxonomy:** by photo type, health group and trend.
- **Stress tests:** blur and angle, CGM gaps, a switch from Dexcom to Libre, missing heart rate, and a sensor-free mode.

There is no overlap with MAAI7102 (1–5-year data-centre power forecasting).
