# GLIMPSE: Does a Meal Photo Preserve the Glucose-Spike Signal? A Pre-Meal Risk Card from Photo, Sensor and Profile Data

**Team:** Maitha Alhosani (solo) · **Course:** MAAI7103 Deep Learning · **Proposal date:** 27 September 2026

Before eating, a person with prediabetes photographs her plate. GLIMPSE (*Glycemic-response Inference from Meal Photos and Sensor Evidence*) combines the photo with the last two hours of her continuous glucose monitor (CGM) and heart-rate data, plus her profile. It returns a calibrated risk of a large glucose spike, or abstains when it cannot judge.

The project asks two research questions:
- **RQ1:** Does a photo recover the predictive value that logged meal information adds beyond sensor and profile data?
- **RQ2:** Does a learned pre-meal sensor representation beat engineered trend features?

## 1. The venture and its user

**Problem.** Some prediabetes and wellness programmes give members a CGM for a few weeks, but a CGM shows a spike only after eating, too late to change the choice. The user is an adult with prediabetes or a wellness member. The buyer is the programme provider, such as an employer or insurer wellness scheme.

**Differentiation.** January AI already predicts glucose responses from food photos without a sensor. GLIMPSE's untested bet is that live pre-meal glucose, calibration and explicit abstention make that guidance more trustworthy.

**Scenario.** At lunch, Mariam photographs rice with chicken. Her glucose has been rising since a mid-morning coffee with dates. The **risk card** shows *caution*: a high spike probability, a predicted-rise interval, and the observed inputs behind it (a large carbohydrate estimate and a rising trend), shown as context rather than causes. A **carbohydrate-sensitivity slider** shows how the *model's* prediction changes with less carbohydrate. It displays model sensitivity, not dietary advice.

**Journey.**
- **Onboarding.** The user enters a profile; labs are optional, with missing labs modelled and evaluated separately. Meals whose windows end within the first 72 hours set a personal prior (spike rate and mean rise). If there are fewer than 5 such meals, a population prior is used instead. The 72-hour wait is acknowledged friction.
- **Each meal.** GLIMPSE takes one action:
  - *low risk*;
  - *caution*;
  - **abstain** ("cannot judge this meal", plus generic guidance). This covers an unreadable or unfamiliar photo, a CGM gap, or low confidence.
- **Learning.** Abstained meals and a 5% audit go to a dietitian **review queue**. The target is ≤0.5 reviews per user per day, assuming about 2 minutes per review.
- **Feedback and monitoring.** Users can correct carbohydrate estimates, and completed CGM windows label each prediction. **Monitoring** tracks confidence, abstention, calibration drift, out-of-distribution (OOD) rate and latency.

**Boundaries.** There are no dosing, hypoglycaemia, causal or health-benefit claims. CGMacros has no medication data, so excluding insulin and sulfonylurea users cannot be verified. The licence (CC BY-NC-SA) limits use to coursework.

## 2. The system concept

Three modalities each have their own encoder: the **before-meal photo**, a **CGM and heart-rate series** (24 five-minute steps), and **tabular context** (profile, labs, time of day).

| Component | Model | Params | Notes |
|---|---|---|---|
| Photo → served carbs, protein, fat (**fine-tuned**) | EfficientNet-B0 | ~5M | Nutrition5k overhead subset (<5,006), then CGMacros development before-photos (<1,640); fibre is added at the CGMacros stage |
| Sensor encoder (**from scratch**) | 1-D CNN | 0.1–1M | The smallest size is a serious candidate |
| Fusion (**from scratch**) | MLP | <1M | Spike logit, rise quantiles, and meal-only and sensor-only auxiliary heads |
| OOD embedding (**frozen**) | DINOv2-S | 22M | Photo distance to training data |

**Training.**
- **Image model:** Huber loss on log-macros. The backbone is frozen first, then its last blocks are unfrozen.
- **Fusion:** cross-entropy for the spike, pinball loss for the 10/50/90% rise quantiles, and 0.3-weighted auxiliary losses, with weight decay and early stopping.
- **Model selection:** a small, fixed set of inner-fold configurations chooses sizes, loss weights and freezing depth.
- **Carbohydrate path:** carbohydrate enters the spike logit only through a non-negative weight, so predicted risk cannot rise when the carbohydrate estimate alone falls.
- **Abstention signals:** head disagreement and DINOv2 distance are kept only if they beat a rule based on the fused probability alone.

**Budget.** About 27M pretrained or fine-tuned parameters (the limit is 1B) and under 2M trained from scratch. The latency target is p95 ≤300 ms, offline, in PyTorch inside arm64 Docker on the author's MacBook Pro CPU.

## 3. The data and learning plan

**Datasets.**
- **CGMacros** (PhysioNet) follows 45 adults (15 healthy, 16 prediabetes, 14 type 2 diabetes) for about 10 days. It records Dexcom and Libre CGM, Fitbit heart rate, before- and after-meal photos, labs, and macros for the *consumed* meal along with the percentage consumed.
- **Nutrition5k** provides 5,006 cafeteria plates, with overhead images for a subset. It has carbohydrate, protein and fat labels but no fibre (per its repository; to be confirmed). How well it transfers to phone photos is evaluated.

**Reference macros are noisy supervision.** The reference is consumed macros ÷ share eaten.
- This assumes proportional consumption, which may fail when particular components are left uneaten.
- Labels are unusable if the share is below 10%, the photo is missing, or a photo-pair audit finds a mismatch.
- Only 8.8% of meals are partly eaten, so an analysis restricted to fully consumed meals tests the assumption.
- Consumed macros serve only as a retrospective comparator.

**Inputs and label.** Inputs are only the before-photo, CGM and heart rate up to the meal, and profile, labs and time. The label is a Dexcom peak rise of at least 50 mg/dL above the 30-minute pre-meal mean, within 2 hours. This is an operational definition, so thresholds of 30 and 70 mg/dL are also reported.

A window is usable when it has:
- at least 3 baseline readings;
- at least 70% of post-meal readings;
- no gap over 20 minutes.

A stricter rule is also reported.

**Evaluation populations.**
- **Isolated-meal cohort (RQ1, RQ2 and pilot):** meals with no other meal within 2 hours and a usable photo.
- **Operational cohort:** *all* eligible meal events, including those followed by further eating, because the app cannot foresee them. Their outcome is the observed two-hour response, not a response attributed to the first meal. Missing or unusable photos count as abstentions.

**Pilot counts.** 1,526 meals pass the window rules. Removing 453 onboarding meals and 19 that cross the 72-hour boundary leaves 1,054 operational meals. Of these, 896 are isolated meals, and 634 of those come from the 31-person **target cohort** (healthy plus prediabetes), which has a 35.2% spike rate.

| Set | People | Use |
|---|---|---|
| Development | 23 target + 14 type 2 diabetes (training only) | **Primary analyses** using nested person-grouped cross-validation (CV). Each outer fold uses only its own training people. Inner 4-fold cross-fitting fits the temperature, conformal quantile and cost thresholds, which are then applied to the refitted fold model. Whether this calibration transfers is tested empirically on the outer fold. |
| Locked final set | 8 target (4 healthy, 4 prediabetes) | Secondary check only. IDs were frozen on 27 September (seeded, with a SHA-256 hash). They appeared in the pilot, so results are internal evidence only. |
| Emirati stress set | 30 dishes (to be collected) + 30 in-distribution photos | Exploratory visual-OOD check. The threshold accepts 95% of development photos. |

**Leakage controls.**
- Every learned stage uses the same person exclusions.
- Fusion trains on **out-of-fold photo predictions**.
- Near-duplicate photos are removed.
- A *macro-profile group* test holds out meals with identical logged macros together. This tests macro profiles, not recipes, because the data has no food identifiers.

**First feasibility gate (week 4).** Carbohydrate mean absolute error (MAE) and bias on at least 100 audited CGMacros photos, using person-excluded predictions only, plus a first RQ1 estimate. So far, a 9-photo spot check has only flagged packaging and drink photos as a hard category.

## 4. Scope, evidence, and success

**Deliverables.**
- An offline containerised Gradio app with the risk card, slider, abstention, review queue and monitoring.
- W&B logs, git history and a reproducibility package. The pilot scripts, logs and locked-set file are already committed.

**Out of scope:** dosing, hypoglycaemia, causal claims, substitution advice, live CGM integration.

**Exploratory pilot.** Gradient-boosted trees (GBM) were trained within each fold on that fold's training people. They were scored by AUROC on held-out isolated target-cohort meals (5-fold person CV × 3 seeds). The locked set is included, so these figures are optimistic.

| Inputs besides sensor, profile and prior | AUROC |
|---|---|
| None (no meal information) | 0.660 |
| Reconstructed served macros | 0.741 |
| Consumed macros (retrospective) | 0.740 |
| Served macros, fully consumed meals only | 0.741 (no-meal: 0.665) |

Meal information adds **+0.081** (paired person-bootstrap 95% CI +0.050 to +0.122). Two comparisons were inconclusive:
- a from-scratch network vs the GBM: −0.013 (CI −0.051 to +0.028);
- learned vs engineered sequence features: −0.021 (CI −0.042 to +0.005).

RQ2 is therefore genuinely open.

**RQ1 (primary: development CV; secondary: locked set).** Three models share identical sensor, profile and prior inputs, with no image embedding. The only difference is the meal information they receive: *none*, *photo-predicted*, or *reconstructed* served macros. All intervals are paired person-bootstrap 95% CIs.

- **Photo vs reconstructed.** For Δ = AUROC(photo) − AUROC(reconstructed):
  - *non-inferior* if the lower bound exceeds −0.03;
  - *inferior* if the upper bound is below −0.03;
  - *inconclusive* otherwise.

  The 0.03 margin is about one-third of the *pilot* meal-information gain.
- **Photo vs none.** For G_photo = AUROC(photo) − AUROC(none):
  - *positive incremental value* only if the **lower bound exceeds zero**;
  - *degradation* if the interval lies wholly below zero;
  - *inconclusive* otherwise.

G_photo and G_reference are reported separately, alongside carbohydrate MAE and bias. The image embedding is tested separately.

**RQ2.** Paired ΔAUROC of the learned sensor encoder vs engineered trends, with other inputs identical.

**Exploratory targets** (target cohort, with counts and denominators):
- **Discrimination:** fusion AUROC ≥ a *fair GBM* given the same photo-predicted macros, sensor features, profile and prior.
- **Rise prediction:** MAE below a GBM rise regressor.
- **Calibration:** expected calibration error (ECE) ≤0.05, with reliability diagrams.
- **Rise intervals:** cross-fitted conformal quantile regression, weighting each person by 1/(their meal count). No guarantee is claimed; pooled coverage should be 75–85% at 80% nominal.
- **Usefulness** (operational cohort; zero cleared meals counts as failure):
  - cost at least 10% below the better of *always caution* and *always low risk*;
  - at least 20% of meals cleared, with at most 10% of those spiking;
  - abstention at most 20%.

**Costs** (spike / no spike) are assumptions, not demonstrated health benefit:

| Action | Spike | No spike |
|---|---|---|
| Low risk | 5 | 0 |
| Caution | 0 | 1 |
| Abstain | 2 | 0.5 |

Sensitivity sweeps vary the missed-spike cost from 3 to 10 and scale the abstain costs by 0.5–4×.

**Other evidence.**
- **Usability:** a 5-person formative think-aloud test checks that users can tell *abstain* from *low risk*.
- **Probes:** photo swaps with the sensor state fixed, flat vs rising traces at equal glucose, and branch masking.
- **Failure analysis:** failures are broken down by photo type, health group and trend.
- **Stress tests:** blur, CGM gaps, a Dexcom-to-Libre switch, missing heart rate or labs, and a sensor-free mode.

There is no overlap with MAAI7102 (1–5-year data-centre power forecasting).
