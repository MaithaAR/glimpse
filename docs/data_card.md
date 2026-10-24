# Data card — GLIMPSE

Numbers below come from `glimpse prepare` (seed 20260927). The generated files in `outputs/data/` are the source of truth.

## Source, provenance and licence
- **CGMacros v1.0.0**, PhysioNet open access ([link](https://physionet.org/content/cgmacros/1.0.0/)), licence **CC BY-NC-SA 4.0** (non-commercial, share-alike).
- 45 adults (15 healthy, 16 prediabetes, 14 type 2 diabetes by A1c) followed for about 10 days: Dexcom G6 CGM (5-min), Fitbit heart rate, before-meal photos, logged macros with share eaten, and lab tests.
- `just install` downloads the participant CSVs and the 1,486 before-meal photos listed in `splits/photo_list.txt` straight from PhysioNet's online archive (resumable). **No participant data or photos are committed.**

## What one example is
One meal: the **2 hours of glucose and heart rate before it** (24 five-minute slots), 8 profile inputs (age, BMI, A1c, fasting glucose, meal hour as two clock values, personal spike rate and mean rise from the person's first 72 h) and the before-meal photo.
**Label:** spike = peak rise of ≥ 50 mg/dL above the 30-min pre-meal mean within 2 h (30 and 70 kept for sensitivity checks).
**Photo label:** served carbohydrate = logged carbohydrate ÷ share eaten (noisy).

## Cleaning and preprocessing
| Step | Rule | Effect |
|---|---|---|
| Share eaten | < 10% dropped (served macros unreliable) | applied first |
| Usable label window | ≥ 3 baseline readings, ≥ 70% post-meal coverage, no gap > 20 min | 1,526 usable meals |
| Schema (`MealRecord`, `MealSample`) | known group, label in {0, 1} and consistent with the rise, plausible glucose and coverage | 0 rejected |
| Pre-meal checks (same rule the product applies) | ≥ 3 readings in the last 30 min, ≥ 70% slots filled, no gap > 20 min | 20 meals flagged (kept, never used for training or scoring) |
| Window | readings averaged into 24 five-minute slots, **before the meal only** | 2.2% of windows have an empty slot |
| Gap filling | straight line between neighbouring readings; nearest reading at the edges | glucose 0.65%, heart rate 5.2% of slots |
| Scaling | z-score with **training-people** means and spreads; unknown profile values → training mean; missing-glucose flag channel | `checkpoints/normaliser.json` |
| Augmentation (training batches only) | random noise: 3 mg/dL glucose, 1 bpm heart rate, on real readings only | stored data never change |

Out-of-range glucose (outside 40–400 mg/dL): 0 readings.

## Splits (by person)
| Group | People | Training meals | Scored meals | Spike rate (scored) |
|---|---|---|---|---|
| Train (14 target + 14 type 2 diabetes) | 28 | 633 | 547 | 0.48 (target 0.39, T2D 0.59) |
| Validation (target) | 4 | 133 | 99 | 0.19 |
| Test = outer fold 1 (target) | 5 | 125 | 96 | 0.29 |
| Locked (target, sealed) | 8 | 181 | 153 | — not used |
| 5 outer CV folds (23 development target people) | 23 | | **481** | 0.24–0.51 per fold |

- **Main estimate:** person-level 5-fold cross-validation over the 23 development target people. Every one of them is tested once (481 scored meals). The fixed split is the rubric's held-out split and the demo's split.
- **Rates differ** between groups because people differ (per-person spike rate 0 to 0.84) and type 2 diabetes people, who spike more, train only. Results are therefore reported for the target cohort, with base-rate-aware measures and calibration. 4 validation people (99 meals) is a noisy basis for choosing settings.

## Leakage controls
- People never cross groups (checked in code). The 8 locked people are verified by SHA-256 and never touched by `glimpse prepare` or training.
- Scaling statistics and missing-value fills come from training people only. The personal prior uses only a person's own first 72 h.
- Inputs end at the meal time; a test changes every after-meal reading and checks the input is identical.
- Photos: 1,221 development photos checked with exact (SHA-256) and look-alike (64-bit dHash, ≤ 4 bits) fingerprints: **0 cross-split duplicates**. With person-level splits this can only catch a file or ID mix-up. Within a person, look-alike photo pairs are rare (rate 0.0001), so repeated meals are not simply the same picture.

## Fine-tuning ledger (project cap 10,000 items; item = one photo used to fine-tune)
| Dataset | Stage | Items |
|---|---|---|
| CGMacros before-meal photos, fixed training people | intermediate | 865 |
| CGMacros before-meal photos, all development people (worst case over CV folds) | intermediate | 1,221 |
| Nutrition5k plates (optional pre-training) | final, planned | 0 |
| **Running total (worst case)** | | **1,221** |
`glimpse prepare` stops with an error if the total exceeds the cap.

## Modalities
- **Sequential:** CGM glucose and heart rate (this step).
- **Tabular:** profile and personal prior (this step).
- **Images:** before-meal photos, joined through the photo model (served-carbohydrate range and estimate) with its own preprocessing (the backbone's resize and normalisation, whole plate kept) and augmentation (flip, ±10% brightness/contrast, crops keeping ≥ 80%; training only).

## Glossary
- **Onboarding meals:** a person's first 72 h. They build the personal prior and are never scored.
- **Isolated:** no other meal within 2 h, so the rise belongs to one meal.
- **Scored meals:** isolated meals that pass the pre-meal checks (used for validation and test).
- **Training meals:** all meals after onboarding that pass the pre-meal checks.
- **Flagged:** fails the pre-meal checks; the product would answer “can't judge” instead.

## Known limitations
45 people from one study; served macros assume proportional eating; no local dishes; small validation and test groups give wide confidence intervals.
