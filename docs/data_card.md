# Data card — GLIMPSE

## Source and licence
- **CGMacros v1.0.0** (PhysioNet, open access; [link](https://physionet.org/content/cgmacros/1.0.0/)). Licence: CC BY-NC-SA 4.0 (non-commercial, share-alike).
- 45 adults (15 healthy, 16 prediabetes, 14 type 2 diabetes by A1c) followed for about 10 days: Dexcom G6 Pro (5-min) and FreeStyle Libre Pro (15-min) CGM, Fitbit Sense heart rate, before/after-meal photos, logged macros with share eaten, and labs.
- Downloaded by `just install` (`glimpse download-data`): participant CSVs plus the 1,486 before-meal photos listed in `splits/photo_list.txt`. **No participant data or photos are committed**; `samples/` holds synthetic inputs only.

## What one sample is
One meal: the before-meal photo, the 2 hours of Dexcom glucose and heart rate before the meal (24 five-minute steps), the profile (age, BMI, A1c, fasting glucose, meal hour) and a personal prior from the person's first 72 hours. **Label:** Dexcom peak rise ≥ 50 mg/dL above the 30-min pre-meal mean within 2 h. **Photo-model label:** served carbohydrate = logged carbohydrate ÷ share eaten (noisy).

## Cleaning and checks (`glimpse prepare`)
| Step | Rule | Effect |
|---|---|---|
| Label window | ≥3 baseline readings, ≥70% post-meal coverage, no gap > 20 min | 1,526 usable meals |
| Share eaten | share < 10% dropped (served macros unreliable) | applied before the above |
| Schema validation | every meal validated as `MealSample` (label in {0,1}, plausible rise, known group) | 0 rejected |
| Pre-meal checks | same rule the product applies at inference (coverage ≥ 70%, ≥3 readings in last 30 min, gap ≤ 20 min) | 20 meals fail; kept, flagged `premeal_ok=False` |
| Missing values | glucose window 0.7%, heart-rate window 5.2%, personal prior 1.8% | gaps interpolated within a window; unknown values imputed with the **training** mean |
| Near-duplicate photos | 64-bit dHash, Hamming ≤ 4 across splits | 0 pairs among 1,221 development photos |

## Splits (person-grouped, seed 20260927, `splits/dev_splits.json`)
| Split | People | Meals (all) | Evaluation meals* | Spike rate* |
|---|---|---|---|---|
| Train (14 target + 14 type 2 diabetes) | 28 | 954 | 561 | 0.449 |
| Validation (target cohort) | 4 | 159 | 86 | 0.291 |
| Test (target cohort) | 5 | 159 | 95 | 0.358 |
| Locked final set (never used before the final run) | 8 | 254 | 153 | 0.425 |

\*Isolated (no further meal within 2 h), after the 72-hour onboarding, passing pre-meal checks. Test and validation people come only from the target cohort (healthy + prediabetes). Five person-grouped folds over the 23 development target people support cross-validated estimates, because 5 test people give wide intervals.

## Leakage controls
- People never cross splits; the 8 locked people are excluded from everything until the final run (SHA-256-checked list).
- Scaling statistics (`checkpoints/normaliser.json`) and imputation means are fitted on training people only, then reused unchanged for validation, test and the demo.
- Sensor windows end at the meal time; nothing after the first bite is a model input. Consumed macros and share eaten are used only to build the served-carbohydrate label.
- The personal prior uses only the person's own first 72 hours, which precede all evaluated meals.

## Fine-tuning item ledger (project cap 10,000)
| Dataset | Stage | Items |
|---|---|---|
| CGMacros before-meal photos (training people, with carbohydrate label) | intermediate | 922 |
| Nutrition5k overhead photos | final, optional | 0 (not used yet) |
| **Total** | | **922** |

## Known limitations
45 people from one study and one cuisine; served macros assume proportional eating; a 5-person test split gives wide confidence intervals; packaging and drink photos are hard to read.
