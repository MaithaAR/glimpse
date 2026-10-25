# Model card: GLIMPSE pre-meal spike warning (intermediate)

**What it does.** It predicts the chance that a meal raises glucose by ≥ 50 mg/dL within 2 h. The inputs are a
before-meal photo, the 2 h of CGM and heart rate before the meal, and a short profile. The answer is *low risk*
(< 0.143), *caution* (≥ 0.20) or *can't judge* (in between, or when an input check fails).

**Intended use.** A wellness nudge, a second look before eating, for adults with prediabetes who already wear a
CGM. **Not** a medical device. Not for type 1 diabetes, insulin dosing or treatment decisions.

**Parts.** The photo model is ConvNeXt-Tiny (27.8M parameters, pretrained on ImageNet-22k and fine-tuned on 865
CGMacros photos of the training people; 1,221 counting all CV refits); it estimates a carb range and grams. The fusion is a logistic regression over profile, CGM trend
and photo inputs. Deterministic input checks run first. The from-scratch SensorCNN is evaluated but not deployed:
it ties the trend features.

**Training data.** 28 CGMacros adults (14 healthy or prediabetes, 14 type 2 diabetes), about 10 days each, in the
USA; CC BY-NC-SA 4.0. Carb labels are "served" carbs (logged ÷ share eaten), which is noisy.

**Evaluation** (5-fold person cross-validation, 23 people; 95% ranges by person):

- **Discrimination:** AUROC 0.724 (0.659–0.796), against 0.686 without the photo.
- **Calibration:** ECE 0.052.
- **Fixed test (5 people):** AUROC 0.742, ECE 0.129 (over-predicts).
- **Decisions on the cross-validation:** 14% of meals cleared, 12% of those spiked; 11% *can't judge*.
- **Speed:** p95 282 ms on a Mac CPU in the committed demo run (`outputs/demo/demo_summary.json`); runs vary by a few ms, and one earlier run reached 303 ms, so it is close to the 300 ms target.

**Limitations:**

- Small and narrow data: 45 people from one study, mostly US foods, no local dishes.
- The model cannot tell rare from frequent spikers (CV without test people: 30% predicted vs 16% observed for rare spikers, 42% vs 54% for frequent ones), so risk is over-predicted for rare spikers (test ECE 0.129) and few meals are cleared (3% on the test). A validation-fitted recalibration made this worse on other people (`outputs/evaluation/recalibration.md`); the final uses a per-person prior.
- The photo's two outputs often disagree, so users are shown only the carb range.
- One photo-model seed (the CV of the sensor models uses 3 seeds).
- The unfamiliar-photo check alone misses some non-meal images (for example, a plain white picture); the blank-photo rule catches that case.

**Fairness and safety.**

- Age, BMI and HbA1c are inputs.
- Results by group are not yet reported (too few people).
- Errors lean toward *caution* (the safer side).
- Every *can't judge* states its reason.

**Licence.** Code is for course use. The model is trained on CC BY-NC-SA 4.0 data, so it is not for commercial use.
