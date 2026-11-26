# External check on a second dataset, 27 Nov 2026 (Part 1: plan and results share one commit; Parts 2 and 3: the plan commit precedes the result)

Exploratory, planned and run on the day of the deadline; not a pre-registered study. The served model (v1)
is unchanged and cannot be scored here: the second dataset has no meal photos, and v1 refuses a meal without
one. What can be scored, exactly as frozen on 26 Nov, is the sensor-first model v2
(checkpoints/sensor_first/): its fusion weights, Platt map, m = 1.0 intercept and cut-offs 0.15 and 0.22.

Why: the report's only clean test has 8 people. A second public dataset gives 16 people no GLIMPSE model has
seen, from another study, with another heart-rate device and a self-kept food log.

Data: BIG IDEAs Lab Glycemic Variability and Wearable Device Data, PhysioNet v1.1.2 (open access). 16 adults,
HbA1c 5.3 to 6.4 (inside the product's scope), Dexcom G6 for 8 to 10 days, Empatica E4 heart rate, a food
log with carbs. `just external-bigideas` needs data/bigideas/ (README has the download lines); the data are
not redistributed in this repository.

Fixed before scoring:
- Meal: every distinct food-log time with calories above zero. Windows, the label (rise of at least 50 mg/dL
  above the 30-minute pre-meal mean within 2 hours), usability (post-meal coverage at least 0.70, no gap over
  20 minutes), the pre-meal checks and "isolated" (no other logged time in the next 2 hours) are computed by
  the same code and config as for CGMacros.
- Onboarding: the first 72 hours from the person's first labelled meal give the two onboarding priors (at
  least 5 meals, as in the config); those meals are not scored.
- Inputs v2 needs and this dataset lacks: age, BMI and fasting glucose. They are left missing and the frozen
  pipeline's own imputer fills them with CGMacros training medians. HbA1c is given. This is a handicap for
  the model and is stated with the result.
- Only the model is scored. The product's refusal rules are not applied beyond the window checks (rule R5
  would refuse a profile without age).
- History for the intercept: each person's earlier scored meals whose 2-hour label has closed, in time order.

Reported, whatever it shows: pooled AUROC of v2's risk with a 95% person-bootstrap interval (2,000 draws);
the same for v2 without the intercept, for a running spike rate per person and for the logged carbs alone;
mean within-person AUROC; Brier score and ECE (15 bins); and at the frozen cut-offs the share cleared, the
spikes among cleared meals, the share referred and the cost reduction against always caution.

Expectation, written now: AUROC between 0.60 and 0.70, below development (0.754), because of the missing
profile fields, the different devices and free-living meals; usefulness targets not met.

## Result (27 Nov, run once on loading the data; outputs/external_bigideas/external_bigideas.json)

Found on loading, before any score: for 4 of the 16 people (7, 13, 15, 16) the food-log dates in the public
files do not overlap their CGM dates, so they have no usable meal (food log against CGM: person 7, 16 to 24 Oct
2019 against 14 to 22 Mar 2020; 13, 30 Dec 2019 to 6 Jan 2020 against 28 May to 4 Jun 2020; 15, 17 to 25 Feb
against 19 to 27 Jul 2020; 16, 21 to 28 Feb against 16 to 24 Jul 2020). No date shift was guessed. 12 people
remain, with 531 usable meals and 201 scored ones (after onboarding, isolated, pre-meal checks passed). Spike
rate 25.4%. All 12 have onboarding priors. Heart rate is missing in 14% of scored windows.

| on 201 meals of 12 unseen people | AUROC (95% over people) | within-person AUROC (11 people) |
|---|---|---|
| v2 as frozen (fusion, Platt, intercept) | 0.628 (0.535–0.704) | 0.473 |
| v2 without the intercept | 0.624 (0.563–0.676) | 0.620 |
| running spike rate per person, no model | 0.553 (0.415–0.652) | – |
| logged carbs alone | 0.734 (0.638–0.818) | 0.725 |

Calibration of v2 as frozen: Brier 0.185 against 0.189 for a constant forecast; ECE 0.053; mean risk minus
spike rate +0.005.

Decisions at the frozen cut-offs (0.15 and 0.22): 16.9% cleared (34 meals, 5 spiked: 14.7%), 20.4% referred,
cost reduction against always caution 3.0% (−13.8%, +22.3%).

Reading.
- The expectation written before scoring (0.60 to 0.70, targets not met) held. The interval excludes 0.5.
- The external AUROC, 0.628 on 12 new people, is close to the locked test's 0.645 on 8 (that was v1; v2 has
  no locked number). Two independent small tests now put the honest discrimination of this family of models
  between 0.62 and 0.65, well under the development 0.75 to 0.76. They are different models (v1 and v2).
- The mean risk carried over to a different study, device and food log: it is 0.5 points above the spike
  rate. ECE is 0.053, just above the 0.05 target, and no noise floor was computed for this set.
- The intercept did not help ranking here (0.628 against 0.624) and again scrambled the order of a person's
  own meals (0.473 against 0.620).
- The carbs the person logged, alone, rank meals better than the whole model (0.734). Free-living logs
  include many low-carb entries, which CGMacros' standardised meals do not, so carbs separate meals far more
  here than the +0.03 they add on CGMacros.
- At the frozen cut-offs the model is not useful on these people: it misses 14.7% of what it clears and its
  cost saving is not distinguishable from zero.

Limits: 12 people; age, BMI and fasting glucose imputed; another wrist device; a self-kept food log with
timing errors; only the model is scored, not the product with its refusal rules; v1, the served model,
cannot be scored without photos.

## How the two datasets are kept apart, and how they are matched

Nothing is combined. No model was trained, tuned or calibrated on BIG IDEAs data. The fusion weights, Platt
map, intercept strength and cut-offs are the files frozen on 26 Nov from CGMacros alone, and BIG IDEAs appears
in no CGMacros table of the report.

| aspect | CGMacros (training) | BIG IDEAs (external) | risk | how it is handled |
|---|---|---|---|---|
| glucose sensor | Dexcom G6 Pro | Dexcom G6 | small | same resampling to 5-minute means, same code |
| heart rate | Fitbit, per minute | Empatica E4, 1 Hz reduced to 5-minute means | device bias | mean of the scored windows 81.3 bpm against 82.4; no window outside the training range |
| meal times | logged in the study app with a photo | self-kept food log | a clock or logging offset would corrupt windows and labels | checked: for each of the 12 people the mean glucose curve is flat before the logged time (+0.7, +0.1, −0.5 mg/dL at −60, −30, 0 min) and peaks 30 to 70 minutes after it (pooled +20 mg/dL at 45 min) |
| what counts as a meal | a logged meal with calories | every distinct food-log time with calories, snacks included | more small meals, lower spike rate (25.4% against 32.8%) | stated; the label rule and the "isolated" rule are the same code |
| label | rise of at least 50 mg/dL over the 30-minute pre-meal mean within 2 hours | the same | none | same function and config |
| age, BMI, fasting glucose | measured | absent | the model loses three profile inputs | left missing; the frozen imputer fills CGMacros medians; stated as a handicap |
| HbA1c | lab | given, 5.3 to 6.4 | none | used as given |
| pre-meal glucose | last reading before the meal 124.6 mg/dL on average (cohort includes type-2 people) | 109.7 mg/dL | a lower, narrower range | inside the training range; stated |
| onboarding priors | first 72 hours | first 72 hours, same rule | 25% of scored meals have a prior rise outside the CGMacros range (24 to 115 mg/dL) | stated; the sensor inputs are inside the training ranges (under 1% of meals outside) |
| people in the cohort | 45, of whom v2 trained on 37 | 16, of whom 12 usable | 4 people lost to a date mismatch in the public files | their date ranges are listed above; no shift was guessed |

These checks are in outputs/external_bigideas/external_bigideas.json under `matching_checks`.

## Part 2, planned on 27 Nov before it was run: does meal content help on free-living meals?

Why. On CGMacros the logged carbs add at most about +0.03 AUROC to the sensor and profile model, which is why
the photo, whose job is to supply the carbs, could not pass its gate. CGMacros meals are largely
standardised. In Part 1 the carbs people logged ranked BIG IDEAs meals at 0.734 on their own. If meal
content adds little on standardised meals and a lot on free-living ones, the photo channel's low ceiling is
a property of the training data, not of the design.

What is run, once:
- Two logistic models are fitted on the 37 non-locked CGMacros people with v2's recipe and v2's own C (no
  tuning): the 15 sensor and profile inputs, and the same plus log(1 + logged carbs). The first must
  reproduce the frozen v2's raw scores on BIG IDEAs to 1e-6, or the run stops.
- Both are scored on the 201 BIG IDEAs meals of Part 1, without the intercept (raw ranking) and with the
  frozen Platt map and intercept replay.
- Endpoint: AUROC with carbs minus without, paired bootstrap over the 12 people, 2,000 draws.
- A simulation of photo-quality carbs: the BIG IDEAs logged carbs are perturbed with errors drawn from the
  fine-tuned photo model's own out-of-person errors on CGMacros (log scale, outputs/photo_cv/photo_oof.csv),
  200 times; the mean gain is reported. This is a simulation, not a photo result: BIG IDEAs has no photos.

Expectation: a gain of +0.03 to +0.10 with carbs; about half of it left under photo-sized errors. Whatever it
shows is reported. Nothing is refitted on BIG IDEAs, and the served model does not change.

## Part 2 result (27 Nov, run once; outputs/external_bigideas/external_part2.json)

The refitted sensor and profile model reproduces the frozen v2 exactly on these meals (largest logit
difference 0.0), so the comparison isolates the carb input.

| 201 meals, 12 people; both models fitted on CGMacros only | without carbs | with logged carbs | gain (95% over people) |
|---|---|---|---|
| raw ranking, no intercept: AUROC | 0.624 | 0.749 | +0.126 (+0.060, +0.188) |
| the same, within-person AUROC | 0.620 | 0.780 | – |
| with the frozen Platt map and intercept: AUROC | 0.628 | 0.720 | +0.093 (+0.053, +0.148) |
| carbs perturbed with the photo model's own CGMacros errors (log-scale SD 0.81), 200 draws | 0.624 | – | mean +0.101 (+0.067 to +0.143 across draws) |

Reading.
- On free-living meals the carb input adds +0.126 AUROC, with an interval well above zero, to a model that
  never saw these people or this dataset. On CGMacros the same input adds about +0.03 at most. The
  expectation (+0.03 to +0.10) was exceeded.
- On this one external dataset meal content is worth far more than on CGMacros. Cautions: 12 people and one
  run; every food-log entry with calories counts as a meal, so carbs partly separate snacks from meals; Part 2
  was planned after Part 1 had shown carbs alone at 0.734; the CGMacros +0.03 is a cross-validated figure, so
  the two gains are not measured the same way; and the model with carbs (0.749) adds only 0.015 to the carbs
  alone (0.734).
- The simulation says carbs with photo-sized errors would keep most of the gain. It draws those errors
  independently of the meal, which real photo errors are not, and its range is a spread across draws, not an
  interval over people. It is a simulation: BIG
  IDEAs has no photos, the errors are borrowed from CGMacros photos, and photos of free-living meals may be
  harder to read. It is not evidence that the photo model works.
- The intercept again costs ranking: with it the carb model scores 0.720, without it 0.749.

Limits as in Part 1, plus: the carbs are self-logged; one model pair, one run, no multiplicity correction
across Parts 1 and 2.

## Part 3, planned on 27 Nov before it was run: do the carb-aware model's decisions work on the external people?

Why. Parts 1 and 2 scored ranking. The product is a decision: clear, caution or refer. v2's decisions were
not shown useful on the external people (Part 1). Part 2 found that a carb-aware model without the
intercept ranks them much better. The open question is whether that turns into decisions that meet the
product's own targets on people it has never seen. This is the third look at the same 12 people, which is
stated with the result.

Fixed before the run, all from CGMacros only:
- Model: the Part 2 model with logged carbs (v2's recipe and C, fitted on the 37 non-locked CGMacros people).
- Platt map: fitted on that recipe's out-of-fold scores for the 23 development people (five outer folds).
- Cut-offs: the pair (low, high) with the lowest declared cost (5 for a missed spike, 1 for a needless
  caution, 2 or 0.5 for a referral) on those out-of-fold risks, grid 0.05 to 0.60 in steps of 0.01, low at or
  below high. No per-person intercept. Nothing is chosen on BIG IDEAs.

Scored once on the 201 external meals, against the product's four targets (fixed in the proposal): cost
reduction against always caution at least 10%, at least 20% of meals cleared, at most 20% referred, at most
10% of cleared meals spiking. Person-bootstrap intervals, 2,000 draws. Brier, ECE and mean risk minus spike
rate are reported too, and the same recipe's out-of-fold numbers on CGMacros for reference.

Protocol labels as in the report: met only if the whole interval meets the target, on track if the point
does, at risk otherwise.

Expectation: more meals cleared than v2 (16.9%) and a positive cost reduction with an interval that still
includes zero; at most two targets on track. Reported whatever it shows. The served model does not change.

## Part 3 result (27 Nov, run once; outputs/external_bigideas/external_part3.json)

From CGMacros alone: Platt map (1.20, −0.09) and cut-offs 0.13 (low risk below) and 0.20 (caution from).

| target (fixed in the proposal) | carb-aware model, 201 external meals of 12 people | label | the same recipe out of fold on CGMacros (481 meals) | v2 as frozen, external (Part 1) |
|---|---|---|---|---|
| cost reduction against always caution, at least 10% | 16.3% (7.9% to 27.4%) | on track; the interval is above zero | 9.6% (−7.7% to 24.6%) | 3.0% (−13.8% to 22.3%) |
| meals cleared, at least 20% | 10.4% (4.1% to 18.1%), 21 meals | at risk | 12.1% | 16.9% |
| meals referred, at most 20% | 14.4% (8.5% to 21.0%) | on track | 15.2% | 20.4% |
| cleared meals that spiked, at most 10% | 4.8% (0.0% to 12.5%), 1 of 21 | on track | 10.3%, 6 of 58 | 14.7%, 5 of 34 |

AUROC 0.749 (Part 2). Brier 0.167 against 0.189 for a constant forecast. ECE 0.089, mean risk 3.3 points
above the spike rate: calibration is worse than v2's (0.053, +0.5), because the Platt map comes from a
cohort that spikes more often.

Reading.
- On this one dataset, and on a third look at the same 12 people, the carb-aware model's decisions save cost
  with an interval that excludes zero: 16.3% against always caution, and 1 of its 21 clears spiked. Three of the four targets are on
  track; none is met by the protocol's whole-interval rule.
- It is cautious: it clears 10.4% of meals, half the target.
- The expectation (more clears than v2, a cost interval through zero, at most two targets on track) was
  wrong in the model's favour on cost and wrong against it on clears.
- What differs from v2 is two things the audits pointed to: the carbs are an input, and there is no
  per-person intercept.

Limits. Third look at the same 12 people, with no correction for that. The carbs are the person's own log,
not a photo estimate; with photos the gain would be smaller by an unknown amount. 21 cleared meals is a
small count. Model only: the product's refusal rules are not applied. The served model is unchanged.

## Follow-up checks (27 Nov, after a grader's questions; outputs/external_bigideas/external_checks.json)

Run after Parts 1 to 3 were known, on the same 201 meals, so they are descriptive.

- Does the intercept change the ranking? AUROC with it minus without it: +0.004 (−0.097 to +0.110). Not shown
  either way.
- How many people does the frozen v2 rank below chance on their own meals? 6 of the 11 with both outcomes.
- Is the carb gain spread over people? 10 of 11 rank better with carbs than without.
- Is the carb gain only snacks against meals? Dropping every food-log entry under 200 kcal leaves 148 meals
  (spike rate 33.1%). The gain is +0.081 (+0.004 to +0.155), smaller and still above zero; carbs alone score
  0.632 there. So part of the +0.126 is carbs telling snacks from meals, and part is not.
- Part 1's plan and results share one commit; the plans of Parts 2 and 3 were committed 7 and 5 minutes before
  their results.
