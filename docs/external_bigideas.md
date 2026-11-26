# External check on a second dataset, 27 Nov 2026 (plan written before any scoring)

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
files do not overlap their CGM dates, so they have no usable meal. No date shift was guessed. 12 people
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
  in the low 0.60s, well under the development 0.75 to 0.76.
- Calibration carried over to a different study, device and food log: the mean risk is within half a point
  of the spike rate.
- The intercept did not help ranking here (0.628 against 0.624) and again scrambled the order of a person's
  own meals (0.473 against 0.620).
- The carbs the person logged, alone, rank meals better than the whole model (0.734). Meal content is the
  missing signal, as the label audit on CGMacros found.
- At the frozen cut-offs the model is not useful on these people: it misses 14.7% of what it clears and its
  cost saving is not distinguishable from zero.

Limits: 12 people; age, BMI and fasting glucose imputed; another wrist device; a self-kept food log with
timing errors; only the model is scored, not the product with its refusal rules; v1, the served model,
cannot be scored without photos.
