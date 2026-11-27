# Serve v1, keep v2 as the tested alternative; four add-on studies (registered 26 Nov 2026, before any run)

The author's decision, made in this session after the round-14 grades and the council review; written by
the assistant at her instruction. Nothing frozen changes: v1's and v2's fusions, Platt maps, m, cut-offs
and the photo model stay as they are, and the locked eight are not rescored.

## The switch
`prototype.product` returns to `photo_fusion` (v1), the model the locked eight measured once. v2 stays in
the code and the report as the pre-registered sensor-first alternative, one config line away. Why: on
development v2 was not shown non-inferior (v2 minus v1 −0.015, −0.045 to +0.010), it lets 12.7% of its
clears spike against v1's 7.3% (prediabetes 6 of 30 against 1 of 25), and it has no clean test, so the
registration's rule ("v2 ships whatever the development result") is set aside by the author: the safety
cap and the clean test outweigh the gate's design preference. This is a post-hoc decision and is reported
as one.

Rule R16 applies to v1 as well: a sensor input (the 7 glucose and heart-rate inputs) beyond the minimum or
maximum of v1's frozen training meals, on the side where it lowers v1's risk, gives can't judge.
Predicted: no development decision of v1 changes; the locked eight are counted, not rescored.

## Add-on studies (descriptive)
| | what | prediction fixed now |
|---|---|---|
| O1 | unfamiliar-photo check as a detector: kNN distance AUROC with in-distribution = the 14 real phone meal photos and 60 held-out training photos (leave-one-out distances), out-of-distribution = 6 synthetic non-meal pictures, the instructor screenshot and 24 generated non-meal images (text pages, app screens, patterns, plain colours, noise) | AUROC above 0.9; the real local dish stays below the cut |
| O2 | out-of-scope inputs: Mahalanobis distance of the 15 sensor and profile inputs to the 23 development people (leave one person out), development meals in-scope against the 14 type-2 diabetes people's meals | AUROC above 0.8, driven by HbA1c and fasting glucose |
| M1 | monitoring power against volume: replay 4's cohort shift with 120, 240, 480 and 960 answers after the switch, calibration CUSUM and caution share | detection within 960 answers above 80% for the calibration CUSUM |
| C1 | the calibration check: draw labels from each served risk, replay the per-person intercept on those labels (it learns from them), and recompute ECE, for v1 and v2; 500 draws | v2's 0.029 is typical under this null (the adaptive intercept, not leakage, explains a below-floor ECE) |

## Course checks (added 26 Nov after reviewing the course material, before any of them ran)
Descriptive, on development only; nothing frozen changes. Predictions fixed now.

| | what | prediction |
|---|---|---|
| S1 | shuffled-label control: the v1 fusion refitted on its committed out-of-fold inputs with labels shuffled within each training fold, scored out of fold, no intercept; 20 shuffles | AUROC 0.45 to 0.55 |
| S2 | learning curve against training people: the v1 fusion (no intercept) refitted on 4, 8, 12 and all of each fold's training people, out-of-fold AUROC, 10 draws per size | rises with diminishing returns; all people within 0.03 of 12 |
| S3 | slices (report the worst slice): v1 served answers by sex, age band (under 40, 40 to 55, over 55) and person; worst three people | the worst person's AUROC below 0.6 |
| S4 | identity shortcut: can v1's four photo inputs name the person? multinomial logistic, meal-level 5-fold CV, accuracy against the majority-person rate | above chance but far from perfect (the cohort eats standardised meals) |
| S5 | train against validation curves: the committed training histories of the photo model and SensorCNN, final gap stated in the report | a small gap at the chosen epoch |
