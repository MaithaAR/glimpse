# Rule ledger

Every input-handling rule is written only after a failure was observed on a real run of
`just predict-folder samples/instructor OUT`. This file records what actually happened first,
then which rule was added in response and where. Rules are never added speculatively.

## Naive run — 2026-11-01, commit 4c72a62 (before any new guard)

All 13 cases returned without a crash and the run exited 0. Decisions:

| case | required behaviour | observed | verdict |
|---|---|---|---|
| 00_clean | answer with risk and carb bin | can't judge: photo unfamiliar (synthetic placeholder flagged by the OOD check) | untestable until the placeholder is a real phone photo |
| 01_mmol | can't judge: "values look like mmol/L; multiply by 18" | can't judge, but the reason says "outside the sensor's range (40-400)" — the user is not told what is wrong or how to fix it | FAIL -> rule R1 |
| 02_unsorted | sort, answer, log the correction | the window check passed (resampling tolerates unsorted input) but nothing was logged or flagged, so the repair is invisible | FAIL -> rule R2 |
| 03_future_timestamps | can't judge naming the inconsistency | can't judge with a generic "glucose data incomplete" coverage message; readings after the meal time are never named | FAIL -> rule R3 |
| 04_gap_in_window | can't judge naming the gap length | can't judge: "longest gap 35 min (max 20)" | pass |
| 05_no_heart_rate | answer, flagged as degraded | flags = [hr_missing] (decision blocked only by the placeholder photo) | pass (flag verified; full path re-checked after photo replacement) |
| 06_profile_implausible | can't judge naming the field | refused, but the reason is a raw pydantic dump with a URL — correct behaviour, unreadable message | FAIL (message) -> rule R4 |
| 07_profile_blank | can't judge, no defaults filled | all-null profile was accepted and the pipeline ran on NaN features | FAIL -> rule R5 |
| 08_dark_photo | can't judge or caution; never low risk | can't judge: "photo is almost blank (grey-level spread 0.0)" | pass |
| 09_screenshot | unfamiliar -> confirm-carbs tier | can't judge, checks_failed=[photo_familiar], escalate_to=confirm_carbs | pass |
| 10_exif_rotated | identical decision to 00_clean | identical, but only because both fail the same OOD check | untestable until the photos are real; EXIF transpose still unproven |
| 11_local_dish | confirm-carbs, not a confident bin | placeholder photo, same path as 00 | untestable until a real local-dish photo replaces it |
| 12_zero_byte | per-item error entry; run continues | can't judge "photo cannot be used (cannot identify image file)", run finished, exit 0 | pass |

## Rules

| id | trigger observed above | rule | lives in | added |
|---|---|---|---|---|
| R1 | 01_mmol | max glucose < 30 mg/dL -> can't judge: "values look like mmol/L; multiply by 18 or export in mg/dL" | read path shared by every entry point | 2026-11-01, S3-C6 |
| R2 | 02_unsorted | unsorted timestamps are sorted before use and the repair is logged + flagged ("timestamps_sorted") | shared CGM reader | 2026-11-01, S3-C6 |
| R3 | 03_future_timestamps | any reading after the meal time -> can't judge naming the count and the latest offending time | shared CGM validation | 2026-11-01, S3-C6 |
| R4 | 06_profile_implausible | implausible profile -> can't judge "age = 7 is outside the plausible range 18-100" (one plain sentence, no stack dump) | input boundary (answer_from_files) | 2026-11-01, S3-C6 |
| R5 | 07_profile_blank | required profile fields all missing/blank -> can't judge naming the missing fields; never filled with defaults | shared profile validation | 2026-11-01, S3-C6 |
| R6 | 10_exif_rotated (unproven) | photos are EXIF-transposed on open so stored rotation cannot change the decision | shared photo opener (open_photo, already present) | 2026-11-01, S3-C7: pixel-identity unit test added; end-to-end proof still waits on real photos |

Photo-set debt (blocks re-testing 00/10/11): replace the synthetic placeholder photos with real
phone photos, including one local dish. Until then those three rows stay "untestable" and are not
cited as passing anywhere.

## Verification run — 2026-11-01 17:25, after rules R1-R5 (commit 5e01186)

All five FAIL rows flipped; observed reasons:

| case | observed after R1-R5 | verdict |
|---|---|---|
| 01_mmol | "the glucose values look like mmol/L; multiply by 18 or export the file in mg/dL" | pass (R1) |
| 02_unsorted | "CGM timestamps were out of order; sorted before use (rule R2)" logged, window check passed; decision still blocked only by the placeholder photo | pass (R2) |
| 03_future_timestamps | "11 glucose readings are timestamped after the meal time (latest 2024-06-01 12:55); check the export and the meal time" | pass (R3) |
| 06_profile_implausible | "unsupported input (age = 7.0 is outside the plausible range 18-100)" — one sentence, no stack dump | pass (R4) |
| 07_profile_blank | "profile fields missing or blank: age, bmi, a1c_pct, fasting_glucose_mgdl" | pass (R5) |

04, 08, 09, 12 behave as in the naive run (pass). 00/10/11 remain untestable until the placeholder
photos are replaced with real phone photos.

## Verification run — 2026-11-01 17:36, real phone photos in 00/10/11 (commit 71abb13)

| case | observed | verdict |
|---|---|---|
| 00_clean | caution, risk 47%, carb bin 30-59 g — a real answer end to end | pass |
| 10_exif_rotated | caution, risk 47%, same carb bin as 00_clean; risk differs from 00 by 0.2 pp (JPEG re-encode noise from storing the rotated copy, not an orientation effect) | pass (R6 end to end: identical decision) |
| 11_local_dish | caution, risk 51%, confident carb bin "under 30 g" (p=0.67), unfamiliar score 0.35 below the cut — the OOD check did NOT flag the local lamb-and-rice dish, and "under 30 g" for a rice dish is likely far too low | FAIL — recorded as R7, remediation in S10 |

| id | trigger observed above | rule | lives in | added |
|---|---|---|---|---|
| R7 | 11_local_dish | the unfamiliar-photo cut must flag local dishes absent from CGMacros before the carb bin is trusted; to be re-tuned on samples/local_photos/ (14 real phone photos) with the ≤2% false-flag budget | OOD check (knn cut) | pending (S10) |

02-07 keep placeholder photos by design (their behaviours are decided before the photo matters);
05's hr_missing flag was verified in the R1-R5 run.

## R7 resolution — 2026-11-01, `just ood` (S10, budget pre-registered at <= 2%)

The cut sweep on the training photos' own leave-one-out distances, scored against all 14 real phone
photos and the key instructor cases:

| quantile | cut | in-dist flagged | real phone photos flagged |
|---|---|---|---|
| 0.99 (deployed) | 0.609 | 1.0% | 0 / 14 |
| 0.98 | 0.556 | 2.1% (over budget) | 1 / 14 |
| 0.95 | 0.501 | 5.1% | 5 / 14 |
| 0.90 | 0.465 | 10.1% | 7 / 14 |

The check does its intended job: the screenshot scores 0.788 and the dark photo 0.682 -- both flagged
at the deployed cut -- and none of the 14 legitimate phone photos false-flag. But the lamb-and-rice
dish scores 0.346: MORE familiar than the clean tuna bowl (0.420), because it looks like the training
set's rice plates. Catching it would need a cut below 0.35, which false-flags half of everything.

**R7 is closed as an accepted limitation, with the cut unchanged (no --apply):** the kNN check detects
VISUALLY unfamiliar images, not visually-familiar-but-nutritionally-misjudgeable dishes. Mitigations in
the shipped design: the carb bin is always shown with its confidence, the confirm-carbs tier lets the
user override the estimate, and mixed local dishes are named as a known failure mode in the report
(§5) and the app's help text. A nutrition-aware embedding could close this properly; it is on the cut
list with the reason recorded.
