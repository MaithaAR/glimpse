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
| R6 | 10_exif_rotated (unproven) | photos are EXIF-transposed on open so stored rotation cannot change the decision | shared photo opener | pending (C7) |

Photo-set debt (blocks re-testing 00/10/11): replace the synthetic placeholder photos with real
phone photos, including one local dish. Until then those three rows stay "untestable" and are not
cited as passing anywhere.

## After rules R1-R5 — verification pending

The folder run is repeated after S3-C6; the five FAIL rows above are expected to flip (01 names the
unit fix, 02 answers with a `timestamps_sorted` flag once its photo is real, 03 names the future
readings, 06 shows one plain sentence, 07 names the blank fields). The observed summary is pasted
here after the run.
