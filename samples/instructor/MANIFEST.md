# Instructor-style input set

One sub-folder per meal: `photo.jpg|png`, `cgm.csv` (timestamp, glucose_mgdl[, hr_bpm]) and
`profile.json` (age, bmi, a1c, fasting_glucose, meal_time). Run with
`just predict-folder samples/instructor outputs/folder_run`. Each case fixes one behaviour from
docs/final_design.md §6; the observed result of the first naive run is recorded in docs/rule_ledger.md.

| case | what it tests | required behaviour |
|---|---|---|
| 00_clean | well-formed input | an answer with risk and carb bin |
| 01_mmol | glucose 4-9 (mmol/L) | can't judge: values look like mmol/L; multiply by 18 |
| 02_unsorted | shuffled timestamps | sort, answer, log the correction |
| 03_future_timestamps | readings after meal_time | can't judge, naming the inconsistency |
| 04_gap_in_window | 30-min hole in the 2 h window | can't judge, naming the gap length |
| 05_no_heart_rate | no HR column | answer, flagged as degraded input |
| 06_profile_implausible | age 7, BMI 85 | can't judge, naming the field |
| 07_profile_blank | all profile fields null | can't judge, no defaults filled |
| 08_dark_photo | near-black photo | can't judge or caution; never low risk |
| 09_screenshot | app screenshot, not food | unfamiliar -> confirm-carbs tier |
| 10_exif_rotated | 00's photo stored rotated with EXIF orientation | identical decision to 00_clean |
| 11_local_dish | unfamiliar local dish | ideally confirm-carbs; the shipped check passes it (documented limitation, report §5 T6) |
| 12_zero_byte | empty photo.jpg | structured can't judge; the run continues |

Real phone photos (EXIF-stripped, resized): 00_clean (tuna rice bowl), 10_exif_rotated (the same
photo stored turned 90 CCW with EXIF orientation 6) and 11_local_dish (lamb with biryani-style
rice, an Emirati home meal). Cases 02-07 keep synthetic plate placeholders: their behaviours are
decided by the CGM or profile before the photo matters. samples/local_photos/ holds 14 more real
phone photos (meals, desserts, a crepe, a croissant, a fruit plate) for the OOD evaluation set.
