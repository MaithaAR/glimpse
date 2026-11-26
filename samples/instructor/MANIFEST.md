# Instructor-style input set

One sub-folder per meal: `photo.jpg|png`, `cgm.csv` (timestamp, glucose_mgdl[, hr_bpm]) and
`profile.json` (age, bmi, a1c, fasting_glucose, meal_time). Run with
`just predict-folder samples/instructor outputs/folder_run`. Each case fixes one behaviour from
docs/final_design.md §6; the observed result of the first naive run is recorded in docs/rule_ledger.md.

Required behaviours were fixed for v1 (photo fusion, served). The last column is what v2, the tested alternative, does:
the photo no longer refuses a meal; it only withholds the carb estimate, with the reason in the summary.

| case | what it tests | required behaviour (v1) | v2 (outputs/v2_sensor_first/folder_run) |
|---|---|---|---|
| 00_clean | well-formed input | an answer with risk and carb bin | caution 48.4%, 30-59 g |
| 01_mmol | glucose 4-9 (mmol/L) | can't judge: values look like mmol/L; multiply by 18 | same |
| 02_unsorted | shuffled timestamps | sort, answer, log the correction | same (caution) |
| 03_future_timestamps | readings after meal_time | can't judge, naming the inconsistency | same |
| 04_gap_in_window | 30-min hole in the 2 h window | can't judge, naming the gap length | same |
| 05_no_heart_rate | no HR column | answer, flagged as degraded input | caution 37.8%, flag `hr_missing` |
| 06_profile_implausible | age 7, BMI 85 | can't judge, naming the field | same |
| 07_profile_blank | all profile fields null | can't judge, no defaults filled | same |
| 08_dark_photo | near-black photo | can't judge or caution; never low risk | caution; carbs withheld (blank) |
| 09_screenshot | app screenshot, not food | unfamiliar photo -> can't judge (the confirm-carbs tier has no input surface) | caution; carbs withheld (unlike a meal) |
| 10_exif_rotated | 00's photo stored rotated with EXIF orientation | identical decision to 00_clean | same decision and risk; carb heads disagree, shown as "about 36 g; range unsure" |
| 11_local_dish | unfamiliar local dish | ideally confirm-carbs; the shipped check passes it (documented limitation, report §5) | caution; shown as "about 33 g; range unsure" |
| 12_zero_byte | empty photo.jpg | structured can't judge; the run continues | caution; carbs withheld (unreadable) |

Real phone photos (EXIF-stripped, resized): 00_clean (tuna rice bowl), 10_exif_rotated (the same
photo stored turned 90 CCW with EXIF orientation 6) and 11_local_dish (lamb with biryani-style
rice, an Emirati home meal). Cases 02-04 and 06-07 keep synthetic plate placeholders: their behaviours are
decided by the CGM or profile before the photo matters; 05_no_heart_rate reuses the 00_clean photo.
(26 Nov) "confirm-carbs tier" for 09 and 11 means the answer's escalate_to field: the app has no input for
typed carbs yet, so it is advice to the user, not a working step (docs/report_appendix.md, failure table). samples/local_photos/ holds 14 more real
phone photos (meals, desserts, a crepe, a croissant, a fruit plate) for the OOD evaluation set.
