# Folder run: 13 meals

- can't judge: 8
- caution: 5

| meal | decision | reason |
|---|---|---|
| 00_clean | caution | risk 36.5% is at or above the caution cut-off 16% |
| 01_mmol | can't judge | the glucose values look like mmol/L; multiply by 18 or export the file in mg/dL |
| 02_unsorted | caution | risk 31.8% is at or above the caution cut-off 16% |
| 03_future_timestamps | can't judge | 11 glucose readings are timestamped after the meal time (latest 2024-06-01 12:55); check the export and the meal time |
| 04_gap_in_window | can't judge | glucose data incomplete: a gap of 35 min (max 20) |
| 05_no_heart_rate | caution | risk 27.6% is at or above the caution cut-off 16% |
| 06_profile_implausible | can't judge | unsupported input (age = 7.0 is outside the plausible range 18-100) |
| 07_profile_blank | can't judge | profile fields missing or blank: age, bmi, a1c_pct, fasting_glucose_mgdl |
| 08_dark_photo | can't judge | the photo is almost blank (grey-level spread 0.0, need 8); no meal visible |
| 09_screenshot | can't judge | the photo looks unlike the meal photos the model learned from |
| 10_exif_rotated | caution | risk 36.9% is at or above the caution cut-off 16% |
| 11_local_dish | caution | risk 38.4% is at or above the caution cut-off 16% |
| 12_zero_byte | can't judge | the photo cannot be used (cannot read photo samples/instructor/12_zero_byte/photo.jpg: cannot identify image file 'samples/instructor/12_zero_byte/photo.jpg') |
