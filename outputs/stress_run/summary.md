# Folder run: 25 meals

- can't judge: 25

| meal | decision | reason | carbs from the photo |
|---|---|---|---|
| 01_semicolon_csv | can't judge | unsupported input (the CSV needs the columns timestamp and glucose_mgdl (missing: ['glucose_mgdl', 'timestamp'])) | – |
| 02_tz_aware_timestamps | can't judge | the photo cannot be used (cannot read photo samples/stress/02_tz_aware_timestamps/photo.jpg: cannot identify image file 'samples/stress/02_tz_aware_timestamps/photo.jpg') | – |
| 03_ampm_meal_time | can't judge | unsupported input (the meal time '06/01/2024 12:00 PM' is not an understood format (use e.g. 2024-06-01T12:00)) | – |
| 04_strings_in_glucose | can't judge | unsupported input (a glucose or heart-rate value is not a number ('high')) | – |
| 05_negative_glucose | can't judge | a glucose reading is zero or negative; the export is broken, check the file | – |
| 06_absurd_glucose | can't judge | a glucose reading is outside the sensor's range (40.0-400.0 mg/dL) | – |
| 07_one_reading | can't judge | glucose data incomplete: 1 readings in the last 30 min (need 3), 4% of the 2 h covered (need 70%), a gap of 115 min (max 20) | – |
| 08_duplicate_timestamps | can't judge | the photo cannot be used (cannot read photo samples/stress/08_duplicate_timestamps/photo.jpg: cannot identify image file 'samples/stress/08_duplicate_timestamps/photo.jpg') | – |
| 09_header_only | can't judge | unsupported input (the CGM export has no glucose readings) | – |
| 10_binary_cgm | can't judge | unsupported input (the CGM file could not be read as a table of readings) | – |
| 11_bom_crlf_csv | can't judge | the photo cannot be used (cannot read photo samples/stress/11_bom_crlf_csv/photo.jpg: cannot identify image file 'samples/stress/11_bom_crlf_csv/photo.jpg') | – |
| 12_wrong_columns | can't judge | unsupported input (the CSV needs the columns timestamp and glucose_mgdl (missing: ['glucose_mgdl', 'timestamp'])) | – |
| 13_profile_strings | can't judge | the photo cannot be used (cannot read photo samples/stress/13_profile_strings/photo.jpg: cannot identify image file 'samples/stress/13_profile_strings/photo.jpg') | – |
| 14_profile_not_json | can't judge | profile.json is not valid JSON (Expecting value at line 1) | – |
| 15_profile_nested | can't judge | unsupported input (the profile value age = {'value': 42} is not a number) | – |
| 16_no_cgm_at_all | can't judge | unsupported input (no CGM export was given) | – |
| 17_empty_folder | can't judge | unsupported input (no CGM export was given) | – |
| 18_future_meal_time | can't judge | 24 glucose readings are timestamped after the meal time (latest 2024-06-01 11:55); check the export and the meal time | – |
| 19_extra_junk | can't judge | the photo cannot be used (cannot read photo samples/stress/19_extra_junk/photo.jpg: cannot identify image file 'samples/stress/19_extra_junk/photo.jpg') | – |
| 20_truncated_photo | can't judge | the photo cannot be used (cannot read photo samples/stress/20_truncated_photo/photo.jpg: image file is truncated (32 bytes not processed)) | – |
| 21_cmyk_photo | can't judge | the photo looks unlike the meal photos the model learned from | – |
| 22_alpha_png | can't judge | the photo looks unlike the meal photos the model learned from | – |
| 23_gif_renamed_jpg | can't judge | the photo looks unlike the meal photos the model learned from | – |
| 24_low_glucose_falling | can't judge | glucose is very low now (48.5 mg/dL); treat the low first and follow your care plan, this tool is not for that (rule R14) | – |
| 25_diabetes_range_a1c | can't judge | HbA1c 7.4% is in the diabetes range, outside what GLIMPSE was built and tested for; ask your care team (rule R15) | – |
