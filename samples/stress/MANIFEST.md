# Stress set (S-stress): hostile inputs for `just stress`

Built 5 Nov 2026 against the instructor's documented folder command. The invariant under test: every
case produces one structured answer row (a decision and a plain-sentence reason), never a crash and
never a raw error row, and the run exits 0. Naive observed behaviour and fixes: docs/rule_ledger.md
(R8 timezone, R9 malformed profile.json, R10 Excel BOM export).

| case | hostile input | expected behaviour |
|---|---|---|
| 01_semicolon_csv | semicolon-delimited CSV | can't judge, names the missing columns |
| 02_tz_aware_timestamps | +04:00 offsets on every timestamp | answered; timezone_dropped flag (R8) |
| 03_ampm_meal_time | meal_time "06/01/2024 12:00 PM" | can't judge, unsupported meal time |
| 04_strings_in_glucose | glucose values "high"/"low" | can't judge, unsupported input |
| 05_negative_glucose | all readings negative | can't judge (reads as not mg/dL, R1 family) |
| 06_absurd_glucose | all readings 1e9 | structured answer or refusal, never a crash |
| 07_one_reading | a single reading 5 min before the meal | can't judge, incomplete window |
| 08_duplicate_timestamps | repeated timestamps appended out of order | answered; timestamps_sorted flag (R2) |
| 09_header_only | CSV with a header and no rows | can't judge, no glucose readings |
| 10_binary_cgm | 2 KB of raw bytes named cgm.csv | can't judge, unsupported input |
| 11_bom_crlf_csv | UTF-8 BOM + CRLF (Excel export) | accepted and answered (R10) |
| 12_wrong_columns | columns time,sugar | can't judge, names the required columns |
| 13_profile_strings | numbers quoted as strings | accepted and answered |
| 14_profile_not_json | profile.json is not JSON | can't judge, says profile.json invalid (R9) |
| 15_profile_nested | age is an object | can't judge, unsupported input |
| 16_no_cgm_at_all | no cgm.csv in the folder | can't judge, no CGM export |
| 17_empty_folder | nothing in the folder | can't judge, no CGM export |
| 18_future_meal_time | meal_time before the readings | can't judge, readings after meal time (R3) |
| 19_extra_junk | stray files and a nested folder | answered; junk ignored |
| 20_truncated_photo | JPEG cut off after 1/3 of its bytes | can't judge, photo cannot be used |
| 21_cmyk_photo | CMYK-mode JPEG | handled: answered or structured refusal |
| 22_alpha_png | RGBA PNG with transparency | handled: answered or structured refusal |
| 23_gif_renamed_jpg | GIF bytes under photo.jpg | handled: answered or structured refusal |

A stray_file.txt at the top level checks that non-folders are skipped. Cases 20-23 carry a clean CGM
and profile so only the photo path is under test; they are exercised by the live pipeline
(`just stress`), while tests/test_stress.py fuzzes the text-level cases on every `just check`.
