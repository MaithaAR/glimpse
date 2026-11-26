# Sample inputs

- `cgm_example.csv`: a **synthetic** CGM export (made up, not from any person): 24 readings every 5 minutes in the
  2 hours before a meal at 2024-03-04 13:00, with heart rate. Columns: `timestamp, glucose_mgdl, hr_bpm`.
- Use it with your own before-meal photo:
  `just predict --cgm samples/cgm_example.csv --photo my_lunch.jpg --age 45 --bmi 28 --a1c 5.9 --fasting 100`
- Without `--photo`, v1 (served) answers a structured "can't judge" (no meal photo); v2
  (`prototype.product: sensor_first`, the alternative) answers from CGM, heart rate and profile alone.
- `instructor/`: 13 meal folders in the grader's format (`photo.*`, `cgm.csv`, `profile.json`), described in
  `instructor/MANIFEST.md`; `just predict-folder samples/instructor outputs/folder_run` answers them without the
  dataset (v1, served: 8 refuse with a named reason, 5 answer caution; v2: 5 refuse, 8 answer).
- `stress/`: 25 hostile folders (timezones, malformed JSON, binary CSVs, broken photos, empty folders), described in
  `stress/MANIFEST.md`; `just stress` runs them; none may crash or produce a raw error row.
- `local_photos/`: the real phone photos behind the unfamiliar-photo study (R7, `just ood`).
- Real CGMacros meals (photos + sensor data) are downloaded by `just install` (PhysioNet, CC BY-NC-SA 4.0) and are
  not stored in this repository; `just demo` and `just app` use them, so those two need the download first.
- Optional `history.csv` in a meal folder (26 Nov): the person's earlier answers as `meal_time,fusion_logit,spiked`
  (`fusion_logit` from an earlier `results.jsonl`, `spiked` the CGM's 0/1 label at +2 h). Meals labelled at
  least two hours before the current meal feed the personal intercept; without the file the answer uses the
  population calibration. Only `returning_user/02_with_priors_and_history` carries one: an illustrative history
  tagged for v1 (`product` = photo_fusion); rows from the other product are skipped and flagged.
- `returning_user/`: one meal at three stages (new user, with onboarding priors, with priors and history),
  described in its MANIFEST; `outputs/returning_user_run/` holds v1's answers (caution 25.6%, can't judge
  at 15.6% between the cut-offs, low risk 9.9%); v2's are in `outputs/v2_sensor_first/returning_user_run/`.
- Optional onboarding priors in `profile.json` (26 Nov note): `prior_spike_rate` (0-1, the share of the person's
  first meals that spiked) and `prior_rise` (mean rise, mg/dL). Rule R13: with neither history nor both priors,
  a would-be low-risk answer becomes can't judge with a named reason, so every instructor and stress sample
  answers caution or can't judge (docs/rule_ledger.md).
