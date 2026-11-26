# Sample inputs

- `cgm_example.csv`: a **synthetic** CGM export (made up, not from any person): 24 readings every 5 minutes in the
  2 hours before a meal at 2024-03-04 13:00, with heart rate. Columns: `timestamp, glucose_mgdl, hr_bpm`.
- Use it with your own before-meal photo:
  `just predict --cgm samples/cgm_example.csv --photo my_lunch.jpg --age 45 --bmi 28 --a1c 5.9 --fasting 100`
- Without `--photo`, v2 (served) answers from CGM, heart rate and profile alone and withholds only the carb
  estimate; v1 (`prototype.product: photo_fusion`) answers a structured "can't judge" (no meal photo).
- `instructor/`: 13 meal folders in the grader's format (`photo.*`, `cgm.csv`, `profile.json`), described in
  `instructor/MANIFEST.md`; `just predict-folder samples/instructor outputs/folder_run` answers them without the
  dataset (v2: 5 refuse with a named reason, 8 answer caution; v1: 8 refuse, 5 answer).
- `stress/`: 23 hostile folders (timezones, malformed JSON, binary CSVs, broken photos, empty folders), described in
  `stress/MANIFEST.md`; `just stress` runs them; none may crash or produce a raw error row.
- `local_photos/`: the real phone photos behind the unfamiliar-photo study (R7, `just ood`).
- Real CGMacros meals (photos + sensor data) are downloaded by `just install` (PhysioNet, CC BY-NC-SA 4.0) and are
  not stored in this repository; `just demo` and `just app` use them, so those two need the download first.
- Optional `history.csv` in a meal folder (26 Nov): the person's earlier answers as `meal_time,fusion_logit,spiked`
  (`fusion_logit` from an earlier `results.jsonl`, `spiked` the CGM's 0/1 label at +2 h). Meals labelled at
  least two hours before the current meal feed the personal intercept; without the file the answer uses the
  population calibration. Only `returning_user/02_with_priors_and_history` carries one: an illustrative history
  written for v2 (the file holds no product tag, so a v1 history must not be reused with v2).
- `returning_user/`: one meal at three stages (new user, with onboarding priors, with priors and history),
  described in its MANIFEST; `outputs/returning_user_run/` holds the answers (caution 35.7%, caution 22.7%,
  low risk 14.8%, just under the 15% cut).
- Optional onboarding priors in `profile.json` (26 Nov note): `prior_spike_rate` (0-1, the share of the person's
  first meals that spiked) and `prior_rise` (mean rise, mg/dL). Rule R13: with neither history nor both priors,
  a would-be low-risk answer becomes can't judge with a named reason, so every instructor and stress sample
  answers caution or can't judge (docs/rule_ledger.md).
