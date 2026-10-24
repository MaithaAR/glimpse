# Sample inputs

- `cgm_example.csv`: a **synthetic** CGM export (made up, not from any person): 24 readings every 5 minutes in the
  2 hours before a meal at 2024-03-04 13:00, with heart rate. Columns: `timestamp, glucose_mgdl, hr_bpm`.
- Use it with your own before-meal photo:
  `just predict --cgm samples/cgm_example.csv --photo my_lunch.jpg --age 45 --bmi 28 --a1c 5.9 --fasting 100`
- Without `--photo` the answer is a structured "can't judge" (no meal photo), which is the abstention path.
- Real CGMacros meals (photos + sensor data) are downloaded by `just install` (PhysioNet, CC BY-NC-SA 4.0) and are
  not stored in this repository; `just demo` and `just app` use them.
