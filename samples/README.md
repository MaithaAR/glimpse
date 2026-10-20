# Sample inputs (synthetic)

All files here are **synthetic** and contain no CGMacros participant data (CGMacros is CC BY-NC-SA and is downloaded by `just install`, never committed). Each folder is one pre-meal request:

| File | Content |
|---|---|
| `meal.json` | meal time, profile (age, BMI, optional A1c and fasting glucose), photo file name |
| `cgm.csv` | `timestamp,glucose_mgdl` readings before the meal |
| `photo.jpg` | synthetic plate image (placeholder for a real meal photo) |

- `normal_case/`: complete 2-hour CGM trace, rising before the meal → expected **answer**.
- `abstain_cgm_gap/`: no CGM readings for the last 50 minutes → expected **abstain** (`insufficient_cgm`, `cgm_gap`).
