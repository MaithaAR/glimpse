# GLIMPSE demo: worked cases (test people, frozen final pipeline)



### Correct warning (caution, and it spiked)
- meal time: 2024-01-28 19:03; photo: 00000040-PHOTO-2024-1-28-19-3-0.jpg
- **CAUTION**: Caution: about 24% chance this meal raises your glucose by 50+ mg/dL within 2 h. A smaller carb portion or a short walk afterwards may help.
- reason: risk 24% is at or above the caution cut-off 16%
- carbs: most likely 90 g or more (34% sure); unfamiliar-photo score 0.350
- what happened: glucose rose 58 mg/dL (spike)
- response time: 278 ms

### Correct all-clear (low risk, no spike)
- meal time: 2024-01-28 15:32; photo: 00000037-PHOTO-2024-1-28-15-32-0.jpg
- **LOW RISK**: Unlikely to spike: about 9% chance this meal raises your glucose by 50+ mg/dL within 2 h.
- reason: risk 9% is below the low-risk cut-off 15%
- carbs: most likely 90 g or more (50% sure); unfamiliar-photo score 0.375
- what happened: glucose fell 33 mg/dL (no spike)
- response time: 285 ms

### Wrong answer (shown on purpose)
- meal time: 2024-01-29 12:47; photo: 00000042-PHOTO-2024-1-29-12-47-0.jpg
- **CAUTION**: Caution: about 37% chance this meal raises your glucose by 50+ mg/dL within 2 h. A smaller carb portion or a short walk afterwards may help.
- reason: risk 37% is at or above the caution cut-off 16%
- carbs: most likely 60-89 g (92% sure); unfamiliar-photo score 0.170
- what happened: glucose rose 35 mg/dL (no spike)
- response time: 304 ms

### Can't judge: sensor gap (same meal, last 40 min of glucose removed)
- meal time: 2024-01-28 19:03; photo: 00000040-PHOTO-2024-1-28-19-3-0.jpg
- **CAN'T JUDGE**: Can't judge this meal: glucose data incomplete: 0 readings in the last 30 min (need 3), 67% of the 2 h covered (need 70%), longest gap 40 min (max 20).
- reason: glucose data incomplete: 0 readings in the last 30 min (need 3), 67% of the 2 h covered (need 70%), longest gap 40 min (max 20)
- response time: 1 ms

### Can't judge: not a meal photo (plain white picture)
- meal time: 2024-01-28 19:03; photo: blank_photo.jpg
- **CAN'T JUDGE**: Can't judge this meal: the photo is almost blank (grey-level spread 0.0, need 8); no meal visible.
- reason: the photo is almost blank (grey-level spread 0.0, need 8); no meal visible
- response time: 1 ms

### Can't judge: borderline risk (between the two cut-offs)
- meal time: 2024-01-31 15:25; photo: 00000064-PHOTO-2024-1-31-15-25-0.jpg
- **CAN'T JUDGE**: Can't judge this meal: the risk (15%) is borderline, between 15% and 16%.
- reason: the risk (15%) is borderline, between 15% and 16%
- carbs: most likely under 30 g (80% sure); unfamiliar-photo score 0.319
- what happened: glucose rose 16 mg/dL (no spike)
- response time: 273 ms



Response time over 96 test meals on Darwin arm64: median 274 ms, 95th percentile 285 ms (target 300 ms).
