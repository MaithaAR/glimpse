# GLIMPSE demo: worked cases (test people, frozen 28-person pipeline)



### Correct warning (caution, and it spiked)
- meal time: 2024-01-28 19:03; photo: 00000040-PHOTO-2024-1-28-19-3-0.jpg
- **CAUTION**: Caution: about 25% chance this meal raises your glucose by 50+ mg/dL within 2 h. A smaller carb portion or a short walk afterwards may help.
- reason: risk 25% is at or above the caution cut-off 20%
- carbs: most likely under 30 g (68% sure); unfamiliar-photo score 0.496
- what happened: glucose rose 58 mg/dL (spike)
- response time: 275 ms

### Correct all-clear (low risk, no spike)
- meal time: 2024-01-28 15:32; photo: 00000037-PHOTO-2024-1-28-15-32-0.jpg
- **LOW RISK**: Unlikely to spike: about 11% chance this meal raises your glucose by 50+ mg/dL within 2 h.
- reason: risk 11% is below the low-risk cut-off 14%
- carbs: most likely under 30 g (58% sure); unfamiliar-photo score 0.493
- what happened: glucose fell 33 mg/dL (no spike)
- response time: 276 ms

### Wrong answer (shown on purpose)
- meal time: 2024-01-29 12:47; photo: 00000042-PHOTO-2024-1-29-12-47-0.jpg
- **CAUTION**: Caution: about 48% chance this meal raises your glucose by 50+ mg/dL within 2 h. A smaller carb portion or a short walk afterwards may help.
- reason: risk 48% is at or above the caution cut-off 20%
- carbs: most likely 60-89 g (92% sure); unfamiliar-photo score 0.191
- what happened: glucose rose 35 mg/dL (no spike)
- response time: 273 ms

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
- meal time: 2024-02-02 12:51; photo: 00000084-PHOTO-2024-2-2-12-51-0.jpg
- **CAN'T JUDGE**: Can't judge this meal: the risk (17%) is borderline, between 14% and 20%.
- reason: the risk (17%) is borderline, between 14% and 20%
- carbs: most likely under 30 g (52% sure); unfamiliar-photo score 0.195
- what happened: glucose fell 2 mg/dL (no spike)
- response time: 281 ms



Response time over 96 test meals on Darwin arm64: median 274 ms, 95th percentile 286 ms (target 300 ms).
