# GLIMPSE demo: worked cases (the intermediate split's test people; the frozen model trained on them, so these are in-sample illustrations, not results)



### Correct warning (caution, and it spiked)
- meal time: 2024-01-31 08:01; photo: 00000058-PHOTO-2024-1-31-8-1-0.jpg
- **CAUTION**: Caution: about 32% chance this meal raises your glucose by 50+ mg/dL within 2 h. A smaller carb portion or a short walk afterwards may help.
- reason: risk 32.0% is at or above the caution cut-off 22%
- carbs: most likely 60-89 g (87% sure); unfamiliar-photo score 0.159
- what happened: glucose rose 53 mg/dL (spike)
- response time: 130 ms

### Correct all-clear (low risk, no spike)
- meal time: 2024-01-28 15:32; photo: 00000037-PHOTO-2024-1-28-15-32-0.jpg
- **LOW RISK**: Unlikely to spike: about 7% chance this meal raises your glucose by 50+ mg/dL within 2 h.
- reason: risk 7.4% is below the low-risk cut-off 15%
- carbs: most likely 90 g or more (50% sure); unfamiliar-photo score 0.375
- what happened: glucose fell 33 mg/dL (no spike)
- response time: 161 ms

### Wrong answer (shown on purpose)
- meal time: 2024-01-29 12:47; photo: 00000042-PHOTO-2024-1-29-12-47-0.jpg
- **CAUTION**: Caution: about 29% chance this meal raises your glucose by 50+ mg/dL within 2 h. A smaller carb portion or a short walk afterwards may help.
- reason: risk 29.0% is at or above the caution cut-off 22%
- carbs: most likely 60-89 g (92% sure); unfamiliar-photo score 0.170
- what happened: glucose rose 35 mg/dL (no spike)
- response time: 144 ms

### Can't judge: sensor gap (same meal, last 40 min of glucose removed)
- meal time: 2024-01-31 08:01; photo: 00000058-PHOTO-2024-1-31-8-1-0.jpg
- **CAN'T JUDGE**: Can't judge this meal: glucose data incomplete: 0 readings in the last 30 min (need 3), 67% of the 2 h covered (need 70%), a gap of 40 min (max 20).
- reason: glucose data incomplete: 0 readings in the last 30 min (need 3), 67% of the 2 h covered (need 70%), a gap of 40 min (max 20)
- response time: 2 ms

### Can't judge: not a meal photo (plain white picture)
- meal time: 2024-01-31 08:01; photo: blank_photo.jpg
- **CAUTION**: Caution: about 32% chance this meal raises your glucose by 50+ mg/dL within 2 h. A smaller carb portion or a short walk afterwards may help.
- reason: risk 32.0% is at or above the caution cut-off 22%
- response time: 4 ms

### Can't judge: borderline risk (between the two cut-offs)
- meal time: 2024-01-28 19:03; photo: 00000040-PHOTO-2024-1-28-19-3-0.jpg
- **CAN'T JUDGE**: Can't judge this meal: the risk (21.0%) is borderline, between 15% and 22%.
- reason: the risk (21.0%) is borderline, between 15% and 22%
- carbs: most likely 90 g or more (34% sure); unfamiliar-photo score 0.350
- what happened: glucose rose 58 mg/dL (spike)
- response time: 161 ms



Response time over 96 test meals on Linux x86_64: median 140 ms, 95th percentile 182 ms (target 300 ms).
