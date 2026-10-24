# GLIMPSE demo: worked cases (test people, frozen 28-person pipeline)



### Correct warning (caution, and it spiked)
- meal time: 2024-01-28 19:03; photo: 00000040-PHOTO-2024-1-28-19-3-0.jpg
- **CAUTION**: Caution: about 25% chance this meal raises your glucose by 50+ mg/dL within 2 h. A smaller carb portion or a short walk afterwards may help.
- reason: risk 25% is at or above the caution cut-off 20%
- estimated carbs: 15 g (bin under 30 g); unfamiliar-photo score 0.496
- what happened: glucose rose 58 mg/dL (spike)
- response time: 282 ms

### Correct all-clear (low risk, no spike)
- meal time: 2024-01-28 15:32; photo: 00000037-PHOTO-2024-1-28-15-32-0.jpg
- **LOW RISK**: Unlikely to spike: about 11% chance this meal raises your glucose by 50+ mg/dL within 2 h.
- reason: risk 11% is below the low-risk cut-off 14%
- estimated carbs: 49 g (bin under 30 g); unfamiliar-photo score 0.493
- what happened: glucose rose -33 mg/dL (no spike)
- response time: 286 ms

### Wrong answer (shown on purpose)
- meal time: 2024-01-29 12:47; photo: 00000042-PHOTO-2024-1-29-12-47-0.jpg
- **CAUTION**: Caution: about 48% chance this meal raises your glucose by 50+ mg/dL within 2 h. A smaller carb portion or a short walk afterwards may help.
- reason: risk 48% is at or above the caution cut-off 20%
- estimated carbs: 100 g (bin 60-89 g); unfamiliar-photo score 0.191
- what happened: glucose rose 35 mg/dL (no spike)
- response time: 280 ms

### Can't judge: sensor gap (same meal, last 40 min of glucose removed)
- meal time: 2024-01-28 19:03; photo: 00000040-PHOTO-2024-1-28-19-3-0.jpg
- **CAN'T JUDGE**: Can't judge this meal: glucose data incomplete: 0 readings in the last 30 min (need 3), 67% of the 2 h covered (need 70%), longest gap 40 min (max 20).
- reason: glucose data incomplete: 0 readings in the last 30 min (need 3), 67% of the 2 h covered (need 70%), longest gap 40 min (max 20)
- response time: 1 ms

### Can't judge: not a meal photo (plain grey picture)
- meal time: 2024-01-28 19:03; photo: blank_photo.jpg
- **CAN'T JUDGE**: Can't judge this meal: the photo looks unlike the meal photos the model learned from.
- reason: the photo looks unlike the meal photos the model learned from
- estimated carbs: 4 g (bin 30-59 g); unfamiliar-photo score 0.749
- response time: 253 ms



Response time over 96 test meals on Darwin arm64: median 277 ms, 95th percentile 297 ms (target 300 ms).
