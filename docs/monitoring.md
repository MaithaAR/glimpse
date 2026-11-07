# S11 — monitoring design (deployed GLIMPSE)

Half a page by design: GLIMPSE's deployment loop already produces everything a monitor needs, because
the CGM labels every meal itself two hours later (`cgm_auto`, the same spike rule as training). The
monitor is therefore a consumer of existing signals, not new modelling.

## What is watched, and with what statistic

1. **Input drift (photo).** Every live answer already computes the kNN distance of the photo's
   embedding to the training reference (the unfamiliar-photo check, rule R6/R7). Logged per answer,
   summarised weekly as the share of photos above the deployed cut (0.614) and the median distance.
   Alert: weekly median above the training LOO 90th percentile (0.445) — the user's camera, lighting
   or cuisine has drifted from the training distribution. The S10 study is the baseline drawn once:
   training LOO distances vs the 14 real phone photos (figure below).
2. **Outcome drift (labels arrive free).** Because `cgm_auto` labels every meal at +2 h, rolling
   30-day AUROC and missed-among-cleared are computable in deployment with no annotation effort.
   Alert: missed-among-cleared above the config cap, or rolling AUROC below the locked-set value.
3. **Behaviour drift.** Weekly shares of low risk / caution / can't judge and the borderline-band
   share, compared to the development shares (23/64/14%). A rising can't-judge share is the system
   telling us its inputs degraded (S9: refusal is the designed response).
4. **Calibration drift.** The personal intercept b_p(t) is re-fitted at every answer from the
   person's own history; its population spread is tracked monthly. Alert: median |b_p| growing over
   three consecutive months — the population calibration (Platt) no longer centres people.

Not built (and why): no automated retraining (a graded prototype; retraining is a human decision on
alert), no dashboard service (the statistics are one `just monitoring` run over logged answers).

## The stub that exists

`just monitoring` renders the drift baseline from committed artifacts: the distribution of training
leave-one-out kNN distances, the deployed cut, and the 14 real phone photos' distances —
outputs/monitoring/drift_baseline.png + monitoring.json (the numbers behind the figure). This is the
panel a deployed monitor would redraw weekly from live logs.
