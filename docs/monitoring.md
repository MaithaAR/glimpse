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

## What a maintainer does on each alert

Photo-drift alert (weekly median above the training q90): inspect the week's logged distances and a
sample of refused photos; if the cause is a camera or app change, rebuild the reference under the
current model on the new photos; if it is new cuisine, route those users to confirm-carbs and queue
the photos for the next labelled fine-tune. Rolling-AUROC or missed-among-cleared drift (from
cgm_auto): freeze clearing (serve caution-only) for the affected people until the per-person
intercepts relearn, and review their recent labels for sensor gaps. Latency alert: profile the cold
path first (model load dominates); nothing else in the pipeline is above 30 ms. Every alert and the
action taken is a dated note in this file; none of this retunes a frozen artifact mid-cycle.

## Replay of the outcome and behaviour rules on logged answers (registered 26 Nov, before the run)

Written after three cold grades found the monitoring demonstration thin (one signal, a six-photo
drift week). This is a check that the rules above compute and fire from logged answers. It is not
new evidence about the model: both input files are committed and their totals (dev 109 cleared, 8
spiked; locked 10 cleared, 2 spiked, 139 of 153 caution) were known when this was written. Nothing in
it changes a frozen artifact, threshold or reported result.

- **Stream.** outputs/evaluation_v2/dev_answers.csv (481 answers, sorted by `row`), then
  outputs/locked_eval/locked_answers.csv (153, sorted by `row`): five months of ordinary use followed
  by a harder population. A "week" is a block of 20 consecutive answers; a final block shorter than
  20 is dropped. A "30-day" window is four consecutive blocks (80 answers), rolling one block at a time;
  windows that span the switch are labelled mixed.
- **Signal 2a, missed-among-cleared** (rule above): spikes among `low risk` answers in the window;
  alert when above `usefulness.max_missed_among_cleared` (0.10), computed only when the window holds at
  least 10 cleared answers (fewer is reported as "too few cleared to judge").
- **Signal 2b, rolling AUROC** of `risk` against `spike` (answers with a risk): alert when below the
  locked-set value (0.645). On locked windows this compares the locked set with itself, so it is
  reported, not interpreted.
- **Signal 3, behaviour.** Weekly shares of low risk / caution / can't judge, reported for every
  block. The design above gives no numeric threshold, so one is added here and marked new: alert when
  a block's caution share exceeds the highest caution share of any dev block. It is written knowing
  the locked total (91% caution), so its firing is a demonstration that it computes, not a test.
- **Calibration-in-the-large** (new, descriptive, standing in for signal 4, which needs months of
  b_p history these files do not hold): mean predicted risk minus observed spike rate per window.
- **Expected:** 2a quiet on dev windows; on locked windows mostly "too few cleared to judge", because
  the frozen cut-offs clear about one meal in fifteen there; signal 3 quiet on dev blocks and firing on
  most locked blocks; calibration-in-the-large near zero on dev and negative (under-prediction) on
  locked. Whatever it shows is reported as measured (`just monitoring-replay` ->
  outputs/monitoring/replay.json).
