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

### Note on the replay above, and a second replay at the deployed operating point (registered 26 Nov, before its run)

Two corrections to the wording above, found by the second grading round. The dev answers sorted by
`row` come person by person, so a "week" of 20 answers holds one to three people: the stream is a
sequence of users arriving, not five months of calendar time. And the dev answers were decided at each
fold's own cut-offs (fold 1: 0.21 / 0.21), not at the deployed (0.15, 0.16), so the dev behaviour shares
and the behaviour threshold describe five different operating points.

Replay 2 keeps every rule of replay 1 (blocks of 20, windows of four blocks, 2a at the 0.10 cap with at
least 10 cleared, 2b against 0.645 reported only, calibration-in-the-large descriptive) and changes one
thing: every dev answer that has a risk is re-decided at the deployed cut-offs (low risk below 0.15,
caution from 0.16, can't judge between), the decision the app would have given; answers without a risk
stay can't judge. The locked answers already used the deployed cut-offs and are unchanged. The behaviour
threshold becomes the highest caution share of any dev block under these decisions. Nothing is refitted.
Expected: more dev cautions than in replay 1 (the deployed caution cut-off 0.16 is below most fold
cut-offs), 2a still noisy on dev windows, locked windows still too few cleared to judge, and the
behaviour rule able to fire only if some locked block exceeds every dev block. Reported as measured in
outputs/monitoring/replay.json (`deployed_operating_point`). Replay 1's calibration-in-the-large
expectation for the locked stream (negative) was wrong: it measured +0.038.

### Results and corrections after the replays (26 Nov)

Measured: replay 1, 7 of 21 dev windows alert; replay 2, 3 of 21; the windows overlap, so each run is one or two
clusters of misses (weeks 12-22), not separate alarms. Every locked-only window has too few cleared answers;
the mixed window spanning the switch (weeks 24-27) alerts in both. Calibration-in-the-large: the registered
expectation (negative on the locked stream) was checked against the median of four overlapping windows,
+0.038; pooled over the whole locked stream it is −0.010, so the two summaries disagree in sign and the
expectation is not cleanly met or missed. Two lines above overstate: "nothing else in the pipeline is above
30 ms" has no per-stage timing behind it (outputs/efficiency/ times whole answers only), and the "latency
alert" names a response without a defined signal; latency is measured, not monitored.
One more correction (26 Nov, third grading round): replay 2 is described above as "the decision the app
would have given". It re-decides the evaluated, with-history risk at the deployed cut-offs; the app and the
folder command answer without history (Platt only), whose usefulness is reported separately in the report §4
(served column, outputs/report_metrics/report_metrics.json `served_operating_point`).

### Replay 3: a sequential calibration monitor on the served risk (registered 26 Nov, before its run)

Written after the fourth grading round, knowing that the served (Platt-only) risk under-predicts the locked
cohort by 11 points overall; so this is a demonstration that a standard sequential rule computes on the
logged stream, not a test. Stream and blocks as in replay 1 (dev answers in `row` order, then locked; blocks
of 20). Per block, x = observed spike rate minus mean served risk (dev: `risk_platt`; locked: the frozen
Platt map on `risk_raw`), over answers with a risk. Standardise with the mean and standard deviation of x over
the dev blocks. One-sided CUSUM for under-prediction, textbook defaults fixed now: S_t = max(0, S_{t-1} +
z_t − 0.5), alarm when S_t > 4, reset after an alarm. The same rule on the caution share (x = caution share)
is reported alongside. Expected: few or no alarms on dev blocks (they define the reference, so a dev alarm
would be a false alarm), and a calibration alarm within the locked blocks. Reported as measured in
outputs/monitoring/replay.json (`cusum_served`).

Result and correction (26 Nov, after the sixth grading round). The calibration CUSUM did not alarm. The
caution-share CUSUM alarmed once, in the 7th locked week, with no dev false alarm; but its dev blocks carry
each fold's own cut-offs and its locked blocks the deployed ones, so a shift at the seam can come from the
cut-off change alone. Two rows added after that result put both streams at the deployed cut-offs, on the
shipped risk (`caution_share_deployed_cutoffs`) and on the served risk (`caution_share_served`): neither
alarms. Seventh-round correction (26 Nov): "artifact" overstated it. At the deployed cut-offs the shipped
risk's statistic still climbs to 2.6 of 4 by the last locked week, so the step mixes the cut-off change with
a real rise in the caution share; the registered calibration expectation (a locked alarm) failed, its
statistic peaking at 2.4. No rule caught the real shift cleanly.

Two older lines are stale (26 Nov): "nothing else in the pipeline is above 30 ms" (alert actions above) is
contradicted by outputs/efficiency/stage_timing.json, where the ConvNeXt pass alone takes 47 ms on a 2-CPU
machine; and the folder command now accepts an optional history.csv, so "answer without history" in the
replay-2 correction holds only when that file is absent.

### Replay 4: detection power under a realistic cohort shift (registered 26 Nov, before its run)

Written by the assistant after the graders asked whether any rule can catch a real shift. Data-free, on
the committed v2 development answers (the served product). 500 simulated streams. Each stream: people in
a random order, their answers in time order, in blocks of 20; the first 12 blocks draw from all 23
development people, then the stream switches to the 12 people with prediabetes only (spike rate 43%
against 33% overall, close to the locked cohort's 42%). Rules, fixed now: the calibration CUSUM of
replay 3 (block spike rate minus block mean risk) and the caution-share CUSUM, both one-sided, k = 0.5,
h = 4, standardised on the stream's own first 8 blocks (a deployment's burn-in). Reported: the share of
streams with a false alarm before the switch, the share detecting within 6 blocks (120 answers) after
it, and the median delay. Expectation: the calibration rule detects in most streams within 6 blocks; the
caution-share rule detects more slowly. Output: outputs/monitoring/power.json.
