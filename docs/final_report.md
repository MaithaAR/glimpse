# GLIMPSE: final report
Before eating: a meal photo + the last 2 h of CGM glucose and heart rate + a short profile → **low risk**,
**caution** or **can't judge** (with a reason). Offline, local weights, one Mac.
Maitha Alhosani · MAAI7103 · November 2026 · `just install` → `just predict-folder <in> <out>`

<!-- BUDGET 3,000 words: bottom line 180 · §1 350 · §2 525 · §3 650 · §4 550 · §5 475 · §6 300 · §7 200.
     Check with `just wordcount`. TODO markers: [S16] filled 22 Nov, [DIET] 19 Nov, [REGEN] 12 Nov. -->

**Bottom line.** The central fix this phase was calibration, and it worked: a per-person intercept,
fitted only on each person's earlier meals and replayed in deployment order, raises AUROC from 0.726
to 0.762 and improves every one of the five folds individually, while halving the per-person
calibration gaps the intermediate diagnosed. The photo's contribution was put through a pre-registered
gate and did not pass: +0.025 AUROC with a 95% interval crossing zero (−0.010, +0.061), so, per the
paragraph committed before the run, the shipped product is sensor + confirmed carbs, with the photo
head as a carb-entry assist rather than an established risk signal. Safety behaves by design: 8 of 109
cleared meals spiked (7.3%, interval 2.6–14.4% against a pre-registered 10% cap), degraded input
triggers refusal rather than a changed answer, and a 23-case adversarial input suite produces zero
crashes and zero raw error rows. [S16: one sentence with the locked-8 result, scored once on 22 Nov —
whatever it shows.]

## 1. System and tooling
<!-- 350 words: condensed user/workflow from the intermediate; ONE scoring path (D2); the instructor
command; efficiency table below; tooling line (uv, pydantic, loguru, ruff, 129 tests, CI, W&B). -->
TODO prose. Must include: every reported number, the demo, the app and the grader's
`just predict-folder` go through the same `Pipeline.answer`; rules run before models.

| Measured on the shipped pipeline | value | cap |
|---|---|---|
| largest model (ConvNeXt-Tiny + heads) | 27.8M | 200M |
| pretrained total | 27.8M | 1B |
| trained from scratch (heads 3,845 + SensorCNN 5,217 + fusion 20) | 9,082 | 50M |
| cold first answer / warm median / p95 | 1.55 s / 264 ms / 270 ms | ≤300 ms p95 |
| peak memory / checkpoint | 797 MB / 31 MB | – |

## 2. Multimodal design and the photo gate
<!-- 525 words: architecture (frozen backbone + two heads, SensorCNN as RQ2 baseline only, LR fusion
with capacity argument: 23 dev people support ~20 params); photo CV quality (22.6 g MAE, Spearman .53,
bin F1 .61 vs ridge-on-true-macros F1 .38); then the gate. -->
TODO prose. MUST-APPEAR SENTENCE 1: "The pre-registered gate returned +0.025 (−0.010, +0.061; shuffled
control −0.010); per pre-commitment (docs/gate_s8.md, committed before the run, losing paragraph
deleted in git) the shipped product is sensor + confirmed carbs, with the photo head as carb-entry
assist, not an established risk signal."
Supporting: RQ1 — true logged carbs gain only +0.002 over profile+trend, an information ceiling no
better photo encoder can break; D8 — excluding training meals that share a macro profile with a held
person's meal kills the gain (−0.005) while a size-matched random drop keeps it (+0.037): the small
gain rides on CGMacros' repeated standardised meals; channels — dropping HR +0.007 (kept: feature set
fixed before the ablation), dropping profile −0.122 (the backbone). RQ2: SensorCNN 0.696 vs
engineered-feature LR 0.691, gain +0.004 (−0.040, +0.045) — the scratch model does not beat the
hand-made trend features.

## 3. Evaluation and ablations
<!-- 650 words: partition v2 (5 outer folds, 4 inner folds = every selection; 14 T2D train-only;
481 dev meals scored exactly once, splits/reconciliation.md; fold 1 declared = intermediate's scored
people); the headline and per-fold tables; calibration method (Platt on inner-OOF + penalised personal
intercept, 2-h label gate, replay); (m, lo, hi) by min expected cost on replayed inner risks; m basin. -->
TODO prose. MUST-APPEAR SENTENCE 3a: "The personalised model beats raw in all five outer folds; the
pooled estimate excluding the previously-scored fold 1 is 0.757 (0.651–0.833) against 0.762 overall."
Pooling-artefact sentence: "Within every fold the Platt AUROC equals the raw AUROC exactly — a
monotone per-fold map cannot reorder meals inside its fold — so the pooled Platt figure (0.714) below
raw (0.726) is an artefact of mixing five fold-specific maps, and the calibration evidence is the
log-loss (0.537 personal vs 0.572/0.567) and the per-person gaps, not AUROC."

| risk (dev, 481 meals) | AUROC (95% person-clustered) | log-loss |
|---|---|---|
| raw fusion | 0.726 (0.659–0.797) | 0.567 |
| + per-fold Platt | 0.714 pooled (= raw within every fold) | 0.572 |
| + personal intercept (shipped) | **0.762 (0.682–0.829)** | **0.537** |
| shipped, excluding fold 1 | 0.757 (0.651–0.833) | 0.547 |

| fold | n | raw | personal | m | cut-offs (lo, hi) |
|---|---|---|---|---|---|
| 1 | 90 | .742 | .790 | 1.0 | .21, .21 |
| 2 | 105 | .614 | .661 | 2.0 | .13, .23 |
| 3 | 104 | .755 | .772 | 0.5 | .13, .20 |
| 4 | 67 | .757 | .787 | 1.0 | .15, .17 |
| 5 | 94 | .788 | .809 | 1.0 | .14, .14 |

Per-person gaps halved: rare spikers predicted 20.9% vs observed 14.2% (was 29.3%); frequent 42.6% vs
52.9% (was 37.0%). Cut-off variation (two folds chose no abstention band) is reported as a finding:
the cost optimiser priced abstention at zero value for those inner distributions. The cost-vs-m curve
is a flat basin (m ∈ [0.5, 2] across folds). [REGEN: deployed config values frozen from the all-dev
replay on 10–12 Nov.]

## 4. Trust and safety
<!-- 550 words: cost matrix as declared assumption (missed spike 5× unnecessary caution; abstain 2,
0.5) with DCA framing (Vickers & Elkin 2006: the threshold IS the harm ratio) and the pre-registered
sensitivity sweep; usefulness vs pre-registered targets; robustness table; stress suite; monitoring. -->
TODO prose. Targets fixed before results (docs/evaluation_protocol.md): cost reduction 14% (min 10%),
cleared 23% (min 20%), abstain 14% (max 20%), missed-among-cleared 7.3% (max 10%).
MUST-APPEAR (soft spot d): "8 of 109 cleared meals spiked (7.3%; person-clustered 95% CI 2.6–14.4%):
within the cap at the point estimate, though with only 17 people contributing cleared meals the
interval's upper bound exceeds it — a limit of cohort size, monitored continuously in deployment,
where the CGM labels every meal at +2 h (cgm_auto). The 8 misses spread across 6 people."

| perturbation (same per-fold pipelines, no history) | unchanged | dangerous flips | newly can't-judge |
|---|---|---|---|
| CGM dropout 30% | 100% | 0.0% | 0% |
| 25-min pre-meal gap | 11% | 0.0% | **89% (refusal, by design)** |
| photo blur r=6 | 94% | 1.3% | 4% |
| photo darken ×0.25 | 97% | 0.2% | 2% |

Stress suite: 23 hostile inputs (timezone offsets, malformed JSON, binary CSVs, truncated/CMYK/GIF
photos, empty folders...) → zero crashes, zero raw error rows; rules R8–R10 added only against
observed failures of a recorded naive run (commits 3ee062b → 3cf9973), like R1–R6 before them;
the invariant is enforced by tests on every commit. Monitoring: four signals with alert rules
(docs/monitoring.md); drift baseline committed. [DIET: two dietitians, three structured questions —
cap acceptability, label wording, the rising-baseline case; responses folded in here, or the honest
pending line.]

## 5. Explainability and failure analysis
<!-- 475 words: probes A+B; R7; D8 as failure insight; S14 taxonomy (false-cleared + abstention
breakdown) TODO 10 Nov; label-definition discussion. -->
Failure taxonomy (every class observed, none hypothetical). **T1 rule refusals:** malformed or
degraded input never reaches a model — 13 instructor and 23 stress cases all end in a structured
refusal naming the problem (§4). **T2 missing-photo abstentions:** 21 of 481 dev meals have no usable
photo; the system abstains rather than guessing carbs. **T3 borderline abstentions:** 45 meals, all in
folds 2–4 — folds 1 and 5 chose no abstention band, so where the band exists is itself a product of
per-fold cost optimisation, and we report it as such rather than as a stable property. **T4 near-cut
false-clears:** 3 of the 8 missed-among-cleared sat within 0.03 of their fold's cut — boundary
sensitivity, the price of any hard threshold. **T5 confident false-clears, the genuine blind spot:**
the other 5 were cleared at risks 0.03–0.13, far from the cut, each with 10–20 meals of personal
history available — personalisation had data and still cleared them; they spread across 5 people, so
no single outlier explains them. **T6 confident-wrong carb estimate (R7):** a visually familiar,
nutritionally unfamiliar local dish passes the kNN check and gets a wrong bin — structural, detailed
below. **T7 label-definition surprise (probe B):** not a model failure but a communication one; also
below. MUST-APPEAR SENTENCE 2: "Probe B failed exactly as a label-correct model
should: the label is baseline-relative, so a rising pre-meal trace lowers predicted spike probability
— the registration, not the model, encoded naive physiology (pre-registered with a committed
expected-fail, 17da63a)." Probe A: risk moves in the photo's direction on 100% of meals, yet the
extreme bottom↔top-decile plate swap leaves 68% of decisions unchanged (median swing 0.102) — the
gate's verdict seen mechanistically. R7 (rule ledger): the kNN check measures visual, not nutritional,
familiarity — a local lamb-and-rice dish scores 0.346 against a 0.609 cut (more familiar than a tuna
bowl, 0.420) and gets a confident wrong carb bin; the cut sweep under a ≤2% false-flag budget shows no
threshold fixes it; mitigations are the confirm-carbs tier and monitoring; a nutrition-aware embedding
is named future work. Clinical note the probe forces: a user already rising into a meal may see "low
risk" while clinically ascending — a rise-over-last-value label is worth future work; the app's
monitoring design includes the separate rising-now warning the dietitians were asked about.

## 6. Reproducibility
<!-- 300 words: fresh-clone chain; one-Mac story + CSCC sentence; W&B; manifest; locked-8 protocol. -->
TODO prose. MUST-APPEAR SENTENCE 3: "The 8 locked subjects were scored once, on 22 Nov, from frozen
artifacts verified by a fresh-clone run; every reported number comes from the same Pipeline.answer
path the instructor command invokes." Fresh-clone dry-run (5 Nov): clone + `uv sync` + one documented
weights step (pinned HF revision, SHA-256) reproduces selftest, the 13-case instructor suite and the
23-case stress suite; the dry run itself found and fixed two onboarding defects. Every number has a
`just` recipe; W&B offline exports committed; cluster access existed but the deployed system trains
and runs on one Mac overnight, and the reproducibility environment was deliberately not split.
PRE-COMMITTED (before S16): the locked-8 result is reported as measured, with the same tables as §3,
whatever it shows; no re-tuning follows from it.

## 7. Limitations and conclusion
<!-- 200 words: n=45 one study; repeated standardised meals (D8); no UAE data; advisory only, no
clinical validation or regulatory review; T2D train-only so no claim for them; HR adds nothing;
photo not established as risk signal. Close on: the discipline (pre-registration, score-once,
refusal-by-design) is the contribution that survives any single number. -->
TODO prose.
