# Model card: GLIMPSE pre-meal spike warning (final, 26 Nov 2026)

The intermediate card is kept at docs/archive/model_card_intermediate.md. Every number here is in
docs/final_report.md or docs/report_appendix.md with its evidence file.

**What it does.** Before a meal it estimates the chance that glucose rises by at least 50 mg/dL above the
30-minute pre-meal mean within 2 h, and answers *low risk*, *caution* or *can't judge* (between the
cut-offs, or when a rule fails), always with a one-sentence reason.

**Intended use.** A wellness nudge for adults with prediabetes who already wear a CGM. Not a medical
device. Not for type 1 diabetes, insulin dosing or treatment decisions. No claim for type 2 diabetes:
those participants were used for training only.

**Two products, one code path** (`prototype.product` in configs/default.yaml).
- **v2, sensor first (served; frozen 26 Nov, docs/v2_sensor_first.md).** Risk from a 16-parameter logistic
  regression over 7 CGM and heart-rate trend features and 8 profile inputs; Platt map (0.98, −0.17);
  per-person intercept from earlier labelled meals (m = 1.0); cut-offs low risk below 0.15, caution from
  0.22, can't judge between. The photo model gives a carb estimate for the user to confirm and never enters
  the risk: a missing, blank, unreadable or unfamiliar photo withholds only that estimate, with a named flag.
- **v1, photo fusion (frozen 6 Nov; the model the locked eight measured).** Adds the photo model's grams and
  carb range as 4 of 19 inputs; Platt (0.96, −0.20), m = 1.0, cut-offs 0.15 and 0.16; refuses a meal
  without a usable photo. Its files are unchanged since the tag.
- Rules R1-R12 run first and refuse with a named reason; output rule R13 withholds low risk when there is
  neither history nor both onboarding priors (docs/rule_ledger.md).
- Photo model: ConvNeXt-Tiny (ImageNet-22k pre-trained, 1k fine-tuned; pinned revision and SHA-256), last
  stage fine-tuned, two heads; 27.8M parameters, 15.5M fine-tuned. It failed its pre-registered risk gate
  (+0.025 AUROC, −0.010 to +0.061), which is why v2 exists. It reads the plate, not the background, but does
  not scale grams with apparent portion size (probes D and E).
- Unfamiliar-photo check: kNN distance to the training photos, cut 0.614.
- SensorCNN (5,217 parameters, from scratch) is evaluated, not deployed: it ties the trend features.

**Training data.** CGMacros v1.0.0 (PhysioNet, CC BY-NC-SA 4.0): 37 non-locked adults (23 healthy or
prediabetic development people, 14 type 2 diabetes), about 10 days each, USA, one study, repeated
standardised meals. 1,221 photos at most touched by fine-tuning (cap 10,000).

**Evaluation.**

| | v2, development (481 meals) | v1, development (481) | v1, locked (8 people, scored once 25 Nov, 153) |
|---|---|---|---|
| AUROC (95% person-clustered) | 0.754 (0.673–0.815) | 0.762 (0.682–0.829) | 0.645 (0.540–0.736) |
| meals answered with a risk | 481 | 460 | 150 |
| ECE, 15 equal-width bins (target ≤ 0.05) | 0.029 (0.032–0.091), on track | 0.052, at the noise floor | 0.061 |
| low risk / caution / can't judge | 25% / 67% / 8% | 23% / 64% / 14% | 7% / 91% / 3% |
| spikes among meals cleared (cap 10%) | 12.7%; prediabetes 6 of 30 | 7.3%; prediabetes 1 of 25 | 2 of 10 |
| cost reduction vs always caution (target ≥ 10%) | 10.2% | 14.4% | −5.7% |

v2 minus v1 on the 460 meals both answer: −0.017 (−0.045, +0.010), not non-inferior at the registered
0.03 margin. v2 has no clean test: the locked eight measured v1 once and are never scored again. The
development estimates carry disclosed selection pressure (fold 1 was scored at the intermediate stage).

**Known limits.** The intercept learns each person's base rate and ranks a person's own meals worse
(within-person AUROC about 0.08 lower in both products). v2 is the riskier product for the intended users:
6 of its 30 clears for people with prediabetes spiked; its 15 false clears are 6 near-cut and 9 confident
(docs/v2_explain.md). A missing heart rate turns 17 development cautions into low risk; wrong history
labels cost the most AUROC. `history.csv` is not tagged with the product that wrote it. Without history and priors (every sample),
rule R13 refers every would-be clear. Flips under CGM dropout and short gaps stay at a handful of meals;
gaps of 20 minutes or more refuse by design. Visually familiar but nutritionally unfamiliar dishes get a
confident wrong carb range (R7). The label is baseline-relative, so a rising pre-meal trace lowers the
predicted risk (probe B).

**Monitoring and feedback.** docs/monitoring.md. Photo drift is demonstrated on a synthetic week. The
outcome and calibration rules are too noisy at these volumes: a simulated cohort shift is caught within
120 answers in under a quarter of streams (replay 4).

**Ethics.** De-identified public data under the source study's approval; no new human data; no end-user
or clinical validation.
