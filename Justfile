# GLIMPSE task runner: short names for the commands used most often. Run `just` to list them.

# list the recipes
default:
    @just --list

# install Python 3.14 + locked libraries, download the data and the pinned photo weights, prepare the dataset
install:
    uv sync --locked
    uv run glimpse download-data
    uv run glimpse download-weights
    uv run glimpse prepare

# rebuild the meal table and the model-ready dataset from the downloaded data
prepare:
    uv run glimpse prepare

# partition v2 (final stage): splits/dev_splits_v2.json + the asserted meal-level reconciliation
splits-v2:
    uv run glimpse splits-v2

# run the test suite
test:
    uv run pytest

# load and validate configs/default.yaml, start logging, set the seed
check-config:
    uv run glimpse check-config

# check code style and errors (no print(), sorted imports, line length)
lint:
    uv run ruff check src tests
    uv run ruff format --check src tests

# lint, then test
check: lint test

# count every model's parameters against the limits (writes outputs/param_budget.csv)
params:
    uv run glimpse params

# the intermediate-stage training runs (old 28-person split); the shipped artifacts come only from `just finalize`
train: train-scratch train-photo train-final

# train the from-scratch SensorCNN: epochs chosen on inner folds, then the 28-person model + checkpoint
train-scratch:
    uv run glimpse train --model scratch

# train the photo model: ConvNeXt-Tiny heads, then the last stage if the inner folds show a gain (intermediate)
train-photo:
    uv run glimpse train --model photo

# cross-validated out-of-fold predictions of the CNN and every comparison model (main estimate) -> outputs/cv/
cv:
    uv run glimpse cv

# logged-carbs check: profile + trend with vs without the logged carbs, same folds; decides whether the photo CV runs
carbs-check:
    uv run glimpse carbs-check

# photo cross-validation: recipe re-chosen inside each outer fold; OOF + nested carb estimates (~1.5 h, resumable)
photo-cv *ARGS:
    uv run glimpse photo-cv {{ARGS}}

# fusion: profile + trend with no meal / photo / logged carbs / shuffled photo (+ post-hoc ridge arm); ~2 min
fusion:
    uv run glimpse fusion

# S6 ablations: D8 profile-excluded training + input-channel ablations (same folds and C as `fusion`)
ablate:
    uv run glimpse ablate

# offline W&B runs regenerated from the committed CSVs/JSONs (wandb sync outputs/wandb/... uploads them)
wandb-export:
    uv run glimpse wandb-export

# S10: unfamiliar-photo check against the 14 real phone photos; --apply writes the re-chosen cut (R7)
ood *ARGS:
    uv run glimpse ood {{ARGS}}

# S9: decision-flip rates under the four pre-registered perturbations (CGM dropout/gap, photo blur/dark)
robustness:
    uv run glimpse robustness

# the two pre-registered counterfactual probes, photo swap and flat-vs-rising (S13)
probes:
    uv run glimpse probes

# parameters vs caps, serving latency and artifact sizes on this machine (S12)
efficiency:
    uv run glimpse efficiency

# drift baseline + the simulated four-week stream with the week-4 alert (S11, rubric 4e)
monitoring:
    uv run glimpse monitoring

# the pre-registered serving-artifact regeneration (docs/final_design.md amendment); stops itself on any criterion miss
finalize:
    uv run glimpse finalize

# the eight figures (seven in the report, Figure A1 in the appendix), rendered from committed evidence only (pre-commitment: nothing retuned from their look)
figures:
    uv run glimpse figures

# pre-registered meal-time shift robustness, 15 minutes either way (7 Nov)
timing:
    uv run glimpse timing

# the 25 hostile input folders through the instructor command (outputs/stress_run)
stress:
    uv run glimpse predict-folder samples/stress outputs/stress_run

# intermediate-stage 28-person fusion + thresholds (superseded by `just finalize`; refuses to overwrite it)
train-final:
    uv run glimpse final

# where serving time goes, stage by stage, random-initialised backbone (no download; 26 Nov, descriptive)
stage-timing:
    uv run glimpse stage-timing

# whole folder-command answers for v1 and v2, with and without a photo (report §1 timing; ~2 min with weights)
end-to-end-timing:
    uv run glimpse stage-timing --end-to-end

# v2 under missing heart rate, missing priors and noisy history (docs/v2_explain.md, V2-S; ~1 min with data)
v2-input-stress:
    uv run glimpse v2-input-stress

# v2 probes B and C and input monitors (docs/v2_hardening.md, Part B; probe B needs the data, ~2 min)
v2-probes:
    uv run glimpse v2-probes

# registered last-day checks (docs/final_switch.md): OOD AUROC, out-of-scope, monitoring power, calibration, course checks (~3 min)
final-checks:
    uv run glimpse final-checks

# rule R16 for v1: training ranges (written once), counts and the heart-rate sweep
v1-input-ranges:
    uv run glimpse v1-input-ranges

# v2 calibration, failure taxonomy and exact SHAP (docs/v2_explain.md; data-free once the features are committed)
v2-explain:
    uv run glimpse v2-explain

# every report table: budget, both dev evaluations, v2 metrics and probes, OOD, monitoring, probes A and B, then
# `report-numbers` (needs data and weights; about 15 min on a laptop CPU). evaluate-all adds the slow studies.
evaluate:
    uv run glimpse params
    uv run glimpse evaluate
    uv run glimpse evaluate --sensor-first
    uv run glimpse v2-metrics
    uv run glimpse v2-probes
    uv run glimpse ood
    uv run glimpse monitoring
    uv run glimpse probes
    just report-numbers

# `evaluate` plus the cheaper compute studies behind the report's other tables (~45 min with data and weights)
evaluate-all:
    just fusion
    just ablate
    just promised
    just evaluate
    just stress-quality
    just v2-input-stress
    just photo-probes
    just end-to-end-timing
    just robustness-levels
    just final-checks
    just report-numbers

# v2 (docs/v2_sensor_first.md): the sensor-first freeze, written once to checkpoints/sensor_first/
finalize-sensor-first:
    uv run glimpse finalize-sensor-first

# v2 beside v1 on development (needs the dataset for the samples' path)
v2-metrics:
    uv run glimpse v2-metrics

# quality under stress at the evaluated operating point, v2 and v1 (registered 26 Nov; ~25 min)
stress-quality:
    uv run glimpse stress-quality

# photo probes D and E and int8 against fp32 (registered 26 Nov; ~10 min)
photo-probes:
    uv run glimpse photo-probes

# monitoring replay 4: CUSUM detection power under a simulated cohort shift (data-free)
monitoring-power:
    uv run glimpse monitoring-power

# every compute study behind the report in dependency order, ~3 h (not `finalize` or `evaluate-locked`)
reproduce-all:
    just prepare
    just splits-v2
    just params
    just cv
    just carbs-check
    just photo-cv
    just fusion
    just ablate
    just promised
    just robustness
    just robustness-levels
    just timing
    just probes
    just ood
    just monitoring
    just efficiency
    just stage-timing
    just stress-quality
    just v2-input-stress
    just photo-probes
    just end-to-end-timing
    just photo-seeds
    just evaluate
    just selftest
    just predict-folder samples/instructor outputs/folder_run
    just predict-folder samples/returning_user outputs/returning_user_run
    just stress
    just demo-cli

# regenerate the data-free outputs and fail if they differ from the committed ones (no dataset or weights needed)
verify-outputs:
    just report-numbers
    uv run glimpse verify-outputs

# re-derive every report table and figure from committed outputs (no dataset or weights needed, about 5 min on a 2-CPU machine)
report-numbers:
    uv run glimpse report-metrics
    uv run glimpse monitoring-replay
    uv run glimpse monitoring-power
    uv run glimpse probe-c
    uv run glimpse v2-explain
    uv run glimpse v2-probes --data-free
    uv run glimpse figures
    just wordcount

# the outcome and behaviour monitoring rules replayed on the logged answers (registered 26 Nov in docs/monitoring.md)
monitoring-replay:
    uv run glimpse monitoring-replay

# descriptive metrics from the committed answer files (outputs/report_metrics/report_metrics.json)
report-metrics:
    uv run glimpse report-metrics

# seed spread of the shipped 12+2 recipe: 3 fresh seeds scored on the validation people (~10-15 min; nothing deployed changes)
photo-seeds:
    uv run glimpse photo-seeds

# degradation across increasing stress (rubric 4d; pre-registered 22 Nov; ~20 min, live photo passes)
robustness-levels:
    uv run glimpse robustness-levels

# S16: the 8 locked people, scored exactly ONCE against the frozen artifacts; the guard refuses a second run
evaluate-locked:
    uv run glimpse evaluate-locked

# the intermediate's promised comparisons, run once against the frozen evidence (pre-registered 8 Nov; ~20 min)
promised:
    uv run glimpse promised

# demo: worked cases (in-sample for the frozen model), two can't-judge cases, response time, then the interface
demo:
    uv run glimpse demo
    @echo "starting the demo interface at http://127.0.0.1:7860 (Ctrl-C to stop; CLI only: just demo-cli)"
    uv run --extra app python -m glimpse.app

# the CLI half of the demo only (no interface; used by reproduce-all)
demo-cli:
    uv run glimpse demo

# answer one meal from your own files, e.g. just predict --cgm my_cgm.csv --photo lunch.jpg --age 45 --bmi 28 --a1c 5.9 --fasting 100
predict *ARGS:
    uv run glimpse predict {{ARGS}}

# answer every meal folder under IN (the instructor's entry point), e.g. just predict-folder samples/instructor outputs/folder_run
predict-folder IN OUT="outputs/folder_run":
    uv run glimpse predict-folder {{IN}} {{OUT}}

# offline check that every frozen part loads and the full path runs (no dataset needed)
selftest:
    uv run glimpse selftest

# Linux container (needs Docker): build, offline self-test, and the demo with your own CGMacros copy mounted read-only
docker-build:
    docker build -t glimpse .

docker-selftest:
    docker run --rm --network none glimpse selftest

docker-demo:
    docker run --rm --network none -v "$PWD/data/cgmacros:/app/data/cgmacros:ro" -v "$PWD/data/dataset.json:/app/data/dataset.json:ro" -v "$PWD/data/dataset.npz:/app/data/dataset.npz:ro" glimpse demo

# the GLIMPSE screen in the browser (http://127.0.0.1:7860); needs the optional `app` extra (Gradio)
app:
    uv run --extra app python -m glimpse.app

# remove generated files (keeps downloaded data, the locked environment, checkpoints and committed results)
clean:
    rm -rf logs .pytest_cache .ruff_cache outputs/feedback
    git checkout -- outputs/demo 2>/dev/null || true  # committed demo files are evidence: restore, never delete
    find . -name "__pycache__" -type d -prune -exec rm -rf {} +

# prose words in the report, tables, figures and their captions excluded (the final rubric's rule)
wordcount:
    uv run python -c "t=open('docs/final_report.md').read(); ls=[l for l in t.splitlines() if not (l.lstrip().startswith(('|','![')) or (l.lstrip().startswith('*') and not l.lstrip().startswith('**')))]; print(len(' '.join(ls).split()), 'words (cap 3,000; tables, figures and their captions excluded per the final rubric)')"

# probe C: profile and onboarding priors counterfactual on the frozen fusion (registered 26 Nov; data-free)
probe-c:
    uv run glimpse probe-c
