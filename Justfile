# GLIMPSE task runner: short names for the commands used most often. Run `just` to list them.

# list the recipes
default:
    @just --list

# install Python 3.14 + locked libraries, download the data and the pinned photo weights, prepare the dataset
install:
    uv sync
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

# the three training runs in order (the deployed artifacts then come from `just finalize`)
train: train-scratch train-photo train-final

# train the from-scratch SensorCNN: epochs chosen on inner folds, then the 28-person model + checkpoint
train-scratch:
    uv run glimpse train --model scratch

# train the photo model: frozen ConvNeXt-Tiny + heads; last stage kept only if the inner folds show a gain
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

# the six report figures, rendered from committed evidence only (pre-commitment: nothing retuned from their look)
figures:
    uv run glimpse figures

# pre-registered meal-time shift robustness, 15 minutes either way (7 Nov)
timing:
    uv run glimpse timing

# the 23 hostile input folders through the instructor command (outputs/stress_run)
stress:
    uv run glimpse predict-folder samples/stress outputs/stress_run

# final 28-person pipeline: fusion + thresholds chosen on the validation people and frozen (~8 min; test untouched)
train-final:
    uv run glimpse final

# cut-off amendment: being rebuilt on partition v2 (S5, D4); the command says so and exits
amend-thresholds:
    uv run glimpse amend-thresholds

# dev evaluation on partition v2 (every risk from Pipeline.answer, one Pipeline per fold), then `report-numbers`
evaluate:
    uv run glimpse evaluate
    just report-numbers

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
    just photo-seeds
    just evaluate
    just selftest
    just predict-folder samples/instructor outputs/folder_run
    just stress
    just demo

# re-derive every report table and figure from committed outputs (no dataset or weights needed, ~1 min)
report-numbers:
    uv run glimpse report-metrics
    uv run glimpse monitoring-replay
    uv run glimpse figures
    just wordcount

# the outcome and behaviour monitoring rules replayed on the logged answers (registered 26 Nov in docs/monitoring.md)
monitoring-replay:
    uv run glimpse monitoring-replay

# descriptive metrics from the committed answer files (outputs/report_metrics/report_metrics.json)
report-metrics:
    uv run glimpse report-metrics

# seed spread of the shipped 12+2 recipe: 3 fresh seeds scored on the validation people (~15 min; nothing deployed changes)
photo-seeds:
    uv run glimpse photo-seeds

# superseded by partition v2: the deployed calibration comes from `just finalize`; the command says so and exits
recalibrate:
    uv run glimpse recalibrate

# degradation across increasing stress (rubric 4d; pre-registered 22 Nov; ~20 min, live photo passes)
robustness-levels:
    uv run glimpse robustness-levels

# S16: the 8 locked people, scored exactly ONCE against the frozen artifacts; the guard refuses a second run
evaluate-locked:
    uv run glimpse evaluate-locked

# the intermediate's promised comparisons, run once against the frozen evidence (pre-registered 8 Nov; ~20 min)
promised:
    uv run glimpse promised

# demo with the frozen pipeline: worked cases from the test people, two can't-judge cases, response time
demo:
    uv run glimpse demo
    @echo "UI: just app -> http://127.0.0.1:7860 (optional Gradio extra)"

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
    git checkout -- outputs/demo  # the demo's committed example files are evidence: restore them, never delete them
    find . -name "__pycache__" -type d -prune -exec rm -rf {} +

wordcount:
    uv run python -c "t=open('docs/final_report.md').read(); ls=[l for l in t.splitlines() if not (l.lstrip().startswith(('|','![')) or (l.lstrip().startswith('*') and not l.lstrip().startswith('**')))]; print(len(' '.join(ls).split()), 'words (cap 3,000; tables, figures and their captions excluded per the final rubric)')"
