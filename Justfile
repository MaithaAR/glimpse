# GLIMPSE task runner. Every recipe runs from a clean checkout; no manual steps.
set dotenv-load := false

config := "configs/default.yaml"

# list recipes
default:
    @just --list

# install dependencies, then download model weights and data (idempotent, resumable)
install:
    uv sync
    uv run glimpse download-weights --config {{config}}
    uv run glimpse download-data --config {{config}}

# train both models (scratch sensor CNN and fine-tuned photo model)
train: train-scratch train-finetune

# train the from-scratch 1-D CNN sensor model
train-scratch:
    uv run glimpse train --model scratch --config {{config}}

# fine-tune the pretrained photo model
train-finetune:
    uv run glimpse train --model finetune --config {{config}}

# evaluate models and baselines; writes metrics and calibration outputs to outputs/
evaluate:
    HF_HUB_OFFLINE=1 uv run glimpse evaluate --config {{config}}

# end-to-end demo on the bundled sample inputs (runs offline)
demo:
    HF_HUB_OFFLINE=1 uv run glimpse demo --input samples/normal_case --config {{config}}
    HF_HUB_OFFLINE=1 uv run glimpse demo --input samples/abstain_cgm_gap --config {{config}}

# run the test suite
test:
    uv run pytest

# lint and format check
lint:
    uv run ruff check src tests
    uv run ruff format --check src tests

# remove generated artifacts (keeps downloaded data, weights and committed checkpoints)
clean:
    rm -rf outputs logs .pytest_cache .ruff_cache .coverage htmlcov
    find . -name "__pycache__" -type d -prune -exec rm -rf {} +
