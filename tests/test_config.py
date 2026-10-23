"""Settings are loaded strictly: the real file passes, mistakes are rejected with a clear error."""

import copy

import pytest
import yaml
from pydantic import ValidationError

from glimpse.config import GlimpseConfig, load_config

CONFIG = "configs/default.yaml"


def _raw() -> dict:
    with open(CONFIG, encoding="utf-8") as f:
        return yaml.safe_load(f)


def test_real_config_is_valid():
    """The committed settings file passes every check."""
    cfg = load_config(CONFIG)
    assert cfg.label.spike_threshold_mgdl == 50 and cfg.seed == 20260927


def test_out_of_range_value_is_rejected():
    """A probability above 1 is impossible and must stop the program."""
    raw = copy.deepcopy(_raw())
    raw["targets"]["ece_max"] = 2.0
    with pytest.raises(ValidationError):
        GlimpseConfig.model_validate(raw)


def test_unknown_key_is_rejected():
    """A misspelled setting is an error, not silently ignored."""
    raw = copy.deepcopy(_raw())
    raw["label"]["spike_treshold_mgdl"] = 50
    with pytest.raises(ValidationError):
        GlimpseConfig.model_validate(raw)


def test_costs_must_make_a_missed_spike_worst():
    """Costs where a missed spike is cheaper than a caution are rejected."""
    raw = copy.deepcopy(_raw())
    raw["costs"]["low_risk"] = [0, 0]
    with pytest.raises(ValidationError):
        GlimpseConfig.model_validate(raw)


def test_config_is_read_only():
    """Settings cannot be changed by accident while the program runs."""
    cfg = load_config(CONFIG)
    with pytest.raises(ValidationError):
        cfg.seed = 1
