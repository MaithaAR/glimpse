"""Tests for configuration validation and Pydantic contracts."""

import pytest
import yaml
from pydantic import ValidationError

from glimpse.config import GlimpseConfig, load_config
from glimpse.schemas import AbstainReason, Profile, RiskCard


def test_default_config_loads():
    """The shipped config is valid."""
    cfg = load_config("configs/default.yaml")
    assert cfg.decision.low_risk_below <= cfg.decision.caution_at_or_above


def test_config_rejects_unknown_key_and_bad_value():
    """Unknown keys and out-of-range values fail at startup with a clear error."""
    raw = yaml.safe_load(open("configs/default.yaml"))
    raw["typo_key"] = 1
    with pytest.raises(ValidationError):
        GlimpseConfig.model_validate(raw)
    raw = yaml.safe_load(open("configs/default.yaml"))
    raw["decision"] = {"low_risk_below": 0.5, "caution_at_or_above": 0.2}
    with pytest.raises(ValidationError):
        GlimpseConfig.model_validate(raw)


def test_profile_bounds():
    """Domain bounds on profile fields."""
    with pytest.raises(ValidationError):
        Profile(age=12, bmi=25.0)


def test_riskcard_consistency():
    """An abstention must carry a reason; an answer must carry a decision."""
    common = {"message": "x", "thresholds": (0.2, 0.3), "model_version": "t", "latency_ms": 1.0}
    with pytest.raises(ValidationError):
        RiskCard(status="abstain", decision=None, reasons=[], **common)
    with pytest.raises(ValidationError):
        RiskCard(status="answer", decision=None, **common)
    card = RiskCard(status="abstain", decision=None, reasons=[AbstainReason.CGM_GAP], **common)
    assert RiskCard.model_validate_json(card.model_dump_json()) == card
