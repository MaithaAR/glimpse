"""Mocked feedback loop: labels follow the training rule, records are validated and appended."""

from datetime import datetime, timedelta

from glimpse.config import load_config
from glimpse.feedback import auto_rise, label_from, record_feedback

MEAL = datetime(2020, 5, 1, 12, 0)


def test_label_rules():
    """Not eaten wins; otherwise the rise decides against the spike threshold; nothing known -> unknown."""
    assert label_from("did not eat it", 80.0, 50) == "not eaten"
    assert label_from("ate it all", 50.0, 50) == "spike"
    assert label_from("ate part of it", 20.0, 50) == "no spike"
    assert label_from(None, None, 50) == "unknown"


def test_cgm_auto_rise_uses_the_training_rule():
    """Peak within 2 h after the meal minus the mean of the 30 min before it; nothing after the horizon counts."""
    cfg = load_config()
    times = [MEAL - timedelta(minutes=20), MEAL, MEAL + timedelta(minutes=60), MEAL + timedelta(minutes=180)]
    assert auto_rise(times, [100.0, 110.0, 170.0, 300.0], MEAL, cfg) == 65.0
    assert auto_rise(times[:2], [100.0, 110.0], MEAL, cfg) is None


def test_feedback_is_appended_as_json_lines(tmp_path):
    """Two records -> two validated lines in the feedback file."""
    cfg = load_config()
    proto = cfg.prototype.model_copy(update={"feedback_file": str(tmp_path / "f.jsonl")})
    cfg = cfg.model_copy(update={"prototype": proto})
    said = {"decision": "caution", "risk": 0.4, "meal_time": "2020-05-01T12:00"}
    assert record_feedback(cfg, said, "ate it all", 62.0).feedback_label == "spike"
    assert record_feedback(cfg, said, "did not eat it", None).feedback_label == "not eaten"
    assert len((tmp_path / "f.jsonl").read_text().splitlines()) == 2
