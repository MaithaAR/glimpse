"""End-to-end pipeline tests on the bundled synthetic samples and on malformed inputs."""

import json
import shutil
from datetime import datetime, timedelta

from glimpse.config import load_config
from glimpse.data.preprocess import cgm_window
from glimpse.pipeline.predict import predict
from glimpse.schemas import AbstainReason, GlucoseReading

CFG = load_config("configs/default.yaml")


def test_cgm_window_shape_and_gap():
    """A full 2-hour trace gives 24 points; a missing last 50 minutes is a gap."""
    meal = datetime(2026, 1, 1, 12, 0)
    full = [GlucoseReading(timestamp=meal - timedelta(minutes=5 * k), glucose_mgdl=100.0) for k in range(23, -1, -1)]
    w = cgm_window(full, meal, 120)
    assert w.values.shape == (24,) and w.coverage == 1.0 and w.max_gap_min <= 5
    w2 = cgm_window([r for r in full if r.timestamp <= meal - timedelta(minutes=50)], meal, 120)
    assert w2.max_gap_min >= 50 and w2.readings_last_30min == 0


def test_normal_case_answers():
    """Complete inputs produce an answer with a probability in [0, 1]."""
    card = predict("samples/normal_case", CFG)
    assert card.status == "answer" and 0 <= card.spike_probability <= 1


def test_gap_case_abstains():
    """A CGM gap produces a structured abstention with explicit reasons."""
    card = predict("samples/abstain_cgm_gap", CFG)
    assert card.status == "abstain" and AbstainReason.CGM_GAP in card.reasons


def test_malformed_and_unsupported_inputs(tmp_path):
    """Missing files, corrupt photos and invalid profiles never crash: they abstain with a reason."""
    assert predict(tmp_path / "does_not_exist", CFG).reasons == [AbstainReason.INVALID_INPUT]
    bad = tmp_path / "bad_photo"
    shutil.copytree("samples/normal_case", bad)
    (bad / "photo.jpg").write_bytes(b"not an image")
    assert AbstainReason.UNREADABLE_PHOTO in predict(bad, CFG).reasons
    meta = json.loads((bad / "meal.json").read_text())
    meta["profile"]["age"] = 7
    (bad / "meal.json").write_text(json.dumps(meta))
    assert predict(bad, CFG).reasons == [AbstainReason.INVALID_INPUT]
