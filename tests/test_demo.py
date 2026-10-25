"""The app's rules: profile order, decision words, glucose checks before the model, unfamiliar-photo distance."""

import math
from datetime import datetime, timedelta
from pathlib import Path

import numpy as np
import torch

from glimpse.config import load_config
from glimpse.data.dataset import PROFILE
from glimpse.data.preprocess import resample_window
from glimpse.demo import decision, knn_distance, leave_one_out_distances, profile_vector, sentence, window_problem

MEAL = datetime(2020, 5, 1, 12, 0)


def _window(minutes_back: list[int], value: float = 100.0):
    times = [MEAL - timedelta(minutes=m) for m in minutes_back]
    return resample_window(times, [value] * len(times), MEAL, 120)


def test_profile_vector_order_and_meal_hour():
    """Same order as training; noon sits at the bottom of the 24-hour circle; unknown prior stays NaN."""
    v = profile_vector({"age": 50, "bmi": 30, "a1c_pct": 5.9, "fasting_glucose_mgdl": 105,
                        "prior_spike_rate": float("nan"), "prior_rise": 40}, MEAL)  # fmt: skip
    assert len(v) == len(PROFILE) and v[0] == 50 and np.isnan(v[PROFILE.index("prior_spike_rate")])
    assert math.isclose(v[PROFILE.index("hour_cos")], -1.0, abs_tol=1e-9)


def test_decision_and_words():
    assert decision(0.10, 0.143, 0.2) == "low risk"
    assert decision(0.17, 0.143, 0.2) == "can't judge"
    assert decision(0.20, 0.143, 0.2) == "caution"
    assert "Caution" in sentence("caution", 0.4, "") and sentence("can't judge", None, "no photo").endswith("no photo.")


def test_glucose_rules_run_before_the_model():
    """Full 5-minute readings pass; a 40-minute hole before the meal and an impossible value do not."""
    cfg = load_config()
    full = list(range(0, 120, 5))
    assert window_problem(_window(full), cfg) is None
    assert "incomplete" in window_problem(_window([m for m in full if m >= 40]), cfg)
    assert "outside" in window_problem(_window(full, 20.0), cfg)


def test_unfamiliar_photo_distance():
    """A copy of a training photo is at distance 0; an opposite direction is far; leave-one-out skips itself."""
    ref = torch.nn.functional.normalize(torch.randn(20, 8), dim=1)
    assert torch.isclose(knn_distance(ref, ref[:1], 1), torch.zeros(1), atol=1e-6).all()
    assert knn_distance(ref, -ref[:1], 1)[0] > 1e-3
    assert (leave_one_out_distances(ref, 1) > 1e-6).all()


def test_cgm_csv_input(tmp_path):
    """The app's CSV: timestamp + glucose_mgdl (+ optional hr_bpm); a wrong header is refused clearly."""
    import pytest

    from glimpse.demo import read_cgm_csv

    f = tmp_path / "cgm.csv"
    f.write_text("timestamp,glucose_mgdl,hr_bpm\n2020-05-01 11:55,101,70\n2020-05-01 12:00,,72\n")
    gt, gv, ht, hv = read_cgm_csv(f)
    assert gv == [101.0] and len(ht) == 2 and hv == [70.0, 72.0]
    f.write_text("time,value\n2020-05-01 12:00,100\n")
    with pytest.raises(ValueError, match="glucose_mgdl"):
        read_cgm_csv(f)


def test_selftest_runs_when_weights_are_present():
    """The container's self-test, run locally when the weights and frozen checkpoints exist."""
    import pytest

    cfg = load_config()
    weights = Path(cfg.photo.weights_dir)
    if not (Path(cfg.prototype.ood_reference_file).is_file() and weights.is_dir() and any(weights.iterdir())):
        pytest.skip("photo weights not downloaded (run `just install`)")
    from glimpse.demo_run import selftest

    assert all(selftest(cfg).values())


def test_meal_input_and_answer_are_validated():
    """Readings must pair up and profile keys must be known; an answer needs a valid decision and a risk in [0, 1]."""
    import pytest
    from pydantic import ValidationError

    from glimpse.demo import Answer, MealInput, cant_judge

    with pytest.raises(ValidationError):
        MealInput(meal_time=MEAL, glucose_times=[MEAL], glucose_mgdl=[], profile={})
    with pytest.raises(ValidationError):
        MealInput(meal_time=MEAL, glucose_times=[], glucose_mgdl=[], profile={"shoe_size": 40.0})
    with pytest.raises(ValidationError):
        Answer(decision="maybe", sentence="x", reason="y")
    with pytest.raises(ValidationError):
        Answer(decision="caution", sentence="x", reason="y", risk=1.5)
    with pytest.raises(ValidationError):
        Answer(decision="caution", sentence="x", reason="y")  # answered without a risk
    a = cant_judge("no meal photo was taken")
    assert a.decision == "can't judge" and '"decision":"can\'t judge"' in a.model_dump_json()


def test_bad_input_files_give_a_structured_cant_judge(tmp_path):
    """Missing, empty or wrong files never crash: they come back as 'can't judge' with the reason."""
    from glimpse.demo import answer_from_files

    empty = tmp_path / "empty.csv"
    empty.write_text("")
    header_only = tmp_path / "header.csv"
    header_only.write_text("timestamp,glucose_mgdl\n")
    wrong = tmp_path / "wrong.csv"
    wrong.write_text("time,value\n2020-05-01 12:00,100\n")
    for cgm in (None, tmp_path / "missing.csv", empty, header_only, wrong):
        a = answer_from_files(None, cgm, None, None, {"age": 50.0})
        assert a.decision == "can't judge" and a.reason.startswith("unsupported input")


def test_sample_cgm_file_is_valid():
    """The synthetic sample in samples/ reads cleanly: 24 readings in the 2 h before the meal."""
    from glimpse.demo import read_cgm_csv

    gt, gv, ht, hv = read_cgm_csv(Path("samples/cgm_example.csv"))
    assert len(gt) == len(gv) == 24 and len(hv) == 24


def test_blank_photo_rule():
    """Plain white, black or grey pictures are caught before the photo model; a textured picture is not."""
    from PIL import Image

    from glimpse.demo import photo_problem

    for v in (0, 128, 255):
        assert "blank" in photo_problem(Image.new("RGB", (200, 200), (v, v, v)), 8.0)
    noise = Image.fromarray(np.random.default_rng(0).integers(0, 256, (200, 200, 3), dtype=np.uint8))
    assert photo_problem(noise, 8.0) is None
