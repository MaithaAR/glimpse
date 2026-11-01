"""The serving contract: schema v2 round-trips, the decision layer runs the same whatever supplies the carb
estimate (the evaluation-equals-deployment seam, D2), and tier 2 (confirm-carbs) answers from the user's number."""

from datetime import datetime, timedelta
from pathlib import Path

import numpy as np
import pytest
from loguru import logger

from glimpse.config import load_config
from glimpse.serve import Answer, MealInput, PhotoEstimate, Pipeline

MEAL = datetime(2024, 1, 28, 12, 0)
PROFILE = {"age": 40.0, "bmi": 28.0, "a1c_pct": 5.6, "fasting_glucose_mgdl": 95.0}


def flat_cgm(n: int = 24, value: float = 100.0) -> dict:
    """A complete, calm 2-hour window ending at the meal."""
    times = [MEAL - timedelta(minutes=5 * (n - i)) for i in range(n)]
    return {"glucose_times": times, "glucose_mgdl": [value] * n, "hr_times": times, "hr_bpm": [70.0] * n}


class _Fusion:
    """A stand-in fusion with a fixed risk, so the decision layer is tested without any checkpoint."""

    def __init__(self, risk: float):
        self.risk = risk

    def predict_proba(self, x):
        return np.array([[1 - self.risk, self.risk]])


def bare_pipeline(risk: float, table: dict | None, calibration: tuple | None = None,
                  m_personal: float = float("inf")) -> Pipeline:  # fmt: skip
    """A Pipeline without loading any model file: config, stub fusion, frozen cut-offs, optional estimate table."""
    p = object.__new__(Pipeline)
    p.cfg = load_config()
    p.fusion = _Fusion(risk)
    p.lo, p.hi = p.cfg.prototype.low_risk_below, p.cfg.prototype.caution_from
    p.table = table if table is not None else {}
    p.calib, p.m_personal = calibration, m_personal
    p.model_versions = {"fusion": "stub", "thresholds": "stub", "photo": "precomputed-table"}
    p.log = logger.bind(stage="predict")
    return p


def test_answer_schema_v2_round_trip():
    """The new contract fields serialize and default correctly."""
    a = Answer(decision="can't judge", sentence="x", reason="y", checks_failed=["cgm_window"],
               escalate_to="confirm_carbs")  # fmt: skip
    again = Answer.model_validate_json(a.model_dump_json())
    assert again.checks_failed == ["cgm_window"] and again.escalate_to == "confirm_carbs"
    assert again.checks_passed == [] and again.carb_source == "none" and again.model_versions == {}


def test_same_decision_code_for_table_and_live_estimates():
    """A precomputed estimate goes through the identical decision path: same checks, same fields."""
    est = PhotoEstimate(grams=75.0, bin_probs=[0.0, 0.0, 1.0, 0.0], unfamiliar_score=0.2)
    p = bare_pipeline(risk=0.45, table={7: est})
    m = MealInput(meal_time=MEAL, profile=PROFILE, **flat_cgm())
    a = p.answer(m, key=7)
    assert a.decision == "caution" and a.carb_source == "photo" and a.carb_bin == "60-89 g"
    assert a.checks_passed[:5] == ["cgm_units", "cgm_times", "cgm_window", "profile", "photo_present"]
    assert a.checks_failed == []


def test_confirm_carbs_tier_uses_the_users_number():
    """No usable photo + a typed carb estimate: the risk is still produced, flagged as user-supplied."""
    p = bare_pipeline(risk=0.45, table={})
    base = dict(meal_time=MEAL, profile=PROFILE, **flat_cgm())
    asked = p.answer(MealInput(**base), key=None)  # no estimate, no user number -> escalate to confirm-carbs
    assert asked.decision == "can't judge" and asked.escalate_to == "confirm_carbs"
    answered = p.answer(MealInput(**base, user_carbs_g=75.0), key=None)
    assert answered.decision == "caution" and answered.carb_source == "user"
    assert "carbs_from_user" in answered.flags and answered.carb_bin == "60-89 g"


def test_borderline_escalates_to_human():
    """A risk between the cut-offs is a structured abstention routed to review, not a guess."""
    p = bare_pipeline(risk=0.17, table={0: PhotoEstimate(grams=20.0, bin_probs=[1.0, 0.0, 0.0, 0.0])})
    m = MealInput(meal_time=MEAL, profile=PROFILE, **flat_cgm())
    a = p.answer(m, key=0)
    assert a.decision == "can't judge" and a.escalate_to == "human" and a.checks_failed == ["borderline"]


@pytest.mark.skipif(not (Path("checkpoints/fusion_final.joblib").exists() and Path("data/dataset.json").exists()),
                    reason="needs the trained pipeline and the prepared dataset")  # fmt: skip
def test_live_and_table_pipelines_agree_meal_by_meal():
    """The parity proof: the live photo model and its own precomputed estimates give identical Answers."""
    from glimpse.data.dataset import MealDataset
    from glimpse.data.splits import Splits, meal_rows
    from glimpse.demo_cases import meal_input

    cfg = load_config()
    ds = MealDataset.load(Path("data/dataset"))
    sp = Splits.load(Path("splits/dev_splits.json"))
    live = Pipeline(cfg)
    table_rows = {}
    answers = {}
    for row in meal_rows(ds, set(sp.test), scored_only=True)[:12]:
        m = meal_input(cfg, ds, int(row))
        est = live.photo_estimate(m)
        if isinstance(est, PhotoEstimate):
            table_rows[int(row)] = est
            answers[int(row)] = (m, live.answer(m))
    assert table_rows, "no scorable test meal found"
    tabled = Pipeline(cfg, photo_table=table_rows)
    for row, (m, a_live) in answers.items():
        a_tab = tabled.answer(m, key=row)
        d1 = a_live.model_dump(exclude={"latency_ms", "model_versions"})
        d2 = a_tab.model_dump(exclude={"latency_ms", "model_versions"})
        assert d1 == d2, f"row {row}: live and table answers differ"


def test_predict_folder_finishes_with_per_item_entries(tmp_path):
    """One good-shaped meal, one broken one: both appear in results.jsonl and the run still finishes."""
    import json

    from glimpse.serve.folder import predict_folder

    good = tmp_path / "in" / "a_good"
    good.mkdir(parents=True)
    rows = ["timestamp,glucose_mgdl,hr_bpm"]
    rows += [f"2024-06-01 {10 + i // 12:02d}:{(i % 12) * 5:02d}:00,{100 + i},70" for i in range(24)]
    (good / "cgm.csv").write_text("\n".join(rows) + "\n")
    (good / "profile.json").write_text(
        json.dumps({"age": 42, "bmi": 27.5, "a1c": 5.8, "fasting_glucose": 102, "meal_time": "2024-06-01T12:00"})
    )
    bad = tmp_path / "in" / "b_broken"
    bad.mkdir()
    (bad / "profile.json").write_text("{not json")
    results = predict_folder(bare_pipeline(risk=0.45, table={}), tmp_path / "in", tmp_path / "out")
    lines = [json.loads(ln) for ln in results.read_text().splitlines()]
    assert [r["meal_id"] for r in lines] == ["a_good", "b_broken"]
    assert lines[0]["decision"] == "can't judge"  # no photo estimate in the bare pipeline -> confirm-carbs
    assert lines[0]["escalate_to"] == "confirm_carbs"
    assert "error" in lines[1]
    assert (tmp_path / "out" / "summary.md").read_text().startswith("# Folder run: 2 meals")


def test_mmol_values_get_the_conversion_hint():
    """Rule R1: a mmol/L export (values 4-9) is named and the fix is stated, not a generic range error."""
    cg = flat_cgm()
    cg["glucose_mgdl"] = [v / 18 for v in cg["glucose_mgdl"]]
    a = bare_pipeline(risk=0.45, table={}).answer(MealInput(meal_time=MEAL, profile=PROFILE, **cg))
    assert a.decision == "can't judge" and "multiply by 18" in a.reason and a.checks_failed == ["cgm_units"]


def test_future_timestamps_are_named():
    """Rule R3: readings after the meal time are called out, not folded into a coverage complaint."""
    cg = flat_cgm()
    cg["glucose_times"] = [t + timedelta(hours=3) for t in cg["glucose_times"]]
    cg["hr_times"] = cg["glucose_times"]
    a = bare_pipeline(risk=0.45, table={}).answer(MealInput(meal_time=MEAL, profile=PROFILE, **cg))
    assert a.decision == "can't judge" and "after the meal time" in a.reason and a.checks_failed == ["cgm_times"]


def test_blank_profile_is_refused_not_defaulted():
    """Rule R5: all-blank required fields -> can't judge naming them; nothing is filled in."""
    blank = {k: float("nan") for k in PROFILE}
    a = bare_pipeline(risk=0.45, table={}).answer(MealInput(meal_time=MEAL, profile=blank, **flat_cgm()))
    assert a.decision == "can't judge" and "missing or blank" in a.reason and a.checks_failed == ["profile"]
    assert all(k in a.reason for k in PROFILE)


def test_unsorted_cgm_is_sorted_and_flagged(tmp_path):
    """Rule R2: out-of-order rows are sorted before use and the repair is visible as a flag."""
    from glimpse.serve import read_cgm_csv

    f = tmp_path / "cgm.csv"
    f.write_text("timestamp,glucose_mgdl\n2024-06-01 11:00,100\n2024-06-01 10:00,98\n2024-06-01 10:30,99\n")
    gt, gv, ht, hv, fl = read_cgm_csv(f)
    assert gv == [98.0, 99.0, 100.0] and fl == ["timestamps_sorted"] and gt == sorted(gt)


def test_implausible_profile_message_is_one_plain_sentence(tmp_path):
    """Rule R4: the boundary turns a validation error into plain words, never a validator stack dump."""
    from glimpse.serve import answer_from_files

    f = tmp_path / "cgm.csv"
    f.write_text("timestamp,glucose_mgdl\n2024-06-01 11:00,100\n")
    p = bare_pipeline(risk=0.45, table={})
    a = answer_from_files(p, f, None, "2024-06-01T12:00", {"age": 7.0, "bmi": 28.0})
    assert a.decision == "can't judge"
    assert "age = 7.0 is outside the plausible range 18-100" in a.reason
    assert "pydantic" not in a.reason and "https://" not in a.reason


def test_exif_rotated_photo_opens_identically(tmp_path):
    """Rule R6: a photo stored rotated with an EXIF orientation tag opens pixel-identical to the upright
    original, so the stored rotation cannot change any decision downstream."""
    import numpy as _np
    from PIL import Image

    from glimpse.data.photos import open_photo

    rng = _np.random.default_rng(0)
    arr = rng.integers(0, 255, size=(120, 160, 3), dtype=_np.uint8)
    up = Image.fromarray(arr)
    up.save(tmp_path / "up.jpg", quality=100, subsampling=0)
    ex = Image.Exif()
    ex[274] = 6  # "rotate 90 CW to display": the stored pixels are the upright image turned 90 CCW
    up.rotate(90, expand=True).save(tmp_path / "rot.jpg", quality=100, subsampling=0, exif=ex)
    a = _np.asarray(open_photo(tmp_path / "up.jpg"), dtype=_np.int16)
    b = _np.asarray(open_photo(tmp_path / "rot.jpg"), dtype=_np.int16)
    assert a.shape == b.shape
    assert float(_np.abs(a - b).mean()) < 3  # identical up to JPEG round-trip noise


def test_heic_photo_opens(tmp_path):
    """An iPhone HEIC photo goes through the same shared opener as JPEG and PNG."""
    pytest.importorskip("pillow_heif")
    import numpy as _np
    from PIL import Image

    from glimpse.data.photos import open_photo

    f = tmp_path / "meal.heic"
    arr = _np.random.default_rng(1).integers(0, 255, size=(120, 160, 3), dtype=_np.uint8)
    Image.fromarray(arr).save(f)
    img = open_photo(f)
    assert img.size == (160, 120) and img.mode == "RGB"


def test_fold_estimates_builds_the_pipeline_table():
    """Out-of-fold CSV rows become validated PhotoEstimates keyed by meal row."""
    import pandas as pd

    from glimpse.evaluation.run import fold_estimates

    df = pd.DataFrame([{"row": 7, "photo_g": 62.5, "p_bin0": 0.1, "p_bin1": 0.2, "p_bin2": 0.6, "p_bin3": 0.1}])
    t = fold_estimates(df)
    assert set(t) == {7} and t[7].grams == 62.5 and abs(sum(t[7].bin_probs) - 1.0) < 1e-9


def _est() -> PhotoEstimate:
    return PhotoEstimate(grams=75.0, bin_probs=[0.0, 0.0, 1.0, 0.0], unfamiliar_score=0.2)


def test_defaults_are_the_identity_calibration():
    """Golden regression for fix 1's plumbing: without calibration and history, the final risk equals the
    raw fusion risk exactly, so demo, predict and the app are provably unchanged."""
    p = bare_pipeline(risk=0.45, table={7: _est()})
    a = p.answer(MealInput(meal_time=MEAL, profile=PROFILE, **flat_cgm()), key=7)
    assert a.risk == a.risk_raw == a.risk_platt == 0.45
    assert abs(a.fusion_logit - np.log(0.45 / 0.55)) < 1e-9


def test_platt_and_history_adjust_the_risk_in_order():
    """Platt moves the logit first; the personal offset (fitted on equally recalibrated history) moves it
    further; the decision is made on the final risk."""
    m = MealInput(meal_time=MEAL, profile=PROFILE, **flat_cgm())
    p = bare_pipeline(risk=0.45, table={7: _est()}, calibration=(1.0, -0.8))
    a = p.answer(m, key=7)
    z = np.log(0.45 / 0.55)
    assert abs(a.risk_platt - 1 / (1 + np.exp(-(z - 0.8)))) < 1e-9
    assert a.risk == a.risk_platt  # no history -> no offset
    p2 = bare_pipeline(risk=0.45, table={7: _est()}, calibration=(1.0, -0.8), m_personal=1.0)
    spiky = [(1.5, 1), (0.8, 1), (1.1, 1)]  # a frequent spiker's past
    b = p2.answer(m, key=7, history=spiky)
    assert b.risk > b.risk_platt  # the offset pushes a frequent spiker's risk up
    p3 = bare_pipeline(risk=0.45, table={7: _est()}, calibration=(1.0, -0.8))
    c = p3.answer(m, key=7, history=spiky)
    assert c.risk == c.risk_platt  # m = inf: the same history changes nothing (D3b population arm)


def test_decision_parity_with_the_vectorised_rule():
    """Fix 2's parity requirement: the vectorised cut-off rule used by the sensitivity sweep and
    Pipeline.answer give the same decision meal by meal on clean inputs, across the whole risk range."""
    for risk in (0.05, 0.142, 0.143, 0.17, 0.2, 0.45, 0.95):
        p = bare_pipeline(risk=risk, table={0: _est()})
        a = p.answer(MealInput(meal_time=MEAL, profile=PROFILE, **flat_cgm()), key=0)
        vec = "low risk" if a.risk < p.lo else "caution" if a.risk >= p.hi else "can't judge"
        assert a.decision == vec, (risk, a.decision, vec)
