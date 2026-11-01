"""The serving contract: schema v2 round-trips, the decision layer runs the same whatever supplies the carb
estimate (the evaluation-equals-deployment seam, D2), and tier 2 (confirm-carbs) answers from the user's number."""

from datetime import datetime, timedelta
from pathlib import Path

import pytest
from loguru import logger

from glimpse.config import load_config
from glimpse.serve import Answer, MealInput, PhotoEstimate, Pipeline

MEAL = datetime(2024, 1, 28, 12, 0)


def flat_cgm(n: int = 24, value: float = 100.0) -> dict:
    """A complete, calm 2-hour window ending at the meal."""
    times = [MEAL - timedelta(minutes=5 * (n - i)) for i in range(n)]
    return {"glucose_times": times, "glucose_mgdl": [value] * n, "hr_times": times, "hr_bpm": [70.0] * n}


class _Fusion:
    """A stand-in fusion with a fixed risk, so the decision layer is tested without any checkpoint."""

    def __init__(self, risk: float):
        self.risk = risk

    def predict_proba(self, x):
        return [[1 - self.risk, self.risk]]


def bare_pipeline(risk: float, table: dict | None) -> Pipeline:
    """A Pipeline without loading any model file: config, stub fusion, frozen cut-offs, optional estimate table."""
    p = object.__new__(Pipeline)
    p.cfg = load_config()
    p.fusion = _Fusion(risk)
    p.lo, p.hi = p.cfg.prototype.low_risk_below, p.cfg.prototype.caution_from
    p.table = table if table is not None else {}
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
    m = MealInput(meal_time=MEAL, profile={"age": 40.0, "bmi": 28.0}, **flat_cgm())
    a = p.answer(m, key=7)
    assert a.decision == "caution" and a.carb_source == "photo" and a.carb_bin == "60-89 g"
    assert a.checks_passed[:2] == ["cgm_window", "photo_present"] and a.checks_failed == []


def test_confirm_carbs_tier_uses_the_users_number():
    """No usable photo + a typed carb estimate: the risk is still produced, flagged as user-supplied."""
    p = bare_pipeline(risk=0.45, table={})
    base = dict(meal_time=MEAL, profile={"age": 40.0, "bmi": 28.0}, **flat_cgm())
    asked = p.answer(MealInput(**base), key=None)  # no estimate, no user number -> escalate to confirm-carbs
    assert asked.decision == "can't judge" and asked.escalate_to == "confirm_carbs"
    answered = p.answer(MealInput(**base, user_carbs_g=75.0), key=None)
    assert answered.decision == "caution" and answered.carb_source == "user"
    assert "carbs_from_user" in answered.flags and answered.carb_bin == "60-89 g"


def test_borderline_escalates_to_human():
    """A risk between the cut-offs is a structured abstention routed to review, not a guess."""
    p = bare_pipeline(risk=0.17, table={0: PhotoEstimate(grams=20.0, bin_probs=[1.0, 0.0, 0.0, 0.0])})
    m = MealInput(meal_time=MEAL, profile={"age": 40.0, "bmi": 28.0}, **flat_cgm())
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
