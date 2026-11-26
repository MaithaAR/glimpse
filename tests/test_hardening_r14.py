"""Serving hardening round 3 (26 Nov, docs/v2_hardening.md Part A): each test is one observed failure."""

from datetime import timedelta

from glimpse.serve import MealInput, PhotoEstimate
from glimpse.serve.folder import _carbs, read_history_flagged
from glimpse.serve.io import answer_from_files, read_cgm_csv
from test_serve import MEAL, PROFILE, bare_pipeline, flat_cgm

PRIORS = {"prior_spike_rate": 0.2, "prior_rise": 30.0}


def _meal(values=None, profile=None, **kw) -> MealInput:
    cgm = flat_cgm()
    if values is not None:
        cgm["glucose_mgdl"] = values
    return MealInput(meal_time=MEAL, profile=profile or {**PROFILE, **PRIORS}, **cgm, **kw)


def test_r14_very_low_glucose_is_referred_not_forecast():
    vals = [100.0] * 18 + [80, 70, 62, 58, 52, 48.5]
    a = bare_pipeline(risk=0.31, table={}).answer(_meal(vals), key=0)
    assert a.decision == "can't judge" and "treat the low first" in a.reason and a.escalate_to == "human"


def test_r14_low_glucose_drops_the_walk_advice():
    vals = [100.0] * 20 + [80, 72, 66, 62]
    a = bare_pipeline(risk=0.31, table={0: PhotoEstimate(grams=45.0, bin_probs=[0.1, 0.8, 0.1, 0.0])}).answer(
        _meal(vals), key=0
    )
    assert a.decision == "caution" and "walk afterwards" not in a.sentence and "treat that first" in a.sentence
    assert "glucose_low" in a.flags


def test_r15_diabetes_range_is_out_of_scope():
    a = bare_pipeline(risk=0.31, table={}).answer(_meal(profile={**PROFILE, **PRIORS, "a1c_pct": 7.2}), key=0)
    assert a.decision == "can't judge" and "rule R15" in a.reason and a.escalate_to == "human"


def test_input_refusals_ask_for_a_fixed_input():
    a = bare_pipeline(risk=0.31, table={}).answer(_meal([6.0] * 24), key=0)  # mmol/L
    assert a.decision == "can't judge" and a.escalate_to == "fix_input"


def test_carb_heads_disagreeing_are_flagged_and_shown_as_unsure():
    p = bare_pipeline(risk=0.45, table={0: PhotoEstimate(grams=36.5, bin_probs=[0.6, 0.3, 0.1, 0.0])})
    a = p.answer(_meal(), key=0)
    assert "carb_heads_disagree" in a.flags
    shown = _carbs(a.model_dump(mode="json"))
    assert "about 36 g" in shown and "under 30 g or 30-59 g" in shown


def test_history_from_another_product_is_skipped_and_counted(tmp_path):
    f = tmp_path / "history.csv"
    f.write_text("meal_time,fusion_logit,spiked,product\n2024-01-27T08:00,-1.0,0,photo_fusion\n"
                 "2024-01-27T13:00,-0.5,1,sensor_first\n2024-01-27T19:00,oops,0,sensor_first\n")  # fmt: skip
    rows, flags = read_history_flagged(f, "sensor_first")
    assert len(rows) == 1 and "history_other_product_skipped_1" in flags and "history_rows_skipped_1" in flags
    g = tmp_path / "h2.csv"
    g.write_text("meal_time,fusion_logit,spiked\n2024-01-27T08:00,-1.0,0\n")
    assert read_history_flagged(g, "sensor_first")[1] == ["history_untagged"]


def test_dexcom_and_libre_headers_are_mapped(tmp_path):
    f = tmp_path / "cgm.csv"
    f.write_text(
        "Timestamp (YYYY-MM-DDThh:mm:ss),Glucose Value (mg/dL)\n2024-06-01T10:00:00,100\n2024-06-01T10:05:00,101"
    )
    assert len(read_cgm_csv(f)[1]) == 2
    f.write_text("Device Timestamp,Historic Glucose mg/dL\n2024-06-01 10:00,100\n2024-06-01 10:15,101")
    assert len(read_cgm_csv(f)[1]) == 2


def test_fifteen_minute_exports_name_the_interval():
    times = [MEAL - timedelta(minutes=15 * (8 - i)) for i in range(8)]
    m = MealInput(meal_time=MEAL, glucose_times=times, glucose_mgdl=[100.0] * 8, profile={**PROFILE, **PRIORS})
    a = bare_pipeline(risk=0.3, table={}).answer(m, key=0)
    assert a.decision == "can't judge" and "every 15 min" in a.reason


def test_bad_profile_field_is_named_and_inferred_meal_time_flagged(tmp_path):
    f = tmp_path / "cgm.csv"
    f.write_text("timestamp,glucose_mgdl\n" + "".join(f"2024-06-01T{10 + i // 12:02d}:{5 * (i % 12):02d},100\n"
                                                    for i in range(24)))  # fmt: skip
    a = answer_from_files(
        bare_pipeline(risk=0.3, table={}), f, None, "2024-06-01T12:00", {**PROFILE, "a1c_pct": "5.2%"}
    )
    assert a.decision == "can't judge" and "a1c_pct" in a.reason
    b = answer_from_files(bare_pipeline(risk=0.3, table={}), f, None, None, {**PROFILE, **PRIORS})
    assert "meal_time_inferred" in b.flags


def test_libreview_title_line_and_mmol_columns(tmp_path):
    f = tmp_path / "cgm.csv"
    f.write_text("Glucose Data,Generated on,2024-06-02\nDevice Timestamp,Historic Glucose mmol/L\n"
                 "2024-06-01 10:00,5.6\n2024-06-01 10:05,5.7\n")  # fmt: skip
    t, g, _, _, _ = read_cgm_csv(f)
    assert len(g) == 2 and max(g) < 30  # mmol/L values reach rule R1, which names the x18 fix


def test_r14_message_never_rounds_up_to_the_threshold():
    vals = [100.0] * 18 + [80, 70, 62, 58, 55, 53.7]
    a = bare_pipeline(risk=0.31, table={}).answer(_meal(vals), key=0)
    assert a.decision == "can't judge" and "53.7 mg/dL" in a.reason
