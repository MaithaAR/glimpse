"""Serving hardening round 2 (8 Nov red-team): each test is one observed escape, now a contract."""

import json

import pytest
from test_serve import bare_pipeline

from glimpse.serve.folder import predict_folder
from glimpse.serve.io import _plain, answer_from_files, read_cgm_csv

GOOD = {"age": 42.0, "bmi": 27.5, "a1c_pct": 5.8, "fasting_glucose_mgdl": 102.0}


def _csv(tmp_path, text):
    f = tmp_path / "cgm.csv"
    f.write_text(text)
    return f


def test_slash_dates_are_refused_as_ambiguous(tmp_path):
    f = _csv(tmp_path, "timestamp,glucose_mgdl\n06/01/2024 10:00,100\n13/01/2024 10:05,101")
    a = answer_from_files(bare_pipeline(risk=0.4, table={}), f, None, None, GOOD)
    assert a.decision == "can't judge" and "ambiguous" in a.reason


def test_epoch_second_timestamps_are_named(tmp_path):
    f = _csv(tmp_path, "timestamp,glucose_mgdl\n1717236000,100\n1717236300,101")
    a = answer_from_files(bare_pipeline(risk=0.4, table={}), f, None, None, GOOD)
    assert a.decision == "can't judge" and "epoch" in a.reason


def test_header_whitespace_is_stripped(tmp_path):
    f = _csv(tmp_path, "timestamp ,glucose_mgdl \n2024-06-01T10:00,100\n2024-06-01T10:05,101")
    t, g, _, _, _ = read_cgm_csv(f)
    assert len(g) == 2


def test_out_of_range_heart_rate_is_dropped_and_flagged(tmp_path):
    f = _csv(tmp_path, "timestamp,glucose_mgdl,hr_bpm\n2024-06-01T10:00,100,-50\n2024-06-01T10:05,101,70")
    _, _, _, hv, flags = read_cgm_csv(f)
    assert hv == [70.0] and "hr_out_of_range_dropped" in flags


def test_mixed_timezone_pandas_message_reads_plain():
    msg = ValueError("Mixed timezones detected. Pass utc=True in to_datetime or tz='UTC' in DatetimeIndex")
    assert _plain(msg) == "the timestamps mix different UTC offsets; export the file in one timezone"


def test_prior_spike_rate_out_of_range_is_refused(tmp_path):
    f = _csv(tmp_path, "timestamp,glucose_mgdl\n2024-06-01T10:00,100\n2024-06-01T10:05,101")
    a = answer_from_files(bare_pipeline(risk=0.4, table={}), f, None, None, GOOD | {"prior_spike_rate": 7.0})
    assert a.decision == "can't judge" and "prior_spike_rate" in a.reason


def test_uppercase_photo_name_is_found_and_unreadable_profile_is_structured(tmp_path):
    case = tmp_path / "in" / "01_upper"
    case.mkdir(parents=True)
    (case / "cgm.csv").write_text("timestamp,glucose_mgdl\n2024-06-01T10:00,100")
    (case / "PHOTO.JPG").write_bytes(b"not a jpeg")
    (case / "profile.json").write_text(json.dumps(GOOD | {"meal_time": "2024-06-01T12:00"}))
    locked = tmp_path / "in" / "02_locked_profile"
    locked.mkdir()
    (locked / "cgm.csv").write_text("timestamp,glucose_mgdl\n2024-06-01T10:00,100")
    (locked / "profile.json").write_text("{}")
    (locked / "profile.json").chmod(0o000)
    try:
        predict_folder(bare_pipeline(risk=0.4, table={}), tmp_path / "in", tmp_path / "out")
    finally:
        (locked / "profile.json").chmod(0o644)
    rows = {json.loads(ln)["meal_id"]: json.loads(ln)
            for ln in (tmp_path / "out" / "results.jsonl").read_text().splitlines()}
    assert "error" not in rows["01_upper"]
    assert "photo" in rows["01_upper"]["reason"]  # the photo WAS found (and then rejected as unreadable)
    locked_row = rows["02_locked_profile"]
    if "error" in locked_row:  # running as root makes chmod 000 readable; only then is this vacuous
        pytest.skip("permissions not enforced (root)")
    assert locked_row["decision"] == "can't judge"


def test_output_dir_inside_input_dir_is_not_a_meal(tmp_path):
    case = tmp_path / "in" / "01_only"
    case.mkdir(parents=True)
    (case / "cgm.csv").write_text("timestamp,glucose_mgdl\n2024-06-01T10:00,100")
    (case / "profile.json").write_text(json.dumps(GOOD))
    app = bare_pipeline(risk=0.4, table={})
    predict_folder(app, tmp_path / "in", tmp_path / "in" / "out")
    predict_folder(app, tmp_path / "in", tmp_path / "in" / "out")
    lines = (tmp_path / "in" / "out" / "results.jsonl").read_text().splitlines()
    assert len(lines) == 1 and json.loads(lines[0])["meal_id"] == "01_only"
