"""Stress contract for the instructor's folder command: hostile inputs become structured answers,
never crashes and never raw error rows. Every case here is text-level (CSV, JSON, meal time, folder
shape) and runs on the table-mode pipeline, so this guards the whole parsing surface on every commit;
the committed samples/stress set exercises the live photo path via `just stress`."""

import json
from datetime import datetime, timedelta
from pathlib import Path

from glimpse.serve.folder import predict_folder
from test_serve import bare_pipeline

MEAL = "2024-06-01T12:00"
GOOD_PROFILE = {"age": 42, "bmi": 27.5, "a1c": 5.8, "fasting_glucose": 102, "meal_time": MEAL}


def _cgm_lines(n: int = 24, value: float = 100.0, tz: str = "") -> list[str]:
    t0 = datetime(2024, 6, 1, 10, 0)
    return ["timestamp,glucose_mgdl,hr_bpm"] + [
        f"{(t0 + timedelta(minutes=5 * i)).isoformat()}{tz},{value + (i % 3)},70" for i in range(n)
    ]


def _write_case(root: Path, name: str, cgm: str | bytes | None, profile: str | None,
                photo: bytes | None = b"not really a jpeg") -> None:  # fmt: skip
    d = root / name
    d.mkdir(parents=True)
    if cgm is not None:
        (d / "cgm.csv").write_bytes(cgm if isinstance(cgm, bytes) else cgm.encode())
    if profile is not None:
        (d / "profile.json").write_text(profile)
    if photo is not None:
        (d / "photo.jpg").write_bytes(photo)


def build_stress_folder(root: Path) -> int:
    """Every hostile text-level input we expect an instructor's stress test to try."""
    ok_prof = json.dumps(GOOD_PROFILE)
    cases: dict[str, tuple] = {
        "01_semicolon_csv": ("\n".join(ln.replace(",", ";") for ln in _cgm_lines()), ok_prof),
        "02_tz_aware_timestamps": ("\n".join(_cgm_lines(tz="+04:00")), ok_prof),
        "03_ampm_meal_time": (
            "\n".join(_cgm_lines()),
            json.dumps({**GOOD_PROFILE, "meal_time": "06/01/2024 12:00 PM"}),
        ),  # fmt: skip
        "04_strings_in_glucose": ("timestamp,glucose_mgdl\n2024-06-01T10:00,high\n2024-06-01T10:05,low", ok_prof),
        "05_negative_glucose": ("\n".join(_cgm_lines(value=-50.0)), ok_prof),
        "06_absurd_glucose": ("\n".join(_cgm_lines(value=1e9)), ok_prof),
        "07_one_reading": ("timestamp,glucose_mgdl\n2024-06-01T11:55,100", ok_prof),
        "08_duplicate_timestamps": (
            "\n".join(_cgm_lines()) + "\n2024-06-01T10:00,99,70\n2024-06-01T10:00,101,70",
            ok_prof,
        ),  # fmt: skip
        "09_header_only": ("timestamp,glucose_mgdl", ok_prof),
        "10_binary_cgm": (bytes(range(256)) * 8, ok_prof),
        "11_bom_crlf_csv": ("﻿" + "\r\n".join(_cgm_lines()), ok_prof),
        "12_wrong_columns": ("time,sugar\n2024-06-01T10:00,100", ok_prof),
        "13_profile_strings": (
            "\n".join(_cgm_lines()),
            json.dumps({"age": "42", "bmi": "27.5", "a1c": "5.8", "fasting_glucose": "102", "meal_time": MEAL}),
        ),  # fmt: skip
        "14_profile_not_json": ("\n".join(_cgm_lines()), "age: 42, bmi: 27.5 <-- not json"),
        "15_profile_nested": ("\n".join(_cgm_lines()), json.dumps({**GOOD_PROFILE, "age": {"value": 42}})),  # fmt: skip
        "16_no_cgm_at_all": (None, ok_prof),
        "17_empty_folder": (None, None, None),
        "18_future_meal_time": (
            "\n".join(_cgm_lines()),
            json.dumps({**GOOD_PROFILE, "meal_time": "2024-06-01T09:00"}),
        ),  # fmt: skip
    }
    for name, spec in cases.items():
        _write_case(root, name, spec[0], spec[1], *(spec[2:] or (b"junk",)))
    (root / "19_extra_junk").mkdir()
    (root / "19_extra_junk" / "cgm.csv").write_text("\n".join(_cgm_lines()))
    (root / "19_extra_junk" / "profile.json").write_text(ok_prof)
    (root / "19_extra_junk" / "notes.txt").write_text("irrelevant")
    (root / "19_extra_junk" / "nested").mkdir()
    (root / "stray_file.txt").write_text("not a meal folder")  # files at the top level are skipped
    return 19


def _rows(out: Path) -> list[dict]:
    return [json.loads(ln) for ln in (out / "results.jsonl").read_text().splitlines()]


def test_hostile_folder_yields_structured_answers_never_errors(tmp_path):
    """The invariant the report claims: any input folder -> one validated answer per meal, no error rows."""
    n = build_stress_folder(tmp_path / "in")
    app = bare_pipeline(risk=0.45, table={})
    predict_folder(app, tmp_path / "in", tmp_path / "out")
    rows = _rows(tmp_path / "out")
    assert len(rows) == n
    errors = [r["meal_id"] for r in rows if "error" in r]
    assert errors == [], f"raw error rows for: {errors} -- each must become a structured can't judge"
    assert all(r["decision"] in ("low risk", "caution", "can't judge") for r in rows)
    assert all(r["reason"] for r in rows if r["decision"] == "can't judge")
    assert (tmp_path / "out" / "summary.md").read_text().startswith("# Folder run: ")


def test_specific_hostile_cases_fail_for_the_right_reason(tmp_path):
    """Spot checks: the reason names the actual problem, and repairs are flagged, not silent."""
    build_stress_folder(tmp_path / "in")
    predict_folder(bare_pipeline(risk=0.45, table={}), tmp_path / "in", tmp_path / "out")
    by = {r["meal_id"]: r for r in _rows(tmp_path / "out")}
    tz = by["02_tz_aware_timestamps"]
    assert "timezone_dropped" in tz["flags"]  # rule R8: the repair is flagged, not silent
    assert by["04_strings_in_glucose"]["decision"] == "can't judge"
    assert "column" in by["12_wrong_columns"]["reason"]
    assert by["14_profile_not_json"]["decision"] == "can't judge"
    assert "CGM" in by["16_no_cgm_at_all"]["reason"]
    assert "after the meal time" in by["18_future_meal_time"]["reason"]  # rule R3


def test_folder_run_is_deterministic(tmp_path):
    """Two identical runs give identical answers (latency aside) -- the instructor can re-run us."""
    build_stress_folder(tmp_path / "in")
    app = bare_pipeline(risk=0.45, table={})
    predict_folder(app, tmp_path / "in", tmp_path / "a")
    predict_folder(app, tmp_path / "in", tmp_path / "b")
    strip = lambda rows: [{k: v for k, v in r.items() if k != "latency_ms"} for r in rows]  # noqa: E731
    assert strip(_rows(tmp_path / "a")) == strip(_rows(tmp_path / "b"))
