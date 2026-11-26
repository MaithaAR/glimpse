"""Rules R17-R19 (docs/rule_ledger.md, registered 27 Nov before the change): a semicolon CSV is read, a
folder's only image is used when no photo.* exists, and Dexcom "Low"/"High" text maps to the sensor limits."""

import json
from datetime import datetime, timedelta
from pathlib import Path

from glimpse.serve.folder import find_photo, predict_folder
from glimpse.serve.io import read_cgm_csv
from test_serve import bare_pipeline

MEAL = "2024-06-01T12:00"
PROFILE = {"age": 42, "bmi": 27.5, "a1c": 5.8, "fasting_glucose": 102, "meal_time": MEAL}


def _lines(sep: str = ",", dec: str = ".", low_at: int | None = None) -> str:
    t0, out = datetime(2024, 6, 1, 10, 0), [sep.join(["timestamp", "glucose_mgdl", "hr_bpm"])]
    for i in range(24):
        g = "Low" if i == low_at else f"{100 + (i % 3)}{dec}5"
        out.append(sep.join([(t0 + timedelta(minutes=5 * i)).isoformat(sep=" "), g, f"70{dec}0"]))
    return "\n".join(out) + "\n"


def test_r17_semicolon_and_decimal_comma(tmp_path: Path):
    for name, text in (("semi", _lines(";")), ("comma", _lines(";", ","))):
        p = tmp_path / f"{name}.csv"
        p.write_text(text)
        gt, gv, _, hv, flags = read_cgm_csv(p)
        assert len(gt) == 24 and gv[0] == 100.5 and hv[0] == 70.0 and "csv_semicolon" in flags
    p = tmp_path / "plain.csv"
    p.write_text(_lines())
    assert "csv_semicolon" not in read_cgm_csv(p)[4]


def test_r19_low_high_text(tmp_path: Path):
    p = tmp_path / "c.csv"
    p.write_text(_lines(low_at=23))
    _, gv, _, _, flags = read_cgm_csv(p)
    assert gv[-1] == 40.0 and "cgm_low_high_text" in flags


def test_r18_photo_discovery(tmp_path: Path):
    one = tmp_path / "one"
    one.mkdir()
    (one / "IMG_0412.JPEG").write_bytes(b"x")
    assert find_photo(one) == (one / "IMG_0412.JPEG", ["photo_name_nonstandard"])
    named = tmp_path / "named"
    named.mkdir()
    (named / "photo.webp").write_bytes(b"x")
    (named / "other.jpg").write_bytes(b"x")
    assert find_photo(named) == (named / "photo.webp", [])
    two = tmp_path / "two"
    two.mkdir()
    (two / "a.jpg").write_bytes(b"x")
    (two / "b.png").write_bytes(b"x")
    assert find_photo(two)[0] == "several"


def test_folder_names_several_unnamed_photos(tmp_path: Path):
    d = tmp_path / "in" / "m"
    d.mkdir(parents=True)
    (d / "cgm.csv").write_text(_lines())
    (d / "profile.json").write_text(json.dumps(PROFILE))
    (d / "a.jpg").write_bytes(b"x")
    (d / "b.jpg").write_bytes(b"x")
    predict_folder(bare_pipeline(risk=0.45, table={}), tmp_path / "in", tmp_path / "out")
    r = json.loads((tmp_path / "out" / "results.jsonl").read_text().splitlines()[0])
    assert r["decision"] == "can't judge" and "a.jpg, b.jpg" in r["reason"]


def test_r20_day_month_resolved_by_meal_time(tmp_path: Path):
    """01-06-2024 read both ways: the meal time decides; without one, or when both fit, it is refused."""
    import pytest

    t0 = datetime(2024, 6, 1, 10, 0)
    rows = ["timestamp,glucose_mgdl"] + [f"{(t0 + timedelta(minutes=5 * i)):%d-%m-%Y %H:%M},100" for i in range(24)]
    p = tmp_path / "c.csv"
    p.write_text("\n".join(rows))
    gt, _, _, _, flags = read_cgm_csv(p, datetime(2024, 6, 1, 12, 0))
    assert gt[0] == t0 and "dates_day_first" in flags
    with pytest.raises(ValueError, match="ambiguous"):
        read_cgm_csv(p)
    with pytest.raises(ValueError, match="ambiguous"):
        read_cgm_csv(p, datetime(2025, 1, 1))


def test_r21_profile_units(tmp_path: Path):
    from glimpse.serve.io import answer_from_files

    p = tmp_path / "c.csv"
    p.write_text(_lines())
    prof = {"age": 42, "bmi": 27.5, "a1c_pct": "5.9 %", "fasting_glucose_mgdl": "104 mg/dL"}
    a = answer_from_files(bare_pipeline(risk=0.45, table={}), p, None, MEAL, prof)
    assert "not a number" not in (a.reason or "")  # read as 5.9 and 104, not refused
