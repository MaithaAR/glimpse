"""Reads user-supplied files into a validated MealInput: the CGM export and the entry point that turns
anything malformed into a structured 'can't judge' instead of a crash. Report: instructor-readiness in §2/6c."""

from __future__ import annotations

from datetime import datetime
from pathlib import Path
from typing import TYPE_CHECKING

from loguru import logger
from pydantic import ValidationError

from glimpse.serve.rules import cant_judge
from glimpse.serve.schemas import Answer, MealInput

if TYPE_CHECKING:
    from glimpse.serve.pipeline import Pipeline


def read_cgm_csv(path: str | Path) -> tuple[list[datetime], list[float], list[datetime], list[float], list[str]]:
    """A sensor export as the app accepts it: columns `timestamp`, `glucose_mgdl` and optionally `hr_bpm`.
    Returns the readings plus the repairs made (rule R2: out-of-order rows are sorted and the fix is flagged)."""
    import pandas as pd

    d = pd.read_csv(path)
    missing = {"timestamp", "glucose_mgdl"} - set(d.columns)
    if missing:
        raise ValueError(f"the CSV needs the columns timestamp and glucose_mgdl (missing: {sorted(missing)})")
    d["timestamp"] = pd.to_datetime(d["timestamp"], format="mixed")  # rule R10: mixed ISO precisions are fine
    if d["timestamp"].dtype == object:  # rows disagree on the UTC offset; converting could misplace the window
        raise ValueError("the timestamps mix different UTC offsets; export the file in one timezone")
    flags: list[str] = []
    if getattr(d["timestamp"].dt, "tz", None) is not None:  # rule R8: offsets dropped, local clock kept
        d["timestamp"] = d["timestamp"].dt.tz_localize(None)
        flags.append("timezone_dropped")
        logger.bind(stage="predict").warning("CGM timestamps carried a UTC offset; dropped before use (rule R8)")
    if not d["timestamp"].is_monotonic_increasing:
        d = d.sort_values("timestamp")
        flags.append("timestamps_sorted")
        logger.bind(stage="predict").warning("CGM timestamps were out of order; sorted before use (rule R2)")
    g = d.dropna(subset=["glucose_mgdl"])
    h = d.dropna(subset=["hr_bpm"]) if "hr_bpm" in d else d.iloc[0:0]
    return (list(g.timestamp.dt.to_pydatetime()), g.glucose_mgdl.astype(float).tolist(),
            list(h.timestamp.dt.to_pydatetime()), h.hr_bpm.astype(float).tolist() if len(h) else [], flags)  # fmt: skip


def answer_from_files(app: Pipeline, cgm: str | Path | None, photo: str | Path | None, meal_time: str | None,
                      profile: dict[str, float | None]) -> Answer:  # fmt: skip
    """The app's entry point for user files: anything malformed, empty or unsupported becomes a structured
    'can't judge' with the reason, never a crash."""
    try:
        if cgm is None:
            raise ValueError("no CGM export was given")
        gt, gv, ht, hv, fl = read_cgm_csv(cgm)
        if not gt:
            raise ValueError("the CGM export has no glucose readings")
        when = datetime.fromisoformat(meal_time) if meal_time else max(gt)
        if when.tzinfo is not None:  # rule R8, same treatment as the readings
            when = when.replace(tzinfo=None)
            fl.append("timezone_dropped")
        prof = {k: float("nan") if v is None else float(v) for k, v in profile.items()}
        m = MealInput(meal_time=when, glucose_times=gt, glucose_mgdl=gv, hr_times=ht, hr_bpm=hv, profile=prof,
                      photo=Path(photo) if photo else None, input_flags=fl)  # fmt: skip
    except ValidationError as e:  # rule R4: one plain sentence per problem, never a validator stack dump
        msgs = "; ".join(err["msg"].removeprefix("Value error, ") for err in e.errors())
        a = cant_judge(f"unsupported input ({msgs})", checks_failed=["input_valid"])
        logger.bind(stage="predict").warning("can't judge: {}", a.reason)
        return a
    except (ValueError, OSError, TypeError) as e:
        a = cant_judge(f"unsupported input ({e})")
        logger.bind(stage="predict").warning("can't judge: {}", a.reason)
        return a
    return app.answer(m)
