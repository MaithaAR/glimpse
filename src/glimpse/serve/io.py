"""Reads user-supplied files into a validated MealInput: the CGM export and the entry point that turns
anything malformed into a structured 'can't judge' instead of a crash. Report: instructor-readiness in §2/6c."""

from __future__ import annotations

from datetime import datetime
from pathlib import Path
from typing import TYPE_CHECKING

from loguru import logger

from glimpse.serve.rules import cant_judge
from glimpse.serve.schemas import Answer, MealInput

if TYPE_CHECKING:
    from glimpse.serve.pipeline import Glimpse

def read_cgm_csv(path: str | Path) -> tuple[list[datetime], list[float], list[datetime], list[float]]:
    """A sensor export as the app accepts it: columns `timestamp`, `glucose_mgdl` and optionally `hr_bpm`."""
    import pandas as pd

    d = pd.read_csv(path)
    missing = {"timestamp", "glucose_mgdl"} - set(d.columns)
    if missing:
        raise ValueError(f"the CSV needs the columns timestamp and glucose_mgdl (missing: {sorted(missing)})")
    d["timestamp"] = pd.to_datetime(d["timestamp"])
    g = d.dropna(subset=["glucose_mgdl"])
    h = d.dropna(subset=["hr_bpm"]) if "hr_bpm" in d else d.iloc[0:0]
    return (list(g.timestamp.dt.to_pydatetime()), g.glucose_mgdl.astype(float).tolist(),
            list(h.timestamp.dt.to_pydatetime()), h.hr_bpm.astype(float).tolist() if len(h) else [])  # fmt: skip


def answer_from_files(app: Glimpse, cgm: str | Path | None, photo: str | Path | None, meal_time: str | None,
                      profile: dict[str, float | None]) -> Answer:  # fmt: skip
    """The app's entry point for user files: anything malformed, empty or unsupported becomes a structured
    'can't judge' with the reason, never a crash."""
    try:
        if cgm is None:
            raise ValueError("no CGM export was given")
        gt, gv, ht, hv = read_cgm_csv(cgm)
        if not gt:
            raise ValueError("the CGM export has no glucose readings")
        when = datetime.fromisoformat(meal_time) if meal_time else max(gt)
        prof = {k: float("nan") if v is None else float(v) for k, v in profile.items()}
        m = MealInput(meal_time=when, glucose_times=gt, glucose_mgdl=gv, hr_times=ht, hr_bpm=hv, profile=prof,
                      photo=Path(photo) if photo else None)  # fmt: skip
    except (ValueError, OSError, TypeError) as e:
        a = cant_judge(f"unsupported input ({e})")
        logger.bind(stage="predict").warning("can't judge: {}", a.reason)
        return a
    return app.predict(m)
