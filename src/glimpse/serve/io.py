"""Reads user-supplied files into a validated MealInput: the CGM export and the entry point that turns
anything malformed into a structured 'can't judge' instead of a crash. Report: §1 and §4 (the stress suite)."""

from __future__ import annotations

from datetime import datetime, timedelta
from pathlib import Path
from typing import TYPE_CHECKING

from loguru import logger
from pydantic import ValidationError

from glimpse.calibration import LABEL_LAG_MIN
from glimpse.serve.rules import cant_judge
from glimpse.serve.schemas import Answer, MealInput

if TYPE_CHECKING:
    from glimpse.serve.pipeline import Pipeline


# common device-export column names (26 Nov) -> the app's names; anything else must use the app's own
EXPORT_COLUMNS = {"Timestamp (YYYY-MM-DDThh:mm:ss)": "timestamp", "Glucose Value (mg/dL)": "glucose_mgdl",
                  "Device Timestamp": "timestamp", "Historic Glucose mg/dL": "glucose_mgdl",
                  "Historic Glucose (mg/dL)": "glucose_mgdl", "Heart Rate (bpm)": "hr_bpm",
                  # mmol/L columns map too, so rule R1 answers with the x18 guidance instead of "missing column"
                  "Historic Glucose mmol/L": "glucose_mgdl", "Historic Glucose (mmol/L)": "glucose_mgdl",
                  "Glucose Value (mmol/L)": "glucose_mgdl"}  # fmt: skip


def read_cgm_csv(path: str | Path) -> tuple[list[datetime], list[float], list[datetime], list[float], list[str]]:
    """A sensor export as the app accepts it: columns `timestamp`, `glucose_mgdl` and optionally `hr_bpm`.
    Returns the readings plus the repairs made (rule R2: out-of-order rows are sorted and the fix is flagged)."""
    import pandas as pd

    d = pd.read_csv(path)
    d.columns = d.columns.str.strip()  # a header like "timestamp " is visually identical to the real one
    if not ({"timestamp", "glucose_mgdl"} | set(EXPORT_COLUMNS)) & set(d.columns):  # LibreView: a title line first
        d = pd.read_csv(path, skiprows=1)
        d.columns = d.columns.str.strip()
    d = d.rename(columns={k: v for k, v in EXPORT_COLUMNS.items() if k in d.columns and v not in d.columns})
    missing = {"timestamp", "glucose_mgdl"} - set(d.columns)
    if missing:
        raise ValueError(f"the CSV needs the columns timestamp and glucose_mgdl (missing: {sorted(missing)})")
    if d["timestamp"].astype(str).str.contains("/").any():  # rule R11: 06/01/2024 is ambiguous (day/month)
        raise ValueError("dates with slashes (06/01/2024) are ambiguous; export ISO dates such as 2024-06-01")
    d["timestamp"] = pd.to_datetime(d["timestamp"], format="mixed")  # rule R10: mixed ISO precisions are fine
    years = d["timestamp"].dt.year.dropna()
    if len(years) and int(years.max()) < 2000:  # rule R11: epoch numbers parse as 1970 nanoseconds
        raise ValueError("the timestamps look like epoch numbers, not dates; export readable dates")
    if d["timestamp"].dtype == object:  # rows disagree on the UTC offset; converting could misplace the window
        raise ValueError("the timestamps mix different UTC offsets; export the file in one timezone")
    flags: list[str] = []
    if getattr(d["timestamp"].dt, "tz", None) is not None:  # rule R8: offsets dropped, local clock kept
        d["timestamp"] = d["timestamp"].dt.tz_localize(None)
        flags.append("timezone_dropped")
        logger.bind(stage="predict").warning("CGM timestamps carried a UTC offset; dropped before use (rule R8)")
    if "hr_bpm" in d:  # rule R11: a heart rate outside 25-250 bpm is sensor noise, dropped and flagged
        bad = d["hr_bpm"].notna() & ~d["hr_bpm"].between(25, 250)
        if bad.any():
            d.loc[bad, "hr_bpm"] = float("nan")
            flags.append("hr_out_of_range_dropped")
    if not d["timestamp"].is_monotonic_increasing:
        d = d.sort_values("timestamp")
        flags.append("timestamps_sorted")
        logger.bind(stage="predict").warning("CGM timestamps were out of order; sorted before use (rule R2)")
    g = d.dropna(subset=["glucose_mgdl"])
    h = d.dropna(subset=["hr_bpm"]) if "hr_bpm" in d else d.iloc[0:0]
    return (list(g.timestamp.dt.to_pydatetime()), g.glucose_mgdl.astype(float).tolist(),
            list(h.timestamp.dt.to_pydatetime()), h.hr_bpm.astype(float).tolist() if len(h) else [], flags)  # fmt: skip


def _plain(e: Exception) -> str:
    """Rule R4 applied to library errors too: the common parser messages in plain words, anything else
    verbatim but on one line, so a reason never shows a developer-facing phrase or breaks a table."""
    t = " ".join(str(e).split())
    if "could not convert string to float" in t:
        return "a glucose or heart-rate value is not a number (" + t.rsplit(":", 1)[-1].strip() + ")"
    if "Invalid isoformat string" in t:
        tail = t.split("Invalid isoformat string:", 1)[-1].strip()
        return "the meal time " + tail + " is not an understood format (use e.g. 2024-06-01T12:00)"
    if "tokenizing" in t.lower():
        return "the CGM file could not be read as a table of readings"
    if "float() argument" in t:
        return "a profile value is not a number"
    if "Mixed timezones" in t:  # pandas >= 2 raises before the object-dtype branch below is reached
        return "the timestamps mix different UTC offsets; export the file in one timezone"
    return t


def answer_from_files(app: Pipeline, cgm: str | Path | None, photo: str | Path | None, meal_time: str | None,
                      profile: dict[str, float | None],
                      history: list[tuple[datetime, float, int]] | None = None) -> Answer:  # fmt: skip
    """The app's entry point for user files: anything malformed, empty or unsupported becomes a structured
    'can't judge' with the reason, never a crash. `history` (optional, 26 Nov): the person's earlier answers
    as (meal time, fusion logit, CGM label); only meals whose label existed two hours before this meal are
    used, the same gate as the evaluation replay. Without it the call is exactly the no-history one."""
    try:
        if cgm is None:
            raise ValueError("no CGM export was given")
        gt, gv, ht, hv, fl = read_cgm_csv(cgm)
        if not gt:
            raise ValueError("the CGM export has no glucose readings")
        when = datetime.fromisoformat(str(meal_time)) if meal_time else max(gt)
        if not meal_time:  # 26 Nov: the meal is taken to be now, at the last reading; say so
            fl.append("meal_time_inferred")
        if when.tzinfo is not None:  # rule R8, same treatment as the readings
            when = when.replace(tzinfo=None)
            fl.append("timezone_dropped")
        prof = {}
        for k, v in profile.items():  # 26 Nov: name the field that is not a number ("5.2%")
            try:
                prof[k] = float("nan") if v is None else float(v)
            except TypeError, ValueError:
                raise ValueError(f"the profile value {k} = {v!r} is not a number") from None
        m = MealInput(meal_time=when, glucose_times=gt, glucose_mgdl=gv, hr_times=ht, hr_bpm=hv, profile=prof,
                      photo=Path(photo) if photo else None, input_flags=fl)  # fmt: skip
    except ValidationError as e:  # rule R4: one plain sentence per problem, never a validator stack dump
        msgs = "; ".join(err["msg"].removeprefix("Value error, ") for err in e.errors())
        a = cant_judge(f"unsupported input ({msgs})", checks_failed=["input_valid"])
        logger.bind(stage="predict").warning("can't judge: {}", a.reason)
        return a
    except (ValueError, OSError, TypeError) as e:
        a = cant_judge(f"unsupported input ({_plain(e)})")
        logger.bind(stage="predict").warning("can't judge: {}", a.reason)
        return a
    if not history:
        return app.answer(m)
    lag = timedelta(minutes=LABEL_LAG_MIN)
    usable = [(z, y) for t, z, y in sorted(history) if t + lag <= when]
    m = m.model_copy(update={"input_flags": [*m.input_flags, f"history_{len(usable)}_meals"]})
    return app.answer(m, history=usable)
