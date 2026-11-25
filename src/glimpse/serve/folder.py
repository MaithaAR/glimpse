"""Batch entry point for a folder of meals, as the instructor will run it: one sub-folder per meal
(photo.*, cgm.csv, profile.json) -> results.jsonl + summary.md. A bad meal becomes a per-item entry,
never a crash, and the per-meal loop always finishes (a missing or empty input folder is still an
error). An optional history.csv per meal feeds the personal intercept (report §1)."""

from __future__ import annotations

import csv
import json
import math
from collections import Counter
from datetime import datetime
from pathlib import Path
from typing import TYPE_CHECKING

from loguru import logger

from glimpse.serve.io import answer_from_files
from glimpse.serve.rules import cant_judge

if TYPE_CHECKING:
    from glimpse.serve.pipeline import Pipeline

# profile.json keys (docs/final_design.md §6) -> the model's profile input names
# rule R11: both spellings accepted, so a profile written with the model-side names is not "missing"
PROFILE_KEYS = {"age": "age", "bmi": "bmi", "a1c": "a1c_pct", "fasting_glucose": "fasting_glucose_mgdl",
                "prior_spike_rate": "prior_spike_rate", "prior_rise": "prior_rise",
                "a1c_pct": "a1c_pct", "fasting_glucose_mgdl": "fasting_glucose_mgdl"}  # fmt: skip
PHOTO_NAMES = ("photo.jpg", "photo.jpeg", "photo.png", "photo.heic")


def predict_folder(app: Pipeline, in_dir: str | Path, out_dir: str | Path) -> Path:
    """Answer every meal folder under `in_dir`; write results.jsonl and summary.md under `out_dir`."""
    out = Path(out_dir)
    cases = sorted(p for p in Path(in_dir).iterdir() if p.is_dir() and p.resolve() != out.resolve())
    if not cases:
        raise ValueError(f"no meal folders under {in_dir}")
    rows = [_one_case(app, case) for case in cases]
    out.mkdir(parents=True, exist_ok=True)
    results = out / "results.jsonl"
    results.write_text("".join(json.dumps(r) + "\n" for r in rows))
    (out / "summary.md").write_text(_summary(rows))
    logger.bind(stage="predict-folder").info("{} meals -> {}", len(rows), results)
    return results


def _one_case(app: Pipeline, case: Path) -> dict:
    """One meal folder -> one validated Answer as a plain dict (or an error entry; the batch continues)."""
    try:
        try:
            raw = json.loads((case / "profile.json").read_text()) if (case / "profile.json").exists() else {}
        except json.JSONDecodeError as e:  # rule R9: a broken profile file is named, never a raw error row
            a = cant_judge(f"profile.json is not valid JSON ({e.msg} at line {e.lineno})",
                           checks_failed=["input_valid"])  # fmt: skip
            return {"meal_id": case.name, **a.model_dump(mode="json")}
        except OSError as e:  # rule R11: an unreadable profile file is named too
            a = cant_judge(f"profile.json cannot be read ({e.strerror or e})", checks_failed=["input_valid"])
            return {"meal_id": case.name, **a.model_dump(mode="json")}
        meal_time = raw.pop("meal_time", None)
        profile = {PROFILE_KEYS[k]: v for k, v in raw.items() if k in PROFILE_KEYS}
        photo = next((f for f in sorted(case.iterdir()) if f.name.lower() in PHOTO_NAMES), None)
        cgm = case / "cgm.csv"
        history = read_history(case / "history.csv")
        a = answer_from_files(app, cgm if cgm.exists() else None, photo, meal_time, profile, history)
        return {"meal_id": case.name, **a.model_dump(mode="json")}
    except Exception as e:  # a malformed meal is reported in its own row; the run must finish
        logger.bind(stage="predict-folder").warning("{}: error ({})", case.name, e)
        return {"meal_id": case.name, "error": f"{type(e).__name__}: {e}"}


def read_history(path: Path) -> list[tuple[datetime, float, int]] | None:
    """Optional history.csv (26 Nov): the person's earlier answers as meal_time, fusion_logit (from an
    earlier results.jsonl) and spiked (0/1, the CGM's label at +2 h). Rows that do not parse are skipped;
    a missing file means no history, the population calibration the samples are scored at."""
    if not path.exists():
        return None
    rows = []
    try:
        with open(path, newline="", encoding="utf-8-sig") as f:
            for r in csv.DictReader(f):
                try:
                    t = datetime.fromisoformat(str(r["meal_time"]).strip()).replace(tzinfo=None)
                    z, y = float(r["fusion_logit"]), int(float(r["spiked"]))
                except KeyError, TypeError, ValueError:
                    continue
                if math.isfinite(z) and y in (0, 1):
                    rows.append((t, z, y))
    except OSError, UnicodeDecodeError, csv.Error:
        return None
    return rows


def _summary(rows: list[dict]) -> str:
    """Counts per decision, then one line per meal with its reason -- the instructor's first look."""
    counts = Counter(r.get("decision", "error") for r in rows)
    lines = [f"# Folder run: {len(rows)} meals", ""]
    lines += [f"- {k}: {n}" for k, n in sorted(counts.items())]
    lines += ["", "| meal | decision | reason |", "|---|---|---|"]
    for r in rows:
        lines.append(f"| {r['meal_id']} | {r.get('decision', 'error')} | {r.get('reason') or r.get('error', '')} |")
    return "\n".join(lines) + "\n"
