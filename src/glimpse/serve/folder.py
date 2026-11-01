"""Batch entry point for a folder of meals, as the instructor will run it: one sub-folder per meal
(photo.*, cgm.csv, profile.json) -> results.jsonl + summary.md. A bad meal becomes a per-item entry,
never a crash, and the run always exits 0. Report: instructor-readiness in §6c."""

from __future__ import annotations

import json
from collections import Counter
from pathlib import Path
from typing import TYPE_CHECKING

from loguru import logger

from glimpse.serve.io import answer_from_files

if TYPE_CHECKING:
    from glimpse.serve.pipeline import Pipeline

# profile.json keys (docs/final_design.md §6) -> the model's profile input names
PROFILE_KEYS = {"age": "age", "bmi": "bmi", "a1c": "a1c_pct", "fasting_glucose": "fasting_glucose_mgdl",
                "prior_spike_rate": "prior_spike_rate", "prior_rise": "prior_rise"}  # fmt: skip
PHOTO_NAMES = ("photo.jpg", "photo.jpeg", "photo.png", "photo.heic")


def predict_folder(app: Pipeline, in_dir: str | Path, out_dir: str | Path) -> Path:
    """Answer every meal folder under `in_dir`; write results.jsonl and summary.md under `out_dir`."""
    cases = sorted(p for p in Path(in_dir).iterdir() if p.is_dir())
    if not cases:
        raise ValueError(f"no meal folders under {in_dir}")
    rows = [_one_case(app, case) for case in cases]
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    results = out / "results.jsonl"
    results.write_text("".join(json.dumps(r) + "\n" for r in rows))
    (out / "summary.md").write_text(_summary(rows))
    logger.bind(stage="predict-folder").info("{} meals -> {}", len(rows), results)
    return results


def _one_case(app: Pipeline, case: Path) -> dict:
    """One meal folder -> one validated Answer as a plain dict (or an error entry; the batch continues)."""
    try:
        raw = json.loads((case / "profile.json").read_text()) if (case / "profile.json").exists() else {}
        meal_time = raw.pop("meal_time", None)
        profile = {PROFILE_KEYS[k]: v for k, v in raw.items() if k in PROFILE_KEYS}
        photo = next((case / n for n in PHOTO_NAMES if (case / n).exists()), None)
        cgm = case / "cgm.csv"
        a = answer_from_files(app, cgm if cgm.exists() else None, photo, meal_time, profile)
        return {"meal_id": case.name, **a.model_dump(mode="json")}
    except Exception as e:  # a malformed meal is reported in its own row; the run must finish
        logger.bind(stage="predict-folder").warning("{}: error ({})", case.name, e)
        return {"meal_id": case.name, "error": f"{type(e).__name__}: {e}"}


def _summary(rows: list[dict]) -> str:
    """Counts per decision, then one line per meal with its reason -- the instructor's first look."""
    counts = Counter(r.get("decision", "error") for r in rows)
    lines = [f"# Folder run: {len(rows)} meals", ""]
    lines += [f"- {k}: {n}" for k, n in sorted(counts.items())]
    lines += ["", "| meal | decision | reason |", "|---|---|---|"]
    for r in rows:
        lines.append(f"| {r['meal_id']} | {r.get('decision', 'error')} | {r.get('reason') or r.get('error', '')} |")
    return "\n".join(lines) + "\n"
