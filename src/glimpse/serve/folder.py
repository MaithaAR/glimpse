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
PHOTO_NAMES = ("photo.jpg", "photo.jpeg", "photo.png", "photo.heic", "photo.webp")
IMAGE_SUFFIXES = (".jpg", ".jpeg", ".png", ".heic", ".webp")


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
        photo, pflags = find_photo(case)
        if photo == "several":
            names = ", ".join(f.name for f in sorted(case.iterdir()) if f.suffix.lower() in IMAGE_SUFFIXES)
            a = cant_judge(f"several photos and none named photo.jpg ({names}); keep one, or name the meal photo",
                           checks_failed=["input_valid"])  # fmt: skip
            return {"meal_id": case.name, **a.model_dump(mode="json")}
        cgm = case / "cgm.csv"
        product = (getattr(app, "model_versions", None) or {}).get("product")
        history, hflags = read_history_flagged(case / "history.csv", product)
        a = answer_from_files(app, cgm if cgm.exists() else None, photo, meal_time, profile, history)
        a.flags.extend(hflags + pflags)
        return {"meal_id": case.name, "product": product, **a.model_dump(mode="json")}
    except Exception as e:  # a malformed meal is reported in its own row; the run must finish
        logger.bind(stage="predict-folder").warning("{}: error ({})", case.name, e)
        return {"meal_id": case.name, "error": f"{type(e).__name__}: {e}"}


def find_photo(case: Path) -> tuple[Path | str | None, list[str]]:
    """The meal photo: photo.* first; otherwise (rule R18) the folder's only image, flagged; "several" when
    more than one unnamed image is present, so the caller can name them instead of guessing."""
    files = sorted(f for f in case.iterdir() if f.is_file())
    named = next((f for f in files if f.name.lower() in PHOTO_NAMES), None)
    if named:
        return named, []
    images = [f for f in files if f.suffix.lower() in IMAGE_SUFFIXES]
    if len(images) == 1:
        return images[0], ["photo_name_nonstandard"]
    return ("several", []) if images else (None, [])


def read_history(path: Path) -> list[tuple[datetime, float, int]] | None:
    """Optional history.csv (26 Nov): the person's earlier answers as meal_time, fusion_logit (from an
    earlier results.jsonl) and spiked (0/1, the CGM's label at +2 h). Rows that do not parse are skipped;
    a missing file means no history, the population calibration the samples are scored at."""
    return read_history_flagged(path, None)[0]


def read_history_flagged(path: Path, product: str | None) -> tuple[list[tuple[datetime, float, int]] | None, list]:
    """read_history plus what it skipped (26 Nov): an optional `product` column (as results.jsonl writes it)
    must match the served product, since a logit from the other product would set the intercept wrong;
    rows from another product and rows that do not parse are skipped and counted in the flags."""
    if not path.exists():
        return None, []
    rows, bad, other, tagged = [], 0, 0, False
    try:
        with open(path, newline="", encoding="utf-8-sig") as f:
            for r in csv.DictReader(f):
                tag = (r.get("product") or "").strip()
                tagged = tagged or bool(tag)
                if product and tag and tag != product:
                    other += 1
                    continue
                try:
                    t = datetime.fromisoformat(str(r["meal_time"]).strip()).replace(tzinfo=None)
                    z, y = float(r["fusion_logit"]), int(float(r["spiked"]))
                except KeyError, TypeError, ValueError:
                    bad += 1
                    continue
                if math.isfinite(z) and y in (0, 1):
                    rows.append((t, z, y))
                else:
                    bad += 1
    except OSError, UnicodeDecodeError, csv.Error:
        return None, ["history_unreadable"]
    flags = ([] if tagged else ["history_untagged"]) + ([f"history_other_product_skipped_{other}"] if other else [])
    return rows, flags + ([f"history_rows_skipped_{bad}"] if bad else [])


def _summary(rows: list[dict]) -> str:
    """Counts per decision, then one line per meal with its reason -- the instructor's first look."""
    counts = Counter(r.get("decision", "error") for r in rows)
    lines = [f"# Folder run: {len(rows)} meals", ""]
    lines += [f"- {k}: {n}" for k, n in sorted(counts.items())]
    lines += ["", "| meal | decision | reason | carbs from the photo |", "|---|---|---|---|"]
    for r in rows:
        lines.append(
            f"| {r['meal_id']} | {r.get('decision', 'error')} | {r.get('reason') or r.get('error', '')} | {_carbs(r)} |"
        )
    return "\n".join(lines) + "\n"


def _carbs(r: dict) -> str:
    """The carb assist as the user would see it, or why it was withheld (v2 answers the risk regardless)."""
    if r.get("carb_bin"):
        if "carb_heads_disagree" in r.get("flags", []):  # 26 Nov: never show 36 g as "under 30 g"
            from glimpse.serve.schemas import CARB_BINS

            other = CARB_BINS[sum(r["carbs_g"] >= e for e in (30, 60, 90))]
            return f"about {r['carbs_g']:.0f} g; range unsure ({r['carb_bin']} or {other}), please confirm"
        return f"{r['carb_bin']} ({r.get('carb_bin_prob', 0):.0%} sure)"
    plain = {"present": "no photo", "readable": "photo unreadable", "not_blank": "photo blank",
             "familiar": "photo unlike a meal"}  # fmt: skip
    why = [plain.get(f.removeprefix("carbs_unavailable_photo_"), f) for f in r.get("flags", [])
           if f.startswith("carbs_unavailable")]  # fmt: skip
    return f"withheld: {', '.join(why)}" if why else "–"
