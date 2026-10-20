"""Data-quality report, cross-split duplicate-photo check and the fine-tuning item ledger."""

from __future__ import annotations

import csv
import json
from itertools import combinations
from pathlib import Path

import numpy as np
from loguru import logger
from PIL import Image
from pydantic import BaseModel, ConfigDict

from glimpse.data.dataset import MealDataset
from glimpse.data.splits import Splits

FINETUNE_CAP = 10_000
DHASH_NEAR_DUP_BITS = 4  # Hamming distance at or below which two photos count as near-duplicates


class SplitRow(BaseModel):
    """One line of the split table."""

    model_config = ConfigDict(extra="forbid", frozen=True)
    split: str
    people: int
    meals_all: int
    meals_eval: int  # isolated, post-onboarding, passing pre-meal checks: the evaluation population
    spike_rate_eval: float
    with_photo: int


class LedgerEntry(BaseModel):
    """Fine-tuning items counted against the project cap."""

    model_config = ConfigDict(extra="forbid", frozen=True)
    dataset: str
    stage: str
    items: int
    status: str


def dhash(path: Path, size: int = 8) -> int:
    """64-bit difference hash: grayscale, resize to (size+1) x size, compare horizontally adjacent pixels."""
    with Image.open(path) as im:
        px = np.asarray(im.convert("L").resize((size + 1, size)), dtype=np.int16)
    bits = (px[:, 1:] > px[:, :-1]).flatten()
    return int("".join("1" if b else "0" for b in bits), 2)


def cross_split_near_duplicates(ds: MealDataset, sp: Splits, photo_root: Path) -> list[tuple[str, str, int]]:
    """Photo pairs from different splits whose dHashes differ in at most DHASH_NEAR_DUP_BITS bits."""
    where = {p: name for name in ("train", "validation", "test") for p in getattr(sp, name)}
    hashes = []
    for s in ds.samples:
        if s.photo_relpath and s.person_id in where and (photo_root / s.photo_relpath).is_file():
            hashes.append((s.photo_relpath, where[s.person_id], dhash(photo_root / s.photo_relpath)))
    pairs = []
    for (pa, sa, ha), (pb, sb, hb) in combinations(hashes, 2):
        if sa != sb and (d := (ha ^ hb).bit_count()) <= DHASH_NEAR_DUP_BITS:
            pairs.append((pa, pb, d))
    logger.bind(stage="quality").info("{} photos hashed, {} cross-split near-duplicate pairs", len(hashes), len(pairs))
    return pairs


def split_table(ds: MealDataset, sp: Splits) -> list[SplitRow]:
    """People, meals and spike rate per split."""
    rows = []
    for name in ("train", "validation", "test", "locked"):
        people = set(getattr(sp, name))
        ss = [s for s in ds.samples if s.person_id in people]
        ev = [s for s in ss if s.isolated and s.premeal_ok]
        rows.append(SplitRow(split=name, people=len(people), meals_all=len(ss), meals_eval=len(ev),
                             spike_rate_eval=round(float(np.mean([s.spike for s in ev])) if ev else 0.0, 3),
                             with_photo=sum(s.photo_relpath is not None for s in ss)))  # fmt: skip
    return rows


def finetune_ledger(ds: MealDataset, sp: Splits) -> list[LedgerEntry]:
    """Count photos that will be used to fine-tune the pretrained model (training people, photo + carb label)."""
    train = set(sp.train)
    n = sum(1 for s in ds.samples if s.person_id in train and s.photo_relpath and s.carbs_served_g is not None)
    return [
        LedgerEntry(dataset="CGMacros before-meal photos", stage="intermediate", items=n, status="used"),
        LedgerEntry(dataset="Nutrition5k overhead photos", stage="final (optional)", items=0, status="not used yet"),
    ]


def write_reports(ds: MealDataset, sp: Splits, photo_root: Path, out_dir: Path) -> None:
    """Write split table, data-quality summary, duplicate list and ledger to outputs/data/ and log the key facts."""
    log = logger.bind(stage="quality")
    out_dir.mkdir(parents=True, exist_ok=True)
    rows = split_table(ds, sp)
    with open(out_dir / "split_table.csv", "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(SplitRow.model_fields))
        w.writeheader()
        w.writerows(r.model_dump() for r in rows)
    for r in rows:
        log.info("split {:<10} people={:>2} meals={:>4} eval={:>3} spike_rate={:.3f}",
                 r.split, r.people, r.meals_all, r.meals_eval, r.spike_rate_eval)  # fmt: skip
    tab_missing = np.isnan(ds.tabular).mean(0)
    missing = {
        "glucose_window_missing_frac": float(np.isnan(ds.glucose).mean()),
        "heart_rate_window_missing_frac": float(np.isnan(ds.heart_rate).mean()),
        **{f"tabular_missing_frac:{n}": float(v) for n, v in zip(ds.tabular_names, tab_missing, strict=True)},
        "meals_failing_premeal_checks": sum(not s.premeal_ok for s in ds.samples),
        "class_counts_all": {
            "spike": int(sum(s.spike for s in ds.samples)),
            "no_spike": int(sum(1 - s.spike for s in ds.samples)),
        },
    }
    log.info("missing values and class counts: {}", missing)
    dups = cross_split_near_duplicates(ds, sp, photo_root)
    with open(out_dir / "cross_split_near_duplicates.csv", "w", newline="") as f:
        csv.writer(f).writerows([("photo_a", "photo_b", "hamming_bits"), *dups])
    ledger = finetune_ledger(ds, sp)
    total = sum(e.items for e in ledger)
    if total > FINETUNE_CAP:
        raise ValueError(f"fine-tuning items {total} exceed the project cap {FINETUNE_CAP}")
    log.info("fine-tuning ledger: {} of {} items", total, FINETUNE_CAP)
    (out_dir / "data_quality.json").write_text(json.dumps(
        {"missing_and_counts": missing, "cross_split_near_duplicate_pairs": len(dups),
         "finetune_ledger": [e.model_dump() for e in ledger], "finetune_total": total, "finetune_cap": FINETUNE_CAP},
        indent=2))  # fmt: skip
