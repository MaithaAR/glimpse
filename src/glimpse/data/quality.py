"""Data-quality report, duplicate-photo check and fine-tuning ledger. Written to outputs/data/ by `glimpse prepare`;
any broken assertion stops the run."""

from __future__ import annotations

import hashlib
import json
from collections import defaultdict
from pathlib import Path

import numpy as np
import pandas as pd
from loguru import logger
from PIL import Image

from glimpse.config import GlimpseConfig
from glimpse.data.dataset import MealDataset
from glimpse.data.splits import Splits, meal_rows

DHASH_MAX_DISTANCE = 4  # differing bits (of 64) for two photos to count as "looks the same"


def dhash(path: Path, size: int = 8) -> int:
    """Perceptual 'difference hash': shrink to 9x8 grey, compare neighbouring pixels -> 64-bit fingerprint.
    Re-saved or resized copies of a photo get (almost) the same fingerprint; different photos do not."""
    with Image.open(path) as im:
        px = np.asarray(im.convert("L").resize((size + 1, size)), dtype=int)
    bits = (px[:, 1:] > px[:, :-1]).flatten()
    return int("".join("1" if b else "0" for b in bits), 2)


def _group_of(sp: Splits) -> dict[int, str]:
    """Person id -> split name."""
    g = {p: "train" for p in sp.train} | {p: "validation" for p in sp.validation} | {p: "test" for p in sp.test}
    return g | {p: "locked" for p in sp.locked}


def duplicate_photos(ds: MealDataset, sp: Splits, photo_root: Path) -> tuple[pd.DataFrame, dict]:
    """Exact copies (SHA-256) and look-alikes (dHash) between different splits, plus how often a person's own
    photos look alike (repeated meals). With person-level splits, a cross-split hit means a file or ID mix-up."""
    group = _group_of(sp)
    items = [
        (s.person_id, photo_root / s.photo_relpath)
        for s in ds.samples
        if s.photo_relpath and (photo_root / s.photo_relpath).is_file() and s.person_id not in sp.locked
    ]
    sha = [hashlib.sha256(p.read_bytes()).hexdigest() for _, p in items]
    dh = [dhash(p) for _, p in items]
    pairs, within = [], defaultdict(lambda: [0, 0])
    for i in range(len(items)):
        for j in range(i + 1, len(items)):
            a, b = items[i][0], items[j][0]
            dist = (dh[i] ^ dh[j]).bit_count()
            same = sha[i] == sha[j] or dist <= DHASH_MAX_DISTANCE
            if a == b:
                within[a][0] += 1
                within[a][1] += same
            elif same and group[a] != group[b]:
                pairs.append(
                    {
                        "photo_a": str(items[i][1]),
                        "photo_b": str(items[j][1]),
                        "split_a": group[a],
                        "split_b": group[b],
                        "exact": sha[i] == sha[j],
                        "dhash_distance": dist,
                    }
                )
    pair_counts = np.array(list(within.values()))
    info = {
        "photos_checked": len(items),
        "cross_split_duplicates": len(pairs),
        "within_person_lookalike_pair_rate": float(pair_counts[:, 1].sum() / max(pair_counts[:, 0].sum(), 1)),
    }
    return pd.DataFrame(pairs, columns=["photo_a", "photo_b", "split_a", "split_b", "exact", "dhash_distance"]), info


def finetune_ledger(ds: MealDataset, sp: Splits, photo_root: Path, cap: int) -> pd.DataFrame:
    """Photos that fine-tuning could ever use, against the project cap. An item = one photo used to fine-tune.
    Cross-validation trains on different people per fold, so the worst case counts every development photo."""

    def n_photos(people: set[int]) -> int:
        return sum(
            1
            for s in ds.samples
            if s.person_id in people
            and s.photo_relpath
            and s.carbs_served_g is not None
            and (photo_root / s.photo_relpath).is_file()
        )

    dev = set(sp.train) | set(sp.validation) | set(sp.test)
    rows = [
        {
            "dataset": "CGMacros before-meal photos, fixed training people",
            "stage": "intermediate",
            "items": n_photos(set(sp.train)),
        },
        {
            "dataset": "CGMacros before-meal photos, all development people (worst case over CV folds)",
            "stage": "intermediate",
            "items": n_photos(dev),
        },
        {"dataset": "Nutrition5k plates (optional pre-training)", "stage": "final, planned", "items": 0},
    ]
    t = pd.DataFrame(rows)
    total = int(t["items"].iloc[1:].sum())  # worst case + later stages (the fixed-split row is a subset)
    t.loc[len(t)] = {"dataset": f"running total (worst case) vs cap {cap:,}", "stage": "", "items": total}
    if total > cap:
        raise ValueError(f"fine-tuning items {total} exceed the project cap {cap}")
    return t


def quality_report(ds: MealDataset, sp: Splits, cfg: GlimpseConfig) -> dict:
    """Missing values, out-of-range readings, label balance by split and cohort, per-person counts, input means."""
    lo, hi = cfg.pipeline.glucose_range_mgdl
    y = ds.labels()
    assert ds.glucose.shape == ds.heart_rate.shape == (len(ds.samples), cfg.pipeline.cgm_lookback_min // 5)
    assert ds.profile.shape == (len(ds.samples), len(ds.profile_names))
    g = ds.glucose[~np.isnan(ds.glucose)]
    rep: dict = {
        "meals": len(ds.samples),
        "missing_share": {
            "glucose_slots": float(np.isnan(ds.glucose).mean()),
            "heart_rate_slots": float(np.isnan(ds.heart_rate).mean()),
            **{n: float(np.isnan(ds.profile[:, k]).mean()) for k, n in enumerate(ds.profile_names)},
        },
        "windows_needing_gap_fill": float(np.isnan(ds.glucose).any(axis=1).mean()),
        "glucose_out_of_range_readings": int(((g < lo) | (g > hi)).sum()),
        "flagged_by_premeal_checks": int(sum(not s.premeal_ok for s in ds.samples)),
    }
    by_split = {}
    for name, people in {"train": sp.train, "validation": sp.validation, "test": sp.test}.items():
        rows = meal_rows(ds, set(people), scored_only=True)
        cohort = np.array([ds.samples[i].group for i in rows])
        by_split[name] = {
            "scored_meals": len(rows),
            "spikes": int(y[rows].sum()),
            "spike_rate_target_cohort": float(y[rows][cohort != "T2D"].mean()),
            "spike_rate_T2D": float(y[rows][cohort == "T2D"].mean()) if (cohort == "T2D").any() else None,
            "mean_glucose_last_slot": float(np.nanmean(ds.glucose[rows, -1])),
            "mean_a1c": float(np.nanmean(ds.profile[rows, ds.profile_names.index("a1c_pct")])),
        }
    rep["by_split"] = by_split
    per = pd.DataFrame([{"person": s.person_id, "spike": s.spike} for s in ds.samples if s.isolated and s.premeal_ok])
    stats = per.groupby("person").spike.agg(["size", "mean"])
    rep["per_person_scored_meals"] = {
        "min": int(stats["size"].min()),
        "median": float(stats["size"].median()),
        "max": int(stats["size"].max()),
    }
    rep["per_person_spike_rate"] = {
        "min": float(stats["mean"].min()),
        "median": float(stats["mean"].median()),
        "max": float(stats["mean"].max()),
    }
    return rep


def write_quality_outputs(ds: MealDataset, sp: Splits, cfg: GlimpseConfig) -> None:
    """Run every check and write outputs/data/{data_quality.json, cross_split_duplicates.csv, finetune_ledger.csv}."""
    log = logger.bind(stage="quality")
    out = Path(cfg.paths.outputs) / "data"
    out.mkdir(parents=True, exist_ok=True)
    root = Path(cfg.data.cgmacros_root)
    rep = quality_report(ds, sp, cfg)
    pairs, dup = duplicate_photos(ds, sp, root)
    rep["duplicate_photos"] = dup
    pairs.to_csv(out / "cross_split_duplicates.csv", index=False)
    ledger = finetune_ledger(ds, sp, root, cfg.finetune_cap_items)
    ledger.to_csv(out / "finetune_ledger.csv", index=False)
    (out / "data_quality.json").write_text(json.dumps(rep, indent=2))
    log.info(
        "missing: glucose {:.2%}, heart rate {:.2%}; {} out-of-range readings; {} flagged meals",
        rep["missing_share"]["glucose_slots"],
        rep["missing_share"]["heart_rate_slots"],
        rep["glucose_out_of_range_readings"],
        rep["flagged_by_premeal_checks"],
    )
    for k, v in rep["by_split"].items():
        log.info(
            "{}: {} scored meals, spike rate target {:.2f}, T2D {}",
            k,
            v["scored_meals"],
            v["spike_rate_target_cohort"],
            "-" if v["spike_rate_T2D"] is None else f"{v['spike_rate_T2D']:.2f}",
        )
    log.info(
        "photos checked {}: {} cross-split duplicates; within-person look-alike pair rate {:.3f}",
        dup["photos_checked"],
        dup["cross_split_duplicates"],
        dup["within_person_lookalike_pair_rate"],
    )
    log.info("fine-tuning ledger:\n{}", ledger.to_string(index=False))
