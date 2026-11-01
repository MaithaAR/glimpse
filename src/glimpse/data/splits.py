"""Person-level splits. Every meal of a person stays in one group, so results are always on unseen people.

- locked: 8 target people sealed in the proposal; never used before the final run.
- outer CV folds: the 23 development target people in 5 groups; each fold is tested once (main estimate).
- fixed split: test = outer fold 1; validation = 4 more target people; train = everyone else, including all
  type 2 diabetes people (they help training but are never scored).
- inner folds: groups inside the training people, used for tuning only."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd
from loguru import logger
from pydantic import BaseModel, ConfigDict, model_validator

from glimpse.config import GlimpseConfig
from glimpse.data.dataset import MealDataset
from glimpse.evaluation import load_locked_set


class Splits(BaseModel):
    """Person ids per group (no data). Saved to splits/dev_splits.json and committed."""

    model_config = ConfigDict(strict=True, extra="forbid", frozen=True)
    seed: int
    locked: list[int]
    outer_folds: list[list[int]]  # target development people per outer fold
    test: list[int]
    validation: list[int]
    train: list[int]
    inner_folds: list[list[int]]  # training people per inner fold

    @model_validator(mode="after")
    def _no_person_twice(self) -> Splits:
        """A person may appear in only one of train / validation / test / locked."""
        groups = [set(self.train), set(self.validation), set(self.test), set(self.locked)]
        if sum(len(g) for g in groups) != len(set().union(*groups)):
            raise ValueError("a person appears in more than one split")
        return self

    def save(self, path: Path) -> None:
        """Write person ids as JSON."""
        path.write_text(self.model_dump_json(indent=2))

    @classmethod
    def load(cls, path: Path) -> Splits:
        """Read and re-check."""
        return cls.model_validate_json(path.read_text())


def _folds(people: list[int], k: int, rng: np.random.Generator) -> list[list[int]]:
    """Shuffle people and deal them into k groups like cards."""
    shuffled = [int(p) for p in rng.permutation(sorted(people))]
    return [sorted(shuffled[i::k]) for i in range(k)]


def make_splits(ds: MealDataset, cfg: GlimpseConfig) -> Splits:
    """Seeded person-level splits (same result on every run with the same seed)."""
    locked = sorted(load_locked_set(cfg.splits.locked_file))
    group = {s.person_id: s.group for s in ds.samples}
    dev = sorted(set(group) - set(locked))
    target = [p for p in dev if group[p] in cfg.cohort.target_groups]
    other = [p for p in dev if p not in target]
    rng = np.random.default_rng(cfg.seed)
    outer = _folds(target, cfg.cv.outer_folds, rng)
    test = outer[0]
    if len(test) != cfg.splits.test_people:
        raise ValueError(f"outer fold 1 has {len(test)} people, config expects {cfg.splits.test_people}")
    rest = [int(p) for p in rng.permutation(sorted(set(target) - set(test)))]
    validation = sorted(rest[: cfg.splits.validation_people])
    train = sorted(rest[cfg.splits.validation_people :] + other)
    inner = [sorted(a + b) for a, b in zip(_folds([p for p in train if p in target], cfg.cv.inner_folds, rng),
                                             _folds(other, cfg.cv.inner_folds, rng), strict=True)]  # fmt: skip
    return Splits(seed=cfg.seed, locked=locked, outer_folds=outer, test=test, validation=validation, train=train,
                  inner_folds=inner)  # fmt: skip


def meal_rows(ds: MealDataset, people: set[int], scored_only: bool) -> np.ndarray:
    """Row indices for these people. Training uses every meal after onboarding that passes the pre-meal checks;
    scoring (validation, test) uses only isolated meals, so each rise belongs to one meal."""
    return np.array([i for i, s in enumerate(ds.samples) if s.person_id in people and s.premeal_ok
                     and not s.onboarding and (s.isolated or not scored_only)], dtype=int)  # fmt: skip


def split_table(ds: MealDataset, sp: Splits) -> pd.DataFrame:
    """People, meals and spike rate per group: the sizes reported in the data card."""
    y = ds.labels()
    rows = []
    groups = {"train": sp.train, "validation": sp.validation, "test (= outer fold 1)": sp.test, "locked": sp.locked}
    groups |= {f"outer fold {k + 1}": f for k, f in enumerate(sp.outer_folds)}
    for name, people in groups.items():
        train_rows, scored = meal_rows(ds, set(people), False), meal_rows(ds, set(people), True)
        rows.append({"group": name, "people": len(people), "training_meals": len(train_rows),
                     "scored_meals": len(scored), "spike_rate_scored": round(float(y[scored].mean()), 3)})  # fmt: skip
    t = pd.DataFrame(rows)
    logger.bind(stage="splits").info("split sizes:\n{}", t.to_string(index=False))
    return t


def save_split_outputs(ds: MealDataset, sp: Splits, cfg: GlimpseConfig) -> None:
    """Commit-able split file + the size table in outputs/data/."""
    sp.save(Path("splits/dev_splits.json"))
    out = Path(cfg.paths.outputs) / "data"
    out.mkdir(parents=True, exist_ok=True)
    split_table(ds, sp).to_csv(out / "split_table.csv", index=False)
    (out / "split_people.json").write_text(json.dumps(sp.model_dump(), indent=2))


class SplitsV2(BaseModel):
    """Partition v2 (design D1, final stage): no fixed validation or test group. Each of the dev folds is
    held out once (development estimate); the inner folds inside a fold's training targets are the
    validation split for every selection. Type 2 diabetes people train in every fold, are never scored,
    and never sit on an inner validation side. The locked people are scored once, at the very end."""

    model_config = ConfigDict(strict=True, extra="forbid", frozen=True)
    seed: int
    locked: list[int]
    dev_folds: list[list[int]]  # target development people; each fold held out once
    t2d: list[int]  # training-only people, in every fold's training set
    inner_folds: list[list[list[int]]]  # [outer fold][inner fold] -> that fold's training TARGETS only

    @model_validator(mode="after")
    def _partition(self) -> SplitsV2:
        """Locked, dev and T2D are disjoint; every dev person sits in exactly one fold; each fold's inner
        folds partition exactly the dev targets outside that fold (T2D never on a validation side)."""
        dev = [p for f in self.dev_folds for p in f]
        if len(dev) != len(set(dev)):
            raise ValueError("a development person appears in more than one fold")
        for a, b in (("locked", "dev"), ("locked", "t2d"), ("dev", "t2d")):
            x = {"locked": set(self.locked), "dev": set(dev), "t2d": set(self.t2d)}
            if x[a] & x[b]:
                raise ValueError(f"{a} and {b} share people: {sorted(x[a] & x[b])}")
        for k, inner in enumerate(self.inner_folds):
            expect = set(dev) - set(self.dev_folds[k])
            flat = [p for f in inner for p in f]
            if len(flat) != len(set(flat)) or set(flat) != expect:
                raise ValueError(f"inner folds of fold {k + 1} do not partition its training targets")
        return self

    def training_people(self, k: int) -> list[int]:
        """Everyone fold k trains on: the dev targets outside the fold plus every T2D person."""
        dev = [p for j, f in enumerate(self.dev_folds) if j != k for p in f]
        return sorted(dev + list(self.t2d))

    def save(self, path: Path) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(self.model_dump_json(indent=2) + "\n")

    @classmethod
    def load(cls, path: Path) -> SplitsV2:
        return cls.model_validate_json(Path(path).read_text())


def make_splits_v2(ds: MealDataset, cfg: GlimpseConfig) -> SplitsV2:
    """Seeded partition v2: same folds on every run with the same seed and dataset."""
    locked = sorted(load_locked_set(cfg.splits.locked_file))
    group = {s.person_id: s.group for s in ds.samples}
    dev = sorted(set(group) - set(locked))
    target = [p for p in dev if group[p] in cfg.cohort.target_groups]
    t2d = sorted(p for p in dev if p not in target)
    rng = np.random.default_rng(cfg.seed)
    folds = _folds(target, cfg.cv.outer_folds, rng)
    inner = [_folds(sorted(set(target) - set(f)), cfg.cv.inner_folds, rng) for f in folds]
    return SplitsV2(seed=cfg.seed, locked=locked, dev_folds=folds, t2d=t2d, inner_folds=inner)


def assert_not_locked(people, locked, context: str) -> None:
    """Structural guard: any locked person reaching a development data path is a hard error. The locked set
    is scored once, at the end, through its own recipe -- never as a side effect."""
    hit = sorted(set(people) & set(locked))
    if hit:
        raise RuntimeError(f"{context}: locked people {hit} must never be loaded here; "
                           f"they are scored once, at the final locked-set run")  # fmt: skip


def reconciliation_v2(ds: MealDataset, old: Splits, new: SplitsV2) -> str:
    """The meal-level accounting that makes partition v2 checkable: every development meal is scored in
    exactly one held-out fold, fold training and held-out meals never overlap, T2D and locked people are
    never scored. Raises on any violation; returns the markdown that is committed as splits/reconciliation.md."""
    y = ds.labels()
    held = [meal_rows(ds, set(f), scored_only=True) for f in new.dev_folds]
    all_scored = meal_rows(ds, {p for f in new.dev_folds for p in f}, scored_only=True)
    flat = np.concatenate(held)
    if len(flat) != len(set(flat.tolist())) or set(flat.tolist()) != set(all_scored.tolist()):
        raise RuntimeError("held-out fold meals do not cover every development meal exactly once")
    for k in range(len(new.dev_folds)):
        tr = meal_rows(ds, set(new.training_people(k)), scored_only=False)
        if set(tr.tolist()) & set(held[k].tolist()):
            raise RuntimeError(f"fold {k + 1}: training meals overlap its held-out meals")
        assert_not_locked(new.training_people(k) + new.dev_folds[k], new.locked, f"fold {k + 1}")
    where = {p: k + 1 for k, f in enumerate(new.dev_folds) for p in f}
    lines = ["# Partition v2 reconciliation", "",
             "Generated by `just splits-v2`; every statement below is asserted in code before this file is",
             "written. Design decision D1 in docs/final_design.md.", "",
             f"- development meals scored exactly once across the {len(new.dev_folds)} folds: "
             f"{len(flat)} held-out meals = {len(all_scored)} isolated development meals, no meal twice",
             "- no fold's training meals overlap its held-out meals (checked at meal level, not person level)",
             f"- type 2 diabetes people ({len(new.t2d)}) train in every fold, are never scored, and never sit",
             "  on an inner validation side (inner folds partition target training people only)",
             f"- locked people ({len(new.locked)}): never loaded by any development recipe (structural guard",
             "  `assert_not_locked`); scored once at the final locked-set run",
             "", "## Fold sizes", "", "| fold | people | held-out meals | spike rate | training people |",
             "|---|---|---|---|---|"]  # fmt: skip
    for k, f in enumerate(new.dev_folds):
        r = held[k]
        lines.append(f"| {k + 1} | {len(f)} | {len(r)} | {float(y[r].mean()):.3f} | "
                     f"{len(new.training_people(k))} |")  # fmt: skip
    lines += ["", "## Where the intermediate's fixed groups landed", "",
              "The intermediate's 4-person validation and 5-person test groups dissolve into development",
              "(D1); their labels informed the intermediate's calibration diagnosis, so development numbers",
              "are optimistic and the locked set is the only clean measurement.", "",
              "| person | was | now in fold |", "|---|---|---|"]  # fmt: skip
    for p in sorted(old.test):
        lines.append(f"| {p} | test (intermediate, scored once) | {where[p]} |")
    for p in sorted(old.validation):
        lines.append(f"| {p} | validation (intermediate) | {where[p]} |")
    return "\n".join(lines) + "\n"
