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
