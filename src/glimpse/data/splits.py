"""Seeded, person-grouped splits inside the development people (the 8 locked people are never used)."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
from loguru import logger
from pydantic import BaseModel, ConfigDict, model_validator

from glimpse.config import GlimpseConfig
from glimpse.data.dataset import MealDataset
from glimpse.evaluation.metrics import load_locked_set

N_TEST_PEOPLE = 5
N_VAL_PEOPLE = 4


class Splits(BaseModel):
    """Person ids per split; test and validation people come only from the target cohort."""

    model_config = ConfigDict(extra="forbid", frozen=True)
    seed: int
    train: list[int]
    validation: list[int]
    test: list[int]
    cv_folds: list[list[int]]  # target-cohort development people per fold (type 2 diabetes people always train)
    locked: list[int]

    @model_validator(mode="after")
    def _disjoint(self) -> Splits:
        """No person may appear in two splits or in the locked set."""
        sets = [set(self.train), set(self.validation), set(self.test), set(self.locked)]
        if sum(len(s) for s in sets) != len(set().union(*sets)):
            raise ValueError("splits overlap")
        return self

    def save(self, path: Path) -> None:
        """Write the split file (person ids only, no data)."""
        path.write_text(self.model_dump_json(indent=2))

    @classmethod
    def load(cls, path: Path) -> Splits:
        """Read and validate a split file."""
        return cls.model_validate(json.loads(path.read_text()))


def make_splits(
    ds: MealDataset, cfg: GlimpseConfig, locked_path: Path = Path("splits/locked_final_set.json")
) -> Splits:
    """Shuffle development target people with the config seed; take test, then validation, rest to train."""
    locked = sorted(load_locked_set(str(locked_path)))
    group = {s.person_id: s.group for s in ds.samples}
    dev = sorted(set(group) - set(locked))
    target = [p for p in dev if group[p] in cfg.cohort.target_groups]
    other = [p for p in dev if p not in target]
    rng = np.random.default_rng(cfg.seed)
    shuffled = [int(p) for p in rng.permutation(target)]
    test, val, train_t = (
        shuffled[:N_TEST_PEOPLE],
        shuffled[N_TEST_PEOPLE : N_TEST_PEOPLE + N_VAL_PEOPLE],
        shuffled[N_TEST_PEOPLE + N_VAL_PEOPLE :],
    )
    folds = [sorted(shuffled[i :: cfg.cv.outer_folds]) for i in range(cfg.cv.outer_folds)]
    sp = Splits(seed=cfg.seed, train=sorted(train_t + other), validation=sorted(val), test=sorted(test),
                cv_folds=folds, locked=locked)  # fmt: skip
    logger.bind(stage="splits").info(
        "people: train {} ({} target), validation {}, test {}, locked {}",
        len(sp.train), len(train_t), len(sp.validation), len(sp.test), len(sp.locked),
    )  # fmt: skip
    return sp
