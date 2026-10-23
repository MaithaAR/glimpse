"""Build the labelled meal table with the proposal's rules (legacy.build_meals, unchanged), then validate every
row against the MealRecord schema. Rows that fail are dropped and logged with the reason."""

from __future__ import annotations

import json
import math
from pathlib import Path

import pandas as pd
from loguru import logger
from pydantic import ValidationError

from glimpse.config import GlimpseConfig
from glimpse.data.legacy import add_personal_prior, build_meals, evaluation_set
from glimpse.schemas import MealRecord


def _native(v):
    """pandas/NumPy values -> plain Python (strict validation does not guess types). NaN -> None."""
    if v is None or (isinstance(v, float) and math.isnan(v)):
        return None
    if hasattr(v, "item"):  # NumPy scalar
        v = v.item()
        return None if isinstance(v, float) and math.isnan(v) else v
    if isinstance(v, pd.Timestamp):
        return v.to_pydatetime()
    return v


def validate_meals(M: pd.DataFrame) -> tuple[pd.DataFrame, int]:
    """Keep rows that pass MealRecord; log each rejected row. Returns (valid rows, number rejected)."""
    log = logger.bind(stage="data")
    keep, rejected = [], 0
    for i, row in enumerate(M.to_dict(orient="records")):
        try:
            MealRecord.model_validate({k: _native(v) for k, v in row.items()})
            keep.append(i)
        except ValidationError as e:
            rejected += 1
            log.warning("rejected meal sid={} t={}: {}", row.get("sid"), row.get("t"), e.errors()[0]["msg"])
    return M.iloc[keep].reset_index(drop=True), rejected


def meal_counts(M: pd.DataFrame, cfg: GlimpseConfig) -> dict[str, int]:
    """The accounting reported in the proposal (usable, onboarding, operational, isolated, target cohort)."""
    target = cfg.cohort.target_groups
    E = evaluation_set(M)
    O = M[M.post_onboarding & M.usable & M.rise.notna()]  # noqa: E741 - O = operational meals
    return {
        "usable": int(M.usable.sum()),
        "onboarding": int((M.usable & M.onboarding).sum()),
        "operational": len(O),
        "operational_target": int(O.group.isin(target).sum()),
        "isolated": len(E),
        "isolated_target": int(E.group.isin(target).sum()),
        "isolated_target_people": int(E[E.group.isin(target)].sid.nunique()),
    }


def build_meal_table(cfg: GlimpseConfig, out_csv: Path = Path("data/meals.csv")) -> pd.DataFrame:
    """Meal table -> schema validation -> personal prior -> saved to data/meals.csv, counts to outputs/data/."""
    log = logger.bind(stage="data")
    raw = cfg.model_dump()
    M = build_meals(cfg.data.cgmacros_root, raw)
    log.info("built {} candidate meals from {} people", len(M), M.sid.nunique())
    M, rejected = validate_meals(M)
    M = add_personal_prior(M, raw)
    counts = meal_counts(M, cfg) | {"rejected_by_schema": rejected}
    log.info("meal accounting: {}", counts)
    out_csv.parent.mkdir(parents=True, exist_ok=True)
    M.to_csv(out_csv, index=False)
    out = Path(cfg.paths.outputs) / "data" / "meal_counts.json"
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(counts, indent=2))
    log.info("meal table written to {} ({} rows); counts to {}", out_csv, len(M), out)
    return M
