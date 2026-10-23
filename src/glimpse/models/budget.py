"""Parameter budget: every model counted by code and checked against the project limits."""

from __future__ import annotations

import csv
from pathlib import Path

from loguru import logger
from pydantic import BaseModel, ConfigDict

from glimpse.models.sensor_cnn import SensorCNN, count_parameters

PER_MODEL_LIMIT = 200_000_000
PRETRAINED_TOTAL_LIMIT = 1_000_000_000
SCRATCH_LIMIT = 50_000_000

# models planned for later stages: (name, role, estimated parameters)
PLANNED = [("photo out-of-distribution check (DINOv2-S)", "pretrained, frozen; planned", 22_000_000)]


class BudgetRow(BaseModel):
    """One model in the budget table."""

    model_config = ConfigDict(strict=True, extra="forbid", frozen=True)
    model: str
    role: str
    status: str
    parameters: int
    limit: int
    within_limit: bool


def budget_table() -> list[BudgetRow]:
    """Count built models exactly; list planned ones with their estimates."""
    n = count_parameters(SensorCNN())
    rows = [BudgetRow(model="SensorCNN (1-D CNN)", role="trained from scratch", status="built", parameters=n,
                      limit=SCRATCH_LIMIT, within_limit=n <= SCRATCH_LIMIT)]  # fmt: skip
    from glimpse.models.photo_model import PhotoCarbModel

    n = sum(p.numel() for p in PhotoCarbModel().parameters())  # all weights (frozen or not); no download needed
    name = "PhotoCarbModel (ConvNeXt-Tiny + 2 heads)"
    rows.append(BudgetRow(model=name, role="pretrained, fine-tuned", status="built", parameters=n,
                          limit=PER_MODEL_LIMIT, within_limit=n <= PER_MODEL_LIMIT))  # fmt: skip
    for name, role, est in PLANNED:
        rows.append(BudgetRow(model=name, role=role, status="planned (estimate)", parameters=est,
                              limit=PER_MODEL_LIMIT, within_limit=est <= PER_MODEL_LIMIT))  # fmt: skip
    return rows


def write_budget(path: Path) -> list[BudgetRow]:
    """Write the table to CSV, log it, and stop if a limit is broken."""
    rows = budget_table()
    pretrained = sum(r.parameters for r in rows if r.role.startswith("pretrained"))
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(BudgetRow.model_fields))
        w.writeheader()
        w.writerows(r.model_dump() for r in rows)
    log = logger.bind(stage="budget")
    for r in rows:
        log.info("{:<44} {:>12,} parameters (limit {:,}; {})", r.model, r.parameters, r.limit, r.status)
    log.info("pretrained + fine-tuned total {:,} of {:,}", pretrained, PRETRAINED_TOTAL_LIMIT)
    if not all(r.within_limit for r in rows) or pretrained > PRETRAINED_TOTAL_LIMIT:
        raise ValueError("parameter budget exceeded")
    return rows
