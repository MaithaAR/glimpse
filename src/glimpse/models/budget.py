"""Parameter budget: every model counted programmatically and checked against the project limits."""

from __future__ import annotations

import csv
from pathlib import Path

from loguru import logger
from pydantic import BaseModel, ConfigDict

from glimpse.models.sensor_cnn import SensorCNN, count_parameters

PER_MODEL_LIMIT = 200_000_000
PRETRAINED_AGGREGATE_LIMIT = 1_000_000_000
SCRATCH_LIMIT = 50_000_000

# (name, timm architecture, role, status); weights are not needed to count parameters
PRETRAINED = [
    ("photo model", "convnext_tiny", "pretrained, fine-tuned (photo -> carbs)", "built"),
    ("photo baseline", "efficientnet_b0", "pretrained, frozen (proposal-pilot baseline)", "built"),
    ("photo OOD embedder", "vit_small_patch14_dinov2", "pretrained, frozen (out-of-distribution check)", "planned"),
]


class BudgetRow(BaseModel):
    """One model in the budget table."""

    model_config = ConfigDict(extra="forbid", frozen=True)
    model: str
    architecture: str
    role: str
    status: str
    parameters: int
    limit: int
    within_limit: bool


def budget_table(n_tabular: int = 8) -> list[BudgetRow]:
    """Count every model's parameters (instantiated without weights) and check each limit."""
    import timm

    rows = [
        BudgetRow(
            model="spike-risk sensor model",
            architecture="SensorCNN (1-D CNN)",
            role="trained from scratch",
            status="built",
            parameters=(n := count_parameters(SensorCNN(n_tabular=n_tabular))),
            limit=SCRATCH_LIMIT,
            within_limit=n <= SCRATCH_LIMIT,
        )  # fmt: skip
    ]
    for name, arch, role, status in PRETRAINED:
        n = count_parameters(timm.create_model(arch, pretrained=False, num_classes=0))
        rows.append(BudgetRow(model=name, architecture=arch, role=role, status=status, parameters=n,
                              limit=PER_MODEL_LIMIT, within_limit=n <= PER_MODEL_LIMIT))  # fmt: skip
    return rows


def write_budget(path: Path) -> list[BudgetRow]:
    """Write the table to CSV, log it, and fail if any limit is exceeded."""
    rows = budget_table()
    agg = sum(r.parameters for r in rows if r.role.startswith("pretrained"))
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(BudgetRow.model_fields))
        w.writeheader()
        w.writerows(r.model_dump() for r in rows)
    log = logger.bind(stage="budget")
    for r in rows:
        log.info("{:<26} {:<26} {:>12,} params  ({}, {})", r.model, r.architecture, r.parameters, r.role, r.status)
    log.info("pretrained/fine-tuned aggregate {:,} of {:,}", agg, PRETRAINED_AGGREGATE_LIMIT)
    if not all(r.within_limit for r in rows) or agg > PRETRAINED_AGGREGATE_LIMIT:
        raise ValueError("parameter budget exceeded")
    return rows
