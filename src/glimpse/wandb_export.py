"""W&B exports (rubric, Tooling/Reproducibility): every training history CSV and every results summary
in outputs/ is replayed into OFFLINE Weights & Biases runs under outputs/wandb/, so the experiment log
is inspectable with the W&B UI (`wandb sync outputs/wandb/wandb/offline-run-*` uploads to any account)
while the repository itself stays the single, network-free source of truth. The CSVs remain the primary
artifact; this export is regenerated from them by `just wandb-export`, so it is reproducible like every
other number."""

from __future__ import annotations

import json
import os
from pathlib import Path

import numpy as np
import pandas as pd
from loguru import logger

SUMMARIES = ["cv/cv_summary.json", "photo_cv/photo_cv_summary.json", "fusion/fusion_summary.json",
             "evaluation_v2/dev_summary.json", "ablations/d8_profile_excluded.json",
             "ablations/channels.json", "demo/demo_summary.json"]  # fmt: skip


def export_wandb(outputs: str | Path) -> Path:
    """Replay outputs/ into offline W&B runs: one run per history CSV (epoch metrics), one per summary
    JSON (final numbers). Returns the manifest path."""
    import wandb

    os.environ["WANDB_MODE"] = "offline"
    os.environ.setdefault("WANDB_SILENT", "true")
    outputs = Path(outputs)
    wdir = outputs / "wandb"
    wdir.mkdir(parents=True, exist_ok=True)
    manifest = []
    for hist in sorted(outputs.glob("**/*history*.csv")):
        df = pd.read_csv(hist)
        num = df.apply(pd.to_numeric, errors="coerce")
        run = wandb.init(project="glimpse", dir=str(wdir), name=f"{hist.parent.name}/{hist.stem}",
                         job_type="training", config={"source": str(hist)}, reinit=True)  # fmt: skip
        for i in range(len(df)):
            row = {c: float(num.iloc[i][c]) for c in num.columns if np.isfinite(num.iloc[i][c])}
            for c in ("model", "run"):
                if c in df.columns:
                    row[c] = str(df.iloc[i][c])
            wandb.log(row, step=i)
        run.finish()
        manifest.append({"name": run.name, "source": str(hist), "rows": int(len(df))})
        logger.bind(stage="wandb").info("exported {} ({} epochs)", run.name, len(df))
    for rel in SUMMARIES:
        p = outputs / rel
        if not p.is_file():
            continue
        run = wandb.init(project="glimpse", dir=str(wdir), name=rel.replace("/", ":").removesuffix(".json"),
                         job_type="results", config={"source": str(p)}, reinit=True)  # fmt: skip
        run.summary.update(json.loads(p.read_text()))
        run.finish()
        manifest.append({"name": run.name, "source": str(p)})
        logger.bind(stage="wandb").info("exported {}", run.name)
    out = wdir / "export_manifest.json"
    out.write_text(json.dumps(manifest, indent=2))
    logger.bind(stage="wandb").info("{} offline runs under {}; upload with `wandb sync` if wanted",
                                    len(manifest), wdir)  # fmt: skip
    return out
