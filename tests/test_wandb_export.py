"""The W&B export replays a history CSV and a summary JSON into offline runs with a manifest."""

import json
from pathlib import Path

import pytest


def test_export_creates_offline_runs_and_manifest(tmp_path):
    pytest.importorskip("wandb")
    from glimpse.wandb_export import export_wandb

    out = tmp_path / "outputs"
    (out / "photo_cv").mkdir(parents=True)
    (out / "fusion").mkdir()
    rows = "model,run,epoch,train_loss,val_loss\nm,r,1,1.0,2.0\nm,r,2,0.5,-\n"
    (out / "photo_cv" / "tiny_history.csv").write_text(rows)
    (out / "fusion" / "fusion_summary.json").write_text(json.dumps({"auroc_pooled": {"photo": 0.72}}))
    manifest = export_wandb(out)
    m = json.loads(Path(manifest).read_text())
    assert len(m) == 2 and m[0]["rows"] == 2
    assert any((out / "wandb").rglob("*.wandb")), "no offline run files written"
