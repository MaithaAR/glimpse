"""S12 efficiency: parameters against the assignment caps, serving latency, and artifact sizes --
measured on the deployed pipeline itself, not quoted from a paper. Report: section 2, one table."""

from __future__ import annotations

import json
import platform
import resource
import time
from pathlib import Path

import numpy as np
from loguru import logger

from glimpse.config import GlimpseConfig
from glimpse.serve import Pipeline
from glimpse.serve.io import answer_from_files
from glimpse.training.final import FINAL, _photo_checkpoint

OUT = Path("outputs/efficiency")
CAPS = {"per_model": 200_000_000, "pretrained_total": 1_000_000_000, "scratch_total": 50_000_000}
CASE, REPS, WARMUP = Path("samples/instructor/00_clean"), 15, 2


def _count(module) -> int:
    return int(sum(p.numel() for p in module.parameters()))


def _case_inputs() -> tuple:
    from glimpse.serve.folder import PROFILE_KEYS

    raw = json.loads((CASE / "profile.json").read_text())
    meal_time = raw.pop("meal_time", None)
    prof = {PROFILE_KEYS[k]: v for k, v in raw.items() if k in PROFILE_KEYS}
    return CASE / "cgm.csv", CASE / "photo.jpg", meal_time, prof


def run_efficiency(cfg: GlimpseConfig) -> dict:
    """Count every parameter in the shipped system, time cold and warm answers, list artifact sizes."""
    from glimpse.models.sensor_cnn import SensorCNN

    log = logger.bind(stage="efficiency")
    t0 = time.perf_counter()
    app = Pipeline(cfg)
    cgm, photo, meal_time, prof = _case_inputs()
    cold_s = None
    times = []
    for i in range(WARMUP + REPS):
        t = time.perf_counter()
        a = answer_from_files(app, cgm, photo, meal_time, dict(prof))
        dt = time.perf_counter() - t
        if i == 0:
            cold_s = time.perf_counter() - t0  # Pipeline construction + first answer (weights paged in)
            assert a.decision in ("low risk", "caution", "can't judge")
        if i >= WARMUP:
            times.append(dt)
    backbone, heads = _count(app.model.backbone), _count(app.model.reg_head) + _count(app.model.bin_head)
    sensor = _count(SensorCNN())
    lr = app.fusion[-1] if hasattr(app.fusion, "steps") else app.fusion  # sklearn Pipeline or bare LR
    fusion_params = int(lr.coef_.size + lr.intercept_.size)
    rss = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss  # bytes on macOS, KiB on Linux
    rss_mb = rss / 1e6 if platform.system() == "Darwin" else rss / 1e3
    arts = {p.name: round(p.stat().st_size / 1e6, 1)
            for p in [_photo_checkpoint(), FINAL, Path(cfg.prototype.ood_reference_file)] if p.is_file()}  # fmt: skip
    res = {"what": "parameters vs the assignment caps, serving latency on instructor case 00_clean, sizes",
           "params": {"photo_backbone_pretrained": backbone, "photo_heads_scratch": heads,
                      "sensor_cnn_scratch": sensor, "fusion_lr_scratch": fusion_params,
                      "largest_model_total": backbone + heads,
                      "pretrained_total": backbone,
                      "scratch_total": heads + sensor + fusion_params},
           "caps": CAPS,
           "caps_ok": {"per_model": backbone + heads <= CAPS["per_model"],
                       "pretrained_total": backbone <= CAPS["pretrained_total"],
                       "scratch_total": heads + sensor + fusion_params <= CAPS["scratch_total"]},
           "latency_s": {"cold_first_answer": round(float(cold_s), 2),
                         "warm_median": round(float(np.median(times)), 3),
                         "warm_p95": round(float(np.quantile(times, 0.95)), 3), "reps": REPS},
           "peak_rss_mb": round(rss_mb, 0), "artifact_sizes_mb": arts,
           "machine": f"{platform.system()} {platform.machine()}, "
                      f"torch threads capped at {min(cfg.training.num_threads, 4)}"}  # fmt: skip
    OUT.mkdir(parents=True, exist_ok=True)
    (OUT / "efficiency.json").write_text(json.dumps(res, indent=2))
    p = res["params"]
    log.info("params: backbone {:.1f}M pretrained + {} head + {} sensorCNN + {} fusion scratch; "
             "caps ok={}", p["photo_backbone_pretrained"] / 1e6, p["photo_heads_scratch"],
             p["sensor_cnn_scratch"], p["fusion_lr_scratch"], res["caps_ok"])  # fmt: skip
    lt = res["latency_s"]
    log.info("latency: cold {:.2f}s, warm median {:.0f} ms, p95 {:.0f} ms; peak RSS {:.0f} MB; "
             "artifacts {}", lt["cold_first_answer"], lt["warm_median"] * 1e3, lt["warm_p95"] * 1e3,
             res["peak_rss_mb"], res["artifact_sizes_mb"])  # fmt: skip
    return res
