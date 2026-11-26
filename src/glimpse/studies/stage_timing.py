"""Where serving time goes, stage by stage (`glimpse stage-timing`, 26 Nov, descriptive).

Times one answer's stages on this machine: photo decode + transform, the ConvNeXt-Tiny pass, the kNN check
against the committed 1,221-photo reference, the heads, and the 20-parameter fusion with its cut-offs. The
backbone is built with random weights (no download): the arithmetic, and so the time, is the same as with the
pinned weights, but its outputs are not used for anything. An int8 dynamic-quantised copy (linear layers) is
timed as a cost-only row: whether it would change any decision needs the real weights and is not measured.
Nothing here changes a served artifact."""

from __future__ import annotations

import json
import os
import platform
import time
from pathlib import Path

import numpy as np
from loguru import logger

OUT = Path("outputs/efficiency")
PHOTO = Path("samples/instructor/00_clean/photo.jpg")
REPS, WARM = 50, 5


def _time(fn, reps: int = REPS) -> dict:
    for _ in range(WARM):
        fn()
    t = []
    for _ in range(reps):
        t0 = time.perf_counter()
        fn()
        t.append(time.perf_counter() - t0)
    return {"median_ms": float(np.median(t) * 1e3), "p95_ms": float(np.quantile(t, 0.95) * 1e3), "reps": reps}


def run_stage_timing(cfg) -> dict:
    """Write outputs/efficiency/stage_timing.json."""
    import joblib
    import timm
    import torch

    from glimpse.data.photos import eval_transform, open_photo
    from glimpse.serve.rules import knn_distance
    from glimpse.training.final import FINAL

    torch.set_num_threads(min(cfg.training.num_threads, os.cpu_count() or 1))
    tf = eval_transform(cfg.photo.arch, cfg.photo.crop_pct)
    img = open_photo(PHOTO, cfg.photo.min_side_px)
    x = tf(img)[None]
    model = timm.create_model(cfg.photo.arch, pretrained=False, num_classes=0).eval()
    ref = torch.load("checkpoints/photo_ood_reference.pt", weights_only=False)
    refe = torch.nn.functional.normalize(ref["emb"].float(), dim=1)
    heads = torch.nn.Linear(768, 5)
    fusion = joblib.load(FINAL)["models"]["photo"]
    xf = np.zeros((1, 19))
    with torch.no_grad():
        emb = model(x)
        q8 = torch.ao.quantization.quantize_dynamic(model, {torch.nn.Linear}, dtype=torch.qint8)
        stages = {
            "photo decode + transform": _time(lambda: tf(open_photo(PHOTO, cfg.photo.min_side_px))),
            "ConvNeXt-Tiny pass (fp32)": _time(lambda: model(x)),
            "ConvNeXt-Tiny pass (int8 dynamic, cost only)": _time(lambda: q8(x)),
            "kNN check (1,221 reference photos)": _time(
                lambda: knn_distance(refe, torch.nn.functional.normalize(emb, dim=1), int(ref["k"]))
            ),  # fmt: skip
            "grams + bin heads": _time(lambda: heads(emb)),
            "fusion + cut-offs (20 parameters)": _time(lambda: fusion.predict_proba(xf)),
        }
    photo_path = sum(v["median_ms"] for k, v in stages.items() if "int8" not in k)
    no_photo = stages["fusion + cut-offs (20 parameters)"]["median_ms"]
    res = {"what": __doc__.splitlines()[0], "machine": f"{platform.machine()}, {os.cpu_count()} CPUs, "
           f"{torch.get_num_threads()} threads (this run, not the author's Mac)",
           "stages": stages, "photo_path_median_ms": photo_path, "no_photo_path_median_ms": no_photo,
           "backbone_share_of_photo_path": stages["ConvNeXt-Tiny pass (fp32)"]["median_ms"] / photo_path}  # fmt: skip
    OUT.mkdir(parents=True, exist_ok=True)
    (OUT / "stage_timing.json").write_text(json.dumps(res, indent=2))
    logger.bind(stage="efficiency").info("photo path {:.1f} ms (backbone {:.0%}); no-photo path {:.2f} ms",
                                         photo_path, res["backbone_share_of_photo_path"], no_photo)  # fmt: skip
    return res


def run_end_to_end(cfg) -> dict:
    """Where one real answer's time goes, with the pinned weights (`glimpse stage-timing --end-to-end`, 26 Nov):
    the folder command's whole per-meal path (profile JSON, CSV parsing and validation, rules, models,
    answer) on samples/instructor/00_clean, with and without its photo, for v2 (served) and v1; and the CSV
    parse alone. Descriptive; this machine, not the author's Mac."""
    import json as _json
    import shutil
    import tempfile

    import torch

    from glimpse.serve import Pipeline
    from glimpse.serve.folder import _one_case
    from glimpse.serve.io import read_cgm_csv

    torch.set_num_threads(min(cfg.training.num_threads, os.cpu_count() or 1))
    case = Path("samples/instructor/00_clean")
    res = {"machine": f"{platform.system()} {platform.machine()}, {os.cpu_count()} CPUs, "
           f"{torch.get_num_threads()} threads (this run)", "reps": REPS,
           "parse_and_validate_csv": _time(lambda: read_cgm_csv(case / "cgm.csv"))}  # fmt: skip
    with tempfile.TemporaryDirectory() as td:
        nophoto = Path(td) / "00_no_photo"
        nophoto.mkdir()
        for f in ("cgm.csv", "profile.json"):
            shutil.copy(case / f, nophoto / f)
        for name, sf in (("v2_sensor_first", True), ("v1_photo_fusion", False)):
            app = Pipeline(cfg, sensor_first=sf)
            res[name] = {"whole_meal_with_photo": _time(lambda a=app: _one_case(a, case)),
                         "whole_meal_without_photo": _time(lambda a=app: _one_case(a, nophoto))}  # fmt: skip
    OUT.mkdir(parents=True, exist_ok=True)
    (OUT / "end_to_end.json").write_text(_json.dumps(res, indent=2))
    logger.bind(stage="efficiency").info("end to end: v2 {:.0f} ms (no photo {:.1f}), v1 {:.0f} ms",
                                         res["v2_sensor_first"]["whole_meal_with_photo"]["median_ms"],
                                         res["v2_sensor_first"]["whole_meal_without_photo"]["median_ms"],
                                         res["v1_photo_fusion"]["whole_meal_with_photo"]["median_ms"])  # fmt: skip
    return res
