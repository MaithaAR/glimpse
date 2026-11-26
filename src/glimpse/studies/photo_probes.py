"""Photo-model probes D (where it looks) and E (portion size), and the int8 study (`glimpse photo-probes`;
registered 26 Nov in docs/photo_probes_int8.md before the run). Frozen photo model; development photos are
in-sample for it, the 14 real phone photos are not."""

from __future__ import annotations

import json
import time
from pathlib import Path

import numpy as np
import torch
from loguru import logger
from PIL import Image

from glimpse.data.photos import eval_transform, open_photo

LOCAL = Path("samples/local_photos")


def _occlude(img: Image.Image, where: str) -> Image.Image:
    """Grey out the central square holding 50% of the area, or an equal-area frame at the border."""
    a = np.asarray(img.convert("RGB")).copy()
    h, w = a.shape[:2]
    if where == "centre":
        s = np.sqrt(0.5)
        y0, x0 = int(h * (1 - s) / 2), int(w * (1 - s) / 2)
        a[y0 : h - y0, x0 : w - x0] = 128
    else:  # border frame with the inner sqrt(0.5) square kept: frame area = 50%
        s = np.sqrt(0.5)
        y0, x0 = int(h * (1 - s) / 2), int(w * (1 - s) / 2)
        keep = a[y0 : h - y0, x0 : w - x0].copy()
        a[:] = 128
        a[y0 : h - y0, x0 : w - x0] = keep
    return Image.fromarray(a)


def _zoom(img: Image.Image, keep: float = 0.6) -> Image.Image:
    w, h = img.size
    cw, ch = int(w * keep), int(h * keep)
    box = ((w - cw) // 2, (h - ch) // 2, (w - cw) // 2 + cw, (h - ch) // 2 + ch)
    return img.crop(box).resize((w, h), Image.BICUBIC)


def run_photo_probes(cfg) -> dict:
    import joblib

    from glimpse.data.dataset import MealDataset
    from glimpse.data.splits import SplitsV2, meal_rows
    from glimpse.demo_cases import meal_input
    from glimpse.serve.rules import decision
    from glimpse.training.final import FINAL, _photo_checkpoint, load_thresholds
    from glimpse.training.photo import load_photo_bundle

    log = logger.bind(stage="photo-probes")
    torch.set_num_threads(2)
    model, _ = load_photo_bundle(cfg, _photo_checkpoint())
    model.eval()
    q8 = torch.ao.quantization.quantize_dynamic(model, {torch.nn.Linear}, dtype=torch.qint8)
    tf = eval_transform(cfg.photo.arch, cfg.photo.crop_pct)

    @torch.no_grad()
    def predict(m, img) -> tuple[float, int]:
        reg, logits = m(tf(img)[None])
        probs.append(torch.softmax(logits, 1)[0].tolist())
        return float(torch.expm1(reg).clamp(min=0)[0]), int(logits.argmax(1)[0])

    probs: list = []

    ds = MealDataset.load(Path("data/dataset"))
    sp = SplitsV2.load(Path("splits/dev_splits_v2.json"))
    dev = {p for f in sp.dev_folds for p in f}
    items = []
    for row in meal_rows(ds, dev, scored_only=True):
        m = meal_input(cfg, ds, int(row))
        if m.photo is not None:
            items.append(("dev", int(row), Path(m.photo), float(ds.samples[int(row)].carbs_served_g)))
    items += [("local", -1, p, float("nan")) for p in sorted(LOCAL.glob("*.jpg"))]
    rec, t32, t8 = [], [], []
    for src, row, path, carbs in items:
        img = open_photo(path, cfg.photo.min_side_px)
        t0 = time.perf_counter()
        g32, b32 = predict(model, img)
        t32.append(time.perf_counter() - t0)
        p32 = probs[-1]
        t0 = time.perf_counter()
        g8, b8 = predict(q8, img)
        t8.append(time.perf_counter() - t0)
        p8 = probs[-1]
        gc, bc = predict(model, _occlude(img, "centre"))
        gb, bb = predict(model, _occlude(img, "border"))
        gz, bz = predict(model, _zoom(img))
        rec.append({"src": src, "row": row, "carbs": carbs, "g32": g32, "b32": b32, "g8": g8, "b8": b8,
                    "p32": p32, "p8": p8,
                    "d_centre": abs(np.log1p(gc) - np.log1p(g32)), "d_border": abs(np.log1p(gb) - np.log1p(g32)),
                    "bin_centre": bc != b32, "bin_border": bb != b32, "zoom_up": gz > g32,
                    "zoom_dlog": float(np.log1p(gz) - np.log1p(g32))})  # fmt: skip
    log.info("{} photos probed", len(rec))

    def probe(sub: list[dict]) -> dict:
        dc, db = np.array([r["d_centre"] for r in sub]), np.array([r["d_border"] for r in sub])
        return {"photos": len(sub), "share_centre_moves_more": float(np.mean(dc > db)),
                "median_abs_dlog_centre": float(np.median(dc)), "median_abs_dlog_border": float(np.median(db)),
                "bin_changes_centre": float(np.mean([r["bin_centre"] for r in sub])),
                "bin_changes_border": float(np.mean([r["bin_border"] for r in sub])),
                "hypothesis_D_holds": bool(np.mean(dc > db) >= 0.6),
                "zoom_share_grams_rise": float(np.mean([r["zoom_up"] for r in sub])),
                "zoom_median_dlog": float(np.median([r["zoom_dlog"] for r in sub]))}  # fmt: skip

    dev_r = [r for r in rec if r["src"] == "dev"]
    probes = {"dev_in_sample": probe(dev_r), "local_unseen": probe([r for r in rec if r["src"] == "local"])}
    # int8 quality side
    g32, g8 = np.array([r["g32"] for r in dev_r]), np.array([r["g8"] for r in dev_r])
    carbs = np.array([r["carbs"] for r in dev_r])
    bundle = joblib.load(FINAL)
    fusion, (a, b) = bundle["models"]["photo"], bundle["calibration"]
    th = load_thresholds()["main"]
    from glimpse.training.baselines import trend_features

    rows = np.array([r["row"] for r in dev_r])
    x = trend_features(ds, rows)

    def v1_decisions(grams: np.ndarray, bin_probs: np.ndarray) -> np.ndarray:
        p = fusion.predict_proba(np.hstack([x, np.log1p(grams)[:, None], bin_probs[:, 1:]]))[:, 1]
        z = np.log(np.clip(p, 1e-6, 1 - 1e-6) / np.clip(1 - p, 1e-6, 1))
        r = 1 / (1 + np.exp(-(a * z + b)))
        return np.array([decision(float(v), th["low_risk_below"], th["caution_from"]) for v in r])

    d32 = v1_decisions(g32, np.array([r["p32"] for r in dev_r]))
    d8 = v1_decisions(g8, np.array([r["p8"] for r in dev_r]))
    int8 = {"photos": len(dev_r), "carb_range_agreement": float(np.mean([r["b32"] == r["b8"] for r in dev_r])),
            "median_abs_grams_difference": float(np.median(np.abs(g32 - g8))),
            "carb_mae_fp32_in_sample": float(np.mean(np.abs(g32 - carbs))),
            "carb_mae_int8_in_sample": float(np.mean(np.abs(g8 - carbs))),
            "v1_decision_agreement": float(np.mean(d32 == d8)),
            "v1_note": "frozen v1 fusion, Platt only (no history), frozen cut-offs, as the folder command serves v1",
            "median_ms_fp32": float(np.median(t32) * 1e3), "median_ms_int8": float(np.median(t8) * 1e3),
            "machine": "cloud x86, 2 threads, whole photo model (transform excluded)"}  # fmt: skip
    out = {"what": __doc__.splitlines()[0], "probes": probes, "int8": int8}
    Path("outputs/probes").mkdir(parents=True, exist_ok=True)
    Path("outputs/probes/photo_probes.json").write_text(json.dumps({"what": out["what"], **probes}, indent=2))
    Path("outputs/efficiency").mkdir(parents=True, exist_ok=True)
    Path("outputs/efficiency/int8.json").write_text(json.dumps(int8, indent=2))
    log.info("probe D dev: centre moves more on {:.0%}; local {:.0%}; int8 bin agreement {:.1%}, v1 decisions {:.1%}",
             probes["dev_in_sample"]["share_centre_moves_more"], probes["local_unseen"]["share_centre_moves_more"],
             int8["carb_range_agreement"], int8["v1_decision_agreement"])  # fmt: skip
    return out
