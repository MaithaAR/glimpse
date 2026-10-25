"""`glimpse demo` (Step 6a): builds the unfamiliar-photo reference once, then shows worked cases from the test
people, two "can't judge" cases, and the response time over all test meals. Results -> outputs/demo/."""

from __future__ import annotations

import json
import platform
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from loguru import logger
from PIL import Image

from glimpse.config import GlimpseConfig
from glimpse.data.dataset import PROFILE, MealDataset, _person_series
from glimpse.data.photos import eval_transform
from glimpse.data.preprocess import resample_window
from glimpse.data.splits import Splits, meal_rows
from glimpse.demo import PERSON_INPUTS, Glimpse, MealInput, knn_distance, leave_one_out_distances
from glimpse.models.photo_model import PhotoCarbModel
from glimpse.models.weights import weights_path
from glimpse.training.final import _photo_checkpoint
from glimpse.training.photo import build_cache, embed_maps, load_photo_bundle, photo_rows

OUT = Path("outputs/demo")


def synthetic_images() -> dict[str, Image.Image]:
    """Clearly non-meal pictures for the unfamiliar-photo check: blank, black, white, noise, gradient, checkerboard."""
    rng = np.random.default_rng(0)
    grad = np.tile(np.linspace(0, 255, 256, dtype=np.uint8), (256, 1))
    board = ((np.indices((256, 256)) // 32).sum(0) % 2 * 255).astype(np.uint8)
    arrays = {"grey": np.full((256, 256, 3), 128, np.uint8), "black": np.zeros((256, 256, 3), np.uint8),
              "white": np.full((256, 256, 3), 255, np.uint8),
              "noise": rng.integers(0, 256, (256, 256, 3), dtype=np.uint8),
              "gradient": np.stack([grad] * 3, -1), "checkerboard": np.stack([board] * 3, -1)}  # fmt: skip
    return {k: Image.fromarray(v) for k, v in arrays.items()}


def build_ood_reference(ds: MealDataset, sp: Splits, cfg: GlimpseConfig) -> dict:
    """Training photos' features (the deployed photo model) + the cut-off; checked on held-out photos and
    synthetic non-meal pictures."""
    log = logger.bind(stage="demo")
    groups = {s.person_id: s.group for s in ds.samples}
    rows_all = photo_rows(ds, set(groups) - set(sp.locked), Path(cfg.data.cgmacros_root))
    person = np.array([ds.samples[i].person_id for i in rows_all])
    fresh = PhotoCarbModel(cfg.photo.arch, weights=weights_path(cfg.photo), dropout=cfg.photo.dropout)
    maps = build_cache(fresh, ds, rows_all, cfg, "maps_eval", Path("data/cache/photo_cv_maps_eval.pt"))
    model, _ = load_photo_bundle(cfg, _photo_checkpoint())
    norm, k, quantile = torch.nn.functional.normalize, cfg.prototype.ood_neighbours, cfg.prototype.ood_quantile
    ref = norm(embed_maps(model, maps[np.isin(person, sp.train)]), dim=1)
    cut = float(torch.quantile(leave_one_out_distances(ref, k), quantile))
    held = norm(embed_maps(model, maps[np.isin(person, sp.validation + sp.test)]), dim=1)
    held_flag = float((knn_distance(ref, held, k) > cut).float().mean())
    tf = eval_transform(cfg.photo.arch, cfg.photo.crop_pct)
    with torch.no_grad():
        syn = {n: float(knn_distance(ref, norm(model.embed(tf(im)[None]), dim=1), k)[0])
               for n, im in synthetic_images().items()}  # fmt: skip
    check = {"training_photos": len(ref), "k": k, "quantile": quantile, "cut": cut,
             "held_out_meal_photos_flagged": held_flag, "held_out_photos": len(held),
             "synthetic_scores": syn, "synthetic_flagged": sum(v > cut for v in syn.values())}  # fmt: skip
    torch.save({"emb": ref.half(), "cut": cut, "k": k, "check": check}, Path(cfg.prototype.ood_reference_file))
    log.info("unfamiliar-photo check: cut-off {:.3f}; held-out meal photos flagged {:.1%} ({}); synthetic "
             "pictures flagged {}/{}", cut, held_flag, len(held), check["synthetic_flagged"], len(syn))  # fmt: skip
    return check


def meal_input(cfg: GlimpseConfig, ds: MealDataset, row: int) -> MealInput:
    """A CGMacros meal as the app would receive it: RAW sensor readings (not the prepared dataset) + photo."""
    s, root = ds.samples[row], Path(cfg.data.cgmacros_root)
    ser = _person_series(root, s.person_id)
    ser = ser[ser.index <= pd.Timestamp(s.meal_time)]  # nothing after the meal is passed in
    g, h = ser.glucose.dropna(), ser.hr.dropna()
    prof = dict(zip(PROFILE, ds.profile[row].tolist(), strict=True))
    return MealInput(meal_time=s.meal_time, glucose_times=list(g.index.to_pydatetime()), glucose_mgdl=g.tolist(),
                     hr_times=list(h.index.to_pydatetime()), hr_bpm=h.tolist(),
                     profile={k: prof[k] for k in PERSON_INPUTS},
                     photo=(root / s.photo_relpath) if s.photo_relpath else None)  # fmt: skip


def card(title: str, m: MealInput, a, truth: str | None) -> str:
    """One case as Markdown."""
    photo = m.photo.name if m.photo else "none"
    lines = [f"### {title}", f"- meal time: {m.meal_time:%Y-%m-%d %H:%M}; photo: {photo}",
             f"- **{a.decision.upper()}**: {a.sentence}", f"- reason: {a.reason}"]  # fmt: skip
    if a.carb_bin is not None:
        lines.append(f"- carbs: most likely {a.carb_bin} ({a.carb_bin_prob:.0%} sure); unfamiliar-photo score "
                     f"{a.unfamiliar_photo_score:.3f}")  # fmt: skip
    if truth:
        lines.append(f"- what happened: {truth}")
    lines.append(f"- response time: {a.latency_ms:.0f} ms")
    return "\n".join(lines)


def run_demo(ds: MealDataset, sp: Splits, cfg: GlimpseConfig) -> dict:
    """All 96 test meals through the app, the worked cases, two can't-judge cases and the response time."""
    log = logger.bind(stage="demo")
    OUT.mkdir(parents=True, exist_ok=True)
    ref_file = Path(cfg.prototype.ood_reference_file)
    check = build_ood_reference(ds, sp, cfg) if not ref_file.is_file() else torch.load(ref_file)["check"]
    app = Glimpse(cfg)
    rows = [int(r) for r in meal_rows(ds, set(sp.test), scored_only=True)]
    app.predict(meal_input(cfg, ds, rows[0]))  # warm-up (first call loads libraries)
    results = []
    for r in rows:
        m, s = meal_input(cfg, ds, r), ds.samples[r]
        a = app.predict(m)
        w = resample_window(m.glucose_times, m.glucose_mgdl, m.meal_time, cfg.pipeline.cgm_lookback_min)
        gap = float(np.max(np.abs(np.nan_to_num(ds.glucose[r]) - np.nan_to_num(w.values))))
        results.append({"row": r, "person": s.person_id, "spike": s.spike, "rise_mgdl": s.rise_mgdl,
                        "window_matches_dataset": gap < 1e-6, **a.model_dump()})  # fmt: skip
    df = pd.DataFrame(results)
    df.to_csv(OUT / "test_meals.csv", index=False)
    lat = df.latency_ms.to_numpy()
    ok = df[df.decision != "can't judge"]

    def first(q: pd.DataFrame) -> int | None:
        """Row of the first meal in row order (fixed rule, not hand-picked)."""
        return int(q.row.iloc[0]) if len(q) else None

    # cases chosen by a fixed rule (first in row order), not by looking for nice examples
    warn, clear = ok.decision == "caution", ok.decision == "low risk"
    picks = {"Correct warning (caution, and it spiked)": first(ok[warn & (ok.spike == 1)]),
             "Correct all-clear (low risk, no spike)": first(ok[clear & (ok.spike == 0)]),
             "Wrong answer (shown on purpose)": first(ok[(clear & (ok.spike == 1))
                                                         | (warn & (ok.spike == 0))])}  # fmt: skip
    md = ["# GLIMPSE demo: worked cases (test people, frozen 28-person pipeline)", ""]
    for title, r in picks.items():
        if r is None:
            md.append(f"### {title}\n- none among the test meals")
            continue
        m, s = meal_input(cfg, ds, r), ds.samples[r]
        truth = f"glucose rose {s.rise_mgdl:.0f} mg/dL ({'spike' if s.spike else 'no spike'})"
        md.append(card(title, m, app.predict(m), truth))
    # mocked feedback loop: the CGM labels each worked case 2 h later (same rule as the training label)
    from glimpse.feedback import auto_rise, record_feedback

    auto = {}
    for title, r in picks.items():
        if r is None:
            continue
        s = ds.samples[r]
        ser = _person_series(Path(cfg.data.cgmacros_root), s.person_id).glucose.dropna()
        rise = auto_rise(list(ser.index.to_pydatetime()), ser.tolist(), s.meal_time, cfg)
        shown = df.loc[df.row == r].iloc[0]
        said = {"decision": shown.decision, "risk": float(shown.risk), "meal_time": s.meal_time.isoformat()}
        rec = record_feedback(cfg, said, "ate it all", rise, source="cgm_auto")
        auto[title] = {"auto_rise_mgdl": rise, "dataset_rise_mgdl": s.rise_mgdl, "label": rec.feedback_label}
    base_row = picks["Correct warning (caution, and it spiked)"] or rows[0]
    m = meal_input(cfg, ds, base_row)
    cut = m.meal_time - pd.Timedelta(minutes=40)
    keep = [i for i, t in enumerate(m.glucose_times) if t <= cut]
    gap_m = m.model_copy(update={"glucose_times": [m.glucose_times[i] for i in keep],
                                 "glucose_mgdl": [m.glucose_mgdl[i] for i in keep]})  # fmt: skip
    title = "Can't judge: sensor gap (same meal, last 40 min of glucose removed)"
    md.append(card(title, gap_m, app.predict(gap_m), None))
    blank = OUT / "blank_photo.jpg"
    synthetic_images()["white"].save(blank)
    blank_m = m.model_copy(update={"photo": blank})
    md.append(card("Can't judge: not a meal photo (plain white picture)", blank_m, app.predict(blank_m), None))
    border = df[df.reason.str.startswith("the risk")]  # first borderline meal in row order (fixed rule)
    if len(border):
        r = int(border.row.iloc[0])
        m_b, s_b = meal_input(cfg, ds, r), ds.samples[r]
        truth = f"glucose rose {s_b.rise_mgdl:.0f} mg/dL ({'spike' if s_b.spike else 'no spike'})"
        md.append(card("Can't judge: borderline risk (between the two cut-offs)", m_b, app.predict(m_b), truth))
    p50, p95 = float(np.percentile(lat, 50)), float(np.percentile(lat, 95))
    summary = {"test_meals": len(df), "decisions": df.decision.value_counts().to_dict(),
               "latency_ms_p50": p50, "latency_ms_p95": p95, "latency_target_ms": cfg.targets.latency_p95_ms,
               "latency_met": p95 <= cfg.targets.latency_p95_ms,
               "machine": f"{platform.system()} {platform.machine()}", "torch_threads": torch.get_num_threads(),
               "raw_window_matches_dataset": bool(df.window_matches_dataset.all()),
               "unfamiliar_photo_check": check, "cases": picks, "cgm_auto_feedback": auto}  # fmt: skip
    md += ["", f"Response time over {len(df)} test meals on {summary['machine']}: median {p50:.0f} ms, "
           f"95th percentile {p95:.0f} ms (target {cfg.targets.latency_p95_ms:.0f} ms)."]  # fmt: skip
    (OUT / "cases.md").write_text("\n\n".join(md) + "\n")
    (OUT / "demo_summary.json").write_text(json.dumps(summary, indent=2, default=str))
    log.info("worked cases (also in {}):\n{}", OUT / "cases.md", "\n\n".join(md))
    log.info("answers on {} test meals: {}; response time median {:.0f} ms, p95 {:.0f} ms (target {:.0f}); raw "
             "windows match the dataset: {}", len(df), summary["decisions"], p50, p95, cfg.targets.latency_p95_ms,
             summary["raw_window_matches_dataset"])  # fmt: skip
    return summary


def write_example(cfg: GlimpseConfig, ds: MealDataset, row: int, folder: Path) -> dict:
    """One meal as the files a user would supply: the 2 h of readings before it as a CSV, plus photo and profile."""
    m = meal_input(cfg, ds, row)
    start = m.meal_time - pd.Timedelta(minutes=cfg.pipeline.cgm_lookback_min)
    g = pd.DataFrame({"timestamp": m.glucose_times, "glucose_mgdl": m.glucose_mgdl})
    h = pd.DataFrame({"timestamp": m.hr_times, "hr_bpm": m.hr_bpm})
    d = g.merge(h, on="timestamp", how="outer").sort_values("timestamp")
    d = d[(d.timestamp > start) & (d.timestamp <= m.meal_time)]
    folder.mkdir(parents=True, exist_ok=True)
    csv = folder / f"meal_{row}.csv"
    d.to_csv(csv, index=False)
    return {"row": row, "csv": csv, "photo": m.photo, "meal_time": m.meal_time, "profile": m.profile}


def selftest(cfg: GlimpseConfig) -> dict:
    """Container check without any dataset: loads every frozen part (offline weights with their hash, fusion,
    frozen cut-offs, unfamiliar-photo reference) and runs the full path on a synthetic meal."""
    import tempfile
    from datetime import datetime, timedelta

    app = Glimpse(cfg)
    folder = Path(tempfile.mkdtemp())
    photo, grey = folder / "noise.jpg", folder / "grey.jpg"
    synthetic_images()["noise"].save(photo)  # textured but not a meal: caught by the unfamiliar-photo check
    synthetic_images()["grey"].save(grey)  # flat picture: caught by the blank-photo rule before any model
    meal = datetime(2024, 1, 1, 12, 0)
    times = [meal - timedelta(minutes=m) for m in range(115, -1, -5)]
    m = MealInput(meal_time=meal, glucose_times=times, glucose_mgdl=[95 + 0.2 * k for k in range(len(times))],
                  profile={"age": 50, "bmi": 29, "a1c_pct": 5.9, "fasting_glucose_mgdl": 104,
                           "prior_spike_rate": float("nan"), "prior_rise": float("nan")}, photo=photo)  # fmt: skip
    unfamiliar = app.predict(m)
    blank = app.predict(m.model_copy(update={"photo": grey}))
    no_photo = app.predict(m.model_copy(update={"photo": None}))
    app.ood_cut = float("inf")  # let the noise picture through, only to exercise the scoring path end to end
    scored = app.predict(m)
    checks = {"unfamiliar photo -> can't judge": unfamiliar.decision == "can't judge" and "unlike" in unfamiliar.reason,
              "blank photo -> can't judge": blank.decision == "can't judge" and "blank" in blank.reason,
              "no photo -> can't judge": no_photo.decision == "can't judge",
              "full path gives a risk in [0, 1]": scored.risk is not None and 0 <= scored.risk <= 1}  # fmt: skip
    log = logger.bind(stage="selftest")
    for k, ok in checks.items():
        log.info("{:<36} {}", k, "OK" if ok else "FAILED")
    log.info("full path: {} (risk {:.2f}) in {:.0f} ms", scored.decision, scored.risk, scored.latency_ms)
    if not all(checks.values()):
        raise RuntimeError("self-test failed")
    return checks
