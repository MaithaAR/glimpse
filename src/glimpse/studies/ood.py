"""S10: the unfamiliar-photo check against real phone photos (rule R7 in docs/rule_ledger.md). The naive
run showed a local lamb-and-rice dish scoring 0.35 -- below the deployed cut -- and receiving a confident
wrong carb bin. Pre-registered remediation: re-choose the cut on the training photos' own leave-one-out
distances at the largest quantile that keeps the in-distribution false-flag budget (<= 2%), then report,
photo by photo, which of the 14 real phone photos and the instructor cases the check now flags. If the
dish still passes inside the budget, that is recorded as an accepted limitation with the confirm-carbs
tier as the mitigation -- the budget is not loosened to chase one photo. Report: §4 trust, §5 failures."""

from __future__ import annotations

import json
from pathlib import Path

import pandas as pd
import torch
from loguru import logger

from glimpse.config import GlimpseConfig
from glimpse.data.photos import eval_transform, open_photo
from glimpse.serve.rules import knn_distance, leave_one_out_distances
from glimpse.training.final import _photo_checkpoint
from glimpse.training.photo import load_photo_bundle

OUT = Path("outputs/ood")
BUDGET = 0.02  # pre-registered in-distribution false-flag budget (rule R7)
QUANTILES = [0.90, 0.92, 0.94, 0.95, 0.96, 0.97, 0.98, 0.99]
INSTRUCTOR = ["00_clean", "08_dark_photo", "09_screenshot", "10_exif_rotated", "11_local_dish"]


def run_ood(cfg: GlimpseConfig, apply: bool = False) -> dict:
    """Score the real-photo sets under the deployed reference, sweep the cut, choose per the budget."""
    log = logger.bind(stage="ood")
    ref_d = torch.load(Path(cfg.prototype.ood_reference_file), weights_only=True)
    ref, cut0, k = ref_d["emb"].float(), float(ref_d["cut"]), int(ref_d["k"])
    model, _ = load_photo_bundle(cfg, _photo_checkpoint())
    model.eval()
    tf, norm = eval_transform(cfg.photo.arch, cfg.photo.crop_pct), torch.nn.functional.normalize

    @torch.no_grad()
    def score(path: Path) -> float:
        emb = norm(model.embed(tf(open_photo(path))[None]), dim=1)
        return float(knn_distance(ref, emb, k)[0])

    rows = [{"set": "local_photos", "photo": p.name, "score": score(p)}
            for p in sorted(Path("samples/local_photos").glob("*.jpg"))]  # fmt: skip
    for case in INSTRUCTOR:
        p = next(Path(f"samples/instructor/{case}").glob("photo.*"))
        rows.append({"set": "instructor", "photo": case, "score": score(p)})
    scores = pd.DataFrame(rows)

    loo = leave_one_out_distances(ref, k)
    sweep = []
    for q in QUANTILES:
        cut = float(torch.quantile(loo, q))
        sweep.append({"quantile": q, "cut": cut, "in_dist_flagged": float((loo > cut).float().mean()),
                      "local_flagged": int((scores[scores.set == "local_photos"].score > cut).sum()),
                      "flags_local_dish": bool(scores.set.eq("instructor").any()
                                               and float(scores[scores.photo == "11_local_dish"].score.iloc[0]) > cut
                                               )})  # fmt: skip
    ok = [r for r in sweep if r["in_dist_flagged"] <= BUDGET]
    chosen = min(ok, key=lambda r: r["cut"])  # the strictest cut the budget allows
    res = {"deployed_cut": cut0, "budget": BUDGET, "chosen": chosen, "sweep": sweep,
           "scores": rows, "applied": bool(apply)}  # fmt: skip
    OUT.mkdir(parents=True, exist_ok=True)
    scores.to_csv(OUT / "real_photo_scores.csv", index=False)
    (OUT / "ood_summary.json").write_text(json.dumps(res, indent=2))
    dish = float(scores[scores.photo == "11_local_dish"].score.iloc[0])
    log.info("deployed cut {:.3f}; strictest cut inside the {:.0%} budget: {:.3f} (q={}), in-dist flags {:.1%}",
             cut0, BUDGET, chosen["cut"], chosen["quantile"], chosen["in_dist_flagged"])  # fmt: skip
    log.info("local dish score {:.3f} -> {} under the chosen cut; {}/14 real phone photos flagged", dish,
             "FLAGGED (confirm-carbs)" if dish > chosen["cut"] else "still passes (accepted limitation)",
             chosen["local_flagged"])  # fmt: skip
    if apply:
        ref_d["cut"] = chosen["cut"]
        ref_d["check"] = dict(ref_d.get("check", {})) | {"r7_retune": chosen, "previous_cut": cut0}
        torch.save(ref_d, Path(cfg.prototype.ood_reference_file))
        log.info("reference cut updated {:.3f} -> {:.3f}; set prototype.ood_quantile to {} to match at the "
                 "next rebuild", cut0, chosen["cut"], chosen["quantile"])  # fmt: skip
    return res
