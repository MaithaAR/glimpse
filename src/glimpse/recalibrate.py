"""`glimpse recalibrate` (added after the test was scored; uses NO test labels).

Rubric 5b asks for a cut-off chosen on validation data. The pre-registered validation cut-off (0.31) failed its
stability check because the 4 validation people rarely spike and the frozen model over-predicts for them. This step
does the validation-chosen rule properly:

1. fit a 2-parameter Platt recalibration on the 4 validation people's predictions from the frozen 28-person model;
2. choose the cut-offs on the recalibrated validation predictions with the pre-registered costs;
3. check the rule on cross-validated predictions that exclude both the test and the validation people (14 people),
   before and after recalibration: ECE, calibration by person, meals cleared, spikes among cleared, cost saving.

The deployed rule stays the cost-only one (switching the demo would re-score the test); this rule is frozen with a
hash as the candidate for the final submission, to be judged once on the locked people."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import joblib
import pandas as pd
from loguru import logger
from pydantic import BaseModel, ConfigDict, Field

from glimpse.config import GlimpseConfig
from glimpse.data.dataset import MealDataset
from glimpse.data.splits import Splits, meal_rows
from glimpse.evaluation import (
    apply_platt,
    calibration_by_person,
    choose_thresholds,
    decide,
    fit_platt,
    reliability,
    usefulness_report,
)
from glimpse.models.photo_model import PhotoCarbModel
from glimpse.models.weights import weights_path
from glimpse.training.final import FINAL, checkpoint_estimates, cost_dict, hash_thresholds, load_thresholds
from glimpse.training.fusion import PHOTO_COLS, photo_table, with_table
from glimpse.training.photo import build_cache, photo_rows

AMENDED = Path("checkpoints/thresholds_amended.json")
FROZEN = Path("checkpoints/thresholds_validation_recal.json")
OUT = Path("outputs/evaluation")


class Usefulness(BaseModel):
    """Decision shares and cost at fixed cut-offs."""

    model_config = ConfigDict(extra="forbid")
    share_low_risk: float = Field(ge=0, le=1)
    share_caution: float = Field(ge=0, le=1)
    share_abstain: float = Field(ge=0, le=1)
    missed_among_cleared: float
    cost_reduction: float
    n_cleared: int = Field(ge=0)


class Recalibration(BaseModel):
    """Everything `glimpse recalibrate` writes, validated before it is written."""

    model_config = ConfigDict(extra="forbid")
    fitted_on: str
    platt: dict[str, float]
    cutoffs: dict[str, float]
    validation: dict[str, Any]
    cv_check: dict[str, Any]
    stability: str
    sha256: str


def _frozen_validation_predictions(ds: MealDataset, sp: Splits, cfg: GlimpseConfig):
    """Main-model risk for the validation people from the frozen 28-person pipeline (same path as the test)."""
    bundle = joblib.load(FINAL)
    groups = {s.person_id: s.group for s in ds.samples}
    rows_all = photo_rows(ds, set(groups) - set(sp.locked), Path(cfg.data.cgmacros_root))
    model = PhotoCarbModel(cfg.photo.arch, weights=weights_path(cfg.photo), dropout=cfg.photo.dropout)
    maps = build_cache(model, ds, rows_all, cfg, "maps_eval", Path("data/cache/photo_cv_maps_eval.pt"))
    table = photo_table(checkpoint_estimates(cfg, ds, rows_all, maps, set(sp.validation)))
    fx = with_table(table, len(PHOTO_COLS))
    r = meal_rows(ds, set(sp.validation), scored_only=True)
    y = ds.labels()[r]
    p = bundle["models"]["photo"].predict_proba(fx(ds, r))[:, 1]
    return y, p


def _slim(u: dict) -> Usefulness:
    return Usefulness(**{k: u[k] for k in Usefulness.model_fields})


def run_recalibration(ds: MealDataset, sp: Splits, cfg: GlimpseConfig) -> dict:
    """Fit on validation, choose cut-offs on validation, check on CV without the test and validation people."""
    log = logger.bind(stage="recalibrate")
    costs, targets = cost_dict(cfg), cfg.targets.model_dump()
    main = load_thresholds(AMENDED)["main"]

    y_v, p_v = _frozen_validation_predictions(ds, sp, cfg)
    slope, intercept = fit_platt(p_v, y_v)
    q_v = apply_platt(p_v, slope, intercept)
    lo, hi, _ = choose_thresholds(q_v, y_v, costs)
    _, ece_v0 = reliability(p_v, y_v, cfg.targets.ece_bins)
    _, ece_v1 = reliability(q_v, y_v, cfg.targets.ece_bins)
    log.info(
        "validation ({} meals): Platt slope {:.2f}, intercept {:+.2f}; ECE {:.3f} -> {:.3f}; "
        "cut-offs low risk < {:.2f}, caution >= {:.2f}",
        len(y_v),
        slope,
        intercept,
        ece_v0,
        ece_v1,
        lo,
        hi,
    )

    oof = pd.read_csv(Path(cfg.paths.outputs) / "fusion" / "fusion_oof.csv")
    cv = oof[(oof.fold != 1) & ~oof.person_id.isin(sp.validation)]  # no test people, no validation people
    y_c, p_c, s_c = cv.spike.to_numpy(), cv.photo.to_numpy(), cv.person_id.to_numpy()
    q_c = apply_platt(p_c, slope, intercept)
    _, ece_c0 = reliability(p_c, y_c, cfg.targets.ece_bins)
    _, ece_c1 = reliability(q_c, y_c, cfg.targets.ece_bins)
    use_val_rule = usefulness_report(decide(q_c, lo, hi), y_c, costs, targets)
    use_cost_rule = usefulness_report(decide(p_c, main["low_risk_below"], main["caution_from"]), y_c, costs, targets)
    u = targets["usefulness"]  # the pre-registered usefulness targets decide, so "passed" means a usable rule
    stable = (
        use_val_rule["cost_reduction"] >= u["min_cost_reduction"]
        and use_val_rule["missed_among_cleared"] <= u["max_missed_among_cleared"]
        and use_val_rule["n_cleared"] > 0
    )
    log.info("CV check ({} people, {} meals): ECE {:.3f} -> {:.3f}; validation rule clears {:.0%} ({:.0%} spike), "
             "cost saving {:+.0%}; cost-only rule clears {:.0%} ({:.0%} spike), cost saving {:+.0%}; stability {}",
             cv.person_id.nunique(), len(cv), ece_c0, ece_c1, use_val_rule["share_low_risk"],
             use_val_rule["missed_among_cleared"], use_val_rule["cost_reduction"], use_cost_rule["share_low_risk"],
             use_cost_rule["missed_among_cleared"], use_cost_rule["cost_reduction"],
             "passed" if stable else "FAILED")  # fmt: skip

    d = {"fitted_on": f"{len(sp.validation)} validation people ({len(y_v)} meals), frozen 28-person model",
         "platt": {"slope": slope, "intercept": intercept},
         "cutoffs": {"low_risk_below": lo, "caution_from": hi},
         "validation": {"meals": int(len(y_v)), "spike_rate": float(y_v.mean()), "ece_before": ece_v0,
                        "ece_after": ece_v1,
                        "validation_rule": _slim(usefulness_report(decide(q_v, lo, hi), y_v, costs,
                                                                   targets)).model_dump(),
                        "cost_only_rule": _slim(usefulness_report(decide(p_v, main["low_risk_below"],
                                                                         main["caution_from"]), y_v, costs,
                                                                  targets)).model_dump()},
         "cv_check": {"people": int(cv.person_id.nunique()), "meals": int(len(cv)), "ece_before": ece_c0,
                      "ece_after": ece_c1, "by_person_before": calibration_by_person(s_c, y_c, p_c),
                      "by_person_after": calibration_by_person(s_c, y_c, q_c),
                      "validation_rule": _slim(use_val_rule).model_dump(),
                      "cost_only_rule": _slim(use_cost_rule).model_dump()},
         "stability": ("passed (meets the pre-registered cost-saving and missed-spike targets on the CV check)"
                       if stable else
                       "failed (misses the pre-registered cost-saving or missed-spike target on the CV check)"),
         }  # fmt: skip
    d["sha256"] = hash_thresholds(d)
    res = Recalibration(**d)
    FROZEN.write_text(
        json.dumps(
            {
                "rule": "validation-chosen on Platt-recalibrated risk (candidate for the final)",
                **res.platt,
                **res.cutoffs,
                "sha256": res.sha256,
            },
            indent=2,
        )
    )
    OUT.mkdir(parents=True, exist_ok=True)
    (OUT / "recalibration.json").write_text(res.model_dump_json(indent=2))
    (OUT / "recalibration.md").write_text(_markdown(res))
    log.info("written {} and {}", FROZEN, OUT / "recalibration.md")
    return res.model_dump()


def _markdown(r: Recalibration) -> str:
    v, c = r.validation, r.cv_check
    bp, ba = c["by_person_before"], c["by_person_after"]

    def row(name: str, u: dict) -> str:
        return (
            f"| {name} | {u['share_low_risk']:.0%} | {u['missed_among_cleared']:.0%} | {u['share_abstain']:.0%} | "
            f"{u['cost_reduction']:+.0%} |"
        )

    return "\n".join(
        [
            "# Validation-chosen cut-offs on recalibrated risk (no test labels used; added after the results)",
            "",
            f"Fitted on: {r.fitted_on}. Platt slope {r.platt['slope']:.2f}, intercept {r.platt['intercept']:+.2f}.",
            "",
            "Cut-offs chosen on the recalibrated validation predictions: "
            f"low risk below {r.cutoffs['low_risk_below']:.2f}, "
            f"caution from {r.cutoffs['caution_from']:.2f}. Frozen in `checkpoints/thresholds_validation_recal.json` "
            f"(sha256 {r.sha256[:12]}). Stability check: {r.stability}.",
            "",
            "| Split | Meals | ECE before | ECE after |",
            "|---|---|---|---|",
            f"| Validation (fitted here) | {v['meals']} | {v['ece_before']:.3f} | {v['ece_after']:.3f} |",
            f"| CV without test and validation people | {c['meals']} ({c['people']} people) | {c['ece_before']:.3f} | "
            f"{c['ece_after']:.3f} |",
            "",
            "| CV check, by person | Predicted before | Observed | Predicted after |",
            "|---|---|---|---|",
            f"| Rare spikers | {bp['rare_spikers']['predicted']:.2f} | {bp['rare_spikers']['observed']:.2f} | "
            f"{ba['rare_spikers']['predicted']:.2f} |",
            f"| Frequent spikers | {bp['frequent_spikers']['predicted']:.2f} | "
            f"{bp['frequent_spikers']['observed']:.2f} | "
            f"{ba['frequent_spikers']['predicted']:.2f} |",
            "",
            "| Rule | Cleared | Spikes among cleared | Can't judge | Cost saving |",
            "|---|---|---|---|---|",
            row("Validation people: validation-chosen, recalibrated", v["validation_rule"]),
            row("Validation people: cost-only (deployed)", v["cost_only_rule"]),
            row("CV check: validation-chosen, recalibrated", c["validation_rule"]),
            row("CV check: cost-only (deployed)", c["cost_only_rule"]),
            "",
        ]
    )
