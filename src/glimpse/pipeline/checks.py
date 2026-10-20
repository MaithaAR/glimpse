"""Deterministic input checks. They run before any learned model and can stop the pipeline with a reason."""

from __future__ import annotations

from pathlib import Path

from loguru import logger
from PIL import Image, UnidentifiedImageError

from glimpse.config import PipelineCfg
from glimpse.data.preprocess import CGMWindow
from glimpse.schemas import AbstainReason, PreMealInput


def check_photo(photo_path: Path | None, cfg: PipelineCfg) -> list[AbstainReason]:
    """The meal photo must exist, decode as an image and be at least min_photo_side_px on each side."""
    if photo_path is None or not Path(photo_path).is_file():
        return [AbstainReason.MISSING_PHOTO]
    try:
        with Image.open(photo_path) as im:
            im.verify()
        with Image.open(photo_path) as im:
            if min(im.size) < cfg.min_photo_side_px:
                return [AbstainReason.UNREADABLE_PHOTO]
    except UnidentifiedImageError, OSError:
        return [AbstainReason.UNREADABLE_PHOTO]
    return []


def check_cgm(inp: PreMealInput, window: CGMWindow, cfg: PipelineCfg) -> list[AbstainReason]:
    """Apply the same window rules used to build training labels, plus the sensor's reportable range."""
    reasons: list[AbstainReason] = []
    lo, hi = cfg.glucose_range_mgdl
    if any(not (lo <= r.glucose_mgdl <= hi) for r in inp.cgm):
        reasons.append(AbstainReason.GLUCOSE_OUT_OF_RANGE)
    if window.readings_last_30min < cfg.min_cgm_readings_last_30min or window.coverage < cfg.min_cgm_coverage:
        reasons.append(AbstainReason.INSUFFICIENT_CGM)
    if window.max_gap_min > cfg.max_cgm_gap_min:
        reasons.append(AbstainReason.CGM_GAP)
    return reasons


def run_checks(inp: PreMealInput, window: CGMWindow, cfg: PipelineCfg) -> list[AbstainReason]:
    """Run every deterministic check and log the outcome."""
    reasons = check_photo(inp.photo_path, cfg) + check_cgm(inp, window, cfg)
    log = logger.bind(stage="checks")
    if reasons:
        log.warning("deterministic checks failed: {}", [r.value for r in reasons])
    else:
        log.info(
            "deterministic checks passed (coverage={:.2f}, max_gap={:.0f} min)", window.coverage, window.max_gap_min
        )
    return reasons
