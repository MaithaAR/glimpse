"""Resumable downloads of CGMacros (open access, PhysioNet) and pretrained weights. Nothing here runs at inference."""

from __future__ import annotations

import threading
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

from loguru import logger

CGMACROS_ZIP = "https://physionet-open.s3.amazonaws.com/cgmacros/1.0.0/CGMacros_dateshifted365.zip"
PHOTO_LIST = Path("splits/photo_list.txt")
PRETRAINED = ("convnext_tiny.fb_in22k_ft_in1k", "efficientnet_b0.ra_in1k")  # fine-tuned model, frozen baseline

_local = threading.local()


def _zip():
    """One RemoteZip handle per thread (HTTP range requests; only the needed members are transferred)."""
    from remotezip import RemoteZip

    if not hasattr(_local, "z"):
        _local.z = RemoteZip(CGMACROS_ZIP)
    return _local.z


def _fetch(member: str, dest_root: Path) -> str:
    """Extract one zip member to dest_root, skipping files already present."""
    dst = dest_root / member
    if dst.is_file() and dst.stat().st_size > 0:
        return "skip"
    dst.parent.mkdir(parents=True, exist_ok=True)
    tmp = dst.with_suffix(dst.suffix + ".part")
    tmp.write_bytes(_zip().read(member))
    tmp.replace(dst)
    return "ok"


def download_cgmacros(dest_root: Path, with_photos: bool = True, workers: int = 8) -> None:
    """Download the participant CSVs, bio.csv and (optionally) the before-meal photos used by the project."""
    log = logger.bind(stage="download")
    names = _zip().namelist()
    csvs = [n for n in names if n.lower().endswith(".csv") and "/photos/" not in n]
    wanted = list(csvs)
    if with_photos:
        wanted += [ln.strip() for ln in PHOTO_LIST.read_text().splitlines() if ln.strip()]
    missing = sorted(set(wanted) - set(names))
    if missing:
        log.warning("{} listed files are not in the archive (first: {})", len(missing), missing[0])
    todo = [w for w in wanted if w in set(names)]
    log.info("fetching {} files ({} CSV, {} photos) into {}", len(todo), len(csvs), len(todo) - len(csvs), dest_root)
    counts = {"ok": 0, "skip": 0}
    with ThreadPoolExecutor(workers) as ex:
        for i, r in enumerate(ex.map(lambda m: _fetch(m, dest_root), todo), 1):
            counts[r] += 1
            if i % 200 == 0 or i == len(todo):
                log.info("progress {}/{} {}", i, len(todo), counts)


def download_weights(cache_dir: Path) -> None:
    """Download pretrained timm/Hugging Face weights once so that training, evaluation and the demo run offline."""
    import timm

    log = logger.bind(stage="download")
    cache_dir.mkdir(parents=True, exist_ok=True)
    for name in PRETRAINED:
        timm.create_model(name, pretrained=True, num_classes=0, cache_dir=str(cache_dir))
        log.info("weights ready: {} (cache {})", name, cache_dir)
