"""Download CGMacros (open access on PhysioNet) without manual steps. Only the needed files are transferred:
the participant CSVs, bio.csv and the before-meal photos listed in splits/photo_list.txt. Re-running skips files
that are already present, so an interrupted download simply resumes."""

from __future__ import annotations

import threading
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

from loguru import logger

CGMACROS_ZIP = "https://physionet-open.s3.amazonaws.com/cgmacros/1.0.0/CGMacros_dateshifted365.zip"
PHOTO_LIST = Path("splits/photo_list.txt")

_local = threading.local()


def _zip():
    """One remote-zip handle per thread: it reads single files from the online zip (no full 4 GB download)."""
    from remotezip import RemoteZip

    if not hasattr(_local, "z"):
        _local.z = RemoteZip(CGMACROS_ZIP)
    return _local.z


def _fetch(member: str, dest_root: Path) -> str:
    """Copy one file out of the online zip into dest_root, unless it is already there."""
    dst = dest_root / member
    if dst.is_file() and dst.stat().st_size > 0:
        return "skip"
    dst.parent.mkdir(parents=True, exist_ok=True)
    tmp = dst.with_suffix(dst.suffix + ".part")  # write to a temporary name first, so a crash never leaves half a file
    tmp.write_bytes(_zip().read(member))
    tmp.replace(dst)
    return "ok"


def download_cgmacros(dest_root: Path, with_photos: bool = True, workers: int = 8) -> dict[str, int]:
    """Fetch the CSVs and (optionally) the listed photos into dest_root; returns how many were new or skipped."""
    log = logger.bind(stage="download")
    names = set(_zip().namelist())
    csvs = sorted(n for n in names if n.lower().endswith(".csv") and "/photos/" not in n)
    wanted = list(csvs)
    if with_photos:
        wanted += [ln.strip() for ln in PHOTO_LIST.read_text().splitlines() if ln.strip()]
    missing = [w for w in wanted if w not in names]
    if missing:
        log.warning("{} listed files are not in the archive (first: {})", len(missing), missing[0])
    todo = [w for w in wanted if w in names]
    log.info("{} files wanted ({} CSV, {} photos) in {}", len(todo), len(csvs), len(todo) - len(csvs), dest_root)
    counts = {"ok": 0, "skip": 0}
    with ThreadPoolExecutor(workers) as ex:
        for i, r in enumerate(ex.map(lambda m: _fetch(m, dest_root), todo), 1):
            counts[r] += 1
            if i % 200 == 0 or i == len(todo):
                log.info("progress {}/{}: {} new, {} already present", i, len(todo), counts["ok"], counts["skip"])
    return counts
