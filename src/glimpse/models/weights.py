"""Pretrained weights: downloaded once (pinned Hugging Face commit), checked against a SHA-256 fingerprint, and from
then on loaded from the local file only - no network is needed to train, evaluate or run the demo."""

from __future__ import annotations

import hashlib
from pathlib import Path

from loguru import logger

from glimpse.config import PhotoCfg

FILENAME = "model.safetensors"


def sha256(path: Path) -> str:
    """Fingerprint of a file (read in 1 MB pieces)."""
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for block in iter(lambda: f.read(1 << 20), b""):
            h.update(block)
    return h.hexdigest()


def weights_path(pc: PhotoCfg, download: bool = False) -> Path:
    """Local path of the pinned weights file. With download=True it is fetched if missing; otherwise it must
    already be on disk (offline). A file whose fingerprint does not match is deleted and an error is raised."""
    from huggingface_hub import hf_hub_download

    path = Path(hf_hub_download(pc.hf_repo, FILENAME, revision=pc.revision, cache_dir=pc.weights_dir,
                                local_files_only=not download))  # fmt: skip
    digest = sha256(path)
    if digest != pc.weights_sha256:
        path.unlink(missing_ok=True)
        raise ValueError(f"weights fingerprint mismatch for {pc.hf_repo}: got {digest}, expected {pc.weights_sha256}")
    return path


def download_weights(pc: PhotoCfg) -> Path:
    """`glimpse download-weights`: fetch (if needed) and verify the photo model's weights."""
    path = weights_path(pc, download=True)
    logger.bind(stage="download").info("weights ready and verified: {} ({:.0f} MB)", path, path.stat().st_size / 1e6)
    return path
