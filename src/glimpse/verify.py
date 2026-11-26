"""`glimpse verify-outputs`: the data-free outputs against the committed ones, numbers within a tolerance.

Byte-level `git diff` flagged 1e-16 float noise between machines; this compares every number in the
committed JSON files to 1e-9 (absolute or relative) and every other value exactly, and fails on any change."""

from __future__ import annotations

import json
import math
import subprocess
from pathlib import Path

from loguru import logger

PATHS = ("outputs/report_metrics", "outputs/monitoring", "outputs/probes", "outputs/v2_explain",
         "outputs/figures/figures.json")  # fmt: skip
TOL = 1e-9


def _diff(a, b, where: str, out: list[str]) -> None:
    if isinstance(a, dict) and isinstance(b, dict):
        for k in sorted(set(a) | set(b)):
            if k not in a or k not in b:
                out.append(f"{where}/{k}: present on one side only")
            else:
                _diff(a[k], b[k], f"{where}/{k}", out)
    elif isinstance(a, list) and isinstance(b, list):
        if len(a) != len(b):
            out.append(f"{where}: length {len(a)} vs {len(b)}")
        for i, (x, y) in enumerate(zip(a, b, strict=False)):
            _diff(x, y, f"{where}[{i}]", out)
    elif isinstance(a, (int, float)) and isinstance(b, (int, float)) and not isinstance(a, bool):
        if not (math.isclose(a, b, rel_tol=TOL, abs_tol=TOL) or (math.isnan(a) and math.isnan(b))):
            out.append(f"{where}: {a} vs {b}")
    elif a != b:
        out.append(f"{where}: {str(a)[:60]} vs {str(b)[:60]}")


def verify_outputs() -> int:
    files = subprocess.run(["git", "ls-files", *PATHS], capture_output=True, text=True, check=True).stdout.split()
    bad: list[str] = []
    for f in files:
        committed = subprocess.run(["git", "show", f"HEAD:{f}"], capture_output=True, check=True).stdout
        now = Path(f).read_bytes() if Path(f).exists() else b""
        if f.endswith(".json"):
            try:
                _diff(json.loads(committed), json.loads(now), f, bad)
            except json.JSONDecodeError:
                bad.append(f"{f}: not valid JSON now")
        elif not f.endswith(".png") and committed != now:  # figures are re-drawn; their numbers are in figures.json
            bad.append(f"{f}: differs")
    for line in bad[:40]:
        logger.bind(stage="verify").error(line)
    logger.bind(stage="verify").info("{} files checked, {} differences beyond {}", len(files), len(bad), TOL)
    return 1 if bad else 0
