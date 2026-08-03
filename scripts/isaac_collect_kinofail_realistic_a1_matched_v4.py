#!/usr/bin/env python3
"""A1 matched collector v4: early shared region, bounded construct horizon, 10 Hz RTX."""

from __future__ import annotations

import hashlib
import types
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
V1 = ROOT / "scripts/isaac_collect_kinofail_realistic_pair_v1.py"
EXPECTED_V1_SHA256 = "3f1b86cfda574dc4185375a6653e7b8ba71be37710d28041033e1643de0f820a"


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main() -> None:
    if _sha256(V1) != EXPECTED_V1_SHA256:
        raise RuntimeError("A1 matched v4 dependency hash mismatch")
    source = V1.read_text(encoding="utf-8")
    patches = {
        "def _region(frame, *, progress_m: float = 1.25, half_length_m: float = 0.55):\n": (
            "def _region(frame, *, progress_m: float = 0.35, half_length_m: float = 0.10):\n"
        ),
        "    capture_stride = 20\n": "    capture_stride = 5\n",
        "    while step < 360 and not obs.fallen:\n": "    while step < 120 and not obs.fallen:\n",
        "        if step % capture_stride == 0:\n": (
            "        if step % capture_stride == 0 and len(rgb[primary_id]) < 16:\n"
        ),
    }
    for old, new in patches.items():
        if source.count(old) != 1:
            raise RuntimeError(f"A1 matched v4 patch point is not unique: {old!r}")
        source = source.replace(old, new)
    implementation = types.ModuleType("kinofail_a1_matched_v4_early_region")
    implementation.__file__ = __file__
    implementation.__package__ = "scripts"
    exec(compile(source, str(V1), "exec"), implementation.__dict__)
    implementation.__file__ = __file__
    implementation.main()


if __name__ == "__main__":
    main()
