#!/usr/bin/env python3
"""A1 matched collector v3: bounded construct horizon and bounded 10 Hz RTX.

A1 consumes only the pre-contact window ending at the first measured operator contact.  This
collector therefore retains 4.4 s (220 control steps) of physics/proprio/telemetry—enough to pass
the fixed region start at 0.70 m under the frozen 0.24 m/s command—and at most 12 synchronized
three-view RTX frames around that contact.  It does not collect the irrelevant later outcome tail.
"""

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
        raise RuntimeError("A1 matched v3 dependency hash mismatch")
    source = V1.read_text(encoding="utf-8")
    patches = {
        "    capture_stride = 20\n": "    capture_stride = 5\n",
        "    while step < 360 and not obs.fallen:\n": "    while step < 220 and not obs.fallen:\n",
        "        if step % capture_stride == 0:\n": (
            "        if (step % capture_stride == 0 and progress >= 0.45 "
            "and len(rgb[primary_id]) < 12):\n"
        ),
    }
    for old, new in patches.items():
        if source.count(old) != 1:
            raise RuntimeError(f"A1 matched v3 patch point is not unique: {old!r}")
        source = source.replace(old, new)
    implementation = types.ModuleType("kinofail_a1_matched_v3_bounded_construct")
    implementation.__file__ = __file__
    implementation.__package__ = "scripts"
    exec(compile(source, str(V1), "exec"), implementation.__dict__)
    implementation.__file__ = __file__
    implementation.main()


if __name__ == "__main__":
    main()
