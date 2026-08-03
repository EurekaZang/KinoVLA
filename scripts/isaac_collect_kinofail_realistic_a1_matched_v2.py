#!/usr/bin/env python3
"""A1 matched collector v2: bounded 10 Hz RTX around downstream first contact.

This is a throughput-only amendment to v1.  Physics still runs for the full frozen horizon and
all proprioception/telemetry are retained.  RTX begins at route progress 0.45 m and stops after
12 synchronized three-view frames, which covers at least 0.4 s before the fixed region beginning
at 0.70 m without rendering hundreds of frames that A1 never consumes.
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
        raise RuntimeError("A1 matched v2 dependency hash mismatch")
    source = V1.read_text(encoding="utf-8")
    patches = {
        "    capture_stride = 20\n": "    capture_stride = 5\n",
        "        if step % capture_stride == 0:\n": (
            "        if (step % capture_stride == 0 and progress >= 0.45 "
            "and len(rgb[primary_id]) < 12):\n"
        ),
    }
    for old, new in patches.items():
        if source.count(old) != 1:
            raise RuntimeError(f"A1 matched v2 patch point is not unique: {old!r}")
        source = source.replace(old, new)
    implementation = types.ModuleType("kinofail_a1_matched_v2_bounded_10hz")
    implementation.__file__ = __file__
    implementation.__package__ = "scripts"
    exec(compile(source, str(V1), "exec"), implementation.__dict__)
    implementation.__file__ = __file__
    implementation.main()


if __name__ == "__main__":
    main()
