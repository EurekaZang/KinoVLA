#!/usr/bin/env python3
"""A1 matched collector: original reachable route region with synchronized 10 Hz RTX.

Unlike the scale-v6 collector, this A1-only collector deliberately does not move the operator
region to the robot's reset footprint.  The frozen A1 snapshot ends at the first measured contact
with the downstream region, so O2 and O4 share the complete pre-contact proprioceptive history.
The underlying scale-v1 implementation is hash checked and only its capture stride is changed
from 2.5 Hz to 10 Hz; physics, controller, camera, runtime gates, and deep reset are unchanged.
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
        raise RuntimeError("A1 matched collector dependency hash mismatch")
    source = V1.read_text(encoding="utf-8")
    old, new = "    capture_stride = 20\n", "    capture_stride = 5\n"
    if source.count(old) != 1:
        raise RuntimeError("frozen v1 capture-stride patch point is not unique")
    implementation = types.ModuleType("kinofail_a1_matched_v1_10hz")
    implementation.__file__ = __file__
    implementation.__package__ = "scripts"
    exec(compile(source.replace(old, new), str(V1), "exec"), implementation.__dict__)
    implementation.__file__ = __file__
    implementation.main()


if __name__ == "__main__":
    main()
