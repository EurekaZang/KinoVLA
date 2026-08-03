#!/usr/bin/env python3
"""Frozen T2 shared-prefix collector with the full nuisance profile applied."""

from __future__ import annotations

import hashlib
import sys
import types
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
IMPLEMENTATION = (
    ROOT / "scripts/isaac_collect_kinofail_realistic_c1_causal_v1.py"
)
EXPECTED_SHA256 = (
    "161b19c59dd7be07f7ef6e9b0e3b8519db1f0c32a00577ae165bbbdcd01d4ce2"
)


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _load() -> types.ModuleType:
    if _sha256(IMPLEMENTATION) != EXPECTED_SHA256:
        raise RuntimeError("confirmatory T2 collector dependency mismatch")
    source = IMPLEMENTATION.read_text(encoding="utf-8")
    replacements = {
        (
            "    backend._start_pos = frame.point(\n"
            '        0.0, float(profile["start_lateral_offset_m"])\n'
            "    )\n"
        ): (
            "    backend._start_pos = frame.point(\n"
            '        float(profile["start_progress_m"]),\n'
            '        float(profile["start_lateral_offset_m"]),\n'
            "    )\n"
        ),
        "            target_lateral_offset_m=0.0,\n": (
            "            target_lateral_offset_m=float(\n"
            '                profile["controller_target_lateral_offset_m"]\n'
            "            ),\n"
        ),
    }
    for old, new in replacements.items():
        if source.count(old) != 1:
            raise RuntimeError(f"T2 patch point is not unique: {old!r}")
        source = source.replace(old, new)
    module = types.ModuleType("kinofail_confirmatory_t2_implementation")
    module.__file__ = str(Path(__file__).resolve())
    module.__package__ = "scripts"
    exec(compile(source, str(IMPLEMENTATION), "exec"), module.__dict__)
    module.__file__ = str(Path(__file__).resolve())
    return module


def main() -> int:
    return int(_load().main())


if __name__ == "__main__":
    raise SystemExit(main())
