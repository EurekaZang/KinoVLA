#!/usr/bin/env python3
"""Uniform pre-event history for O3-severe and every O8 scale-v8 pair.

O3 needs the repair only at severe collapse strength.  O8 can collide
immediately at either severity because its frozen region overlaps some route
entry profiles, so the complete O8 operator block is admitted.
"""

from __future__ import annotations

import hashlib
import types
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
BASE = ROOT / "scripts/isaac_collect_kinofail_realistic_o3_temporal_repair_v1.py"
EXPECTED_BASE_SHA256 = "a6d72be978c67a29a3e2b575632f5f614eb070615836dcba61c21b7e7c80d5a1"


def _sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _load_complete_block_repair():
    if _sha(BASE) != EXPECTED_BASE_SHA256:
        raise RuntimeError("temporal-repair-v4 dependency mismatch")
    source = BASE.read_text(encoding="utf-8")
    replacements = {
        'record["target_operator"] == "O3_collapse"': (
            'record["target_operator"] in {"O3_collapse", "O8_invisible_collider"}'
        ),
        'manifest["collection"]["o3_temporal_repair"]': (
            'manifest["collection"]["temporal_window_repair"]'
        ),
        (
            '"selection_cell": "O3 severe / physical nuisance profile 1 / all scenes",'
        ): (
            '"selection_cell": "O3-severe/O8-all-severity / all profiles / all scenes",'
        ),
        (
            '        if (\n'
            '            representative["target_operator"] != "O3_collapse"\n'
            '            or representative["severity_id"] != "severe"\n'
            '            or index_by_scene_seed.get(key) != 1\n'
            '        ):\n'
            '            raise RuntimeError("pair is outside the frozen O3 severe/profile-1 repair cell")\n'
        ): (
            '        operator = representative["target_operator"]\n'
            '        admitted = (\n'
            '            (operator == "O3_collapse" and representative["severity_id"] == "severe")\n'
            '            or operator == "O8_invisible_collider"\n'
            '        )\n'
            '        if not admitted:\n'
            '            raise RuntimeError("pair is outside the frozen temporal-repair-v4 blocks")\n'
        ),
        '"O3 temporal repair requires exactly five scene seeds per scene"': (
            '"temporal repair v4 requires exactly five scene seeds per scene"'
        ),
        '"kinofail_o3_temporal_repair_impl"': '"kinofail_temporal_repair_v4_impl"',
    }
    for old, new in replacements.items():
        if source.count(old) != 1:
            raise RuntimeError(f"temporal-repair-v4 patch point is not unique: {old!r}")
        source = source.replace(old, new)
    module = types.ModuleType("kinofail_temporal_repair_v4")
    module.__file__ = __file__
    module.__package__ = "scripts"
    exec(compile(source, str(BASE), "exec"), module.__dict__)
    return module


if __name__ == "__main__":
    _load_complete_block_repair().main()
