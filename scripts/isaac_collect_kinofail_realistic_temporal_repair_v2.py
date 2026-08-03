#!/usr/bin/env python3
"""Uniform temporal-window repair for the two frozen scale-v8 failure cells.

This wrapper reuses the hash-pinned O3 repair collector and broadens only its
predeclared selection to:

* O3 severe / physical nuisance profile 1 / all nine scenes
* O8 severe / physical nuisance profile 3 / all nine scenes

Both cells receive the same 30-control-step (0.6 s) recorded pre-roll.  No
operator strength, route, controller, camera, appearance, or pairing changes.
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


def _load_combined_repair():
    if _sha(BASE) != EXPECTED_BASE_SHA256:
        raise RuntimeError("temporal-repair-v2 dependency mismatch")
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
            '"selection_cell": ('
            '"O3 severe / profile 1 / all scenes" '
            'if record["target_operator"] == "O3_collapse" '
            'else "O8 severe / profile 3 / all scenes"),'
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
            '        profile_index = index_by_scene_seed.get(key)\n'
            '        admitted = (\n'
            '            representative["severity_id"] == "severe"\n'
            '            and (\n'
            '                (operator == "O3_collapse" and profile_index == 1)\n'
            '                or (operator == "O8_invisible_collider" and profile_index == 3)\n'
            '            )\n'
            '        )\n'
            '        if not admitted:\n'
            '            raise RuntimeError("pair is outside the frozen temporal-repair-v2 cells")\n'
        ),
        '"O3 temporal repair requires exactly five scene seeds per scene"': (
            '"temporal repair v2 requires exactly five scene seeds per scene"'
        ),
        '"kinofail_o3_temporal_repair_impl"': '"kinofail_temporal_repair_v2_impl"',
    }
    for old, new in replacements.items():
        if source.count(old) != 1:
            raise RuntimeError(f"temporal-repair-v2 patch point is not unique: {old!r}")
        source = source.replace(old, new)
    module = types.ModuleType("kinofail_temporal_repair_v2")
    module.__file__ = __file__
    module.__package__ = "scripts"
    exec(compile(source, str(BASE), "exec"), module.__dict__)
    return module


if __name__ == "__main__":
    _load_combined_repair().main()
