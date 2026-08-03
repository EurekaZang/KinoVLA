#!/usr/bin/env python3
"""Run one model-blind KiNO-v4 confirmation indoor-scene candidate."""

from __future__ import annotations

import hashlib
import types
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
IMPLEMENTATION = ROOT / "scripts/run_kinofail_confirmatory_indoor_scene_v1.py"
EXPECTED_IMPLEMENTATION_SHA256 = (
    "d8673b05b0376a4e93fcf64d40d6a1fb98c741108e01a0eaccb72ee7981efc97"
)


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main() -> int:
    if _sha256(IMPLEMENTATION) != EXPECTED_IMPLEMENTATION_SHA256:
        raise RuntimeError("indoor admission implementation hash mismatch")
    source = IMPLEMENTATION.read_text(encoding="utf-8")
    old = '    return f"confirm_v1_{domain}_pbr_{index % 10:02d}"\n'
    new = '    return f"kino_v4_confirm_{domain}_pbr_{index % 4:02d}"\n'
    if source.count(old) != 1:
        raise RuntimeError("material namespace patch point is not unique")
    source = source.replace(old, new)
    module = types.ModuleType("kinofail_kino_v4_confirmation_indoor")
    module.__file__ = str(Path(__file__).resolve())
    exec(compile(source, str(IMPLEMENTATION), "exec"), module.__dict__)
    return int(module.main())


if __name__ == "__main__":
    raise SystemExit(main())
