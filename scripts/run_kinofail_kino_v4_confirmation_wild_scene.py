#!/usr/bin/env python3
"""Run one model-blind KiNO-v4 confirmation wild-scene candidate."""

from __future__ import annotations

import hashlib
import types
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
IMPLEMENTATION = ROOT / "scripts/run_kinofail_confirmatory_wild_scene_v1.py"
EXPECTED_IMPLEMENTATION_SHA256 = (
    "c3e1ec52e61b756656af69375c2dd864ac099767df06b6974790f324984558b0"
)


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main() -> int:
    if _sha256(IMPLEMENTATION) != EXPECTED_IMPLEMENTATION_SHA256:
        raise RuntimeError("wild admission implementation hash mismatch")
    source = IMPLEMENTATION.read_text(encoding="utf-8")
    replacements = {
        'f"confirm_v1_wild_pbr_{index:02d}"': (
            'f"kino_v4_confirm_wild_pbr_{index % 4:02d}"'
        ),
        'f"confirm_v1_wild_pbr_{(index + 5) % 10:02d}"': (
            'f"kino_v4_confirm_wild_pbr_{(index + 2) % 4:02d}"'
        ),
        'ROOT\n        / "outputs/kinofail_confirmatory_v1/scenes/wild"': (
            'Path("/data/eureka/kinofail_kino_v4_confirmation_v1/scenes/wild")'
        ),
        "confirmatory-v1": "kino-v4-confirmation-v1",
        "confirmatory_f1_candidate_model_blind": (
            "kino_v4_confirmation_candidate_model_blind"
        ),
    }
    for old, new in replacements.items():
        if old not in source:
            raise RuntimeError(f"wild adapter patch point absent: {old}")
        source = source.replace(old, new)
    module = types.ModuleType("kinofail_kino_v4_confirmation_wild")
    module.__file__ = str(Path(__file__).resolve())
    exec(compile(source, str(IMPLEMENTATION), "exec"), module.__dict__)
    return int(module.main())


if __name__ == "__main__":
    raise SystemExit(main())
