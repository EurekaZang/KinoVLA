#!/usr/bin/env python3
"""Prepare KiNO-v4 features using the F4f mixed-corpus observation seal."""

from __future__ import annotations

import hashlib
import sys
import types
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
PREDECESSOR = ROOT / "scripts/prepare_kino_v4_confirmation_features_v1.py"
EXPECTED_PREDECESSOR_SHA256 = "279796aaa4c68d3ae2929689e0f0083334b55a305bf8af2d33bab47742f820aa"


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _load() -> types.ModuleType:
    if _sha256(PREDECESSOR) != EXPECTED_PREDECESSOR_SHA256:
        raise RuntimeError("F4f feature predecessor hash drift")
    source = PREDECESSOR.read_text()
    replacements = {
        'CORPUS = Path("/data/eureka/kinofail_kino_v4_confirmation_v1/corpus")\n': (
            'CORPUS = Path("/data/eureka/kinofail_kino_v4_confirmation_v1/corpus")\n'
            'T3_CORPUS = Path("/data/eureka/kinofail_kino_v4_confirmation_t3_runin_f4e/corpus")\n'
        ),
        'F4 = ROOT / "outputs/freeze/kino_v4_confirmation_observations_f4/observation_seal.json"\n': (
            'F4 = ROOT / "outputs/freeze/kino_v4_confirmation_observations_f4f/observation_seal.json"\n'
        ),
        '            "--t3-corpus",\n'
        '            str(CORPUS),\n': (
            '            "--t3-corpus",\n'
            '            str(T3_CORPUS),\n'
        ),
        '        "--t3-corpus",\n'
        '        str(CORPUS),\n': (
            '        "--t3-corpus",\n'
            '        str(T3_CORPUS),\n'
        ),
    }
    for old, new in replacements.items():
        if source.count(old) != 1:
            raise RuntimeError(f"F4f feature patch point is not unique: {old!r}")
        source = source.replace(old, new)
    module = types.ModuleType("kino_v4_confirmation_features_f4f")
    module.__file__ = str(Path(__file__).resolve())
    module.__package__ = "scripts"
    exec(compile(source, str(PREDECESSOR), "exec"), module.__dict__)
    return module


def main() -> int:
    return int(_load().main())


if __name__ == "__main__":
    raise SystemExit(main())
