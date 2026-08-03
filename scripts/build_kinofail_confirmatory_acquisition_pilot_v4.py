#!/usr/bin/env python3
"""Freeze excluded collector-v4 O8 acquisition QA from the v3 design builder."""

from __future__ import annotations

import hashlib
import types
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
V3_BUILDER = ROOT / "scripts/build_kinofail_confirmatory_acquisition_pilot_v3.py"
EXPECTED_V3_BUILDER_SHA256 = (
    "c07c3ed0c895846c2dc11affa613b2ee6ad198d72f2a8c5d3f2c5fa360a6ab9a"
)


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main() -> int:
    if _sha256(V3_BUILDER) != EXPECTED_V3_BUILDER_SHA256:
        raise RuntimeError("acquisition pilot v3 builder dependency mismatch")
    source = V3_BUILDER.read_text(encoding="utf-8")
    replacements = {
        "isaac_collect_kinofail_confirmatory_pair_v3.py": (
            "isaac_collect_kinofail_confirmatory_pair_v4.py"
        ),
        "outputs/kinofail_acquisition_pilot_v3/design": (
            "outputs/kinofail_acquisition_pilot_v4/design"
        ),
        "kinofail.acquisition-pilot-registry.v3": (
            "kinofail.acquisition-pilot-registry.v4"
        ),
        "kinofail-acquisition-pilot-v3-physical-terminal-sensing": (
            "kinofail-acquisition-pilot-v4-physical-terminal-sensing"
        ),
        "kinofail.acquisition-pilot-freeze.v3": (
            "kinofail.acquisition-pilot-freeze.v4"
        ),
        '"collector_v3": _sha256(COLLECTOR),': (
            '"collector_v4": _sha256(COLLECTOR),'
        ),
    }
    for old, new in replacements.items():
        if source.count(old) != 1:
            raise RuntimeError(f"v4 pilot builder patch point is not unique: {old}")
        source = source.replace(old, new)
    module = types.ModuleType("kinofail_confirmatory_acquisition_pilot_v4")
    module.__file__ = str(Path(__file__).resolve())
    exec(compile(source, str(V3_BUILDER), "exec"), module.__dict__)
    return int(module.main())


if __name__ == "__main__":
    raise SystemExit(main())
