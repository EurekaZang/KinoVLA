#!/usr/bin/env python3
"""Build the F0/F1-bound independent reconfirmation schedules.

This adapter pins the audited v1 schedule compiler and applies only the
preregistered namespace, path, seed, and freshness-auditor substitutions.
All row schemas, counterfactual construction, snapshot alignment, counts, and
operator interpolation code remain inherited from the pinned implementation.
"""

from __future__ import annotations

import hashlib
import types
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
IMPLEMENTATION = ROOT / "scripts/build_kinofail_unified_confirmatory_v1_schedules.py"
EXPECTED_IMPLEMENTATION_SHA256 = (
    "186d7cb4048504041d9e5c1d5b75a0a21a9bb8cca0f3497606b04bd56b47d684"
)


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _load() -> types.ModuleType:
    if _sha256(IMPLEMENTATION) != EXPECTED_IMPLEMENTATION_SHA256:
        raise RuntimeError("v1 schedule compiler hash mismatch")
    source = IMPLEMENTATION.read_text(encoding="utf-8")
    replacements = {
        "configs/data/kinofail_unified_confirmatory_design_v1.json": (
            "configs/data/kinofail_unified_reconfirmation_design_v2.json"
        ),
        "outputs/freeze/unified_moe_v3_confirmatory_f0/freeze_manifest.json": (
            "outputs/freeze/unified_moe_v3_reconfirmation_f0/freeze_manifest.json"
        ),
        "outputs/kinofail_confirmatory_v1/scene_registry.json": (
            "outputs/kinofail_reconfirmation_v2/scene_registry.json"
        ),
        "outputs/assets/terrain_pbr_confirmatory_v1/terrain_assets.lock.json": (
            "outputs/assets/terrain_pbr_confirmatory_v2/terrain_assets.lock.json"
        ),
        "scripts/isaac_collect_kinofail_confirmatory_pair_v1.py": (
            "scripts/isaac_collect_kinofail_confirmatory_pair_v5.py"
        ),
        "outputs/kinofail_confirmatory_v1/schedules": (
            "outputs/kinofail_reconfirmation_v2/schedules"
        ),
        "scripts.audit_kinofail_confirmatory_freshness_v1": (
            "scripts.audit_kinofail_reconfirmation_freshness_v2"
        ),
        "2_030_000_000": "2_060_000_000",
        "2_040_000_000": "2_070_000_000",
        "confirmatory-v1": "reconfirmation-v2",
        "kinofail_unified_confirmatory_v1": "kinofail_unified_reconfirmation_v2",
        "outputs/kinofail_confirmatory_v1/corpus": (
            "outputs/kinofail_reconfirmation_v2/corpus"
        ),
        "outputs/eval/unified_moe_v3_confirmatory_v1": (
            "outputs/eval/unified_moe_v3_reconfirmation_v2"
        ),
    }
    for old, new in replacements.items():
        if old not in source:
            raise RuntimeError(f"schedule patch point absent: {old}")
        source = source.replace(old, new)
    module = types.ModuleType("kinofail_unified_reconfirmation_v2_impl")
    module.__file__ = str(Path(__file__).resolve())
    exec(compile(source, str(IMPLEMENTATION), "exec"), module.__dict__)
    return module


_IMPL = _load()
interpolate_parameters = _IMPL.interpolate_parameters
physical_nuisance = _IMPL.physical_nuisance
validate_inputs = _IMPL.validate_inputs
build_schedules = _IMPL.build_schedules


def main() -> int:
    return int(_IMPL.main())


if __name__ == "__main__":
    raise SystemExit(main())
