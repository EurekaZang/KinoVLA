#!/usr/bin/env python3
"""Final O9 repair collector after mechanism-only moderate-height calibration."""

from __future__ import annotations

import hashlib
import importlib.util
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
V3 = ROOT / "scripts/isaac_collect_kinofail_realistic_o9_semantic_repair_v3.py"
EXPECTED_V3_SHA256 = "896e05404a019ee7115f0f4f778a835e0b68a4196c6c0088c188e4b3dd04c8d6"


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _load_v3():
    actual = _sha256(V3)
    if actual != EXPECTED_V3_SHA256:
        raise RuntimeError(f"O9 semantic repair v4 dependency mismatch: {actual}")
    spec = importlib.util.spec_from_file_location("kinofail_o9_semantic_repair_v3", V3)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"cannot load {V3}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def main() -> None:
    v3 = _load_v3()
    v2 = v3._load_v2()
    v1 = v2._load_v1()
    v8 = v1._load_v8()
    v4 = v8._load_v4()
    v4._install_runtime_validation_adapter()
    implementation = v8._load_10hz_nuisance_implementation()
    v8._install_balanced_nuisance_contract(implementation)
    v4._install_reachable_exposure_contract(implementation)
    v1._install_beached_start_contract(implementation)
    v3._install_pair_shared_diagnostic_creep(implementation)
    v1._install_route_transverse_pallet_crossbar(implementation)
    v1._install_direct_semantic_gate(implementation)
    implementation.__file__ = __file__
    implementation.main()


if __name__ == "__main__":
    main()
