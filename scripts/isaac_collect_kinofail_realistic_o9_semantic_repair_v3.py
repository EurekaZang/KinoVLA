#!/usr/bin/env python3
"""Final O9 semantic repair with pair-shared 0.08 m/s diagnostic creep.

The v2 pilot kept the nominal stable but produced 18.3 N peak sustained base
load, just below the predeclared 20 N admission threshold.  V3 retains the
threshold and raises the pair-shared nonzero command from 0.04 to 0.08 m/s.
The change is calibrated only from excluded pilot mechanism/stability telemetry,
not from attribution-model predictions.
"""

from __future__ import annotations

import hashlib
import importlib.util
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
V2 = ROOT / "scripts/isaac_collect_kinofail_realistic_o9_semantic_repair_v2.py"
EXPECTED_V2_SHA256 = "1e659f67f4299c93480a9706a4fa5d23be7d86734b2702663c1421729887e082"
DIAGNOSTIC_CREEP_MPS = 0.08


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _load_v2():
    actual = _sha256(V2)
    if actual != EXPECTED_V2_SHA256:
        raise RuntimeError(f"O9 semantic repair v3 dependency mismatch: {actual}")
    spec = importlib.util.spec_from_file_location("kinofail_o9_semantic_repair_v2", V2)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"cannot load {V2}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _install_pair_shared_diagnostic_creep(implementation) -> None:
    prior_nuisance = implementation._physical_nuisance

    def physical_nuisance(record):
        profile = dict(prior_nuisance(record))
        profile["pre_repair_forward_speed_mps"] = float(profile["forward_speed_mps"])
        profile["forward_speed_mps"] = DIAGNOSTIC_CREEP_MPS
        profile["o9_motion_contract"] = (
            "pair-shared 0.08 m/s diagnostic creep, frozen after excluded "
            "mechanism/stability pilots and before any v3 outcome"
        )
        return profile

    implementation._physical_nuisance = physical_nuisance


def main() -> None:
    v2 = _load_v2()
    v1 = v2._load_v1()
    v8 = v1._load_v8()
    v4 = v8._load_v4()
    v4._install_runtime_validation_adapter()
    implementation = v8._load_10hz_nuisance_implementation()
    v8._install_balanced_nuisance_contract(implementation)
    v4._install_reachable_exposure_contract(implementation)
    v1._install_beached_start_contract(implementation)
    _install_pair_shared_diagnostic_creep(implementation)
    v1._install_route_transverse_pallet_crossbar(implementation)
    v1._install_direct_semantic_gate(implementation)
    implementation.__file__ = __file__
    implementation.main()


if __name__ == "__main__":
    main()
