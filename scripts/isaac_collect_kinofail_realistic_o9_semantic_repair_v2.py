#!/usr/bin/env python3
"""O9 semantic repair v2 with a pair-shared low-speed diagnostic creep.

The v1 preflight proved the desired belly-on-crossbar mechanism, but the legacy
0.22 m/s nuisance profile drove the unobstructed nominal off the short audited
route after the attribution window.  V2 keeps every v1 semantic intervention
and uniformly caps both members of every pair at 0.04 m/s.  This remains a
nonzero navigation command while keeping the nominal counterfactual inside the
audited route for the full recording.
"""

from __future__ import annotations

import hashlib
import importlib.util
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
V1 = ROOT / "scripts/isaac_collect_kinofail_realistic_o9_semantic_repair_v1.py"
EXPECTED_V1_SHA256 = "df5c75693e9a0f93e4b6baf0998575c9479f9dc582d076e498859e2697489b04"
DIAGNOSTIC_CREEP_MPS = 0.04


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _load_v1():
    actual = _sha256(V1)
    if actual != EXPECTED_V1_SHA256:
        raise RuntimeError(f"O9 semantic repair v2 dependency mismatch: {actual}")
    spec = importlib.util.spec_from_file_location("kinofail_o9_semantic_repair_v1", V1)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"cannot load {V1}")
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
            "pair-shared nonzero diagnostic creep; nominal remains on the "
            "audited route for the full attribution recording"
        )
        return profile

    implementation._physical_nuisance = physical_nuisance


def main() -> None:
    v1 = _load_v1()
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
