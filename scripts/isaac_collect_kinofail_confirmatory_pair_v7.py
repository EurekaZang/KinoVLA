#!/usr/bin/env python3
"""Bind nuisance validation to the exact F0-frozen 16-profile table.

The scientific v5 collector, v6 provenance correction, simulation dynamics,
and output validation remain unchanged.  This wrapper replaces only legacy
range checks that were narrower than values already sealed in F0.
"""

from __future__ import annotations

import hashlib
import json
import os
import sys
import traceback
from pathlib import Path
from typing import Any

from scripts.isaac_collect_kinofail_confirmatory_pair_v4 import _load_v4_wrapper


ROOT = Path(__file__).resolve().parents[1]
V5 = ROOT / "scripts/isaac_collect_kinofail_confirmatory_pair_v5.py"
V6 = ROOT / "scripts/isaac_collect_kinofail_confirmatory_pair_v6.py"
DESIGN = ROOT / "configs/data/kinofail_unified_reconfirmation_design_v2.json"
EXPECTED_V5_SHA256 = (
    "88f8e3abe6805bb1e96a87f333dad29d285b49ec1dbf40e4b626b020bb36419d"
)
EXPECTED_V6_SHA256 = (
    "81345fb350695f0f2c91704d42ee326a14bf897f747010951f356d8021a7745b"
)
EXPECTED_DESIGN_SHA256 = (
    "1a1fd3c53590dd745f12b7ca4582c6b186514077ca168f712ae8338938d428c2"
)
NUISANCE_KEYS = {
    "profile_index",
    "start_progress_m",
    "start_lateral_offset_m",
    "start_heading_offset_rad",
    "forward_speed_mps",
    "controller_target_lateral_offset_m",
    "physics_seed",
    "pair_shared",
}
PROFILE_VALUE_KEYS = (
    "start_progress_m",
    "start_lateral_offset_m",
    "start_heading_offset_rad",
    "forward_speed_mps",
)


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _install_f0_nuisance_contract(implementation: Any) -> None:
    design = json.loads(DESIGN.read_text(encoding="utf-8"))
    contract = design["scale"]["physical_nuisance"]
    if (
        set(contract["exact_keys"]) != NUISANCE_KEYS
        or contract["pair_shared"] is not True
        or contract["physics_seed_equals_operator_seed"] is not True
        or float(contract["controller_target_lateral_offset_m"]) != 0.0
    ):
        raise RuntimeError("F0 nuisance contract structure changed")
    profiles = {
        int(row["profile_index"]): dict(row)
        for row in contract["profile_table"]
    }
    if set(profiles) != set(range(16)):
        raise RuntimeError("F0 nuisance profile table is incomplete")

    def physical_nuisance(record: dict[str, Any]) -> dict[str, Any]:
        value = record.get("physical_nuisance")
        if not isinstance(value, dict) or set(value) != NUISANCE_KEYS:
            raise RuntimeError(
                "confirmatory schedule must provide the exact F0 nuisance keys"
            )
        nuisance = dict(value)
        index = int(nuisance["profile_index"])
        if index not in profiles:
            raise RuntimeError("nuisance profile is absent from F0")
        expected = profiles[index]
        if any(
            float(nuisance[key]) != float(expected[key])
            for key in PROFILE_VALUE_KEYS
        ):
            raise RuntimeError("nuisance values differ from the F0 profile table")
        if (
            nuisance["pair_shared"] is not True
            or int(nuisance["physics_seed"]) != int(record["operator_seed"])
            or float(nuisance["controller_target_lateral_offset_m"]) != 0.0
        ):
            raise RuntimeError("nuisance pair-sharing contract differs from F0")
        return nuisance

    implementation._physical_nuisance = physical_nuisance


def main() -> None:
    try:
        for path, expected in (
            (V5, EXPECTED_V5_SHA256),
            (V6, EXPECTED_V6_SHA256),
            (DESIGN, EXPECTED_DESIGN_SHA256),
        ):
            if _sha256(path) != expected:
                raise RuntimeError(f"v7 exact-hash dependency mismatch: {path}")
        wrapper = _load_v4_wrapper()
        wrapper.__file__ = str(V5)
        wrapper._install_nuisance_contract = _install_f0_nuisance_contract
        result = wrapper.main()
        status = int(result) if result is not None else 0
    except BaseException:
        traceback.print_exc()
        status = 1
    finally:
        sys.stdout.flush()
        sys.stderr.flush()
    os._exit(status)


if __name__ == "__main__":
    main()
