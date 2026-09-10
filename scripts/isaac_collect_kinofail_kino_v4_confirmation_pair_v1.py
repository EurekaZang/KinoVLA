#!/usr/bin/env python3
"""Bind the audited pair collector to the KiNO-v4 F1 nuisance table."""

from __future__ import annotations

import hashlib
import json
import os
import sys
import traceback
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from scripts import isaac_collect_kinofail_confirmatory_pair_v7 as v7
from scripts import isaac_collect_kinofail_confirmatory_pair_v8 as v8
from scripts import isaac_collect_kinofail_confirmatory_pair_v9 as v9
from scripts.isaac_collect_kinofail_confirmatory_pair_v4 import _load_v4_wrapper


F1 = ROOT / "outputs/freeze/kino_v4_confirmation_v1_f1/freeze_manifest.json"
EXPECTED_F1_SHA256 = "de7fcdf73fe192cab0e18d68a3b5cea9844226f068ee4bfcf13479b1a6680d19"
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


def _install_f1_nuisance_contract(implementation: Any) -> None:
    if _sha256(F1) != EXPECTED_F1_SHA256:
        raise RuntimeError("KiNO-v4 F1 hash mismatch")
    design = json.loads(F1.read_text())["design"]
    contract = design["scale"]["physical_nuisance"]
    profiles = {int(row["profile_index"]): dict(row) for row in contract["profile_table"]}
    if set(profiles) != set(range(8)):
        raise RuntimeError("KiNO-v4 F1 nuisance table is incomplete")

    def physical_nuisance(record: dict[str, Any]) -> dict[str, Any]:
        value = record.get("physical_nuisance")
        if not isinstance(value, dict) or set(value) != NUISANCE_KEYS:
            raise RuntimeError("schedule does not provide the exact F1 nuisance keys")
        nuisance = dict(value)
        index = int(nuisance["profile_index"])
        if index not in profiles:
            raise RuntimeError("nuisance profile is absent from KiNO-v4 F1")
        expected = profiles[index]
        if any(
            float(nuisance[key]) != float(expected[key])
            for key in PROFILE_VALUE_KEYS
        ):
            raise RuntimeError("nuisance values differ from the KiNO-v4 F1 profile table")
        if (
            nuisance["pair_shared"] is not True
            or int(nuisance["physics_seed"]) != int(record["operator_seed"])
            or float(nuisance["controller_target_lateral_offset_m"]) != 0.0
        ):
            raise RuntimeError("nuisance pair-sharing contract differs from KiNO-v4 F1")
        return nuisance

    implementation._physical_nuisance = physical_nuisance


def main() -> None:
    try:
        for path, expected in (
            (v9.V8, v9.EXPECTED_V8_SHA256),
            (v9.BACKEND, v9.EXPECTED_BACKEND_SHA256),
            (v8.V7, v8.EXPECTED_V7_SHA256),
            (v7.V5, v7.EXPECTED_V5_SHA256),
            (v7.V6, v7.EXPECTED_V6_SHA256),
            (F1, EXPECTED_F1_SHA256),
        ):
            if _sha256(path) != expected:
                raise RuntimeError(f"exact-hash dependency mismatch: {path}")
        wrapper = _load_v4_wrapper()
        wrapper.__file__ = str(v7.V5)
        wrapper._install_nuisance_contract = _install_f1_nuisance_contract
        v8._install_registry_scene_source_adapter(wrapper)
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
