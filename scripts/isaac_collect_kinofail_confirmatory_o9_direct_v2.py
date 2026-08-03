#!/usr/bin/env python3
"""Direct-contact O9 collector with a locomotion-stable approach speed.

F32/F32b established the direct contact mechanism at 0.08 m/s, while their
excluded pilots showed that this speed is below the stable operating region of
the frozen locomotion policy in some realistic scenes.  This adapter changes
only the frozen pair-shared approach speed to 0.18 m/s.
"""

from __future__ import annotations

import os
import sys
import traceback
from pathlib import Path
from typing import Any

from scripts import isaac_collect_kinofail_confirmatory_o9_direct_v1 as direct


ROOT = Path(__file__).resolve().parents[1]


def install_speed_contract(implementation: Any) -> None:
    def physical_nuisance(record: dict[str, Any]) -> dict[str, Any]:
        value = record.get("physical_nuisance")
        if not isinstance(value, dict) or set(value) != direct.NUISANCE_KEYS:
            raise RuntimeError("O9 direct-v2 schedule has an invalid nuisance contract")
        nuisance = dict(value)
        if nuisance["pair_shared"] is not True:
            raise RuntimeError("O9 direct-v2 nuisance must be pair-shared")
        if int(nuisance["physics_seed"]) != int(record["operator_seed"]):
            raise RuntimeError("O9 direct-v2 operator and physics seeds differ")
        if abs(float(nuisance["start_progress_m"]) - 0.35) > 1.0e-9:
            raise RuntimeError("O9 direct-v2 start progress drift")
        if abs(float(nuisance["forward_speed_mps"]) - 0.18) > 1.0e-9:
            raise RuntimeError("O9 direct-v2 approach speed drift")
        if abs(float(nuisance["spawn_base_height_m"]) - 0.43) > 1.0e-9:
            raise RuntimeError("O9 direct-v2 spawn height drift")
        if abs(float(nuisance["start_lateral_offset_m"])) > 0.003:
            raise RuntimeError("O9 direct-v2 lateral nuisance outside frozen range")
        if abs(float(nuisance["start_heading_offset_rad"])) > 0.002:
            raise RuntimeError("O9 direct-v2 heading nuisance outside frozen range")
        return nuisance

    implementation._physical_nuisance = physical_nuisance


def main() -> None:
    try:
        direct._install_o9_nuisance_contract = install_speed_contract
        direct.__file__ = __file__
        direct.main()
        status = 0
    except BaseException:
        traceback.print_exc()
        status = 1
    finally:
        sys.stdout.flush()
        sys.stderr.flush()
    os._exit(status)


if __name__ == "__main__":
    main()
