#!/usr/bin/env python3
"""Collect result-blind T3 pairs with the original v5-compatible run-in.

F37 established before any prediction that the generic reachable-region
adapter starts every expanded T3 episode inside the operator region.  This
collector changes only the O7/O8 region placement for the new F38 corpus:
the original 1.25 m center and 0.55 m route half-length provide observable
pre-interaction history.  The frozen v5 temporal function, simulator,
operators, sensors, appearance interventions, nuisance table, and thresholds
remain unchanged.
"""

from __future__ import annotations

import hashlib
import os
import sys
import traceback
from pathlib import Path
from typing import Any

from scripts import isaac_collect_kinofail_confirmatory_pair_v7 as v7
from scripts import isaac_collect_kinofail_confirmatory_pair_v8 as v8
from scripts.isaac_collect_kinofail_confirmatory_pair_v4 import _load_v4_wrapper


ROOT = Path(__file__).resolve().parents[1]
V8 = ROOT / "scripts/isaac_collect_kinofail_confirmatory_pair_v8.py"
BACKEND = ROOT / "kino_vla/sim/isaac_policy_backend.py"
EXPECTED_V8_SHA256 = "ccef4cd59126976c6fb9c9d6870432a4f3972ca2abb5f6edf7bae5cfe13dfa39"
EXPECTED_BACKEND_SHA256 = "ac1f5d3fee3c938462d529a0a08f68cf93713ec60cbe32e557793709863d42a3"
RUNIN_CENTER_PROGRESS_M = 1.25
RUNIN_HALF_LENGTH_M = 0.55


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def install_runin_region(wrapper: Any) -> None:
    prior_load_v4 = wrapper._load_v4

    def load_v4() -> Any:
        module = prior_load_v4()
        prior_install = module._install_reachable_exposure_contract

        def install_contract(implementation: Any) -> None:
            prior_install(implementation)

            def runin_region(
                frame: Any,
                *,
                progress_m: float = RUNIN_CENTER_PROGRESS_M,
                half_length_m: float = RUNIN_HALF_LENGTH_M,
            ) -> Any:
                from kino_vla.utils.geometry import Rect

                progress = min(progress_m, frame.route_length_m - half_length_m - 0.10)
                progress = max(progress, half_length_m + 0.10)
                half_width = min(0.40, 0.5 * frame.surface_width_m - 0.05)
                center = frame.point(progress, 0.0)
                if abs(float(frame.direction[0])) >= 1.0 - 1.0e-6:
                    return Rect(float(center[0]), float(center[1]), half_length_m, half_width)
                if abs(float(frame.direction[1])) >= 1.0 - 1.0e-6:
                    return Rect(float(center[0]), float(center[1]), half_width, half_length_m)
                raise ValueError("F38 run-in requires an axis-aligned audited route")

            implementation._region = runin_region

        module._install_reachable_exposure_contract = install_contract
        return module

    wrapper._load_v4 = load_v4


def main() -> None:
    try:
        for path, expected in (
            (V8, EXPECTED_V8_SHA256),
            (BACKEND, EXPECTED_BACKEND_SHA256),
            (v7.V5, v7.EXPECTED_V5_SHA256),
            (v7.V6, v7.EXPECTED_V6_SHA256),
            (v7.DESIGN, v7.EXPECTED_DESIGN_SHA256),
        ):
            if sha256(path) != expected:
                raise RuntimeError(f"F38 exact-hash dependency mismatch: {path}")
        wrapper = _load_v4_wrapper()
        wrapper.__file__ = str(v7.V5)
        wrapper._install_nuisance_contract = v7._install_f0_nuisance_contract
        install_runin_region(wrapper)
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
