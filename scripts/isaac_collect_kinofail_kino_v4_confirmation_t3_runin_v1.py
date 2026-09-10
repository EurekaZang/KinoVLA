#!/usr/bin/env python3
"""Collect KiNO-v4 T3 pairs with the pre-existing v5-compatible run-in.

The first KiNO-v4 T3 acquisition placed the Go2 footprint inside the operator
region at the first telemetry sample.  This result-blind successor preserves
the F1 schedules, nuisance profiles, simulator, sensors, operators, appearance
interventions, and validity rules, while applying the 0.82 m / 0.30 m run-in
geometry established before KiNO-v4 confirmation collection by F38.
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
from scripts import isaac_collect_kinofail_confirmatory_pair_v9 as v9
from scripts.isaac_collect_kinofail_confirmatory_pair_v4 import _load_v4_wrapper
from scripts.isaac_collect_kinofail_kino_v4_confirmation_pair_v1 import (
    EXPECTED_F1_SHA256,
    F1,
    _install_f1_nuisance_contract,
)


ROOT = Path(__file__).resolve().parents[1]
CURRENT = ROOT / "scripts/isaac_collect_kinofail_kino_v4_confirmation_pair_v1.py"
EXPECTED_CURRENT_SHA256 = "aa795945b1e2fa8b5821ccef0228ac3ecff27f98b19cde345f55335e55e60383"
RUNIN_CENTER_PROGRESS_M = 0.82
RUNIN_HALF_LENGTH_M = 0.30
FOOTPRINT_MARGIN_M = 0.35


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _install_runin_and_o8_qa(wrapper: Any) -> None:
    prior_load_v4 = wrapper._load_v4

    def load_v4() -> Any:
        module = prior_load_v4()
        prior_install_contract = module._install_reachable_exposure_contract

        def install_contract(implementation: Any) -> None:
            prior_install_contract(implementation)

            def runin_region(
                frame: Any,
                *,
                progress_m: float = RUNIN_CENTER_PROGRESS_M,
                half_length_m: float = RUNIN_HALF_LENGTH_M,
            ) -> Any:
                from kino_vla.utils.geometry import Rect

                progress = min(
                    progress_m, frame.route_length_m - half_length_m - 0.10
                )
                progress = max(progress, half_length_m + 0.10)
                half_width = min(0.40, 0.5 * frame.surface_width_m - 0.05)
                center = frame.point(progress, 0.0)
                if abs(float(frame.direction[0])) >= 1.0 - 1.0e-6:
                    return Rect(
                        float(center[0]),
                        float(center[1]),
                        half_length_m,
                        half_width,
                    )
                if abs(float(frame.direction[1])) >= 1.0 - 1.0e-6:
                    return Rect(
                        float(center[0]),
                        float(center[1]),
                        half_width,
                        half_length_m,
                    )
                raise ValueError("KiNO-v4 T3 run-in requires an axis-aligned route")

            implementation._region = runin_region
            prior_install_operator = implementation._install_operator
            prior_telemetry = implementation._telemetry

            def install_operator(
                backend: Any,
                record: dict[str, Any],
                frame: Any,
                seed: int,
            ) -> Any:
                operator, region = prior_install_operator(
                    backend, record, frame, seed
                )
                if (
                    operator is not None
                    and record["target_operator"] == "O8_invisible_collider"
                ):
                    operator._kino_v4_o8_footprint_exposure_steps = 0
                    prior_on_step = operator.on_step

                    def footprint_tracked_on_step(
                        runtime_backend: Any, timestamp_s: float
                    ) -> Any:
                        observation = runtime_backend._make_obs()
                        reached = (
                            abs(float(observation.pos[0]) - region.cx)
                            <= region.hx + FOOTPRINT_MARGIN_M
                            and abs(float(observation.pos[1]) - region.cy)
                            <= region.hy + FOOTPRINT_MARGIN_M
                        )
                        if reached:
                            operator._kino_v4_o8_footprint_exposure_steps += 1
                        return prior_on_step(runtime_backend, timestamp_s)

                    operator.on_step = footprint_tracked_on_step
                return operator, region

            def telemetry(
                backend: Any, operator_id: str, operator: Any
            ) -> dict[str, Any]:
                value = prior_telemetry(backend, operator_id, operator)
                if operator_id == "O8_invisible_collider" and operator is not None:
                    value = {
                        **value,
                        "scale_region_exposure_steps": int(
                            getattr(
                                operator,
                                "_kino_v4_o8_footprint_exposure_steps",
                                0,
                            )
                        ),
                        "scale_region_exposure_measurement": (
                            "go2_footprint_margin_0.35m"
                        ),
                        "scale_region_exposure_margin_m": FOOTPRINT_MARGIN_M,
                    }
                return value

            implementation._install_operator = install_operator
            implementation._telemetry = telemetry

        module._install_reachable_exposure_contract = install_contract
        return module

    wrapper._load_v4 = load_v4


def main() -> None:
    try:
        for path, expected in (
            (CURRENT, EXPECTED_CURRENT_SHA256),
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
        _install_runin_and_o8_qa(wrapper)
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
