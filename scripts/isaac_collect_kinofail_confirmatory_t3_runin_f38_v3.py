#!/usr/bin/env python3
"""F38 run-in v3 with Go2-footprint-aligned O8 exposure QA."""

from __future__ import annotations

import hashlib
from pathlib import Path
from typing import Any

from scripts import isaac_collect_kinofail_confirmatory_t3_runin_f38 as base


ROOT = Path(__file__).resolve().parents[1]
PREDECESSOR = ROOT / "scripts/isaac_collect_kinofail_confirmatory_t3_runin_f38.py"
EXPECTED_PREDECESSOR_SHA256 = "2236ca71c30ca8196872150bbe1c81a5394d36c6aba72648bf19e42b9f857ef4"
FOOTPRINT_MARGIN_M = 0.35


def install_runin_and_o8_qa(wrapper: Any) -> None:
    prior_load_v4 = wrapper._load_v4

    def load_v4() -> Any:
        module = prior_load_v4()
        prior_install_contract = module._install_reachable_exposure_contract

        def install_contract(implementation: Any) -> None:
            prior_install_contract(implementation)

            def runin_region(frame: Any, *, progress_m: float = 0.82, half_length_m: float = 0.30) -> Any:
                from kino_vla.utils.geometry import Rect

                progress = min(progress_m, frame.route_length_m - half_length_m - 0.10)
                progress = max(progress, half_length_m + 0.10)
                half_width = min(0.40, 0.5 * frame.surface_width_m - 0.05)
                center = frame.point(progress, 0.0)
                if abs(float(frame.direction[0])) >= 1.0 - 1.0e-6:
                    return Rect(float(center[0]), float(center[1]), half_length_m, half_width)
                if abs(float(frame.direction[1])) >= 1.0 - 1.0e-6:
                    return Rect(float(center[0]), float(center[1]), half_width, half_length_m)
                raise ValueError("F38-v3 requires an axis-aligned audited route")

            implementation._region = runin_region
            prior_install_operator = implementation._install_operator
            prior_telemetry = implementation._telemetry

            def install_operator(backend: Any, record: dict[str, Any], frame: Any, seed: int) -> Any:
                operator, region = prior_install_operator(backend, record, frame, seed)
                if operator is not None and record["target_operator"] == "O8_invisible_collider":
                    operator._f38_o8_footprint_exposure_steps = 0
                    prior_on_step = operator.on_step

                    def footprint_tracked_on_step(runtime_backend: Any, timestamp_s: float) -> Any:
                        observation = runtime_backend._make_obs()
                        reached = (
                            abs(float(observation.pos[0]) - region.cx) <= region.hx + FOOTPRINT_MARGIN_M
                            and abs(float(observation.pos[1]) - region.cy) <= region.hy + FOOTPRINT_MARGIN_M
                        )
                        if reached:
                            operator._f38_o8_footprint_exposure_steps += 1
                        return prior_on_step(runtime_backend, timestamp_s)

                    operator.on_step = footprint_tracked_on_step
                return operator, region

            def telemetry(backend: Any, operator_id: str, operator: Any) -> dict[str, Any]:
                value = prior_telemetry(backend, operator_id, operator)
                if operator_id == "O8_invisible_collider" and operator is not None:
                    value = {
                        **value,
                        "scale_region_exposure_steps": int(
                            getattr(operator, "_f38_o8_footprint_exposure_steps", 0)
                        ),
                        "scale_region_exposure_measurement": "go2_footprint_margin_0.35m",
                        "scale_region_exposure_margin_m": FOOTPRINT_MARGIN_M,
                    }
                return value

            implementation._install_operator = install_operator
            implementation._telemetry = telemetry

        module._install_reachable_exposure_contract = install_contract
        return module

    wrapper._load_v4 = load_v4


def main() -> None:
    if hashlib.sha256(PREDECESSOR.read_bytes()).hexdigest() != EXPECTED_PREDECESSOR_SHA256:
        raise RuntimeError("F38-v3 predecessor collector drift")
    base.RUNIN_CENTER_PROGRESS_M = 0.82
    base.RUNIN_HALF_LENGTH_M = 0.30
    base.install_runin_region = install_runin_and_o8_qa
    base.main()


if __name__ == "__main__":
    main()
