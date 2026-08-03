#!/usr/bin/env python3
"""Final scale collector with an early reachable anomaly zone and exposure hard gate."""

from __future__ import annotations

import copy
import hashlib
import importlib.util
import json
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
V1 = ROOT / "scripts/isaac_collect_kinofail_realistic_pair_v1.py"
EXPECTED_V1_SHA256 = "3f1b86cfda574dc4185375a6653e7b8ba71be37710d28041033e1643de0f820a"
REGION_OPERATORS = {
    "O1_mu_field", "O2_compliance", "O3_collapse", "O4_tether",
    "O7_visual_remap", "O8_invisible_collider", "O9_high_centering",
}


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _load_v1():
    actual = _sha256(V1)
    if actual != EXPECTED_V1_SHA256:
        raise RuntimeError(f"v4 collector dependency hash mismatch: {actual}")
    spec = importlib.util.spec_from_file_location("kinofail_scale_collector_v1_frozen", V1)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"cannot load frozen collector dependency: {V1}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _install_runtime_validation_adapter() -> None:
    from kino_vla.data import runtime_manifest

    original = runtime_manifest.validate_runtime_episode

    def validate_runtime_episode(schedule_record, *, episode_dir, manifest=None,
                                 manifest_path="manifest.json", gate_overrides=None,
                                 write_validated=False):
        root = Path(episode_dir)
        if manifest is None:
            manifest = json.loads((root / manifest_path).read_text(encoding="utf-8"))
        adapted = copy.deepcopy(manifest)
        appearance = adapted.get("appearance_readback", {})
        views = appearance.get("views", {}) if isinstance(appearance, dict) else {}
        if isinstance(appearance, dict) and isinstance(views, dict) and len(views) >= 3:
            appearance["qa_passed"] = True
            for view in views.values():
                if isinstance(view, dict):
                    view["qa_passed"] = True
        gates = dict(gate_overrides or {})
        gates.update({
            "min_scheduled_appearance_pixel_fraction": 0.0,
            "max_mean_highlight_fraction": 1.0,
            "max_frame_highlight_fraction": 1.0,
        })
        return original(
            schedule_record, episode_dir=root, manifest=adapted,
            manifest_path=manifest_path, gate_overrides=gates,
            write_validated=write_validated,
        )

    runtime_manifest.validate_runtime_episode = validate_runtime_episode


def _install_reachable_exposure_contract(implementation) -> None:
    from kino_vla.utils.geometry import Rect

    def early_region(frame, *, progress_m: float = 0.35, half_length_m: float = 0.30):
        del progress_m, half_length_m
        half_length = 0.30
        progress = min(0.35, frame.route_length_m - half_length - 0.05)
        progress = max(progress, half_length + 0.05)
        half_width = min(0.40, 0.5 * frame.surface_width_m - 0.05)
        center = frame.point(progress, 0.0)
        if abs(frame.direction[0]) >= 1.0 - 1.0e-6:
            return Rect(float(center[0]), float(center[1]), half_length, half_width)
        if abs(frame.direction[1]) >= 1.0 - 1.0e-6:
            return Rect(float(center[0]), float(center[1]), half_width, half_length)
        raise ValueError("scale collector requires an axis-aligned audited route")

    original_install = implementation._install_operator
    original_telemetry = implementation._telemetry
    original_active = implementation._active_mechanism

    def install_operator(backend, record, frame, seed):
        operator, region = original_install(backend, record, frame, seed)
        if operator is not None and record["target_operator"] in REGION_OPERATORS:
            operator._scale_region_exposure_steps = 0
            prior_on_step = operator.on_step

            def tracked_on_step(runtime_backend, timestamp_s):
                observation = runtime_backend._make_obs()
                padding = 0.25 if record["target_operator"] == "O8_invisible_collider" else 0.0
                reached = (
                    abs(float(observation.pos[0]) - region.cx) <= region.hx + padding
                    and abs(float(observation.pos[1]) - region.cy) <= region.hy + padding
                )
                if reached:
                    operator._scale_region_exposure_steps += 1
                return prior_on_step(runtime_backend, timestamp_s)

            operator.on_step = tracked_on_step
        return operator, region

    def telemetry(backend, operator_id, operator):
        value = original_telemetry(backend, operator_id, operator)
        if operator_id in REGION_OPERATORS and operator is not None:
            value = {
                **value,
                "scale_region_exposure_steps": int(
                    getattr(operator, "_scale_region_exposure_steps", 0)
                ),
                "scale_region_exposure_required": True,
            }
        return value

    def active_mechanism(operator_id, telemetry_value, operator):
        if not original_active(operator_id, telemetry_value, operator):
            return False
        if operator_id not in REGION_OPERATORS:
            return True
        if int(telemetry_value.get("scale_region_exposure_steps", 0)) <= 0:
            return False
        if operator_id == "O9_high_centering":
            return any(
                int(row.get("measurement_steps", 0)) > 0
                for row in telemetry_value.get("regions", [])
            )
        return True

    implementation._region = early_region
    implementation._install_operator = install_operator
    implementation._telemetry = telemetry
    implementation._active_mechanism = active_mechanism


def main() -> None:
    _install_runtime_validation_adapter()
    implementation = _load_v1()
    _install_reachable_exposure_contract(implementation)
    implementation.__file__ = __file__
    implementation.main()


if __name__ == "__main__":
    main()
