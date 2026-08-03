#!/usr/bin/env python3
"""Frozen confirmatory pair collector with explicit assets and nuisance inputs.

The implementation reuses the previously audited scale collector by exact
content hash, then applies four deterministic adapters before execution:

* the confirmatory material lock is an explicit required argument;
* the schedule supplies the complete physical nuisance profile;
* RGB is sampled at 10 Hz as in scale-v8;
* reachable-region and route-aligned O9 contracts are retained.

No model, feature extractor, label prediction, or endpoint statistic is loaded.
"""

from __future__ import annotations

import hashlib
import importlib.util
import sys
import types
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
V1 = ROOT / "scripts/isaac_collect_kinofail_realistic_pair_v1.py"
V4 = ROOT / "scripts/isaac_collect_kinofail_realistic_pair_v4.py"
EXPECTED_V1_SHA256 = (
    "3f1b86cfda574dc4185375a6653e7b8ba71be37710d28041033e1643de0f820a"
)
EXPECTED_V4_SHA256 = (
    "7f8956a162ffd2cda6a5207efefc1cd87805df9d48881072f555c13dc17985b0"
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


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _load_v4() -> Any:
    if _sha256(V4) != EXPECTED_V4_SHA256:
        raise RuntimeError("confirmatory collector dependency mismatch: v4")
    spec = importlib.util.spec_from_file_location(
        "kinofail_confirmatory_v4", V4
    )
    if spec is None or spec.loader is None:
        raise RuntimeError(f"cannot load {V4}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _load_confirmatory_implementation() -> Any:
    if _sha256(V1) != EXPECTED_V1_SHA256:
        raise RuntimeError("confirmatory collector dependency mismatch: v1")
    source = V1.read_text(encoding="utf-8")
    replacements = {
        '    preliminary.add_argument("--corpus-root", required=True)\n': (
            '    preliminary.add_argument("--corpus-root", required=True)\n'
            '    preliminary.add_argument("--asset-lock", required=True)\n'
        ),
        '    parser.add_argument("--corpus-root", default=pre.corpus_root)\n': (
            '    parser.add_argument("--corpus-root", default=pre.corpus_root)\n'
            '    parser.add_argument("--asset-lock", default=pre.asset_lock)\n'
        ),
        (
            '        asset_lock_path = ROOT / '
            '"outputs/assets/terrain_pbr_v1/terrain_assets.lock.json"\n'
        ): "        asset_lock_path = (ROOT / args.asset_lock).resolve()\n",
        "    capture_stride = 20\n": "    capture_stride = 5\n",
        (
            '    seed = int(record["operator_seed"])\n'
            "    obs = backend.deep_reset(seed)\n"
        ): (
            "    nuisance = _physical_nuisance(record)\n"
            "    backend._start_pos = frame.point(\n"
            '        float(nuisance["start_progress_m"]),\n'
            '        float(nuisance["start_lateral_offset_m"]),\n'
            "    )\n"
            "    backend._start_heading = (\n"
            '        float(frame.heading_rad)\n'
            '        + float(nuisance["start_heading_offset_rad"])\n'
            "    )\n"
            '    seed = int(nuisance["physics_seed"])\n'
            "    obs = backend.deep_reset(seed)\n"
        ),
        "            velocity_body_xy_mps=obs.vel_body, forward_speed_mps=0.24,\n": (
            "            velocity_body_xy_mps=obs.vel_body,\n"
            '            forward_speed_mps=float(nuisance["forward_speed_mps"]),\n'
        ),
        "            target_lateral_offset_m=0.0,\n": (
            '            target_lateral_offset_m=float(\n'
            '                nuisance["controller_target_lateral_offset_m"]\n'
            "            ),\n"
        ),
        "    write_collected_manifest(manifest, episode_dir)\n": (
            '    manifest["collection"]["physical_nuisance"] = dict(nuisance)\n'
            "    write_collected_manifest(manifest, episode_dir)\n"
        ),
    }
    for old, new in replacements.items():
        if source.count(old) != 1:
            raise RuntimeError(
                f"confirmatory collector patch point is not unique: {old!r}"
            )
        source = source.replace(old, new)
    module = types.ModuleType("kinofail_confirmatory_pair_implementation")
    module.__file__ = str(V1)
    module.__package__ = "scripts"
    exec(compile(source, str(V1), "exec"), module.__dict__)
    return module


def _install_nuisance_contract(implementation: Any) -> None:
    def physical_nuisance(record: dict[str, Any]) -> dict[str, Any]:
        value = record.get("physical_nuisance")
        if not isinstance(value, dict) or set(value) != NUISANCE_KEYS:
            raise RuntimeError(
                "confirmatory schedule must provide the exact physical nuisance contract"
            )
        nuisance = dict(value)
        if nuisance["pair_shared"] is not True:
            raise RuntimeError("physical nuisance must be shared within each pair")
        if int(nuisance["physics_seed"]) != int(record["operator_seed"]):
            raise RuntimeError("operator_seed must equal frozen physics_seed")
        if not 0.0 <= float(nuisance["start_progress_m"]) <= 0.14:
            raise RuntimeError("confirmatory start progress is outside the frozen band")
        if abs(float(nuisance["start_lateral_offset_m"])) > 0.05:
            raise RuntimeError("confirmatory lateral nuisance is outside the frozen band")
        if abs(float(nuisance["start_heading_offset_rad"])) > 0.035:
            raise RuntimeError("confirmatory heading nuisance is outside the frozen band")
        if not 0.21 <= float(nuisance["forward_speed_mps"]) <= 0.27:
            raise RuntimeError("confirmatory speed nuisance is outside the frozen band")
        return nuisance

    implementation._physical_nuisance = physical_nuisance


def _install_route_aligned_o9(implementation: Any) -> None:
    prior_install = implementation._install_operator

    def install_operator(backend: Any, record: dict[str, Any], frame: Any, seed: int):
        if (
            record["target_operator"] != "O9_high_centering"
            or record["condition"] != "anomaly"
        ):
            return prior_install(backend, record, frame, seed)
        if record["physical_realization"] != "rounded_ridge":
            raise RuntimeError("confirmatory O9 alignment expects rounded_ridge")

        import kino_vla.sim.operators as operators

        original = operators.HighCentering

        def route_aligned_high_centering(
            region,
            residual_support=0.15,
            *,
            ridge_height_m=None,
            ridge_width_m=None,
            geometry_kind="box_ridge",
        ):
            if geometry_kind != "rounded_ridge":
                raise RuntimeError(f"unexpected O9 geometry {geometry_kind}")
            if abs(float(frame.direction[0])) >= 1.0 - 1.0e-6:
                actual_kind = "rounded_ridge"
            elif abs(float(frame.direction[1])) >= 1.0 - 1.0e-6:
                actual_kind = "longitudinal_rounded_ridge"
            else:
                raise ValueError(
                    "confirmatory collector requires an axis-aligned audited route"
                )
            return original(
                region,
                residual_support,
                ridge_height_m=ridge_height_m,
                ridge_width_m=ridge_width_m,
                geometry_kind=actual_kind,
            )

        operators.HighCentering = route_aligned_high_centering
        try:
            return prior_install(backend, record, frame, seed)
        finally:
            operators.HighCentering = original

    implementation._install_operator = install_operator


def main() -> None:
    v4 = _load_v4()
    v4._install_runtime_validation_adapter()
    implementation = _load_confirmatory_implementation()
    _install_nuisance_contract(implementation)
    v4._install_reachable_exposure_contract(implementation)
    _install_route_aligned_o9(implementation)
    implementation.__file__ = __file__
    implementation.main()


if __name__ == "__main__":
    main()
