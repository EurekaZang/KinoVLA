#!/usr/bin/env python3
"""Scale-v8 collector with paired, predeclared physical nuisance profiles.

The five scheduled scene seeds previously changed appearance and O11 noise, but the inference
reset deliberately fixes Go2 joint pose.  This collector consumes those five seeds as five
balanced route-entry profiles so the replication axis is physical rather than visual-only.
Operator parameters, scene geometry, controller gains and nominal/anomaly pairing stay unchanged.
"""

from __future__ import annotations

import hashlib
import importlib.util
import types
from collections import defaultdict
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
V1 = ROOT / "scripts/isaac_collect_kinofail_realistic_pair_v1.py"
V4 = ROOT / "scripts/isaac_collect_kinofail_realistic_pair_v4.py"
EXPECTED_V1_SHA256 = "3f1b86cfda574dc4185375a6653e7b8ba71be37710d28041033e1643de0f820a"
EXPECTED_V4_SHA256 = "7f8956a162ffd2cda6a5207efefc1cd87805df9d48881072f555c13dc17985b0"

# Each scene receives every profile exactly once at every operator/severity cell.  Values are
# intentionally small relative to the audited route width and nominal 0.24 m/s command.
PHYSICAL_NUISANCE_PROFILES = (
    {
        "profile_index": 0,
        "start_progress_m": 0.00,
        "start_lateral_offset_m": -0.035,
        "start_heading_offset_rad": 0.025,
        "forward_speed_mps": 0.24,
        "controller_target_lateral_offset_m": 0.0,
    },
    {
        "profile_index": 1,
        "start_progress_m": 0.03,
        "start_lateral_offset_m": -0.0175,
        "start_heading_offset_rad": -0.0125,
        "forward_speed_mps": 0.26,
        "controller_target_lateral_offset_m": 0.0,
    },
    {
        "profile_index": 2,
        "start_progress_m": 0.06,
        "start_lateral_offset_m": 0.0,
        "start_heading_offset_rad": 0.0,
        "forward_speed_mps": 0.22,
        "controller_target_lateral_offset_m": 0.0,
    },
    {
        "profile_index": 3,
        "start_progress_m": 0.09,
        "start_lateral_offset_m": 0.0175,
        "start_heading_offset_rad": 0.0125,
        "forward_speed_mps": 0.25,
        "controller_target_lateral_offset_m": 0.0,
    },
    {
        "profile_index": 4,
        "start_progress_m": 0.12,
        "start_lateral_offset_m": 0.035,
        "start_heading_offset_rad": -0.025,
        "forward_speed_mps": 0.23,
        "controller_target_lateral_offset_m": 0.0,
    },
)


def _sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _load_v4():
    if _sha(V4) != EXPECTED_V4_SHA256:
        raise RuntimeError("scale-v8 dependency mismatch: v4")
    spec = importlib.util.spec_from_file_location("kinofail_scale_v8_v4", V4)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"cannot load {V4}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _load_10hz_nuisance_implementation():
    if _sha(V1) != EXPECTED_V1_SHA256:
        raise RuntimeError("scale-v8 dependency mismatch: v1")
    source = V1.read_text(encoding="utf-8")
    replacements = {
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
            '        float(frame.heading_rad) + float(nuisance["start_heading_offset_rad"])\n'
            "    )\n"
            '    seed = int(record["operator_seed"])\n'
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
            raise RuntimeError(f"scale-v8 patch point is not unique: {old!r}")
        source = source.replace(old, new)
    module = types.ModuleType("kinofail_scale_v8_10hz_nuisance")
    module.__file__ = str(V1)
    module.__package__ = "scripts"
    exec(compile(source, str(V1), "exec"), module.__dict__)
    return module


def _install_balanced_nuisance_contract(implementation) -> None:
    prior_pair = implementation._pair
    index_by_scene_seed: dict[tuple[str, int], int] = {}

    def paired_rows(schedule, pair_id):
        seeds_by_scene: dict[str, set[int]] = defaultdict(set)
        for row in schedule:
            seeds_by_scene[str(row["scene_family"])].add(int(row["scene_seed"]))
        if any(len(values) != len(PHYSICAL_NUISANCE_PROFILES) for values in seeds_by_scene.values()):
            raise RuntimeError("scale-v8 requires exactly five scene seeds per scene")
        index_by_scene_seed.clear()
        for scene, seeds in sorted(seeds_by_scene.items()):
            for index, seed in enumerate(sorted(seeds)):
                index_by_scene_seed[(scene, seed)] = index
        return prior_pair(schedule, pair_id)

    def physical_nuisance(record):
        key = (str(record["scene_family"]), int(record["scene_seed"]))
        if key not in index_by_scene_seed:
            raise RuntimeError(f"unregistered physical nuisance seed: {key}")
        profile = dict(PHYSICAL_NUISANCE_PROFILES[index_by_scene_seed[key]])
        profile.update(
            {
                "scene_seed": int(record["scene_seed"]),
                "operator_seed": int(record["operator_seed"]),
                "pair_shared": True,
                "derivation": "rank(scene_seed within frozen five-seed scene block)",
            }
        )
        return profile

    implementation._pair = paired_rows
    implementation._physical_nuisance = physical_nuisance


def _install_route_aligned_o9(implementation) -> None:
    prior_install = implementation._install_operator

    def install_operator(backend, record, frame, seed):
        if record["target_operator"] != "O9_high_centering" or record["condition"] != "anomaly":
            return prior_install(backend, record, frame, seed)
        if record["physical_realization"] != "rounded_ridge":
            raise RuntimeError("scale-v8 O9 alignment expects rounded_ridge")

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
                raise ValueError("scale-v8 requires an axis-aligned audited route")
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
    implementation = _load_10hz_nuisance_implementation()
    _install_balanced_nuisance_contract(implementation)
    v4._install_reachable_exposure_contract(implementation)
    _install_route_aligned_o9(implementation)
    implementation.__file__ = __file__
    implementation.main()


if __name__ == "__main__":
    main()
