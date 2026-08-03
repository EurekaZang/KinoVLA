#!/usr/bin/env python3
"""Scale-v8 O3 severe/profile-1 repair with a uniform pre-operator observation window.

The severe collapse mechanism can correctly trigger and terminate an episode before the frozen
0.5 s model-input history exists.  This collector changes only the activation time for the
predeclared nine-scene O3/severe/profile-1 cell: locomotion and recording begin normally, while
the collapse operator is installed after 30 control steps (0.6 s).  Physics parameters, route,
controller, camera, appearance, pairing, and physical nuisance remain unchanged.
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
PRE_OPERATOR_STEPS = 30
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
        raise RuntimeError("O3 temporal-repair dependency mismatch: v4")
    spec = importlib.util.spec_from_file_location("kinofail_o3_temporal_repair_v4", V4)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"cannot load {V4}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _load_implementation():
    if _sha(V1) != EXPECTED_V1_SHA256:
        raise RuntimeError("O3 temporal-repair dependency mismatch: v1")
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
        "    operator, region = _install_operator(backend, record, frame, seed)\n": (
            "    delayed_operator_install = (\n"
            '        record["target_operator"] == "O3_collapse"\n'
            '        and record["condition"] == "anomaly"\n'
            "    )\n"
            "    if delayed_operator_install:\n"
            "        operator, region = None, _region(frame)\n"
            "    else:\n"
            "        operator, region = _install_operator(backend, record, frame, seed)\n"
        ),
        "    while step < 360 and not obs.fallen:\n        if operator is not None:\n": (
            "    while step < 360 and not obs.fallen:\n"
            "        if delayed_operator_install and operator is None and step == "
            f"{PRE_OPERATOR_STEPS}:\n"
            "            operator, region = _install_operator(backend, record, frame, seed)\n"
            "        if operator is not None:\n"
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
            '    manifest["collection"]["o3_temporal_repair"] = {\n'
            '        "pair_shared_pre_operator_steps": 30,\n'
            '        "pre_operator_duration_s": 0.6,\n'
            '        "selection_cell": "O3 severe / physical nuisance profile 1 / all scenes",\n'
            "    }\n"
            "    write_collected_manifest(manifest, episode_dir)\n"
        ),
    }
    for old, new in replacements.items():
        if source.count(old) != 1:
            raise RuntimeError(f"O3 temporal-repair patch point is not unique: {old!r}")
        source = source.replace(old, new)
    module = types.ModuleType("kinofail_o3_temporal_repair_impl")
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
        if any(
            len(values) != len(PHYSICAL_NUISANCE_PROFILES)
            for values in seeds_by_scene.values()
        ):
            raise RuntimeError("O3 temporal repair requires exactly five scene seeds per scene")
        index_by_scene_seed.clear()
        for scene, seeds in sorted(seeds_by_scene.items()):
            for index, seed in enumerate(sorted(seeds)):
                index_by_scene_seed[(scene, seed)] = index
        pair = prior_pair(schedule, pair_id)
        representative = pair[0]
        key = (
            str(representative["scene_family"]),
            int(representative["scene_seed"]),
        )
        if (
            representative["target_operator"] != "O3_collapse"
            or representative["severity_id"] != "severe"
            or index_by_scene_seed.get(key) != 1
        ):
            raise RuntimeError("pair is outside the frozen O3 severe/profile-1 repair cell")
        return pair

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


def main() -> None:
    v4 = _load_v4()
    v4._install_runtime_validation_adapter()
    implementation = _load_implementation()
    _install_balanced_nuisance_contract(implementation)
    v4._install_reachable_exposure_contract(implementation)
    implementation.__file__ = __file__
    implementation.main()


if __name__ == "__main__":
    main()
