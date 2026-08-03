#!/usr/bin/env python3
"""Compile F1-bound, scene-sharded confirmatory Scale and C2 schedules.

The compiler is a design-only operation: it does not start Isaac, generate a
scene, download an asset, extract a feature, or run model inference.  It binds
every schedule and snapshot protocol to the F0 manifest, realized F1 registry,
new material lock, collector, and runtime validator by SHA-256.  Publication is
atomic and an existing output path is always an error.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import shutil
import tempfile
from collections import defaultdict
from collections.abc import Iterable, Mapping, Sequence
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import yaml


ROOT = Path(__file__).resolve().parents[1]
DESIGN_PATH = ROOT / "configs/data/kinofail_unified_confirmatory_design_v1.json"
F0_PATH = ROOT / "outputs/freeze/unified_moe_v3_confirmatory_f0/freeze_manifest.json"
REGISTRY_PATH = ROOT / "outputs/kinofail_confirmatory_v1/scene_registry.json"
LOCK_PATH = ROOT / "outputs/assets/terrain_pbr_confirmatory_v1/terrain_assets.lock.json"
COLLECTOR_PATH = ROOT / "scripts/isaac_collect_kinofail_confirmatory_pair_v1.py"
RUNTIME_MANIFEST_PATH = ROOT / "kino_vla/data/runtime_manifest.py"
BASE_OPERATOR_PATH = ROOT / "configs/data/kinofail_realistic.yaml"
T2_COLLECTOR_PATH = (
    ROOT / "scripts/isaac_collect_kinofail_confirmatory_t2_v1.py"
)
T2_VISUAL_HELPER_PATH = ROOT / "kino_vla/sim/c1_causal_visuals.py"
DEFAULT_OUTPUT = ROOT / "outputs/kinofail_confirmatory_v1/schedules"
CONDITIONS = ("nominal_counterfactual", "anomaly")
CAMERAS = ("go2_front_calib_a", "go2_front_calib_b", "go2_front_calib_c")
STATES = {
    "life": ("clean", "dusty", "scuffed", "damp"),
    "production": ("clean", "dusty", "oil_stained", "wet"),
    "wild": ("dry", "damp", "leaf_litter", "muddy"),
}
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
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _json(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise TypeError(f"expected JSON object: {path}")
    return value


def _stable_id(prefix: str, *parts: object) -> str:
    payload = "\x1f".join(str(part) for part in parts).encode("utf-8")
    return f"{prefix}_{hashlib.sha256(payload).hexdigest()[:20]}"


def _display(path: Path, *, root: Path) -> str:
    resolved = path.resolve()
    try:
        return str(resolved.relative_to(root.resolve()))
    except ValueError:
        return str(resolved)


def _write_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(value, indent=2, sort_keys=True, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )


def _write_jsonl(path: Path, rows: Iterable[Mapping[str, Any]]) -> int:
    path.parent.mkdir(parents=True, exist_ok=True)
    count = 0
    with path.open("x", encoding="utf-8") as stream:
        for row in rows:
            stream.write(
                json.dumps(
                    dict(row), sort_keys=True, separators=(",", ":"), ensure_ascii=False
                )
                + "\n"
            )
            count += 1
    return count


def _normalize(value: Any) -> Any:
    if isinstance(value, bool):
        return value
    if isinstance(value, (int, float)):
        number = float(value)
        if not math.isfinite(number):
            raise ValueError("operator parameter is not finite")
        return round(number, 12)
    if isinstance(value, Mapping):
        return {
            str(key): _normalize(item)
            for key, item in sorted(value.items(), key=lambda pair: str(pair[0]))
        }
    if isinstance(value, Sequence) and not isinstance(value, (str, bytes, bytearray)):
        return [_normalize(item) for item in value]
    return value


def interpolate_parameters(moderate: Any, severe: Any, value: float) -> Any:
    """Apply the frozen componentwise M--S interpolation."""

    if isinstance(moderate, Mapping) and isinstance(severe, Mapping):
        if set(moderate) != set(severe):
            raise ValueError("M/S parameter keys differ")
        return {
            str(key): interpolate_parameters(moderate[key], severe[key], value)
            for key in sorted(moderate)
        }
    if (
        isinstance(moderate, Sequence)
        and not isinstance(moderate, (str, bytes, bytearray))
        and isinstance(severe, Sequence)
        and not isinstance(severe, (str, bytes, bytearray))
    ):
        if len(moderate) != len(severe):
            raise ValueError("M/S array lengths differ")
        return [
            interpolate_parameters(left, right, value)
            for left, right in zip(moderate, severe, strict=True)
        ]
    if (
        isinstance(moderate, (int, float))
        and not isinstance(moderate, bool)
        and isinstance(severe, (int, float))
        and not isinstance(severe, bool)
    ):
        return round((1.0 - value) * float(moderate) + value * float(severe), 12)
    if moderate != severe:
        raise ValueError("non-numeric M/S values differ")
    return moderate


def _operator_specs(path: Path) -> list[dict[str, Any]]:
    value = yaml.safe_load(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise TypeError(path)
    operators = []
    for raw in value["operators"]:
        operator = dict(raw)
        moderate = next(
            dict(row["parameters"])
            for row in operator["severities"]
            if row["id"] == "moderate"
        )
        severe = next(
            dict(row["parameters"])
            for row in operator["severities"]
            if row["id"] == "severe"
        )
        nominal = dict(operator["nominal"])
        physical_realization = str(operator["physical_realizations"][0])
        geometry_profile = str(operator["geometry_profiles"][0])
        if operator["id"] == "O4_tether":
            nominal = {
                "attachment_enabled": 0.0,
                "tangential_force_cap_n": 0.0,
                "normal_force_cap_n": 0.0,
                "peel_height_m": 0.025,
                "unload_steps_to_peel": 3.0,
            }
            moderate = {
                "attachment_enabled": 1.0,
                "tangential_force_cap_n": 18.0,
                "normal_force_cap_n": 8.0,
                "peel_height_m": 0.025,
                "unload_steps_to_peel": 3.0,
            }
            severe = {
                "attachment_enabled": 1.0,
                "tangential_force_cap_n": 24.0,
                "normal_force_cap_n": 10.0,
                "peel_height_m": 0.025,
                "unload_steps_to_peel": 3.0,
            }
            physical_realization = "adhesive_foot_contact"
            geometry_profile = "irregular_protective_film"
        operators.append(
            {
                "id": str(operator["id"]),
                "category": str(operator["category"]),
                "nominal": nominal,
                "M": moderate,
                "S": severe,
                "physical_realization": physical_realization,
                "geometry_profile": geometry_profile,
                "counterfactual_inherit": list(
                    operator.get("counterfactual_inherit", [])
                ),
            }
        )
    return operators


def _counterfactual(
    operator: Mapping[str, Any], anomaly: Mapping[str, Any]
) -> dict[str, Any]:
    result = json.loads(json.dumps(operator["nominal"]))
    for key in operator.get("counterfactual_inherit", []):
        result[str(key)] = anomaly[str(key)]
    return result


def _scene_seed(scene: Mapping[str, Any]) -> int:
    for key in ("source_scene_seed", "metric_geometry_seed", "scene_seed"):
        value = scene.get(key)
        if isinstance(value, int):
            return int(value)
    raise ValueError(f"scene has no physical source seed: {scene.get('scene_id')}")


def _appearance_views(
    *,
    scene: Mapping[str, Any],
    primary_material: Mapping[str, Any],
    alternate_material: Mapping[str, Any],
    seed: int,
    appearance_key: str,
) -> list[dict[str, Any]]:
    # Both swaps use the frozen contrast material.  Independent continuous
    # UV/state/photometric transforms keep their rendered sequences distinct,
    # while the contrast-offset-five pairing avoids visually negligible
    # adjacent assets observed during the pre-F0 development smoke.
    materials = (primary_material, alternate_material, alternate_material)
    states = STATES[str(scene["domain"])]
    views = []
    for index, material in enumerate(materials):
        view_id = "primary" if index == 0 else f"swap_{index:02d}"
        unit = ((seed + index * 997) % 10007) / 10006.0
        unit_b = ((seed + index * 1597) % 9973) / 9972.0
        views.append(
            {
                "appearance_view_id": view_id,
                "appearance_view_index": index,
                "is_primary": index == 0,
                "appearance_id": _stable_id(
                    "app", scene["scene_id"], appearance_key, index, seed
                ),
                "appearance_seed": seed,
                "material_family": str(material["id"]),
                "material_asset_id": str(material["source_asset_id"]),
                "material_semantics": str(material["semantic_family"]),
                "material_source": str(material["source"]),
                "material_license": str(material["license"]),
                "physical_size_m": list(material["physical_size_m"]),
                "surface_state": states[(seed + index) % len(states)],
                "uv_scale": round(0.78 + 0.54 * unit, 6),
                "uv_rotation_deg": (0, 90, 180, 270)[(seed + index) % 4],
                "uv_offset": [round(unit, 6), round(unit_b, 6)],
                "albedo_brightness_multiplier": round(0.88 + 0.24 * unit_b, 6),
                "normal_strength": round(0.75 + 0.5 * unit, 6),
                "roughness_multiplier": round(0.82 + 0.36 * unit_b, 6),
                "render_tier": "structured_pbr",
                "triplanar": True,
                "visual_intervention_only": True,
            }
        )
    return views


def physical_nuisance(
    design: Mapping[str, Any], *, profile_index: int, physics_seed: int
) -> dict[str, Any]:
    table = design["scale"]["physical_nuisance"]["profile_table"]
    profile = dict(table[profile_index % len(table)])
    result = {
        "profile_index": int(profile["profile_index"]),
        "start_progress_m": float(profile["start_progress_m"]),
        "start_lateral_offset_m": float(profile["start_lateral_offset_m"]),
        "start_heading_offset_rad": float(profile["start_heading_offset_rad"]),
        "forward_speed_mps": float(profile["forward_speed_mps"]),
        "controller_target_lateral_offset_m": 0.0,
        "physics_seed": int(physics_seed),
        "pair_shared": True,
    }
    if set(result) != NUISANCE_KEYS:
        raise RuntimeError("physical nuisance key drift")
    return result


def _outputs(battery: str, material_id: str, operator: str, episode: str) -> dict[str, Any]:
    base = f"{battery}/{material_id}/{operator}/{episode}"
    return {
        "episode_manifest": f"{base}/manifest.json",
        "rgb": f"{base}/rgb/",
        "rgb_views": {
            "primary": f"{base}/rgb/",
            "swap_01": f"{base}/rgb_views/swap_01/",
            "swap_02": f"{base}/rgb_views/swap_02/",
        },
        "proprio": f"{base}/proprio.npz",
        "telemetry": f"{base}/privileged.jsonl",
        "depth_optional": f"{base}/depth/",
    }


def _scale_context_rows(
    *,
    design: Mapping[str, Any],
    f0: Mapping[str, Any],
    scene: Mapping[str, Any],
    context: Mapping[str, Any],
    materials: Mapping[str, Mapping[str, Any]],
    operators: Sequence[Mapping[str, Any]],
) -> list[dict[str, Any]]:
    scene_index = int(
        next(
            row["scene_index"]
            for row in f0["design"]["scenes"]
            if row["scene_id"] == scene["scene_id"]
        )
    )
    material_slot = int(context["material_slot"])
    context_index = int(context["context_index"])
    primary_id = str(context["material_id"])
    scene_material_ids = [str(value) for value in scene["material_ids"]]
    if primary_id not in scene_material_ids or len(scene_material_ids) != 2:
        raise RuntimeError("F1 material context does not match F0")
    alternate_id = next(value for value in scene_material_ids if value != primary_id)
    moderate = [float(value) for value in design["scale"]["moderate_lambda_points"]]
    hard = [float(value) for value in design["scale"]["hard_lambda_points"]]
    rows: list[dict[str, Any]] = []
    for operator_index, operator in enumerate(operators):
        operator_id = str(operator["id"])
        for replicate_index in range(16):
            band = "moderate" if replicate_index < 8 else "hard"
            point_index = replicate_index if replicate_index < 8 else replicate_index - 8
            interpolation_lambda = (
                moderate[point_index] if band == "moderate" else hard[point_index]
            )
            physical_seed = 2_030_000_000 + (
                ((scene_index * 2 + material_slot) * 11 + operator_index) * 16
                + replicate_index
            )
            expected_seed = 2_030_000_000 + context_index * 11 * 16
            expected_seed += operator_index * 16 + replicate_index
            if physical_seed != expected_seed:
                raise RuntimeError("F0 scale seed formula drift")
            nuisance = physical_nuisance(
                design, profile_index=replicate_index, physics_seed=physical_seed
            )
            anomaly = interpolate_parameters(
                operator["M"], operator["S"], interpolation_lambda
            )
            group_id = _stable_id(
                "cf",
                "confirmatory-v1",
                scene["scene_id"],
                primary_id,
                operator_id,
                replicate_index,
            )
            views = _appearance_views(
                scene=scene,
                primary_material=materials[primary_id],
                alternate_material=materials[alternate_id],
                seed=physical_seed,
                appearance_key=f"scale-{operator_id}-{replicate_index}",
            )
            for condition in CONDITIONS:
                active = condition == "anomaly"
                physics = anomaly if active else _counterfactual(operator, anomaly)
                episode_id = f"{group_id}_{'anomaly' if active else 'nominal'}"
                primary = views[0]
                rows.append(
                    {
                        "schema_version": "kinofail.unified-confirmatory-schedule.v1",
                        "benchmark_id": "kinofail_unified_confirmatory_v1",
                        "battery": "scale",
                        "design_mode": "independent_confirmation",
                        "artifact_state": "planned",
                        "evaluation_eligible": False,
                        "episode_id": episode_id,
                        "counterfactual_group_id": group_id,
                        "condition": condition,
                        "active_operator": operator_id if active else None,
                        "target_operator": operator_id,
                        "attribution_category": (
                            str(operator["category"]) if active else "nominal"
                        ),
                        "severity_id": band,
                        "severity_rank": 2 if band == "moderate" else 3,
                        "parameter_interpolation": {
                            "lambda": interpolation_lambda,
                            "point_index_within_band": point_index,
                            "rule": "P=(1-lambda)*M+lambda*S",
                        },
                        "physics_parameters": physics,
                        "physical_realization": str(operator["physical_realization"]),
                        "geometry_profile": str(operator["geometry_profile"]),
                        "geometry_id": (
                            f"{scene['scene_id']}::{operator['geometry_profile']}"
                        ),
                        "geometry_hash": str(scene["geometry_hash"]),
                        "scene_id": str(scene["scene_id"]),
                        "scene_index": scene_index,
                        "scene_family": str(scene["scene_id"]),
                        "source_scene_id": str(scene["source_scene_id"]),
                        "scene_source": str(scene["source"]),
                        "scene_seed": _scene_seed(scene),
                        "runtime_seed": int(
                            scene.get("runtime_seed", _scene_seed(scene))
                        ),
                        "operator_seed": physical_seed,
                        "physical_seed": physical_seed,
                        "physical_nuisance": nuisance,
                        "domain": str(scene["domain"]),
                        "split": "confirmatory",
                        "camera_profile": CAMERAS[scene_index % 3],
                        "material_id": primary_id,
                        "material_slot": material_slot,
                        "cluster_material": primary_id,
                        "operator_index": operator_index,
                        "replicate_index": replicate_index,
                        "appearance_view_count": 3,
                        "appearance_views": views,
                        "appearance_id": primary["appearance_id"],
                        "appearance_seed": primary["appearance_seed"],
                        "material_family": primary_id,
                        "material_asset_id": primary["material_asset_id"],
                        "source_material_asset_id": primary[
                            "material_asset_id"
                        ],
                        "material_semantics": primary["material_semantics"],
                        "material_source": primary["material_source"],
                        "material_license": primary["material_license"],
                        "material_physical_size_m": primary["physical_size_m"],
                        "surface_state": primary["surface_state"],
                        "uv_scale": primary["uv_scale"],
                        "uv_rotation_deg": primary["uv_rotation_deg"],
                        "uv_offset": primary["uv_offset"],
                        "albedo_brightness_multiplier": primary[
                            "albedo_brightness_multiplier"
                        ],
                        "normal_strength": primary["normal_strength"],
                        "roughness_multiplier": primary[
                            "roughness_multiplier"
                        ],
                        "render_tier": "structured_pbr",
                        "triplanar": True,
                        "texture_swap_group_id": _stable_id(
                            "ts", group_id, condition
                        ),
                        "required_outputs": _outputs(
                            "scale", primary_id, operator_id, episode_id
                        ),
                    }
                )
    return rows


def _t2_case(
    *,
    scene: Mapping[str, Any],
    context: Mapping[str, Any],
    materials: Mapping[str, Mapping[str, Any]],
    case_seed: int,
    local_case: int,
    context_index: int,
) -> dict[str, Any]:
    primary_id = str(context["material_id"])
    alternate_id = next(
        str(value) for value in scene["material_ids"] if str(value) != primary_id
    )
    profile = local_case % 16
    nuisance = physical_nuisance(
        _CURRENT_DESIGN, profile_index=profile, physics_seed=case_seed
    )
    case_id = _stable_id(
        "c2t2", scene["scene_id"], primary_id, local_case, case_seed
    )
    return {
        "schema_version": "kinofail.unified-confirmatory-c2-t2.v1",
        "benchmark_id": "kinofail_unified_confirmatory_v1",
        "cell": "T2_vision_decisive",
        "case_id": case_id,
        "case_seed": case_seed,
        "reset_seed": case_seed,
        "cause_visual_seed": case_seed,
        "scene_id": str(scene["scene_id"]),
        "scene_index": int(
            next(
                row["scene_index"]
                for row in _CURRENT_F0["design"]["scenes"]
                if row["scene_id"] == scene["scene_id"]
            )
        ),
        "scene_cluster": str(scene["scene_id"]),
        "scene_family": str(scene["scene_id"]),
        "source_scene_id": str(scene["source_scene_id"]),
        "scene_seed": _scene_seed(scene),
        "runtime_seed": int(scene.get("runtime_seed", _scene_seed(scene))),
        "domain": str(scene["domain"]),
        "split": "confirmatory",
        "material_id": primary_id,
        "cluster_material": primary_id,
        "material_slot": int(context["material_slot"]),
        "context_index": context_index,
        "local_case_index": local_case,
        "profile_index": profile,
        "camera_profile": CAMERAS[
            next(
                index
                for index, row in enumerate(_CURRENT_F0["design"]["scenes"])
                if row["scene_id"] == scene["scene_id"]
            )
            % 3
        ],
        "physical_nuisance": nuisance,
        "decision_progress_m": 0.22,
        "cause_region": {
            "start_progress_m": 0.9,
            "length_m": 0.62,
            "half_width_m": 0.38,
            "base_z_m": 0.006,
        },
        "causes": [
            {
                "target_operator": "O2_compliance",
                "attribution_category": "compliant_terrain",
                "visible_physical_cue": "soft_surface_deformation",
            },
            {
                "target_operator": "O4_tether",
                "attribution_category": "adhesion",
                "visible_physical_cue": "overlapping_film_and_creases",
            },
        ],
        "appearance_views": _appearance_views(
            scene=scene,
            primary_material=materials[primary_id],
            alternate_material=materials[alternate_id],
            seed=case_seed,
            appearance_key=f"t2-{local_case}",
        ),
        "shared_prefix_contract": {
            "collector": "shared_prefix_C1",
            "one_physics_rollout_for_both_causes": True,
            "visual_rerenders_do_not_advance_physics": True,
            "proprio_bytes_reused_without_relabeling_or_perturbation": True,
            "operator_physics_activates_only_after_the_decision_boundary": True,
        },
    }


def _t3_case_rows(
    *,
    scene: Mapping[str, Any],
    context: Mapping[str, Any],
    materials: Mapping[str, Mapping[str, Any]],
    operators: Mapping[str, Mapping[str, Any]],
    case_seed: int,
    local_case: int,
    context_index: int,
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    primary_id = str(context["material_id"])
    alternate_id = next(
        str(value) for value in scene["material_ids"] if str(value) != primary_id
    )
    profile = local_case % 16
    nuisance = physical_nuisance(
        _CURRENT_DESIGN, profile_index=profile, physics_seed=case_seed
    )
    band = "moderate" if local_case < 13 else "hard"
    point_index = (local_case if band == "moderate" else local_case - 13) % 8
    interpolation_lambda = float(
        _CURRENT_DESIGN["scale"][
            "moderate_lambda_points"
            if band == "moderate"
            else "hard_lambda_points"
        ][point_index]
    )
    case_id = _stable_id(
        "c2t3", scene["scene_id"], primary_id, local_case, case_seed
    )
    views = _appearance_views(
        scene=scene,
        primary_material=materials[primary_id],
        alternate_material=materials[alternate_id],
        seed=case_seed,
        appearance_key=f"t3-{local_case}",
    )
    rows: list[dict[str, Any]] = []
    groups: dict[str, str] = {}
    for operator_id in ("O7_visual_remap", "O8_invisible_collider"):
        operator = operators[operator_id]
        anomaly = interpolate_parameters(
            operator["M"], operator["S"], interpolation_lambda
        )
        group_id = _stable_id(
            "cf", "t3", scene["scene_id"], primary_id, local_case, operator_id
        )
        groups[operator_id] = group_id
        for condition in CONDITIONS:
            active = condition == "anomaly"
            episode_id = f"{group_id}_{'anomaly' if active else 'nominal'}"
            primary = views[0]
            rows.append(
                {
                    "schema_version": "kinofail.unified-confirmatory-schedule.v1",
                    "benchmark_id": "kinofail_unified_confirmatory_v1",
                    "battery": "c2_t3",
                    "cell": "T3_proprio_decisive",
                    "case_id": case_id,
                    "case_seed": case_seed,
                    "episode_id": episode_id,
                    "counterfactual_group_id": group_id,
                    "condition": condition,
                    "active_operator": operator_id if active else None,
                    "target_operator": operator_id,
                    "attribution_category": (
                        str(operator["category"]) if active else "nominal"
                    ),
                    "severity_id": band,
                    "severity_rank": 2 if band == "moderate" else 3,
                    "parameter_interpolation": {
                        "lambda": interpolation_lambda,
                        "point_index_within_band": point_index,
                        "rule": "P=(1-lambda)*M+lambda*S",
                    },
                    "physics_parameters": (
                        anomaly if active else _counterfactual(operator, anomaly)
                    ),
                    "physical_realization": str(
                        operator["physical_realization"]
                    ),
                    "geometry_profile": str(operator["geometry_profile"]),
                    "geometry_id": (
                        f"{scene['scene_id']}::{operator['geometry_profile']}"
                    ),
                    "geometry_hash": str(scene["geometry_hash"]),
                    "scene_id": str(scene["scene_id"]),
                    "scene_index": int(
                        next(
                            row["scene_index"]
                            for row in _CURRENT_F0["design"]["scenes"]
                            if row["scene_id"] == scene["scene_id"]
                        )
                    ),
                    "scene_family": str(scene["scene_id"]),
                    "scene_cluster": str(scene["scene_id"]),
                    "source_scene_id": str(scene["source_scene_id"]),
                    "scene_seed": _scene_seed(scene),
                    "runtime_seed": int(
                        scene.get("runtime_seed", _scene_seed(scene))
                    ),
                    "operator_seed": case_seed,
                    "physical_seed": case_seed,
                    "physical_nuisance": nuisance,
                    "domain": str(scene["domain"]),
                    "split": "confirmatory",
                    "camera_profile": CAMERAS[
                        next(
                            index
                            for index, row in enumerate(
                                _CURRENT_F0["design"]["scenes"]
                            )
                            if row["scene_id"] == scene["scene_id"]
                        )
                        % 3
                    ],
                    "material_id": primary_id,
                    "material_slot": int(context["material_slot"]),
                    "cluster_material": primary_id,
                    "local_case_index": local_case,
                    "appearance_view_count": 3,
                    "appearance_views": views,
                    "appearance_id": primary["appearance_id"],
                    "appearance_seed": primary["appearance_seed"],
                    "material_family": primary_id,
                    "material_asset_id": primary["material_asset_id"],
                    "source_material_asset_id": primary[
                        "material_asset_id"
                    ],
                    "material_semantics": primary["material_semantics"],
                    "material_source": primary["material_source"],
                    "material_license": primary["material_license"],
                    "material_physical_size_m": primary["physical_size_m"],
                    "surface_state": primary["surface_state"],
                    "uv_scale": primary["uv_scale"],
                    "uv_rotation_deg": primary["uv_rotation_deg"],
                    "uv_offset": primary["uv_offset"],
                    "albedo_brightness_multiplier": primary[
                        "albedo_brightness_multiplier"
                    ],
                    "normal_strength": primary["normal_strength"],
                    "roughness_multiplier": primary[
                        "roughness_multiplier"
                    ],
                    "render_tier": "structured_pbr",
                    "triplanar": True,
                    "texture_swap_group_id": _stable_id(
                        "ts", case_id, condition
                    ),
                    "required_outputs": _outputs(
                        "c2_t3", primary_id, operator_id, episode_id
                    ),
                }
            )
    case = {
        "schema_version": "kinofail.unified-confirmatory-c2-t3-case.v1",
        "benchmark_id": "kinofail_unified_confirmatory_v1",
        "cell": "T3_proprio_decisive",
        "case_id": case_id,
        "case_seed": case_seed,
        "scene_id": str(scene["scene_id"]),
        "scene_index": int(
            next(
                row["scene_index"]
                for row in _CURRENT_F0["design"]["scenes"]
                if row["scene_id"] == scene["scene_id"]
            )
        ),
        "scene_cluster": str(scene["scene_id"]),
        "scene_family": str(scene["scene_id"]),
        "source_scene_id": str(scene["source_scene_id"]),
        "domain": str(scene["domain"]),
        "material_id": primary_id,
        "cluster_material": primary_id,
        "material_slot": int(context["material_slot"]),
        "context_index": context_index,
        "local_case_index": local_case,
        "profile_index": profile,
        "physical_nuisance": nuisance,
        "severity_id": band,
        "severity": band,
        "lambda": interpolation_lambda,
        "point_index_within_band": point_index,
        "o7_source_physics_group_id": groups["O7_visual_remap"],
        "o8_source_physics_group_id": groups["O8_invisible_collider"],
        "shared_visual_source": "same frozen three-view appearance schedule",
        "visual_intervention_advances_physics": False,
    }
    return rows, case


def _conflict_model_rows(
    case: Mapping[str, Any],
    *,
    cell: str,
    candidates: Sequence[str],
) -> list[dict[str, Any]]:
    rows = []
    for candidate in candidates:
        for view_index in range(3):
            rows.append(
                {
                    "schema_version": "kinofail.unified-confirmatory-conflict-record.v1",
                    "benchmark_id": "kinofail_unified_confirmatory_v1",
                    "cell": cell,
                    "case_id": str(case["case_id"]),
                    "case_seed": int(case["case_seed"]),
                    "scene_index": int(case["scene_index"]),
                    "scene_id": str(
                        case.get("scene_id")
                        or case.get("scene_cluster")
                        or case["scene_family"]
                    ),
                    "scene_cluster": str(
                        case.get("scene_cluster") or case["scene_family"]
                    ),
                    "scene_family": str(case["scene_family"]),
                    "domain": str(case["domain"]),
                    "material_id": str(case["material_id"]),
                    "material_slot": int(case["material_slot"]),
                    "cluster_material": str(case["material_id"]),
                    "local_case_index": int(case["local_case_index"]),
                    "physical_nuisance": dict(case["physical_nuisance"]),
                    "record_role": "candidate_attribution_view",
                    "cause_id": candidate,
                    "view_id": (
                        "primary" if view_index == 0 else f"swap_{view_index:02d}"
                    ),
                    "candidate_operator": candidate,
                    "appearance_view_index": view_index,
                    "appearance_view_id": (
                        "primary" if view_index == 0 else f"swap_{view_index:02d}"
                    ),
                    "artifact_state": "planned",
                }
            )
    return rows


def _collection_protocol(
    *,
    protocol_id: str,
    schedule_final_path: Path,
    schedule_stage_path: Path,
    registry_path: Path,
    lock_path: Path,
    collector_path: Path,
    runtime_manifest_path: Path,
    f0_path: Path,
    design_path: Path,
    scene_id: str,
    pairs: int,
    operators: Sequence[str],
    root: Path,
) -> dict[str, Any]:
    schedule_rows = [
        json.loads(line)
        for line in schedule_stage_path.read_text(encoding="utf-8").splitlines()
        if line
    ]
    if (
        len(schedule_rows) != pairs * 2
        or {str(row["scene_family"]) for row in schedule_rows} != {scene_id}
    ):
        raise RuntimeError("collection protocol schedule scope drift")
    return {
        "schema_version": "kinofail.formal-collection-protocol.v1",
        "protocol_id": protocol_id,
        "status": "frozen",
        "confirmatory": True,
        "benchmark_id": "kinofail_unified_confirmatory_v1",
        "scene_id": scene_id,
        "schedule_path": _display(schedule_final_path, root=root),
        "schedule_sha256": _sha256(schedule_stage_path),
        "scene_registry_path": _display(registry_path, root=root),
        "scene_registry_sha256": _sha256(registry_path),
        "material_lock_path": _display(lock_path, root=root),
        "material_lock_sha256": _sha256(lock_path),
        "collector_path": _display(collector_path, root=root),
        "collector_sha256": _sha256(collector_path),
        "runtime_manifest_path": _display(runtime_manifest_path, root=root),
        "runtime_manifest_sha256": _sha256(runtime_manifest_path),
        "f0_manifest_path": _display(f0_path, root=root),
        "f0_manifest_sha256": _sha256(f0_path),
        "design_path": _display(design_path, root=root),
        "design_sha256": _sha256(design_path),
        "allowed": {
            "scene_families": [scene_id],
            "target_operators": list(operators),
            "conditions": list(CONDITIONS),
            "counterfactual_group_ids": sorted(
                {
                    str(row["counterfactual_group_id"])
                    for row in schedule_rows
                }
            ),
            "physical_realizations": sorted(
                {str(row["physical_realization"]) for row in schedule_rows}
            ),
            "geometry_profiles": sorted(
                {str(row["geometry_profile"]) for row in schedule_rows}
            ),
            "severity_ids": sorted(
                {str(row["severity_id"]) for row in schedule_rows}
            ),
        },
        "collection_contract": {
            "counterfactual_pairs": pairs,
            "physical_episodes": pairs * 2,
            "appearance_views_per_episode": 3,
            "per_pair_physical_nuisance_exact_and_shared": True,
            "operator_seed_equals_physics_seed": True,
            "per_scene_or_outcome_parameter_tuning_forbidden": True,
            "outcome_strength_is_reported_not_gated": True,
            "a8_in_scope": False,
        },
    }


def _snapshot_protocol(
    *,
    protocol_id: str,
    collection_protocol_id: str,
    schedule_final_path: Path,
    schedule_stage_path: Path,
    corpus_root: Path,
    output_dir: Path,
    registry_path: Path,
    lock_path: Path,
    collector_path: Path,
    runtime_manifest_path: Path,
    f0_path: Path,
    operators: Sequence[str],
    root: Path,
) -> dict[str, Any]:
    return {
        "schema_version": "kinofail.realistic-snapshot-protocol.v2",
        "protocol_id": protocol_id,
        "status": "frozen_before_collection",
        "development_only": False,
        "a8_in_scope": False,
        "source_schedule": _display(schedule_final_path, root=root),
        "source_schedule_sha256": _sha256(schedule_stage_path),
        "source_corpus_root": _display(corpus_root, root=root),
        "output_dir": _display(output_dir, root=root),
        "appearance_interventions": {
            "include_all_manifest_views": True,
            "count_views_as_independent_physical_samples": False,
            "require_timestamp_identity_across_views": True,
            "share_proprio_across_views": True,
        },
        "operator_event_adapters": {
            "coverage": list(operators),
            "selection_rule": (
                "predeclared measured mechanism events only; no outcome-maximum heuristic"
            ),
        },
        "selection": {
            "require_complete_counterfactual_pair": True,
            "require_evaluation_eligible": False,
            "require_runtime_validation_passed": True,
            "required_collection_protocol_id": collection_protocol_id,
            "operators_with_admitted_event_adapter": list(operators),
            "allow_nonblocking_runtime_issue_suffixes": [
                "appearance_effect_too_small",
                "rgb_spatial_contrast_too_low",
            ],
            "on_temporal_alignment_failure": "exclude_complete_pair_and_audit",
        },
        "temporal_alignment": {
            "anchor": "first_predeclared_privileged_operator_event_in_anomaly_only",
            "decision_delay_s": 0.0,
            "decision_delay_s_by_operator": {
                "O5_payload": 0.4,
                "O10_effort_decay": 0.7,
                "O11_obs_bias": 0.4,
            },
            "minimum_decision_time_s": 0.42,
            "nominal_alignment": "reuse_anomaly_event_and_decision_times_exactly",
            "rgb_offsets_from_decision_s": [-0.4, -0.3, -0.2, -0.1, 0.0],
            "max_rgb_target_skew_s": 0.041,
            "proprio_samples": 21,
            "proprio_window_s": 0.5,
            "max_proprio_end_skew_s": 0.021,
        },
        "freeze_provenance": {
            "f0_manifest": _display(f0_path, root=root),
            "f0_manifest_sha256": _sha256(f0_path),
            "scene_registry": _display(registry_path, root=root),
            "scene_registry_sha256": _sha256(registry_path),
            "material_lock": _display(lock_path, root=root),
            "material_lock_sha256": _sha256(lock_path),
            "collector": _display(collector_path, root=root),
            "collector_sha256": _sha256(collector_path),
            "runtime_manifest": _display(runtime_manifest_path, root=root),
            "runtime_manifest_sha256": _sha256(runtime_manifest_path),
            "model_outcomes_available_at_freeze": False,
            "physical_collection_started_at_freeze": False,
        },
        "publication_guard": {
            "may_satisfy_realistic_a0_a7": True,
            "reason": "Independent F0/F1-bound confirmatory scene shard.",
        },
    }


def _t2_collection_protocol(
    *,
    schedule_path: Path,
    audit_path: Path,
    schedule_final_path: Path,
    audit_final_path: Path,
    registry_path: Path,
    lock_path: Path,
    root: Path,
) -> dict[str, Any]:
    return {
        "schema_version": "kinofail.confirmatory-c1-collection-protocol.v1",
        "protocol_id": "kinofail-unified-confirmatory-v1-t2-shared-prefix",
        "status": "frozen",
        "design_tag": "kinofail_unified_confirmatory_v1",
        "schedule_path": _display(schedule_final_path, root=root),
        "schedule_sha256": _sha256(schedule_path),
        "design_audit_path": _display(audit_final_path, root=root),
        "design_audit_sha256": _sha256(audit_path),
        "scene_registry_path": _display(registry_path, root=root),
        "scene_registry_sha256": _sha256(registry_path),
        "asset_lock_path": _display(lock_path, root=root),
        "asset_lock_sha256": _sha256(lock_path),
        "collector_path": _display(T2_COLLECTOR_PATH, root=root),
        "collector_sha256": _sha256(T2_COLLECTOR_PATH),
        "visual_helper_path": _display(T2_VISUAL_HELPER_PATH, root=root),
        "visual_helper_sha256": _sha256(T2_VISUAL_HELPER_PATH),
        "model_or_endpoint_outcomes_available_at_freeze": False,
        "cases": 1_500,
        "samples": 9_000,
        "post_freeze_retry_or_parameter_change": False,
    }


def validate_inputs(
    *,
    design_path: Path,
    f0_path: Path,
    registry_path: Path,
    lock_path: Path,
    collector_path: Path,
    runtime_manifest_path: Path,
) -> tuple[dict[str, Any], dict[str, Any], dict[str, Any], dict[str, Any]]:
    for path in (
        design_path,
        f0_path,
        registry_path,
        lock_path,
        collector_path,
        runtime_manifest_path,
        BASE_OPERATOR_PATH,
        T2_COLLECTOR_PATH,
        T2_VISUAL_HELPER_PATH,
    ):
        if not path.is_file():
            raise FileNotFoundError(path)
    design = _json(design_path)
    f0 = _json(f0_path)
    registry = _json(registry_path)
    lock = _json(lock_path)
    if design.get("schema_version") != "kinofail.unified-confirmatory-freeze-input.v1":
        raise RuntimeError("unsupported authoritative design")
    if (
        f0.get("status") != "frozen_before_new_scene_generation"
        or f0.get("external_design", {}).get("sha256") != _sha256(design_path)
    ):
        raise RuntimeError("F0 is absent, invalid, or bound to another design")
    frozen = {
        str(row["path"]): str(row["sha256"])
        for row in f0.get("frozen_files", [])
    }
    for path in (
        collector_path,
        runtime_manifest_path,
        T2_COLLECTOR_PATH,
        T2_VISUAL_HELPER_PATH,
        Path(__file__).resolve(),
    ):
        if frozen.get(_display(path, root=ROOT)) != _sha256(path):
            raise RuntimeError(f"F0 does not bind current pipeline file: {path}")
    scenes = [dict(row) for row in registry.get("scenes", [])]
    materials = [dict(row) for row in lock.get("materials", [])]
    planned_scenes = {
        str(row["scene_id"]) for row in f0["design"]["scenes"]
    }
    planned_materials = {
        str(row["material_id"]) for row in f0["design"]["materials"]
    }
    if (
        len(scenes) != 30
        or {str(row["scene_id"]) for row in scenes} != planned_scenes
        or registry.get("selection_uses_model_predictions") is not False
    ):
        raise RuntimeError("F1 scene registry does not realize the 30 F0 slots")
    if any(
        not isinstance(row.get("source_scene_id"), str)
        or not isinstance(row.get("geometry_hash"), str)
        or len(str(row["geometry_hash"])) != 64
        or any(
            character not in "0123456789abcdef"
            for character in str(row["geometry_hash"]).lower()
        )
        for row in scenes
    ):
        raise RuntimeError(
            "F1 registry must retain source_scene_id and a 64-hex geometry_hash"
        )
    if (
        len(materials) != 30
        or {str(row["id"]) for row in materials} != planned_materials
    ):
        raise RuntimeError("material lock does not realize the 30 F0 aliases")
    if (
        f0["design"]["scale"]["counterfactual_pairs"] != 10_560
        or f0["design"]["conflict"]["cases_per_cell"] != 1_500
    ):
        raise RuntimeError("F0 count contract drift")
    return design, f0, registry, lock


_CURRENT_DESIGN: dict[str, Any] = {}
_CURRENT_F0: dict[str, Any] = {}


def build_schedules(
    *,
    design_path: Path,
    f0_path: Path,
    registry_path: Path,
    lock_path: Path,
    collector_path: Path,
    runtime_manifest_path: Path,
    output_root: Path,
    repo_root: Path = ROOT,
) -> dict[str, Any]:
    """Build all schedules and protocols, then atomically publish them."""

    output_root = output_root.resolve()
    if output_root.exists() or output_root.is_symlink():
        raise FileExistsError(f"refusing to overwrite {output_root}")
    design, f0, registry, lock = validate_inputs(
        design_path=design_path.resolve(),
        f0_path=f0_path.resolve(),
        registry_path=registry_path.resolve(),
        lock_path=lock_path.resolve(),
        collector_path=collector_path.resolve(),
        runtime_manifest_path=runtime_manifest_path.resolve(),
    )
    global _CURRENT_DESIGN, _CURRENT_F0
    _CURRENT_DESIGN = design
    _CURRENT_F0 = f0
    scenes = {
        str(row["scene_id"]): dict(row) for row in registry["scenes"]
    }
    materials = {
        str(row["id"]): dict(row) for row in lock["materials"]
    }
    operators = _operator_specs(BASE_OPERATOR_PATH)
    operator_map = {str(row["id"]): row for row in operators}
    if [row["id"] for row in operators] != f0["design"]["operators"]:
        raise RuntimeError("operator order differs from F0")
    contexts = sorted(
        (dict(row) for row in f0["design"]["scene_material_assignments"]),
        key=lambda row: int(row["context_index"]),
    )

    scale_by_scene: dict[str, list[dict[str, Any]]] = defaultdict(list)
    t2_by_scene: dict[str, list[dict[str, Any]]] = defaultdict(list)
    t3_by_scene: dict[str, list[dict[str, Any]]] = defaultdict(list)
    t3_cases_by_scene: dict[str, list[dict[str, Any]]] = defaultdict(list)
    conflict_by_scene: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for context in contexts:
        scene_id = str(context["scene_id"])
        scene = scenes[scene_id]
        scale_by_scene[scene_id].extend(
            _scale_context_rows(
                design=design,
                f0=f0,
                scene=scene,
                context=context,
                materials=materials,
                operators=operators,
            )
        )
        context_index = int(context["context_index"])
        for local_case in range(25):
            t2_seed = 2_040_000_000 + context_index * 25 + local_case
            t2_case = _t2_case(
                scene=scene,
                context=context,
                materials=materials,
                case_seed=t2_seed,
                local_case=local_case,
                context_index=context_index,
            )
            t2_by_scene[scene_id].append(t2_case)
            conflict_by_scene[scene_id].extend(
                _conflict_model_rows(
                    t2_case,
                    cell="T2_vision_decisive",
                    candidates=("O2_compliance", "O4_tether"),
                )
            )
            t3_seed = 2_040_000_000 + 1_500 + context_index * 25 + local_case
            physical_rows, t3_case = _t3_case_rows(
                scene=scene,
                context=context,
                materials=materials,
                operators=operator_map,
                case_seed=t3_seed,
                local_case=local_case,
                context_index=context_index,
            )
            t3_by_scene[scene_id].extend(physical_rows)
            t3_cases_by_scene[scene_id].append(t3_case)
            conflict_by_scene[scene_id].extend(
                _conflict_model_rows(
                    t3_case,
                    cell="T3_proprio_decisive",
                    candidates=("O7_visual_remap", "O8_invisible_collider"),
                )
            )

    scale_rows = [
        row
        for scene_id in sorted(scale_by_scene)
        for row in scale_by_scene[scene_id]
    ]
    conflict_rows = [
        row
        for scene_id in sorted(conflict_by_scene)
        for row in conflict_by_scene[scene_id]
    ]
    t2_rows = [
        row for scene_id in sorted(t2_by_scene) for row in t2_by_scene[scene_id]
    ]
    t3_rows = [
        row for scene_id in sorted(t3_by_scene) for row in t3_by_scene[scene_id]
    ]
    t3_case_rows = [
        row
        for scene_id in sorted(t3_cases_by_scene)
        for row in t3_cases_by_scene[scene_id]
    ]
    scale_groups = {
        str(row["counterfactual_group_id"]) for row in scale_rows
    }
    if len(scale_rows) != 21_120 or len(scale_groups) != 10_560:
        raise RuntimeError("scale count drift")
    if (
        sum(len(value) for value in t2_by_scene.values()) != 1_500
        or sum(len(value) for value in t3_cases_by_scene.values()) != 1_500
        or sum(len(value) for value in t3_by_scene.values()) != 6_000
        or len(conflict_rows) != 18_000
    ):
        raise RuntimeError("C2 count drift")
    if any(
        set(row.get("physical_nuisance", {})) != NUISANCE_KEYS
        or row["physical_nuisance"]["pair_shared"] is not True
        or int(row["physical_nuisance"]["physics_seed"])
        != int(row["operator_seed"])
        for row in scale_rows
        + [item for rows in t3_by_scene.values() for item in rows]
    ):
        raise RuntimeError("physical nuisance contract drift")

    # Reuse the F0-aware audit implementation before publishing any schedule.
    from scripts.audit_kinofail_confirmatory_freshness_v1 import (  # noqa: PLC0415
        DEFAULT_PRIOR_EXPOSURES,
        _collect_exposure,
        audit_freshness,
    )

    prior_paths = [
        ROOT / value for value in DEFAULT_PRIOR_EXPOSURES if (ROOT / value).is_file()
    ]
    prior = _collect_exposure(prior_paths)
    freshness_checks, freshness_details = audit_freshness(
        f0=f0,
        scene_registry=registry,
        material_lock=lock,
        scale_rows=scale_rows,
        conflict_rows=conflict_rows,
        prior=prior,
    )
    if not all(freshness_checks.values()):
        failed = sorted(key for key, value in freshness_checks.items() if not value)
        raise RuntimeError(f"confirmatory freshness audit failed: {failed}")

    output_root.parent.mkdir(parents=True, exist_ok=True)
    staging = Path(
        tempfile.mkdtemp(
            prefix=f".{output_root.name}.tmp.", dir=str(output_root.parent)
        )
    )
    try:
        global_scale = staging / "scale_schedule.jsonl"
        global_conflict = staging / "conflict_schedule.jsonl"
        _write_jsonl(global_scale, scale_rows)
        _write_jsonl(global_conflict, conflict_rows)
        global_t2 = staging / "c2_t2/schedule.jsonl"
        global_t2_audit = staging / "c2_t2/audit.json"
        global_t3 = staging / "c2_t3/schedule.jsonl"
        global_t3_cases = staging / "c2_t3/case_schedule.jsonl"
        global_t3_audit = staging / "c2_t3/audit.json"
        _write_jsonl(global_t2, t2_rows)
        _write_jsonl(global_t3, t3_rows)
        _write_jsonl(global_t3_cases, t3_case_rows)
        _write_json(
            global_t2_audit,
            {
                "schema_version": "kinofail.unified-confirmatory-t2-design-audit.v1",
                "passed": True,
                "schedule_sha256": _sha256(global_t2),
                "cases": len(t2_rows),
                "samples_after_three_views_two_causes": len(t2_rows) * 6,
            },
        )
        _write_json(
            global_t3_audit,
            {
                "schema_version": "kinofail.unified-confirmatory-t3-design-audit.v1",
                "passed": True,
                "schedule_sha256": _sha256(global_t3),
                "case_schedule_sha256": _sha256(global_t3_cases),
                "cases": len(t3_case_rows),
                "physical_episodes": len(t3_rows),
            },
        )
        global_t2_protocol = staging / "c2_t2/collection_protocol.json"
        _write_json(
            global_t2_protocol,
            _t2_collection_protocol(
                schedule_path=global_t2,
                audit_path=global_t2_audit,
                schedule_final_path=output_root / "c2_t2/schedule.jsonl",
                audit_final_path=output_root / "c2_t2/audit.json",
                registry_path=registry_path,
                lock_path=lock_path,
                root=repo_root,
            ),
        )
        shard_records = []
        for scene_id in sorted(scenes):
            scene_root = staging / "scenes" / scene_id
            final_scene_root = output_root / "scenes" / scene_id
            scale_stage = scene_root / "scale" / "schedule.jsonl"
            t2_stage = scene_root / "c2_t2" / "case_schedule.jsonl"
            t3_stage = scene_root / "c2_t3" / "schedule.jsonl"
            t3_case_stage = scene_root / "c2_t3" / "case_schedule.jsonl"
            conflict_stage = scene_root / "conflict_schedule.jsonl"
            _write_jsonl(scale_stage, scale_by_scene[scene_id])
            _write_jsonl(t2_stage, t2_by_scene[scene_id])
            _write_jsonl(t3_stage, t3_by_scene[scene_id])
            _write_jsonl(t3_case_stage, t3_cases_by_scene[scene_id])
            _write_jsonl(conflict_stage, conflict_by_scene[scene_id])

            scale_collection_id = f"confirmatory-v1-scale-{scene_id}"
            t3_collection_id = f"confirmatory-v1-c2-t3-{scene_id}"
            scale_collection = _collection_protocol(
                protocol_id=scale_collection_id,
                schedule_final_path=final_scene_root / "scale/schedule.jsonl",
                schedule_stage_path=scale_stage,
                registry_path=registry_path,
                lock_path=lock_path,
                collector_path=collector_path,
                runtime_manifest_path=runtime_manifest_path,
                f0_path=f0_path,
                design_path=design_path,
                scene_id=scene_id,
                pairs=352,
                operators=[str(row["id"]) for row in operators],
                root=repo_root,
            )
            t3_collection = _collection_protocol(
                protocol_id=t3_collection_id,
                schedule_final_path=final_scene_root / "c2_t3/schedule.jsonl",
                schedule_stage_path=t3_stage,
                registry_path=registry_path,
                lock_path=lock_path,
                collector_path=collector_path,
                runtime_manifest_path=runtime_manifest_path,
                f0_path=f0_path,
                design_path=design_path,
                scene_id=scene_id,
                pairs=100,
                operators=["O7_visual_remap", "O8_invisible_collider"],
                root=repo_root,
            )
            scale_collection_path = scene_root / "scale/collection_protocol.json"
            t3_collection_path = scene_root / "c2_t3/collection_protocol.json"
            _write_json(scale_collection_path, scale_collection)
            _write_json(t3_collection_path, t3_collection)
            scale_snapshot = _snapshot_protocol(
                protocol_id=f"confirmatory-v1-scale-snapshot-{scene_id}",
                collection_protocol_id=scale_collection_id,
                schedule_final_path=final_scene_root / "scale/schedule.jsonl",
                schedule_stage_path=scale_stage,
                corpus_root=(
                    ROOT
                    / "outputs/kinofail_confirmatory_v1/corpus"
                    / scene_id
                ),
                output_dir=(
                    ROOT
                    / "outputs/eval/unified_moe_v3_confirmatory_v1/shards"
                    / scene_id
                    / "scale/snapshots"
                ),
                registry_path=registry_path,
                lock_path=lock_path,
                collector_path=collector_path,
                runtime_manifest_path=runtime_manifest_path,
                f0_path=f0_path,
                operators=[str(row["id"]) for row in operators],
                root=repo_root,
            )
            t3_snapshot = _snapshot_protocol(
                protocol_id=f"confirmatory-v1-c2-t3-snapshot-{scene_id}",
                collection_protocol_id=t3_collection_id,
                schedule_final_path=final_scene_root / "c2_t3/schedule.jsonl",
                schedule_stage_path=t3_stage,
                corpus_root=(
                    ROOT
                    / "outputs/kinofail_confirmatory_v1/corpus"
                    / scene_id
                ),
                output_dir=(
                    ROOT
                    / "outputs/eval/unified_moe_v3_confirmatory_v1/shards"
                    / scene_id
                    / "c2_t3/snapshots"
                ),
                registry_path=registry_path,
                lock_path=lock_path,
                collector_path=collector_path,
                runtime_manifest_path=runtime_manifest_path,
                f0_path=f0_path,
                operators=["O7_visual_remap", "O8_invisible_collider"],
                root=repo_root,
            )
            scale_snapshot_path = scene_root / "scale/snapshot_protocol.json"
            t3_snapshot_path = scene_root / "c2_t3/snapshot_protocol.json"
            _write_json(scale_snapshot_path, scale_snapshot)
            _write_json(t3_snapshot_path, t3_snapshot)
            manifest = {
                "schema_version": "kinofail.unified-confirmatory-shard-manifest.v1",
                "scene_id": scene_id,
                "source_scene_id": scenes[scene_id]["source_scene_id"],
                "domain": scenes[scene_id]["domain"],
                "passed": True,
                "counts": {
                    "scale_pairs": 352,
                    "scale_physical_episodes": 704,
                    "t2_cases": 50,
                    "t3_cases": 50,
                    "t3_pairs": 100,
                    "t3_physical_episodes": 200,
                    "conflict_model_records": 600,
                },
                "bindings": {
                    "f0_manifest_sha256": _sha256(f0_path),
                    "scene_registry_sha256": _sha256(registry_path),
                    "material_lock_sha256": _sha256(lock_path),
                    "collector_sha256": _sha256(collector_path),
                    "runtime_manifest_sha256": _sha256(runtime_manifest_path),
                },
                "artifacts": {
                    "scale_schedule": _sha256(scale_stage),
                    "scale_collection_protocol": _sha256(scale_collection_path),
                    "scale_snapshot_protocol": _sha256(scale_snapshot_path),
                    "t2_case_schedule": _sha256(t2_stage),
                    "t3_schedule": _sha256(t3_stage),
                    "t3_case_schedule": _sha256(t3_case_stage),
                    "t3_collection_protocol": _sha256(t3_collection_path),
                    "t3_snapshot_protocol": _sha256(t3_snapshot_path),
                    "conflict_schedule": _sha256(conflict_stage),
                },
            }
            manifest_path = scene_root / "shard_manifest.json"
            _write_json(manifest_path, manifest)
            shard_records.append(
                {
                    "scene_id": scene_id,
                    "manifest": f"scenes/{scene_id}/shard_manifest.json",
                    "manifest_sha256": _sha256(manifest_path),
                }
            )

        freshness_audit = {
            "schema_version": "kinofail.unified-confirmatory-schedule-freshness.v1",
            "passed": True,
            "checks": freshness_checks,
            "details": freshness_details,
            "prior_sources": [
                {
                    "path": _display(path, root=repo_root),
                    "sha256": _sha256(path),
                }
                for path in prior_paths
            ],
        }
        _write_json(staging / "freshness_audit.json", freshness_audit)
        root_manifest = {
            "schema_version": "kinofail.unified-confirmatory-schedule-manifest.v1",
            "created_utc": datetime.now(UTC).isoformat(),
            "status": "planned_schedules_only_no_runtime_data",
            "passed": True,
            "counts": {
                "scene_shards": 30,
                "material_contexts": 60,
                "scale_counterfactual_pairs": 10_560,
                "scale_physical_episodes": 21_120,
                "t2_cases": 1_500,
                "t3_cases": 1_500,
                "t3_counterfactual_pairs": 3_000,
                "t3_physical_episodes": 6_000,
                "conflict_model_records": 18_000,
            },
            "bindings": {
                "design_sha256": _sha256(design_path),
                "f0_manifest_sha256": _sha256(f0_path),
                "scene_registry_sha256": _sha256(registry_path),
                "material_lock_sha256": _sha256(lock_path),
                "collector_sha256": _sha256(collector_path),
                "runtime_manifest_sha256": _sha256(runtime_manifest_path),
                "builder_sha256": _sha256(Path(__file__).resolve()),
            },
            "global_artifacts": {
                "scale_schedule": {
                    "path": "scale_schedule.jsonl",
                    "sha256": _sha256(global_scale),
                },
                "conflict_schedule": {
                    "path": "conflict_schedule.jsonl",
                    "sha256": _sha256(global_conflict),
                },
                "t2_schedule": {
                    "path": "c2_t2/schedule.jsonl",
                    "sha256": _sha256(global_t2),
                },
                "t2_design_audit": {
                    "path": "c2_t2/audit.json",
                    "sha256": _sha256(global_t2_audit),
                },
                "t2_collection_protocol": {
                    "path": "c2_t2/collection_protocol.json",
                    "sha256": _sha256(global_t2_protocol),
                },
                "t3_schedule": {
                    "path": "c2_t3/schedule.jsonl",
                    "sha256": _sha256(global_t3),
                },
                "t3_case_schedule": {
                    "path": "c2_t3/case_schedule.jsonl",
                    "sha256": _sha256(global_t3_cases),
                },
                "t3_design_audit": {
                    "path": "c2_t3/audit.json",
                    "sha256": _sha256(global_t3_audit),
                },
                "freshness_audit": {
                    "path": "freshness_audit.json",
                    "sha256": _sha256(staging / "freshness_audit.json"),
                },
            },
            "shards": shard_records,
            "freeze_policy": {
                "output_exists": "fail_closed",
                "atomic_publish": True,
                "model_or_anomaly_outcomes_read": False,
                "posthoc_retry_or_parameter_change": False,
            },
        }
        _write_json(staging / "manifest.json", root_manifest)
        os.replace(staging, output_root)
    except BaseException:
        shutil.rmtree(staging, ignore_errors=True)
        raise
    return {
        "passed": True,
        "output": str(output_root),
        "manifest": str(output_root / "manifest.json"),
        "manifest_sha256": _sha256(output_root / "manifest.json"),
        "counts": root_manifest["counts"],
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--design", type=Path, default=DESIGN_PATH)
    parser.add_argument("--f0-manifest", type=Path, default=F0_PATH)
    parser.add_argument("--scene-registry", type=Path, default=REGISTRY_PATH)
    parser.add_argument("--material-lock", type=Path, default=LOCK_PATH)
    parser.add_argument("--collector", type=Path, default=COLLECTOR_PATH)
    parser.add_argument(
        "--runtime-manifest", type=Path, default=RUNTIME_MANIFEST_PATH
    )
    parser.add_argument("--out", type=Path, default=DEFAULT_OUTPUT)
    args = parser.parse_args()
    result = build_schedules(
        design_path=args.design,
        f0_path=args.f0_manifest,
        registry_path=args.scene_registry,
        lock_path=args.material_lock,
        collector_path=args.collector,
        runtime_manifest_path=args.runtime_manifest,
        output_root=args.out,
    )
    print(json.dumps(result, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
