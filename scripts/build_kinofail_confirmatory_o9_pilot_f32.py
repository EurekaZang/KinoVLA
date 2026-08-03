#!/usr/bin/env python3
"""Freeze an excluded 12-pair pilot for direct-contact O9 recollection.

The pilot spans all three scene domains and both O9 severity bands.  It is
used only to verify mechanism activation and nominal stability; every pilot
pair is permanently excluded from the later 960-pair confirmation cohort.
No attribution-model artifact is read.
"""

from __future__ import annotations

import copy
import hashlib
import json
import random
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from kino_vla.data.o9_semantics import DEFAULT_THRESHOLDS


ROOT = Path(__file__).resolve().parents[1]
SOURCE_ROOT = ROOT / "outputs/kinofail_reconfirmation_v2"
SOURCE_SCHEDULE = SOURCE_ROOT / "schedules/scale_schedule.jsonl"
OUTPUT = ROOT / "outputs/kinofail_confirmatory_o9_pilot_f32"
CORPUS = Path("/data/eureka/KinoVLA/outputs/kinofail_confirmatory_o9_pilot_f32/corpus")
COLLECTOR = ROOT / "scripts/isaac_collect_kinofail_confirmatory_o9_direct_v1.py"
RUNNER = ROOT / "scripts/run_kinofail_confirmatory_o9_pilot_f32.py"
REGISTRY = SOURCE_ROOT / "scene_registry.json"
ASSET_LOCK = ROOT / "outputs/assets/terrain_pbr_confirmatory_v2/terrain_assets.lock.json"
RUNTIME = ROOT / "kino_vla/data/runtime_manifest.py"
SEMANTICS = ROOT / "kino_vla/data/o9_semantics.py"
SEED_BASE = 2_330_000_000
SURFACE_STATES = ("clean", "dusty", "scuffed", "damp")


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def canonical(value: Any) -> str:
    return json.dumps(value, sort_keys=True, separators=(",", ":"))


def make_id(prefix: str, value: Any) -> str:
    return prefix + hashlib.sha256(canonical(value).encode()).hexdigest()[:20]


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    return [json.loads(line) for line in path.read_text().splitlines() if line]


def write_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n")


def write_jsonl(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("".join(canonical(row) + "\n" for row in rows))


def replace_token(value: Any, old: str, new: str) -> Any:
    if isinstance(value, str):
        return value.replace(old, new)
    if isinstance(value, list):
        return [replace_token(item, old, new) for item in value]
    if isinstance(value, dict):
        return {key: replace_token(item, old, new) for key, item in value.items()}
    return value


def fresh_views(source: list[dict[str, Any]], seed: int, pair_id: str) -> list[dict[str, Any]]:
    views: list[dict[str, Any]] = []
    for index, item in enumerate(source):
        row = copy.deepcopy(item)
        rng = random.Random(seed + 1009 * index)
        row.update(
            {
                "appearance_seed": seed,
                "surface_state": SURFACE_STATES[rng.randrange(len(SURFACE_STATES))],
                "uv_scale": round(rng.uniform(0.72, 1.36), 6),
                "uv_rotation_deg": int(rng.choice((0, 90, 180, 270))),
                "uv_offset": [round(rng.random(), 6), round(rng.random(), 6)],
                "albedo_brightness_multiplier": round(rng.uniform(0.86, 1.14), 6),
                "normal_strength": round(rng.uniform(0.75, 1.25), 6),
                "roughness_multiplier": round(rng.uniform(0.78, 1.22), 6),
            }
        )
        row["appearance_id"] = make_id(
            "app_f32p_", {"pair_id": pair_id, "view": index, "seed": seed, "row": row}
        )
        views.append(row)
    return views


def clone_pair(source: list[dict[str, Any]], index: int) -> list[dict[str, Any]]:
    if len(source) != 2 or {row["condition"] for row in source} != {
        "nominal_counterfactual",
        "anomaly",
    }:
        raise RuntimeError("source O9 pair is incomplete")
    original_id = str(source[0]["counterfactual_group_id"])
    seed = SEED_BASE + index
    pair_id = make_id(
        "cf_f32p_", {"original": original_id, "seed": seed, "pilot_attempt": 1}
    )
    views = fresh_views(list(source[0]["appearance_views"]), seed, pair_id)
    primary = next(row for row in views if row["is_primary"] is True)
    source_lambda = float(source[0]["parameter_interpolation"]["lambda"])
    continuum = max(0.0, min(1.0, (source_lambda - 0.2025) / (0.96375 - 0.2025)))
    width = round(0.12 + 0.02 * continuum, 6)
    residual = round(0.45 - 0.35 * continuum, 6)
    geometry_spec = {
        "kind": "route_transverse_pallet_belly_crossbar",
        "ridge_height_m": 0.36,
        "ridge_width_m": width,
        "residual_support": residual,
    }
    output: list[dict[str, Any]] = []
    for original in source:
        row = replace_token(copy.deepcopy(original), original_id, pair_id)
        condition = str(row["condition"])
        row.update(
            {
                "benchmark_id": "kinofail_confirmatory_o9_pilot_f32",
                "counterfactual_group_id": pair_id,
                "episode_id": f"{pair_id}_{'nominal' if condition == 'nominal_counterfactual' else 'anomaly'}",
                "design_mode": "excluded_physical_mechanism_pilot",
                "split": "excluded_o9_mechanism_pilot",
                "evaluation_eligible": False,
                "artifact_state": "planned",
                "operator_seed": seed,
                "physical_seed": seed,
                "runtime_seed": seed,
                "appearance_seed": seed,
                "appearance_views": copy.deepcopy(views),
                "appearance_id": primary["appearance_id"],
                "surface_state": primary["surface_state"],
                "uv_scale": primary["uv_scale"],
                "uv_rotation_deg": primary["uv_rotation_deg"],
                "uv_offset": primary["uv_offset"],
                "albedo_brightness_multiplier": primary["albedo_brightness_multiplier"],
                "normal_strength": primary["normal_strength"],
                "roughness_multiplier": primary["roughness_multiplier"],
                "texture_swap_group_id": make_id("ts_f32p_", {"pair": pair_id}),
                "geometry_id": f"{row['scene_id']}::transverse_pallet_belly_crossbar_f32",
                "geometry_profile": "transverse_pallet_belly_crossbar",
                "geometry_hash": hashlib.sha256(canonical(geometry_spec).encode()).hexdigest(),
                "physical_realization": "pallet_edge",
                "physical_nuisance": {
                    "profile_index": index,
                    "start_progress_m": 0.35,
                    "start_lateral_offset_m": round(random.Random(seed + 17).uniform(-0.02, 0.02), 6),
                    "start_heading_offset_rad": round(random.Random(seed + 31).uniform(-0.015, 0.015), 6),
                    "forward_speed_mps": 0.08,
                    "controller_target_lateral_offset_m": 0.0,
                    "physics_seed": seed,
                    "pair_shared": True,
                    "spawn_base_height_m": 0.43,
                },
                "parameter_interpolation": {
                    "source_lambda": source_lambda,
                    "direct_o9_continuum": round(continuum, 8),
                    "rule": "height=0.36; width=0.12+0.02*t; residual_support=0.45-0.35*t",
                },
                "o9_semantic_recollection": {
                    "phase": "excluded_pilot",
                    "source_counterfactual_group_id": original_id,
                    "model_prediction_or_score_read": False,
                    "direct_contact_required": True,
                },
            }
        )
        row["physics_parameters"] = (
            {
                "residual_support": 1.0,
                "ridge_height_m": 0.0,
                "ridge_width_m": 0.0,
            }
            if condition == "nominal_counterfactual"
            else {
                "residual_support": residual,
                "ridge_height_m": 0.36,
                "ridge_width_m": width,
            }
        )
        output.append(row)
    return sorted(output, key=lambda row: row["condition"] == "anomaly")


def select_pilot(groups: dict[str, list[dict[str, Any]]]) -> list[list[dict[str, Any]]]:
    representatives = [rows[0] for rows in groups.values()]
    scenes_by_domain: dict[str, list[str]] = {}
    for row in representatives:
        scenes_by_domain.setdefault(str(row["domain"]), []).append(str(row["scene_id"]))
    selected: list[list[dict[str, Any]]] = []
    for domain in sorted(scenes_by_domain):
        scenes = sorted(set(scenes_by_domain[domain]))[:2]
        for scene in scenes:
            candidates = [
                rows
                for rows in groups.values()
                if rows[0]["scene_id"] == scene
            ]
            moderate = min(
                (rows for rows in candidates if rows[0]["severity_id"] == "moderate"),
                key=lambda rows: float(rows[0]["parameter_interpolation"]["lambda"]),
            )
            hard = max(
                (rows for rows in candidates if rows[0]["severity_id"] == "hard"),
                key=lambda rows: float(rows[0]["parameter_interpolation"]["lambda"]),
            )
            selected.extend((moderate, hard))
    if len(selected) != 12:
        raise RuntimeError(f"expected 12 pilot pairs, selected {len(selected)}")
    return selected


def main() -> int:
    if OUTPUT.exists() or CORPUS.exists():
        raise FileExistsError("refusing to overwrite F32 pilot or corpus")
    rows = read_jsonl(SOURCE_SCHEDULE)
    groups: dict[str, list[dict[str, Any]]] = {}
    for row in rows:
        if row["target_operator"] == "O9_high_centering":
            groups.setdefault(str(row["counterfactual_group_id"]), []).append(row)
    if len(groups) != 960:
        raise RuntimeError(f"expected 960 source O9 pairs, found {len(groups)}")
    selected = select_pilot(groups)
    pilot_rows = [
        row for index, source in enumerate(selected) for row in clone_pair(source, index)
    ]
    pilot_rows.sort(key=lambda row: (row["counterfactual_group_id"], row["condition"]))

    design_path = OUTPUT / "design.json"
    design = {
        "schema_version": "kinofail.confirmatory-o9-pilot-f32-design.v1",
        "created_utc": datetime.now(UTC).isoformat(),
        "status": "fixed_before_pilot_collection",
        "scientific_role": "excluded physical-mechanism and nominal-stability pilot",
        "evaluation_eligible": False,
        "all_pilot_pairs_permanently_excluded_from_final_confirmation": True,
        "selection": "two lexicographically first scenes per domain; lowest moderate and highest hard source parameter point",
        "pairs": 12,
        "physical_episodes": 24,
        "domains": sorted({row["domain"] for row in pilot_rows}),
        "scenes": sorted({row["scene_id"] for row in pilot_rows}),
        "semantic_thresholds": DEFAULT_THRESHOLDS,
        "new_parameter_interval": {
            "ridge_height_m": [0.36, 0.36],
            "ridge_width_m": [0.12, 0.14],
            "residual_support": [0.10, 0.45],
            "continuum_points_in_final_design": 16,
        },
        "model_prediction_feature_label_or_score_read": False,
        "source_schedule": str(SOURCE_SCHEDULE.relative_to(ROOT)),
        "source_schedule_sha256": sha256(SOURCE_SCHEDULE),
    }
    write_json(design_path, design)
    schedule_path = OUTPUT / "schedule.jsonl"
    write_jsonl(schedule_path, pilot_rows)
    pair_ids = sorted({str(row["counterfactual_group_id"]) for row in pilot_rows})
    protocol_path = OUTPUT / "collection_protocol.json"
    protocol = {
        "schema_version": "kinofail.formal-collection-protocol.v1",
        "protocol_id": "kinofail-confirmatory-o9-direct-pilot-f32",
        "benchmark_id": "kinofail_confirmatory_o9_pilot_f32",
        "status": "frozen",
        "confirmatory": False,
        "collector_path": str(COLLECTOR.relative_to(ROOT)),
        "collector_sha256": sha256(COLLECTOR),
        "schedule_path": str(schedule_path.relative_to(ROOT)),
        "schedule_sha256": sha256(schedule_path),
        "runtime_manifest_path": str(RUNTIME.relative_to(ROOT)),
        "runtime_manifest_sha256": sha256(RUNTIME),
        "scene_registry_path": str(REGISTRY.relative_to(ROOT)),
        "scene_registry_sha256": sha256(REGISTRY),
        "material_lock_path": str(ASSET_LOCK.relative_to(ROOT)),
        "material_lock_sha256": sha256(ASSET_LOCK),
        "design_path": str(design_path.relative_to(ROOT)),
        "design_sha256": sha256(design_path),
        "allowed": {
            "conditions": ["nominal_counterfactual", "anomaly"],
            "counterfactual_group_ids": pair_ids,
            "geometry_profiles": ["transverse_pallet_belly_crossbar"],
            "physical_realizations": ["pallet_edge"],
            "scene_families": sorted({str(row["scene_family"]) for row in pilot_rows}),
            "severity_ids": ["hard", "moderate"],
            "target_operators": ["O9_high_centering"],
        },
        "collection_contract": {
            "a8_in_scope": False,
            "counterfactual_pairs": 12,
            "physical_episodes": 24,
            "appearance_views_per_episode": 3,
            "operator_seed_equals_physics_seed": True,
            "per_pair_physical_nuisance_exact_and_shared": True,
            "outcome_strength_is_reported_not_gated": True,
            "direct_o9_semantic_gate_required": True,
            "pilot_excluded_from_final_evaluation": True,
        },
    }
    write_json(protocol_path, protocol)
    dependencies = [
        COLLECTOR,
        RUNNER,
        ROOT / "scripts/isaac_collect_kinofail_confirmatory_pair_v1.py",
        SEMANTICS,
        RUNTIME,
    ]
    seal = {
        "schema_version": "kinofail.confirmatory-o9-pilot-f32-seal.v1",
        "created_utc": datetime.now(UTC).isoformat(),
        "status": "sealed_before_collection",
        "passed": True,
        "model_prediction_feature_label_or_score_read": False,
        "result_dependent_retry": False,
        "maximum_concurrent_isaac_processes": 3,
        "counterfactual_pairs": 12,
        "physical_episodes": 24,
        "corpus_root": str(CORPUS),
        "schedule": str(schedule_path.relative_to(ROOT)),
        "schedule_sha256": sha256(schedule_path),
        "protocol": str(protocol_path.relative_to(ROOT)),
        "protocol_sha256": sha256(protocol_path),
        "design": str(design_path.relative_to(ROOT)),
        "design_sha256": sha256(design_path),
        "scene_registry": str(REGISTRY.relative_to(ROOT)),
        "scene_registry_sha256": sha256(REGISTRY),
        "material_lock": str(ASSET_LOCK.relative_to(ROOT)),
        "material_lock_sha256": sha256(ASSET_LOCK),
        "semantic_module": str(SEMANTICS.relative_to(ROOT)),
        "semantic_module_sha256": sha256(SEMANTICS),
        "dependencies": [
            {"path": str(path), "sha256": sha256(path)} for path in dependencies
        ],
    }
    seal_path = OUTPUT / "seal_manifest.json"
    write_json(seal_path, seal)
    (OUTPUT / "seal_manifest.sha256").write_text(
        f"{sha256(seal_path)}  {seal_path.name}\n"
    )
    print(json.dumps(seal, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
