#!/usr/bin/env python3
"""Compile the F0/F1-bound KiNO-v4 independent-confirmation schedules."""

from __future__ import annotations

import hashlib
import json
import sys
from collections import defaultdict
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from scripts.build_kinofail_unified_confirmatory_v1_schedules import (  # noqa: E402
    CAMERAS,
    CONDITIONS,
    _appearance_views,
    _counterfactual,
    _operator_specs,
    _outputs,
    _stable_id,
    interpolate_parameters,
    physical_nuisance,
)


F0 = ROOT / "outputs/freeze/kino_v4_dinov2large_terrain448_f0/freeze_manifest.json"
F1 = ROOT / "outputs/freeze/kino_v4_confirmation_v1_f1/freeze_manifest.json"
CANDIDATES = ROOT / "configs/data/kinofail_kino_v4_confirmation_scene_candidates_v1.json"
OPERATORS = ROOT / "configs/data/kinofail_realistic.yaml"
LOCK = Path("/data/eureka/kinofail_kino_v4_confirmation_v1/assets/terrain_pbr/terrain_assets.lock.json")
DATA_ROOT = Path("/data/eureka/kinofail_kino_v4_confirmation_v1")
COLLECTOR = ROOT / "scripts/isaac_collect_kinofail_kino_v4_confirmation_pair_v1.py"
T2_COLLECTOR = ROOT / "scripts/isaac_collect_kinofail_confirmatory_t2_v1.py"
T2_VISUAL_HELPER = ROOT / "kino_vla/sim/c1_causal_visuals.py"
RUNTIME_MANIFEST = ROOT / "kino_vla/data/runtime_manifest.py"
OUTPUT = ROOT / "outputs/kinofail_kino_v4_confirmation_v1"
DERIVED = ROOT / "outputs/eval/kino_v4_confirmation_v1/shards"
BENCHMARK = "kinofail_kino_v4_confirmation_v1"
PROTOCOL = "kinofail-kino-v4-independent-confirmation-v1"


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _json(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text())
    if not isinstance(value, dict):
        raise TypeError(path)
    return value


def _write_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n")


def _write_jsonl(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("".join(json.dumps(row, sort_keys=True) + "\n" for row in rows))


def _scene_audit(candidate: dict[str, Any]) -> Path:
    scene_id = str(candidate["scene_id"])
    if candidate["domain"] == "wild":
        return DATA_ROOT / "scenes/wild" / scene_id / "terminal_scene_admission.json"
    return DATA_ROOT / "scenes/embodiedgen_compiled" / scene_id / "terminal_scene_admission.json"


def _registry(candidates: dict[str, Any]) -> dict[str, Any]:
    streams = {
        "life": [row for row in candidates["indoor_candidates"] if row["domain"] == "life"],
        "production": [row for row in candidates["indoor_candidates"] if row["domain"] == "production"],
        "wild": list(candidates["wild_scenes"]),
    }
    scenes: list[dict[str, Any]] = []
    attrition: dict[str, Any] = {}
    for domain, stream in streams.items():
        admitted = 0
        attempts = []
        for candidate_index, candidate in enumerate(stream):
            audit_path = _scene_audit(candidate)
            if not audit_path.is_file():
                raise RuntimeError(f"candidate stream is incomplete at {audit_path}")
            audit = _json(audit_path)
            accepted = audit.get("admitted") is True
            attempts.append(
                {
                    "candidate_index": candidate_index,
                    "candidate": candidate,
                    "admitted": accepted,
                    "failure_stage": audit.get("failure_stage"),
                    "terminal_audit": str(audit_path),
                    "terminal_audit_sha256": _sha256(audit_path),
                }
            )
            if accepted:
                episode = Path(str(audit["episode_usd"]))
                evidence_key = "compiled_scene" if domain == "wild" else "terrain_audit"
                compiled = Path(str(audit["evidence"][evidence_key]["path"]))
                if not episode.is_file() or not compiled.is_file():
                    raise RuntimeError(f"admitted scene artifacts absent: {candidate['scene_id']}")
                episode_hash = _sha256(episode)
                compiled_hash = _sha256(compiled)
                geometry_hash = hashlib.sha256(
                    f"{episode_hash}\n{compiled_hash}\n".encode()
                ).hexdigest()
                source_scene_id = (
                    _json(compiled).get("source_scene_id")
                    if domain == "wild"
                    else f"{candidate['room_type']}_seed{candidate['source_seed']}"
                )
                if "material_ids" in audit:
                    material_ids = list(audit["material_ids"])
                else:
                    candidate_slot = int(str(candidate["scene_id"]).rsplit("_", 1)[1]) - 1
                    primary_slot = candidate_slot % 4
                    material_ids = [
                        f"kino_v4_confirm_{domain}_pbr_{primary_slot:02d}",
                        f"kino_v4_confirm_{domain}_pbr_{(primary_slot + 2) % 4:02d}",
                    ]
                scenes.append(
                    {
                        "scene_id": str(candidate["scene_id"]),
                        "scene_index": len(scenes),
                        "domain": domain,
                        "source": (
                            "metric-wild-hybrid-with-HDRI"
                            if domain == "wild"
                            else "EmbodiedGen-v2-new-RoomGen-source"
                        ),
                        "source_scene_id": str(source_scene_id),
                        "source_scene_seed": int(
                            candidate.get("source_seed", candidate.get("metric_geometry_seed"))
                        ),
                        "runtime_seed": int(candidate["runtime_seed"]),
                        "episode_usd": str(episode),
                        "episode_sha256": episode_hash,
                        "compiled_audit": str(compiled),
                        "compiled_audit_sha256": compiled_hash,
                        "geometry_hash": geometry_hash,
                        "material_ids": material_ids,
                        "terminal_scene_admission": str(audit_path),
                        "terminal_scene_admission_sha256": _sha256(audit_path),
                        "selected_by_model_blind_prefix_rule": True,
                    }
                )
                admitted += 1
            if admitted == 4:
                break
        if admitted != 4:
            raise RuntimeError(f"{domain}: frozen stream produced only {admitted} admissions")
        attrition[domain] = {
            "attempted_prefix": len(attempts),
            "admitted_in_attempted_prefix": admitted,
            "attempts": attempts,
        }
    if len(scenes) != 12 or [row["scene_index"] for row in scenes] != list(range(12)):
        raise RuntimeError("scene registry does not realize 12 ordered slots")
    return {
        "schema_version": "kinofail.kino-v4-confirmation-scene-registry.v1",
        "created_utc": datetime.now(UTC).isoformat(),
        "status": "frozen_model_blind_registry",
        "selection_uses_model_predictions": False,
        "scene_count": len(scenes),
        "scenes": scenes,
        "attrition": attrition,
    }


def _view_fields(primary: dict[str, Any]) -> dict[str, Any]:
    return {
        "appearance_id": primary["appearance_id"],
        "appearance_seed": primary["appearance_seed"],
        "material_family": primary["material_family"],
        "material_asset_id": primary["material_asset_id"],
        "source_material_asset_id": primary["material_asset_id"],
        "material_semantics": primary["material_semantics"],
        "material_source": primary["material_source"],
        "material_license": primary["material_license"],
        "material_physical_size_m": primary["physical_size_m"],
        "surface_state": primary["surface_state"],
        "uv_scale": primary["uv_scale"],
        "uv_rotation_deg": primary["uv_rotation_deg"],
        "uv_offset": primary["uv_offset"],
        "albedo_brightness_multiplier": primary["albedo_brightness_multiplier"],
        "normal_strength": primary["normal_strength"],
        "roughness_multiplier": primary["roughness_multiplier"],
        "render_tier": "structured_pbr",
        "triplanar": True,
    }


def _base(
    scene: dict[str, Any], material_id: str, material_slot: int, views: list[dict[str, Any]]
) -> dict[str, Any]:
    return {
        "benchmark_id": BENCHMARK,
        "design_mode": "independent_post_freeze_confirmation",
        "artifact_state": "planned",
        "evaluation_eligible": False,
        "scene_id": scene["scene_id"],
        "scene_index": scene["scene_index"],
        "scene_family": scene["scene_id"],
        "source_scene_id": scene["source_scene_id"],
        "scene_source": scene["source"],
        "scene_seed": scene["source_scene_seed"],
        "runtime_seed": scene["runtime_seed"],
        "domain": scene["domain"],
        "split": "independent_confirmation",
        "camera_profile": CAMERAS[int(scene["scene_index"]) % 3],
        "material_id": material_id,
        "material_slot": material_slot,
        "cluster_material": material_id,
        "appearance_view_count": 3,
        "appearance_views": views,
        **_view_fields(views[0]),
    }


def _scale_rows(
    design: dict[str, Any], scene: dict[str, Any], materials: dict[str, dict[str, Any]],
    operators: list[dict[str, Any]], material_id: str, material_slot: int,
) -> list[dict[str, Any]]:
    alternate_id = next(value for value in scene["material_ids"] if value != material_id)
    context_index = int(scene["scene_index"]) * 2 + material_slot
    rows = []
    for operator_index, operator in enumerate(operators):
        for replicate in range(8):
            band = "moderate" if replicate < 4 else "hard"
            point = replicate if replicate < 4 else replicate - 4
            lam = float(design["scale"][f"{band}_lambda_points"][point])
            seed = int(design["scale"]["seed_namespace_start"]) + (
                (context_index * 11 + operator_index) * 8 + replicate
            )
            nuisance = physical_nuisance(design, profile_index=replicate, physics_seed=seed)
            anomaly = interpolate_parameters(operator["M"], operator["S"], lam)
            group = _stable_id(
                "cf", PROTOCOL, scene["scene_id"], material_id, operator["id"], replicate
            )
            views = _appearance_views(
                scene=scene,
                primary_material=materials[material_id],
                alternate_material=materials[alternate_id],
                seed=seed,
                appearance_key=f"scale-{operator['id']}-{replicate}",
            )
            common = _base(scene, material_id, material_slot, views)
            for condition in CONDITIONS:
                active = condition == "anomaly"
                episode_id = f"{group}_{'anomaly' if active else 'nominal'}"
                rows.append(
                    {
                        "schema_version": "kinofail.kino-v4-confirmation-schedule.v1",
                        **common,
                        "battery": "scale",
                        "episode_id": episode_id,
                        "counterfactual_group_id": group,
                        "condition": condition,
                        "active_operator": operator["id"] if active else None,
                        "target_operator": operator["id"],
                        "attribution_category": operator["category"] if active else "nominal",
                        "severity_id": band,
                        "severity_rank": 2 if band == "moderate" else 3,
                        "parameter_interpolation": {
                            "lambda": lam,
                            "point_index_within_band": point,
                            "rule": "P=(1-lambda)*M+lambda*S",
                        },
                        "physics_parameters": anomaly if active else _counterfactual(operator, anomaly),
                        "physical_realization": operator["physical_realization"],
                        "geometry_profile": operator["geometry_profile"],
                        "geometry_id": f"{scene['scene_id']}::{operator['geometry_profile']}",
                        "geometry_hash": scene["geometry_hash"],
                        "operator_seed": seed,
                        "physical_seed": seed,
                        "physical_nuisance": nuisance,
                        "operator_index": operator_index,
                        "replicate_index": replicate,
                        "texture_swap_group_id": _stable_id("ts", group, condition),
                        "required_outputs": _outputs("scale", material_id, operator["id"], episode_id),
                    }
                )
    return rows


def _t2_case(
    design: dict[str, Any], scene: dict[str, Any], materials: dict[str, dict[str, Any]],
    material_id: str, material_slot: int, local_case: int,
) -> dict[str, Any]:
    alternate_id = next(value for value in scene["material_ids"] if value != material_id)
    context = int(scene["scene_index"]) * 2 + material_slot
    seed = int(design["conflict"]["seed_namespace_start"]) + context * 12 + local_case
    nuisance = physical_nuisance(design, profile_index=local_case, physics_seed=seed)
    case_id = _stable_id("c2t2", scene["scene_id"], material_id, local_case, seed)
    return {
        "schema_version": "kinofail.kino-v4-confirmation-c2-t2.v1",
        "benchmark_id": BENCHMARK,
        "cell": "T2_vision_decisive",
        "case_id": case_id,
        "case_seed": seed,
        "reset_seed": seed,
        "cause_visual_seed": seed,
        "scene_id": scene["scene_id"],
        "scene_index": scene["scene_index"],
        "scene_cluster": scene["scene_id"],
        "scene_family": scene["scene_id"],
        "source_scene_id": scene["source_scene_id"],
        "scene_seed": scene["source_scene_seed"],
        "runtime_seed": scene["runtime_seed"],
        "domain": scene["domain"],
        "split": "independent_confirmation",
        "material_id": material_id,
        "cluster_material": material_id,
        "material_slot": material_slot,
        "context_index": context,
        "local_case_index": local_case,
        "profile_index": local_case % 8,
        "camera_profile": CAMERAS[int(scene["scene_index"]) % 3],
        "physical_nuisance": nuisance,
        "decision_progress_m": 0.22,
        "cause_region": {
            "start_progress_m": 0.9,
            "length_m": 0.62,
            "half_width_m": 0.38,
            "base_z_m": 0.006,
        },
        "causes": [
            {"target_operator": "O2_compliance", "attribution_category": "compliant_terrain", "visible_physical_cue": "soft_surface_deformation"},
            {"target_operator": "O4_tether", "attribution_category": "adhesion", "visible_physical_cue": "overlapping_film_and_creases"},
        ],
        "appearance_views": _appearance_views(
            scene=scene,
            primary_material=materials[material_id],
            alternate_material=materials[alternate_id],
            seed=seed,
            appearance_key=f"t2-{local_case}",
        ),
        "shared_prefix_contract": {
            "one_physics_rollout_for_both_causes": True,
            "visual_rerenders_do_not_advance_physics": True,
            "proprio_bytes_reused_without_relabeling_or_perturbation": True,
            "operator_physics_activates_only_after_the_decision_boundary": True,
        },
    }


def _t3(
    design: dict[str, Any], scene: dict[str, Any], materials: dict[str, dict[str, Any]],
    operator_map: dict[str, dict[str, Any]], material_id: str, material_slot: int,
    local_case: int,
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    alternate_id = next(value for value in scene["material_ids"] if value != material_id)
    context = int(scene["scene_index"]) * 2 + material_slot
    seed = int(design["conflict"]["seed_namespace_start"]) + 288 + context * 12 + local_case
    nuisance = physical_nuisance(design, profile_index=local_case, physics_seed=seed)
    band = "moderate" if local_case < 6 else "hard"
    point = (local_case if local_case < 6 else local_case - 6) % 4
    lam = float(design["scale"][f"{band}_lambda_points"][point])
    case_id = _stable_id("c2t3", scene["scene_id"], material_id, local_case, seed)
    views = _appearance_views(
        scene=scene,
        primary_material=materials[material_id],
        alternate_material=materials[alternate_id],
        seed=seed,
        appearance_key=f"t3-{local_case}",
    )
    common = _base(scene, material_id, material_slot, views)
    rows = []
    groups = {}
    for operator_id in ("O7_visual_remap", "O8_invisible_collider"):
        operator = operator_map[operator_id]
        anomaly = interpolate_parameters(operator["M"], operator["S"], lam)
        group = _stable_id("cf", "t3", scene["scene_id"], material_id, local_case, operator_id)
        groups[operator_id] = group
        for condition in CONDITIONS:
            active = condition == "anomaly"
            episode_id = f"{group}_{'anomaly' if active else 'nominal'}"
            rows.append(
                {
                    "schema_version": "kinofail.kino-v4-confirmation-schedule.v1",
                    **common,
                    "battery": "c2_t3",
                    "cell": "T3_proprio_decisive",
                    "case_id": case_id,
                    "case_seed": seed,
                    "episode_id": episode_id,
                    "counterfactual_group_id": group,
                    "condition": condition,
                    "active_operator": operator_id if active else None,
                    "target_operator": operator_id,
                    "attribution_category": operator["category"] if active else "nominal",
                    "severity_id": band,
                    "severity_rank": 2 if band == "moderate" else 3,
                    "parameter_interpolation": {"lambda": lam, "point_index_within_band": point, "rule": "P=(1-lambda)*M+lambda*S"},
                    "physics_parameters": anomaly if active else _counterfactual(operator, anomaly),
                    "physical_realization": operator["physical_realization"],
                    "geometry_profile": operator["geometry_profile"],
                    "geometry_id": f"{scene['scene_id']}::{operator['geometry_profile']}",
                    "geometry_hash": scene["geometry_hash"],
                    "operator_seed": seed,
                    "physical_seed": seed,
                    "physical_nuisance": nuisance,
                    "local_case_index": local_case,
                    "texture_swap_group_id": _stable_id("ts", case_id, condition),
                    "required_outputs": _outputs("c2_t3", material_id, operator_id, episode_id),
                }
            )
    case = {
        "benchmark_id": BENCHMARK,
        "cell": "T3_proprio_decisive",
        "case_id": case_id,
        "case_seed": seed,
        "scene_id": scene["scene_id"],
        "scene_index": scene["scene_index"],
        "scene_cluster": scene["scene_id"],
        "scene_family": scene["scene_id"],
        "source_scene_id": scene["source_scene_id"],
        "domain": scene["domain"],
        "material_id": material_id,
        "cluster_material": material_id,
        "material_slot": material_slot,
        "context_index": context,
        "local_case_index": local_case,
        "physical_nuisance": nuisance,
        "severity_id": band,
        "lambda": lam,
        "o7_source_physics_group_id": groups["O7_visual_remap"],
        "o8_source_physics_group_id": groups["O8_invisible_collider"],
    }
    return rows, case


def _conflict_rows(case: dict[str, Any], candidates: tuple[str, str]) -> list[dict[str, Any]]:
    return [
        {
            "schema_version": "kinofail.kino-v4-confirmation-conflict-record.v1",
            "benchmark_id": BENCHMARK,
            "cell": case["cell"],
            "case_id": case["case_id"],
            "case_seed": case["case_seed"],
            "scene_index": case["scene_index"],
            "scene_id": case["scene_id"],
            "scene_cluster": case["scene_id"],
            "scene_family": case["scene_id"],
            "domain": case["domain"],
            "material_id": case["material_id"],
            "material_slot": case["material_slot"],
            "cluster_material": case["material_id"],
            "local_case_index": case["local_case_index"],
            "physical_nuisance": case["physical_nuisance"],
            "cause_id": candidate,
            "candidate_operator": candidate,
            "appearance_view_index": view,
            "appearance_view_id": "primary" if view == 0 else f"swap_{view:02d}",
            "artifact_state": "planned",
        }
        for candidate in candidates
        for view in range(3)
    ]


def _collection_protocol(
    scene_id: str, battery: str, schedule: Path, registry: Path, pairs: int
) -> dict[str, Any]:
    return {
        "schema_version": "kinofail.formal-collection-protocol.v1",
        "protocol_id": f"{PROTOCOL}-{scene_id}-{battery}",
        "status": "frozen",
        "confirmatory": True,
        "benchmark_id": BENCHMARK,
        "scene_id": scene_id,
        "schedule_path": str(schedule),
        "schedule_sha256": _sha256(schedule),
        "scene_registry_path": str(registry),
        "scene_registry_sha256": _sha256(registry),
        "material_lock_path": str(LOCK),
        "material_lock_sha256": _sha256(LOCK),
        "collector_path": str(COLLECTOR),
        "collector_sha256": _sha256(COLLECTOR),
        "runtime_manifest_path": str(RUNTIME_MANIFEST),
        "runtime_manifest_sha256": _sha256(RUNTIME_MANIFEST),
        "f0_manifest_path": str(F0),
        "f0_manifest_sha256": _sha256(F0),
        "design_path": str(F1),
        "design_sha256": _sha256(F1),
        "collection_contract": {
            "counterfactual_pairs": pairs,
            "physical_episodes": pairs * 2,
            "appearance_views_per_episode": 3,
            "per_pair_physical_nuisance_exact_and_shared": True,
            "result_dependent_retry_permitted": False,
        },
    }


def _snapshot_protocol(
    scene_id: str, battery: str, schedule: Path, collection: dict[str, Any]
) -> dict[str, Any]:
    corpus = DATA_ROOT / "corpus" / scene_id
    output = DERIVED / scene_id / battery / "snapshots"
    return {
        "schema_version": "kinofail.realistic-snapshot-protocol.v2",
        "protocol_id": f"{PROTOCOL}-{scene_id}-{battery}-snapshots",
        "status": "frozen_before_collection",
        "development_only": False,
        "a8_in_scope": False,
        "source_schedule": str(schedule),
        "source_schedule_sha256": _sha256(schedule),
        "source_corpus_root": str(corpus),
        "output_dir": str(output),
        "appearance_interventions": {
            "include_all_manifest_views": True,
            "count_views_as_independent_physical_samples": False,
            "require_timestamp_identity_across_views": True,
            "share_proprio_across_views": True,
        },
        "selection": {
            "require_complete_counterfactual_pair": True,
            "require_evaluation_eligible": False,
            "require_runtime_validation_passed": True,
            "required_collection_protocol_id": collection["protocol_id"],
            "operators_with_admitted_event_adapter": [row["id"] for row in _operator_specs(OPERATORS)],
            "allow_nonblocking_runtime_issue_suffixes": ["appearance_effect_too_small", "rgb_spatial_contrast_too_low"],
            "on_temporal_alignment_failure": "exclude_complete_pair_and_audit",
        },
        "temporal_alignment": {
            "anchor": "first_predeclared_privileged_operator_event_in_anomaly_only",
            "decision_delay_s": 0.0,
            "decision_delay_s_by_operator": {"O5_payload": 0.4, "O10_effort_decay": 0.7, "O11_obs_bias": 0.4},
            "minimum_decision_time_s": 0.42,
            "nominal_alignment": "reuse_anomaly_event_and_decision_times_exactly",
            "rgb_offsets_from_decision_s": [-0.4, -0.3, -0.2, -0.1, 0.0],
            "max_rgb_target_skew_s": 0.041,
            "proprio_samples": 21,
            "proprio_window_s": 0.5,
            "max_proprio_end_skew_s": 0.021,
        },
    }


def main() -> int:
    if OUTPUT.exists():
        raise FileExistsError(OUTPUT)
    for path in (F0, F1, CANDIDATES, OPERATORS, LOCK, COLLECTOR, T2_COLLECTOR, T2_VISUAL_HELPER):
        if not path.is_file():
            raise FileNotFoundError(path)
    f0, f1, candidates, lock = _json(F0), _json(F1), _json(CANDIDATES), _json(LOCK)
    if f0["status"] != "frozen_before_independent_confirmation_generation":
        raise RuntimeError("invalid F0")
    if f1["status"] != "frozen_before_material_sync_and_scene_generation":
        raise RuntimeError("invalid F1")
    design = f1["design"]
    registry = _registry(candidates)
    OUTPUT.mkdir(parents=True)
    registry_path = OUTPUT / "scene_registry.json"
    _write_json(registry_path, registry)
    materials = {row["id"]: row for row in lock["materials"]}
    operators = _operator_specs(OPERATORS)
    operator_map = {row["id"]: row for row in operators}
    scale_by_scene, t3_by_scene = defaultdict(list), defaultdict(list)
    t2_cases, t3_cases, conflict = [], [], []
    for scene in registry["scenes"]:
        for material_slot, material_id in enumerate(scene["material_ids"]):
            scale_by_scene[scene["scene_id"]].extend(
                _scale_rows(design, scene, materials, operators, material_id, material_slot)
            )
            for local_case in range(12):
                t2 = _t2_case(design, scene, materials, material_id, material_slot, local_case)
                t2_cases.append(t2)
                conflict.extend(_conflict_rows(t2, ("O2_compliance", "O4_tether")))
                t3_rows, t3 = _t3(
                    design, scene, materials, operator_map, material_id, material_slot, local_case
                )
                t3_by_scene[scene["scene_id"]].extend(t3_rows)
                t3_cases.append(t3)
                conflict.extend(_conflict_rows(t3, ("O7_visual_remap", "O8_invisible_collider")))
    scale = [row for scene in registry["scenes"] for row in scale_by_scene[scene["scene_id"]]]
    if len(scale) != 4224 or len({row["counterfactual_group_id"] for row in scale}) != 2112:
        raise RuntimeError("scale count drift")
    if len(t2_cases) != 288 or len(t3_cases) != 288 or len(conflict) != 3456:
        raise RuntimeError("conflict count drift")
    if sum(map(len, t3_by_scene.values())) != 1152:
        raise RuntimeError("T3 physical-row count drift")
    t2_root = OUTPUT / "schedules/c2_t2"
    t2_schedule = t2_root / "schedule.jsonl"
    _write_jsonl(t2_schedule, t2_cases)
    t2_audit = {
        "schema_version": "kinofail.kino-v4-confirmation-t2-design-audit.v1",
        "passed": True,
        "cases": len(t2_cases),
        "model_rows": len(t2_cases) * 6,
        "scene_count": 12,
        "material_contexts": 24,
    }
    _write_json(t2_root / "audit.json", t2_audit)
    t2_protocol = {
        "schema_version": "kinofail.confirmatory-c1-collection-protocol.v1",
        "protocol_id": f"{PROTOCOL}-t2-shared-prefix",
        "status": "frozen",
        "design_tag": BENCHMARK,
        "schedule_path": str(t2_schedule),
        "schedule_sha256": _sha256(t2_schedule),
        "design_audit_path": str(t2_root / "audit.json"),
        "design_audit_sha256": _sha256(t2_root / "audit.json"),
        "scene_registry_path": str(registry_path),
        "scene_registry_sha256": _sha256(registry_path),
        "asset_lock_path": str(LOCK),
        "asset_lock_sha256": _sha256(LOCK),
        "collector_path": str(T2_COLLECTOR),
        "collector_sha256": _sha256(T2_COLLECTOR),
        "visual_helper_path": str(T2_VISUAL_HELPER),
        "visual_helper_sha256": _sha256(T2_VISUAL_HELPER),
        "model_or_endpoint_outcomes_available_at_freeze": False,
        "cases": 288,
        "samples": 1728,
    }
    _write_json(t2_root / "collection_protocol.json", t2_protocol)
    for scene in registry["scenes"]:
        scene_id = scene["scene_id"]
        for battery, rows in (("scale", scale_by_scene[scene_id]), ("c2_t3", t3_by_scene[scene_id])):
            root = OUTPUT / "schedules/scenes" / scene_id / battery
            schedule = root / "schedule.jsonl"
            _write_jsonl(schedule, rows)
            pairs = len({row["counterfactual_group_id"] for row in rows})
            collection = _collection_protocol(scene_id, battery, schedule, registry_path, pairs)
            _write_json(root / "collection_protocol.json", collection)
            _write_json(root / "snapshot_protocol.json", _snapshot_protocol(scene_id, battery, schedule, collection))
    global_root = OUTPUT / "schedules/global"
    _write_jsonl(global_root / "scale_schedule.jsonl", scale)
    _write_jsonl(global_root / "conflict_model_records.jsonl", conflict)
    _write_jsonl(global_root / "t3_cases.jsonl", t3_cases)
    audit = {
        "schema_version": "kinofail.kino-v4-confirmation-schedule-audit.v1",
        "created_utc": datetime.now(UTC).isoformat(),
        "status": "sealed_before_physical_collection",
        "passed": True,
        "selection_uses_model_predictions": False,
        "counts": {
            "scenes": 12,
            "materials": 12,
            "scene_material_contexts": 24,
            "scale_counterfactual_pairs": 2112,
            "scale_physical_episodes": 4224,
            "scale_model_rows": 12672,
            "conflict_cases": 576,
            "conflict_model_rows": 3456,
            "t2_cases": 288,
            "t3_cases": 288,
            "t3_counterfactual_pairs": 576,
            "t3_physical_episodes": 1152,
        },
        "source_sha256": {
            "f0": _sha256(F0),
            "f1": _sha256(F1),
            "registry": _sha256(registry_path),
            "material_lock": _sha256(LOCK),
            "collector": _sha256(COLLECTOR),
            "t2_collector": _sha256(T2_COLLECTOR),
            "script": _sha256(Path(__file__).resolve()),
        },
    }
    _write_json(OUTPUT / "schedule_audit.json", audit)
    print(json.dumps(audit, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
