#!/usr/bin/env python3
"""Freeze the model-blind Scale/T2 breadth extension to all 191 scenes.

The original twelve-scene confirmation corpus remains the dense core.  This
extension adds every one of the 179 already-admitted T3 extension scenes to
Scale and T2.  Sampling is fixed before collection and does not depend on any
model prediction or endpoint outcome.
"""

from __future__ import annotations

import copy
import hashlib
import json
import sys
from collections import Counter
from datetime import UTC, datetime
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import scripts.build_kinofail_kino_v4_confirmation_schedules_v1 as base  # noqa: E402


F0 = ROOT / "outputs/freeze/kino_v4_dinov2large_terrain448_f0/freeze_manifest.json"
F1 = ROOT / "outputs/freeze/kino_v4_confirmation_v1_f1/freeze_manifest.json"
DESIGN = ROOT / "configs/data/kinofail_kino_v4_confirmation_design_v1.json"
OPERATORS = ROOT / "configs/data/kinofail_realistic.yaml"
LOCK = Path(
    "/data/eureka/kinofail_kino_v4_confirmation_v1/assets/terrain_pbr/"
    "terrain_assets.lock.json"
)
CORE_REGISTRY = ROOT / "outputs/kinofail_kino_v4_confirmation_v1/scene_registry.json"
EXTENSION_REGISTRY = (
    ROOT
    / "outputs/kinofail_kino_v4_confirmation_t3_extension_f4h/design/scene_registry.json"
)
CORE_SCALE = (
    ROOT
    / "outputs/kinofail_kino_v4_confirmation_v1/schedules/global/scale_schedule.jsonl"
)
CORE_T2 = (
    ROOT
    / "outputs/kinofail_kino_v4_confirmation_v1/schedules/c2_t2/schedule.jsonl"
)
COLLECTOR = ROOT / "scripts/isaac_collect_kinofail_kino_v4_confirmation_pair_v1.py"
T2_COLLECTOR = ROOT / "scripts/isaac_collect_kinofail_confirmatory_t2_v1.py"
T2_VISUAL_HELPER = ROOT / "kino_vla/sim/c1_causal_visuals.py"
RUNTIME_MANIFEST = ROOT / "kino_vla/data/runtime_manifest.py"

OUTPUT = ROOT / "outputs/kinofail_kino_v4_all191_scale_t2_extension_v1"
DATA_ROOT = Path("/data/eureka/kinofail_kino_v4_all191_scale_t2_extension_v1")
DERIVED_ROOT = Path("/data/eureka/kinovla_outputs/kino_v4_all191_scale_t2_extension_v1")
BENCHMARK = "kinofail_kino_v4_all191_v1"
PROTOCOL = "kinofail-kino-v4-all191-scale-t2-extension-v1"

SCALE_REPLICATES = (0, 4)
T2_LOCAL_CASES = (0, 6)
SCALE_SEED_NAMESPACE = 2_384_000_000
CONFLICT_SEED_NAMESPACE = 2_385_000_000


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _json(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise TypeError(path)
    return value


def _jsonl(path: Path) -> list[dict[str, Any]]:
    return [
        json.loads(line)
        for line in path.read_text(encoding="utf-8").splitlines()
        if line
    ]


def _write_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(value, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )


def _write_jsonl(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        "".join(json.dumps(row, sort_keys=True) + "\n" for row in rows),
        encoding="utf-8",
    )


def _collection_protocol(
    *, scene_id: str, schedule: Path, registry: Path, pairs: int
) -> dict[str, Any]:
    return {
        "schema_version": "kinofail.formal-collection-protocol.v1",
        "protocol_id": f"{PROTOCOL}-{scene_id}-scale",
        "status": "frozen",
        "confirmatory": True,
        "benchmark_id": BENCHMARK,
        "scene_id": scene_id,
        "battery": "scale",
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
        "design_path": str(DESIGN),
        "design_sha256": _sha256(DESIGN),
        "collection_contract": {
            "counterfactual_pairs": pairs,
            "physical_episodes": pairs * 2,
            "appearance_views_per_episode": 3,
            "per_pair_physical_nuisance_exact_and_shared": True,
            "result_dependent_retry_permitted": False,
        },
    }


def _snapshot_protocol(
    *, scene_id: str, schedule: Path, collection: dict[str, Any]
) -> dict[str, Any]:
    return {
        "schema_version": "kinofail.realistic-snapshot-protocol.v2",
        "protocol_id": f"{PROTOCOL}-{scene_id}-scale-snapshots",
        "status": "frozen_before_collection",
        "development_only": False,
        "a8_in_scope": False,
        "source_schedule": str(schedule),
        "source_schedule_sha256": _sha256(schedule),
        "source_corpus_root": str(DATA_ROOT / "corpus" / scene_id),
        "output_dir": str(DERIVED_ROOT / "shards" / scene_id / "scale/snapshots"),
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
            "operators_with_admitted_event_adapter": [
                row["id"] for row in base._operator_specs(OPERATORS)
            ],
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
    }


def main() -> int:
    if OUTPUT.exists():
        raise FileExistsError(OUTPUT)
    required = (
        F0,
        F1,
        DESIGN,
        OPERATORS,
        LOCK,
        CORE_REGISTRY,
        EXTENSION_REGISTRY,
        CORE_SCALE,
        CORE_T2,
        COLLECTOR,
        T2_COLLECTOR,
        T2_VISUAL_HELPER,
        RUNTIME_MANIFEST,
    )
    for path in required:
        if not path.is_file():
            raise FileNotFoundError(path)

    core_registry = _json(CORE_REGISTRY)
    extension_registry = _json(EXTENSION_REGISTRY)
    core_scenes = list(core_registry["scenes"])
    extension_scenes = list(extension_registry["scenes"])
    core_ids = {str(row["scene_id"]) for row in core_scenes}
    extension_ids = {str(row["scene_id"]) for row in extension_scenes}
    if len(core_scenes) != 12 or len(extension_scenes) != 179:
        raise RuntimeError("expected the frozen 12 + 179 scene partition")
    if core_ids & extension_ids or len(core_ids | extension_ids) != 191:
        raise RuntimeError("scene partition does not form 191 unique scenes")
    for scene in extension_scenes:
        for key in ("episode_usd", "compiled_audit", "terminal_scene_admission"):
            if not Path(str(scene[key])).is_file():
                raise FileNotFoundError(scene[key])
        audits = scene.get("registered_front_view_multianchor_audits", {})
        if set(audits) != {"route_progress_0.20_m", "route_progress_0.50_m"}:
            raise RuntimeError(f"missing multianchor visual gate: {scene['scene_id']}")
        for record in audits.values():
            path = Path(str(record["path"]))
            if not path.is_file() or _sha256(path) != str(record["sha256"]):
                raise RuntimeError(f"multianchor audit drift: {scene['scene_id']}")

    design = copy.deepcopy(_json(F1)["design"])
    design["scale"]["seed_namespace_start"] = SCALE_SEED_NAMESPACE
    design["conflict"]["seed_namespace_start"] = CONFLICT_SEED_NAMESPACE
    materials = {row["id"]: row for row in _json(LOCK)["materials"]}
    operators = base._operator_specs(OPERATORS)

    # The imported builders intentionally reuse the exact F0/F1 physics,
    # rendering, and counterfactual contracts while changing only this new
    # benchmark namespace.
    base.PROTOCOL = PROTOCOL
    base.BENCHMARK = BENCHMARK

    OUTPUT.mkdir(parents=True)
    registry_path = OUTPUT / "design/scene_registry.json"
    registry = {
        "schema_version": "kinofail.kino-v4-all191-extension-scene-registry.v1",
        "created_utc": datetime.now(UTC).isoformat(),
        "status": "frozen_before_scale_or_t2_collection",
        "selection_uses_model_predictions": False,
        "scene_count": 179,
        "final_union_scene_count": 191,
        "core_scene_ids": sorted(core_ids),
        "scenes": extension_scenes,
    }
    _write_json(registry_path, registry)

    scale_rows: list[dict[str, Any]] = []
    t2_cases: list[dict[str, Any]] = []
    for scene in extension_scenes:
        scene_id = str(scene["scene_id"])
        material_slot = int(scene["scene_index"]) % 2
        material_id = str(scene["material_ids"][material_slot])
        dense = base._scale_rows(
            design,
            scene,
            materials,
            operators,
            material_id,
            material_slot,
        )
        rows = [row for row in dense if int(row["replicate_index"]) in SCALE_REPLICATES]
        for row in rows:
            row["design_mode"] = "frozen_all191_breadth_extension"
            row["split"] = "all191_scale_t2_extension"
        if len(rows) != 44:
            raise RuntimeError(f"Scale count drift for {scene_id}: {len(rows)}")
        scale_rows.extend(rows)

        scene_root = OUTPUT / "schedules/scenes" / scene_id / "scale"
        schedule = scene_root / "schedule.jsonl"
        _write_jsonl(schedule, rows)
        collection = _collection_protocol(
            scene_id=scene_id,
            schedule=schedule,
            registry=registry_path,
            pairs=22,
        )
        _write_json(scene_root / "collection_protocol.json", collection)
        _write_json(
            scene_root / "snapshot_protocol.json",
            _snapshot_protocol(
                scene_id=scene_id,
                schedule=schedule,
                collection=collection,
            ),
        )

        for slot, candidate_material in enumerate(scene["material_ids"]):
            for local_case in T2_LOCAL_CASES:
                case = base._t2_case(
                    design,
                    scene,
                    materials,
                    str(candidate_material),
                    slot,
                    local_case,
                )
                case["design_mode"] = "frozen_all191_breadth_extension"
                case["split"] = "all191_scale_t2_extension"
                t2_cases.append(case)

    scale_schedule = OUTPUT / "schedules/global/scale_schedule.jsonl"
    t2_schedule = OUTPUT / "schedules/c2_t2/schedule.jsonl"
    _write_jsonl(scale_schedule, scale_rows)
    _write_jsonl(t2_schedule, t2_cases)

    t2_audit = {
        "schema_version": "kinofail.kino-v4-all191-t2-design-audit.v1",
        "passed": True,
        "selection_uses_model_predictions": False,
        "cases": len(t2_cases),
        "physical_cause_units": len(t2_cases) * 2,
        "model_rows": len(t2_cases) * 6,
        "extension_scene_count": 179,
        "final_union_scene_count": 191,
        "material_contexts_per_scene": 2,
        "local_case_indices": list(T2_LOCAL_CASES),
    }
    t2_audit_path = OUTPUT / "schedules/c2_t2/audit.json"
    _write_json(t2_audit_path, t2_audit)
    t2_protocol = {
        "schema_version": "kinofail.confirmatory-c1-collection-protocol.v1",
        "protocol_id": f"{PROTOCOL}-t2-shared-prefix",
        "status": "frozen",
        "design_tag": BENCHMARK,
        "schedule_path": str(t2_schedule),
        "schedule_sha256": _sha256(t2_schedule),
        "design_audit_path": str(t2_audit_path),
        "design_audit_sha256": _sha256(t2_audit_path),
        "scene_registry_path": str(registry_path),
        "scene_registry_sha256": _sha256(registry_path),
        "asset_lock_path": str(LOCK),
        "asset_lock_sha256": _sha256(LOCK),
        "collector_path": str(T2_COLLECTOR),
        "collector_sha256": _sha256(T2_COLLECTOR),
        "visual_helper_path": str(T2_VISUAL_HELPER),
        "visual_helper_sha256": _sha256(T2_VISUAL_HELPER),
        "model_or_endpoint_outcomes_available_at_freeze": False,
        "cases": len(t2_cases),
        "samples": len(t2_cases) * 6,
    }
    _write_json(OUTPUT / "schedules/c2_t2/collection_protocol.json", t2_protocol)

    core_scale = _jsonl(CORE_SCALE)
    core_t2 = _jsonl(CORE_T2)
    scale_pair_ids = {str(row["counterfactual_group_id"]) for row in scale_rows}
    core_scale_pair_ids = {str(row["counterfactual_group_id"]) for row in core_scale}
    t2_ids = {str(row["case_id"]) for row in t2_cases}
    core_t2_ids = {str(row["case_id"]) for row in core_t2}
    if scale_pair_ids & core_scale_pair_ids or t2_ids & core_t2_ids:
        raise RuntimeError("extension identifiers overlap the dense core")
    if len(scale_pair_ids) != 179 * 22 or len(t2_ids) != 179 * 4:
        raise RuntimeError("extension identifier uniqueness failed")

    per_scene_scale = Counter(str(row["scene_id"]) for row in scale_rows)
    per_scene_t2 = Counter(str(row["scene_id"]) for row in t2_cases)
    per_scene_operator = Counter(
        (str(row["scene_id"]), str(row["target_operator"]))
        for row in scale_rows
        if row["condition"] == "anomaly"
    )
    if set(per_scene_scale.values()) != {44} or set(per_scene_t2.values()) != {4}:
        raise RuntimeError("per-scene schedule balance failed")
    if set(per_scene_operator.values()) != {2} or len(per_scene_operator) != 179 * 11:
        raise RuntimeError("Scale operator-by-severity coverage failed")

    audit = {
        "schema_version": "kinofail.kino-v4-all191-extension-freeze.v1",
        "created_utc": datetime.now(UTC).isoformat(),
        "status": "sealed_before_scale_or_t2_collection",
        "passed": True,
        "selection_uses_model_predictions": False,
        "result_dependent_replacement_permitted": False,
        "scientific_contract": {
            "dense_core_preserved": True,
            "dense_core_scenes": 12,
            "breadth_extension_scenes": 179,
            "final_scenes_per_battery": 191,
            "scale": {
                "physical_material_contexts_per_extension_scene": 1,
                "appearance_views_per_episode": 3,
                "all_11_operators_per_extension_scene": True,
                "severity_bands_per_operator": ["moderate", "hard"],
                "replicate_indices": list(SCALE_REPLICATES),
                "counterfactual_pairs_per_extension_scene": 22,
            },
            "t2": {
                "physical_material_contexts_per_extension_scene": 2,
                "appearance_views_per_cause": 3,
                "local_case_indices": list(T2_LOCAL_CASES),
                "shared_prefix_counterfactual_certificate": True,
            },
            "t3": {
                "reuse_existing_model_blind_191_scene_confirmation": True,
                "new_physical_collection_required": False,
            },
        },
        "counts": {
            "final_union_scenes": 191,
            "extension_scenes": 179,
            "scale_extension_counterfactual_pairs": len(scale_pair_ids),
            "scale_extension_physical_episodes": len(scale_rows),
            "scale_extension_model_views": len(scale_rows) * 3,
            "scale_final_planned_pairs_including_dense_core": len(scale_pair_ids)
            + len(core_scale_pair_ids),
            "scale_final_planned_episodes_including_dense_core": len(scale_rows)
            + len(core_scale),
            "t2_extension_cases": len(t2_cases),
            "t2_extension_physical_cause_units": len(t2_cases) * 2,
            "t2_extension_model_views": len(t2_cases) * 6,
            "t2_final_planned_cases_including_dense_core": len(t2_cases) + len(core_t2),
        },
        "domain_counts": dict(Counter(str(row["domain"]) for row in extension_scenes)),
        "seed_namespaces": {
            "scale": SCALE_SEED_NAMESPACE,
            "t2": CONFLICT_SEED_NAMESPACE,
        },
        "source_sha256": {
            "f0": _sha256(F0),
            "f1": _sha256(F1),
            "design": _sha256(DESIGN),
            "core_registry": _sha256(CORE_REGISTRY),
            "extension_registry": _sha256(EXTENSION_REGISTRY),
            "extension_registry_copy": _sha256(registry_path),
            "core_scale_schedule": _sha256(CORE_SCALE),
            "core_t2_schedule": _sha256(CORE_T2),
            "material_lock": _sha256(LOCK),
            "scale_collector": _sha256(COLLECTOR),
            "t2_collector": _sha256(T2_COLLECTOR),
            "builder": _sha256(Path(__file__).resolve()),
        },
    }
    _write_json(OUTPUT / "design/freeze_manifest.json", audit)
    print(json.dumps(audit, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
