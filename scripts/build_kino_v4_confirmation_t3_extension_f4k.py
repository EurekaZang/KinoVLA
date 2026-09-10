#!/usr/bin/env python3
"""Build and seal T3 schedules for model-blind F4h-admitted scenes."""

from __future__ import annotations

import hashlib
import json
import math
import os
import sys
from datetime import UTC, datetime
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from scripts import build_kinofail_kino_v4_confirmation_schedules_v1 as base  # noqa: E402
from scripts.build_kinofail_unified_confirmatory_v1_schedules import _operator_specs  # noqa: E402


F0 = ROOT / "outputs/freeze/kino_v4_dinov2large_terrain448_f0/freeze_manifest.json"
F1 = ROOT / "outputs/freeze/kino_v4_confirmation_v1_f1/freeze_manifest.json"
F4H = ROOT / "outputs/freeze/kino_v4_confirmation_t3_extension_f4h/seal_manifest.json"
F4I = ROOT / "outputs/freeze/kino_v4_confirmation_t3_extension_f4i/amendment_manifest.json"
F4J = ROOT / "outputs/freeze/kino_v4_confirmation_t3_extension_f4j/handoff_manifest.json"
F4P = ROOT / "outputs/freeze/kino_v4_confirmation_t3_reserve_f4p/amendment_manifest.json"
F4Q = ROOT / "outputs/freeze/kino_v4_confirmation_t3_extension_f4q/handoff_manifest.json"
F4R = ROOT / "outputs/freeze/kino_v4_confirmation_t3_extension_f4r/amendment_manifest.json"
F4S = ROOT / "outputs/freeze/kino_v4_confirmation_t3_extension_f4s/amendment_manifest.json"
F4T = ROOT / "outputs/freeze/kino_v4_confirmation_t3_reserve_f4t/amendment_manifest.json"
F4U = ROOT / "outputs/freeze/kino_v4_confirmation_t3_extension_f4u/handoff_manifest.json"
F4V = ROOT / "outputs/freeze/kino_v4_confirmation_t3_reserve_order_f4v/amendment_manifest.json"
F4W = ROOT / "outputs/freeze/kino_v4_confirmation_t3_extension_f4w/handoff_manifest.json"
F4X = ROOT / "outputs/freeze/kino_v4_confirmation_t3_reserve_f4x/amendment_manifest.json"
F4Y = ROOT / "outputs/freeze/kino_v4_confirmation_t3_extension_f4y/handoff_manifest.json"
F4Z = ROOT / "outputs/freeze/kino_v4_confirmation_t3_reserve_f4z/amendment_manifest.json"
F4AA = ROOT / "outputs/freeze/kino_v4_confirmation_t3_extension_f4aa/handoff_manifest.json"
ADMISSION = ROOT / "outputs/kinofail_kino_v4_confirmation_t3_extension_f4h/scene_admission_audit.json"
CANDIDATES = ROOT / "outputs/freeze/kino_v4_confirmation_t3_reserve_f4z/scene_candidates.json"
OPERATORS = ROOT / "configs/data/kinofail_realistic.yaml"
LOCK = Path("/data/eureka/kinofail_kino_v4_confirmation_v1/assets/terrain_pbr/terrain_assets.lock.json")
RUNTIME = ROOT / "kino_vla/data/runtime_manifest.py"
COLLECTOR_IDENTITY = ROOT / "scripts/isaac_collect_kinofail_confirmatory_pair_v5.py"
COLLECTOR_WRAPPER = ROOT / "scripts/isaac_collect_kinofail_kino_v4_confirmation_t3_runin_v1.py"
RUNNER = ROOT / "scripts/run_kinofail_t3_runin_f38_collection_v2.py"
TEMPORAL = ROOT / "kino_vla/eval/c2_temporal_v5.py"
OUT = ROOT / "outputs/kinofail_kino_v4_confirmation_t3_extension_f4h/design"
CORPUS = Path("/data/eureka/kinofail_kino_v4_confirmation_t3_extension_f4h/corpus")
RUN_OUT = ROOT / "outputs/kinofail_kino_v4_confirmation_t3_extension_f4h/collection"
PROTOCOL_ID = "kinofail-kino-v4-independent-confirmation-t3-extension-f4h"
BENCHMARK_ID = "kinofail_kino_v4_confirmation_t3_extension_f4h"


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def load(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise TypeError(path)
    return value


def write_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    os.replace(temporary, path)


def write_jsonl(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(
        "".join(json.dumps(row, sort_keys=True, separators=(",", ":")) + "\n" for row in rows),
        encoding="utf-8",
    )
    os.replace(temporary, path)


def material_id(domain: str, index: int) -> str:
    return f"kino_v4_confirm_{domain}_pbr_{index % 4:02d}"


def scene_entry(result: dict[str, Any], candidate: dict[str, Any], streams: dict[str, list[dict[str, Any]]]) -> dict[str, Any]:
    scene_id = str(candidate["scene_id"])
    domain = str(candidate["domain"])
    index = next(i for i, row in enumerate(streams[domain]) if row["scene_id"] == scene_id)
    if domain == "wild":
        root = Path("/data/eureka/kinofail_kino_v4_confirmation_v1/scenes/wild") / scene_id
        episode = root / "composition/episode.usda"
        compiled = root / "composition/compiled_scene_audit.json"
        terminal = root / "terminal_scene_admission.json"
        source = "metric-wild-hybrid-with-HDRI"
        source_scene_id = str(load(compiled)["source_scene_id"])
        source_seed = int(candidate["metric_geometry_seed"])
    else:
        root = Path("/data/eureka/kinofail_kino_v4_confirmation_v1/scenes/embodiedgen_compiled") / scene_id
        episode = root / "route_surface_v4/terrain_route_v2/episode_terrain_v2.usda"
        compiled = root / "route_surface_v4/terrain_route_v2/compiled_scene_audit.json"
        terminal = root / "terminal_scene_admission.json"
        source = "EmbodiedGen-v2-new-RoomGen-source"
        source_scene_id = f"{candidate['room_type']}_seed{candidate['source_seed']}"
        source_seed = int(candidate["source_seed"])
    front = root / "f4h_registered_front_view_gate/audit.json"
    front_p020 = root / "f4r_registered_front_view_gate_p020/audit.json"
    front_p050 = root / "f4r_registered_front_view_gate_p050/audit.json"
    for path in (episode, compiled, terminal, front, front_p020, front_p050):
        if not path.is_file():
            raise FileNotFoundError(path)
    if (
        any(load(path).get("passed") is not True for path in (front, front_p020, front_p050))
        or result.get("passed") is not True
    ):
        raise RuntimeError(f"non-admitted scene requested: {scene_id}")
    episode_hash = sha256(episode)
    compiled_hash = sha256(compiled)
    return {
        "scene_id": scene_id,
        "scene_index": 12 + int(result["candidate_position"]),
        "domain": domain,
        "source": source,
        "source_scene_id": source_scene_id,
        "source_scene_seed": source_seed,
        "runtime_seed": int(candidate["runtime_seed"]),
        "episode_usd": str(episode),
        "episode_sha256": episode_hash,
        "compiled_audit": str(compiled),
        "compiled_audit_sha256": compiled_hash,
        "geometry_hash": hashlib.sha256(f"{episode_hash}\n{compiled_hash}\n".encode()).hexdigest(),
        "material_ids": [material_id(domain, index), material_id(domain, index + 2)],
        "terminal_scene_admission": str(terminal),
        "terminal_scene_admission_sha256": sha256(terminal),
        "registered_front_view_audit": str(front),
        "registered_front_view_audit_sha256": sha256(front),
        "registered_front_view_multianchor_audits": {
            "route_progress_0.20_m": {"path": str(front_p020), "sha256": sha256(front_p020)},
            "route_progress_0.50_m": {"path": str(front_p050), "sha256": sha256(front_p050)},
        },
        "selected_without_model_outcomes": True,
    }


def protocol(scene: dict[str, Any], schedule: Path, registry: Path, groups: list[str]) -> dict[str, Any]:
    return {
        "schema_version": "kinofail.formal-collection-protocol.v1",
        "protocol_id": f"{PROTOCOL_ID}-{scene['scene_id']}",
        "status": "frozen",
        "confirmatory": True,
        "benchmark_id": BENCHMARK_ID,
        "scene_id": scene["scene_id"],
        "schedule_path": str(schedule),
        "schedule_sha256": sha256(schedule),
        "scene_registry_path": str(registry),
        "scene_registry_sha256": sha256(registry),
        "material_lock_path": str(LOCK),
        "material_lock_sha256": sha256(LOCK),
        "collector_path": str(COLLECTOR_IDENTITY),
        "collector_sha256": sha256(COLLECTOR_IDENTITY),
        "operational_wrapper_path": str(COLLECTOR_WRAPPER),
        "operational_wrapper_sha256": sha256(COLLECTOR_WRAPPER),
        "runtime_manifest_path": str(RUNTIME),
        "runtime_manifest_sha256": sha256(RUNTIME),
        "f0_manifest_path": str(F0),
        "f0_manifest_sha256": sha256(F0),
        "design_path": str(F1),
        "design_sha256": sha256(F1),
        "extension_seal_path": str(F4H),
        "extension_seal_sha256": sha256(F4H),
        "extension_admission_path": str(ADMISSION),
        "extension_admission_sha256": sha256(ADMISSION),
        "collection_contract": {
            "counterfactual_pairs": len(groups),
            "physical_episodes": 2 * len(groups),
            "appearance_views_per_episode": 3,
            "per_pair_physical_nuisance_exact_and_shared": True,
            "result_dependent_retry_permitted": False,
        },
        "allowed": {
            "conditions": ["anomaly", "nominal_counterfactual"],
            "counterfactual_group_ids": sorted(groups),
            "geometry_profiles": ["same_surface_counterfactual", "transparent_panel"],
            "physical_realizations": ["appearance_physics_swap", "transparent_acrylic"],
            "scene_families": [scene["scene_id"]],
            "severity_ids": ["hard", "moderate"],
            "target_operators": ["O7_visual_remap", "O8_invisible_collider"],
        },
    }


def main() -> int:
    if OUT.exists() or RUN_OUT.exists() or CORPUS.exists():
        raise FileExistsError("F4k output boundary already exists")
    for path in (F0, F1, F4H, F4I, F4J, F4P, F4Q, F4R, F4S, F4T, F4U, F4V, F4W, F4X, F4Y, F4Z, F4AA, ADMISSION, CANDIDATES, OPERATORS, LOCK, RUNTIME, COLLECTOR_IDENTITY, COLLECTOR_WRAPPER, RUNNER, TEMPORAL):
        if not path.is_file():
            raise FileNotFoundError(path)
    for forbidden in (
        ROOT / "outputs/eval/kino_v4_confirmation_v1/feature_seal.json",
        ROOT / "outputs/eval/kino_v4_confirmation_v1_score_once/report.json",
    ):
        if forbidden.exists():
            raise RuntimeError(f"prediction-stage artifact already exists: {forbidden}")
    admission = load(ADMISSION)
    if admission.get("passed") is not True or admission.get("model_prediction_truth_key_or_score_read") is not False:
        raise RuntimeError("extension scene-admission gate failed")
    f4x = load(F4X)
    f4y = load(F4Y)
    f4z = load(F4Z)
    f4aa = load(F4AA)
    if (
        f4x.get("passed") is not True
        or f4y.get("passed") is not True
        or f4z.get("passed") is not True
        or sha256(CANDIDATES) != f4z.get("candidate_config_sha256")
        or f4aa.get("passed") is not True
        or f4aa.get("parent_f4z_sha256") != sha256(F4Z)
        or f4aa.get("schedule_builder_sha256") != sha256(Path(__file__).resolve())
    ):
        raise RuntimeError("F4x--F4aa reserve handoff drift")
    candidates = load(CANDIDATES)
    rows = {
        str(row["scene_id"]): dict(row)
        for key in ("indoor_candidates", "wild_scenes")
        for row in candidates[key]
    }
    streams = {
        domain: [row for key in ("indoor_candidates", "wild_scenes") for row in candidates[key] if row["domain"] == domain]
        for domain in ("life", "production", "wild")
    }
    scenes = [scene_entry(result, rows[str(result["scene_id"])], streams) for result in admission["results"] if result["passed"]]
    if len(scenes) < int(admission["adaptive_target_admitted_scenes"]):
        raise RuntimeError("fewer extension scenes than the frozen adaptive target")
    registry_value = {
        "schema_version": "kinofail.kino-v4-confirmation-t3-extension-registry.v1",
        "created_utc": datetime.now(UTC).isoformat(),
        "passed": True,
        "selection_uses_model_predictions": False,
        "scene_count": len(scenes),
        "scenes": scenes,
        "failed_candidates_preserved_in_admission_audit": admission["counts"]["failed_scene_admissions"],
    }
    OUT.mkdir(parents=True, exist_ok=False)
    registry = OUT / "scene_registry.json"
    write_json(registry, registry_value)
    lock = load(LOCK)
    materials = {str(row["id"]): dict(row) for row in lock["materials"]}
    design = load(F1)["design"]
    operators = _operator_specs(OPERATORS)
    operator_map = {row["id"]: row for row in operators}
    base.BENCHMARK = BENCHMARK_ID
    base.PROTOCOL = PROTOCOL_ID
    schedules: dict[str, str] = {}
    protocols: dict[str, str] = {}
    cases = []
    selected = []
    for scene in scenes:
        scene_rows = []
        for material_slot, mat in enumerate(scene["material_ids"]):
            for local_case in range(12):
                physical_rows, case = base._t3(
                    design,
                    scene,
                    materials,
                    operator_map,
                    mat,
                    material_slot,
                    local_case,
                )
                scene_rows.extend(physical_rows)
                cases.append(case)
        root = OUT / "schedules/scenes" / str(scene["scene_id"]) / "c2_t3"
        schedule = root / "schedule.jsonl"
        write_jsonl(schedule, scene_rows)
        groups = sorted({str(row["counterfactual_group_id"]) for row in scene_rows})
        if len(scene_rows) != 96 or len(groups) != 48:
            raise RuntimeError(f"extension T3 schedule count drift: {scene['scene_id']}")
        protocol_path = root / "collection_protocol.json"
        write_json(protocol_path, protocol(scene, schedule, registry, groups))
        schedules[str(schedule.relative_to(ROOT))] = sha256(schedule)
        protocols[str(protocol_path.relative_to(ROOT))] = sha256(protocol_path)
        for case in [row for row in cases if row["scene_id"] == scene["scene_id"]]:
            selected.append(
                {
                    "case_id": case["case_id"],
                    "development_only": False,
                    "pair_ids": [case["o7_source_physics_group_id"], case["o8_source_physics_group_id"]],
                    "protocol": str(protocol_path.relative_to(ROOT)),
                    "scene_id": scene["scene_id"],
                    "schedule": str(schedule.relative_to(ROOT)),
                }
            )
    planned = 24 * len(scenes)
    if len(cases) != planned or len(selected) != planned:
        raise RuntimeError("extension selected-case count drift")
    cases_path = OUT / "schedules/global/t3_cases.jsonl"
    selected_path = OUT / "selected_cases.jsonl"
    write_jsonl(cases_path, cases)
    write_jsonl(selected_path, selected)
    seal = {
        "schema_version": "kinofail.kino-v4-confirmation-t3-extension-f4k-seal.v1",
        "audit_schema_version": "kinofail.kino-v4-confirmation-t3-extension-f4k-audit.v1",
        "created_utc": datetime.now(UTC).isoformat(),
        "status": "sealed_before_extension_physical_collection",
        "passed": True,
        "counts_as_confirmatory_evidence": True,
        "development_only": False,
        "model_prediction_truth_key_or_score_read": False,
        "result_dependent_retry": False,
        "strict_all_cases_pass": False,
        "maximum_case_attrition_rate": 0.05,
        "minimum_accepted_cases": math.floor(0.95 * planned) + 1,
        "maximum_concurrent_isaac_processes": 3,
        "counts": {
            "pairs_to_collect": 2 * planned,
            "physical_episodes": 4 * planned,
            "planned_cases": planned,
            "selected_formal_cases": planned,
        },
        "selected_cases": str(selected_path.relative_to(ROOT)),
        "selected_cases_sha256": sha256(selected_path),
        "scene_registry": str(registry.relative_to(ROOT)),
        "scene_registry_sha256": sha256(registry),
        "material_lock": str(LOCK),
        "material_lock_sha256": sha256(LOCK),
        "collector": str(COLLECTOR_WRAPPER.relative_to(ROOT)),
        "collector_sha256": sha256(COLLECTOR_WRAPPER),
        "runner_sha256": sha256(RUNNER),
        "v5_feature_function_sha256": sha256(TEMPORAL),
        "output_root": str(RUN_OUT.relative_to(ROOT)),
        "corpus_root": str(CORPUS),
        "source_sha256": {
            "f0": sha256(F0),
            "f1": sha256(F1),
            "f4h": sha256(F4H),
            "f4i": sha256(F4I),
            "f4j": sha256(F4J),
            "f4p": sha256(F4P),
            "f4q": sha256(F4Q),
            "f4r": sha256(F4R),
            "f4s": sha256(F4S),
            "f4t": sha256(F4T),
            "f4u": sha256(F4U),
            "f4v": sha256(F4V),
            "f4w": sha256(F4W),
            "f4x": sha256(F4X),
            "f4y": sha256(F4Y),
            "f4z": sha256(F4Z),
            "f4aa": sha256(F4AA),
            "scene_admission": sha256(ADMISSION),
            "global_t3_cases": sha256(cases_path),
            "schedules": schedules,
            "protocols": protocols,
            "schedule_builder": sha256(Path(__file__).resolve()),
        },
    }
    seal_path = OUT / "seal_manifest.json"
    write_json(seal_path, seal)
    (OUT / "seal_manifest.sha256").write_text(f"{sha256(seal_path)}  {seal_path.name}\n", encoding="utf-8")
    print(json.dumps({"passed": True, "scenes": len(scenes), "cases": planned, "seal": str(seal_path)}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
