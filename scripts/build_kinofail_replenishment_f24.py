#!/usr/bin/env python3
"""Freeze a one-shot, model-blind replenishment schedule for Scale attrition.

F24 never edits or retries an episode in the independent reconfirmation.  It
maps every Scale pair that was declared attrited by F13 to exactly one new
counterfactual pair ID, fresh stochastic seeds, and a fresh visual realization.
The resulting cohort is explicitly post-hoc and must be reported alongside,
not in place of, the original attrition ledger.
"""

from __future__ import annotations

import argparse
import copy
import hashlib
import json
import random
from datetime import UTC, datetime
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parents[1]
SOURCE = ROOT / "outputs/kinofail_reconfirmation_v2"
DEFAULT_OUTPUT = ROOT / "outputs/kinofail_replenishment_f24"
DEFAULT_SEED_BASE = 2_170_000_000
EXPECTED_FAILED_PAIRS = 1_417
SURFACE_STATES = ("clean", "dusty", "scuffed", "damp")


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _canonical(value: Any) -> str:
    return json.dumps(value, sort_keys=True, separators=(",", ":"))


def _id(prefix: str, value: Any, length: int = 20) -> str:
    return prefix + hashlib.sha256(_canonical(value).encode()).hexdigest()[:length]


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


def _replace_token(value: Any, old: str, new: str) -> Any:
    if isinstance(value, str):
        return value.replace(old, new)
    if isinstance(value, list):
        return [_replace_token(item, old, new) for item in value]
    if isinstance(value, dict):
        return {key: _replace_token(item, old, new) for key, item in value.items()}
    return value


def _fresh_views(
    views: list[dict[str, Any]], *, seed: int, pair_id: str
) -> list[dict[str, Any]]:
    fresh: list[dict[str, Any]] = []
    for index, source in enumerate(views):
        row = copy.deepcopy(source)
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
        row["appearance_id"] = _id(
            "app_f24_",
            {
                "pair_id": pair_id,
                "view": row["appearance_view_id"],
                "material_asset_id": row["material_asset_id"],
                "appearance_seed": seed,
                "surface_state": row["surface_state"],
                "uv_scale": row["uv_scale"],
                "uv_rotation_deg": row["uv_rotation_deg"],
                "uv_offset": row["uv_offset"],
                "albedo_brightness_multiplier": row[
                    "albedo_brightness_multiplier"
                ],
                "normal_strength": row["normal_strength"],
                "roughness_multiplier": row["roughness_multiplier"],
            },
        )
        fresh.append(row)
    return fresh


def _fresh_pair(
    rows: list[dict[str, Any]],
    *,
    scene_id: str,
    original_pair_id: str,
    global_index: int,
    attrition: dict[str, Any],
    cohort_id: str,
    seed_base: int,
) -> list[dict[str, Any]]:
    if len(rows) != 2 or {row["condition"] for row in rows} != {
        "nominal_counterfactual",
        "anomaly",
    }:
        raise RuntimeError(f"incomplete source pair: {original_pair_id}")
    seed = seed_base + global_index
    pair_id = _id(
        "cf_f24_",
        {
            "protocol": cohort_id,
            "scene_id": scene_id,
            "original_pair_id": original_pair_id,
            "attempt_index": 1,
        },
    )
    views = _fresh_views(
        list(rows[0]["appearance_views"]), seed=seed, pair_id=pair_id
    )
    primary = next(row for row in views if row["is_primary"] is True)
    output: list[dict[str, Any]] = []
    for source in rows:
        row = _replace_token(copy.deepcopy(source), original_pair_id, pair_id)
        condition = str(row["condition"])
        row.update(
            {
                "benchmark_id": "kinofail_replenishment_f24",
                "counterfactual_group_id": pair_id,
                "episode_id": f"{pair_id}_{'nominal' if condition == 'nominal_counterfactual' else 'anomaly'}",
                "design_mode": "posthoc_fixed_replenishment",
                "split": "supplementary_replenishment",
                "evaluation_eligible": False,
                "artifact_state": "planned",
                "appearance_seed": seed,
                "operator_seed": seed,
                "physical_seed": seed,
                "runtime_seed": seed,
                "appearance_views": copy.deepcopy(views),
                "appearance_id": primary["appearance_id"],
                "surface_state": primary["surface_state"],
                "uv_scale": primary["uv_scale"],
                "uv_rotation_deg": primary["uv_rotation_deg"],
                "uv_offset": primary["uv_offset"],
                "albedo_brightness_multiplier": primary[
                    "albedo_brightness_multiplier"
                ],
                "normal_strength": primary["normal_strength"],
                "roughness_multiplier": primary["roughness_multiplier"],
                "texture_swap_group_id": _id(
                    "ts_f24_", {"pair_id": pair_id, "seed": seed}
                ),
                "replenishment": {
                    "protocol": cohort_id,
                    "attempt_index": 1,
                    "fixed_before_collection": True,
                    "replaces_counterfactual_group_id": original_pair_id,
                    "original_attrition_reason": attrition["reason"],
                    "result_dependent_retry": False,
                    "fresh_visual_realization": True,
                    "fresh_physics_seed": True,
                    "original_scene_and_stratum_preserved": True,
                },
            }
        )
        nuisance = dict(row["physical_nuisance"])
        nuisance["physics_seed"] = seed
        row["physical_nuisance"] = nuisance
        output.append(row)
    return sorted(output, key=lambda row: row["condition"] == "anomaly")


def build(
    output_root: Path,
    *,
    smoke: bool,
    cohort_id: str,
    seed_base: int,
) -> dict[str, Any]:
    ledger_paths = sorted((SOURCE / "attrition_ledgers_f13").glob("*/scale.json"))
    if len(ledger_paths) != 30:
        raise RuntimeError(f"expected 30 Scale ledgers, found {len(ledger_paths)}")

    source_pairs: list[tuple[str, str, dict[str, Any], list[dict[str, Any]]]] = []
    source_hashes: dict[str, dict[str, str]] = {}
    all_original_ids: set[str] = set()
    for ledger_path in ledger_paths:
        ledger = _json(ledger_path)
        scene_id = str(ledger["scene_id"])
        schedule_path = SOURCE / "schedules/scenes" / scene_id / "scale/schedule.jsonl"
        schedule = _jsonl(schedule_path)
        groups: dict[str, list[dict[str, Any]]] = {}
        for row in schedule:
            groups.setdefault(str(row["counterfactual_group_id"]), []).append(row)
        for attrition in ledger["attrition"]:
            original_id = str(attrition["counterfactual_group_id"])
            if original_id in all_original_ids:
                raise RuntimeError(f"duplicate attrited pair ID: {original_id}")
            all_original_ids.add(original_id)
            source_pairs.append(
                (scene_id, original_id, dict(attrition), groups[original_id])
            )
        source_hashes[scene_id] = {
            "attrition_ledger": str(ledger_path.relative_to(ROOT)),
            "attrition_ledger_sha256": _sha256(ledger_path),
            "source_schedule": str(schedule_path.relative_to(ROOT)),
            "source_schedule_sha256": _sha256(schedule_path),
        }
    if len(source_pairs) != EXPECTED_FAILED_PAIRS:
        raise RuntimeError(
            f"expected {EXPECTED_FAILED_PAIRS} failed pairs, found {len(source_pairs)}"
        )

    source_pairs.sort(key=lambda item: (item[0], item[1]))
    if smoke:
        wanted = {"confirm_v2_life_scene_00", "confirm_v2_life_scene_01"}
        chosen = []
        for item in source_pairs:
            if (
                item[0] in wanted
                and item[3][0]["target_operator"] == "O4_tether"
                and all(existing[0] != item[0] for existing in chosen)
            ):
                chosen.append(item)
        # scene 00 has no attrited O4 pair; use one fixed eligible O4 source only
        # for diagnostic contrast.  It never enters the F24 replacement cohort.
        if all(item[0] != "confirm_v2_life_scene_00" for item in chosen):
            scene_id = "confirm_v2_life_scene_00"
            schedule_path = SOURCE / "schedules/scenes" / scene_id / "scale/schedule.jsonl"
            rows = _jsonl(schedule_path)
            gid = next(
                row["counterfactual_group_id"]
                for row in rows
                if row["condition"] == "anomaly"
                and row["target_operator"] == "O4_tether"
            )
            chosen.append(
                (
                    scene_id,
                    str(gid),
                    {"reason": "smoke_control_not_a_replacement"},
                    [row for row in rows if row["counterfactual_group_id"] == gid],
                )
            )
        source_pairs = sorted(chosen, key=lambda item: item[0])

    created = datetime.now(UTC).isoformat()
    design_path = output_root / "design/design.json"
    design = {
        "schema_version": "kinofail.f24-replenishment-design.v1",
        "created_utc": created,
        "status": "fixed_before_f24_collection",
        "cohort_id": cohort_id,
        "cohort_role": "posthoc_supplementary_replenishment",
        "confirmatory_claim_permitted": False,
        "model_prediction_or_score_read": False,
        "selection_basis": "all F13-declared Scale attrition slots, fixed once",
        "replacement_of_replacement_permitted": False,
        "attempts_per_original_attrition_slot": 1,
        "original_scale_planned_pairs": 10_560,
        "original_scale_attrited_pairs": 1_417,
        "original_t3_planned_cases": 3_000,
        "original_t3_attrited_cases": 114,
        "strict_scale_target_max_remaining_attrition_pairs": 527,
        "minimum_required_successful_replacements": 890,
        "scheduled_replenishment_pairs": len(source_pairs),
        "fresh_seed_base": seed_base,
        "visual_randomization": {
            "material_asset_identity_preserved": True,
            "fresh_surface_state_uv_brightness_normal_roughness": True,
            "visual_intervention_only": True,
        },
        "o4_collection_correction": {
            "applies_only_to": "O4_tether F24 records",
            "cause": "reachable-exposure QA required a Python operator object although O4 is a backend-native operator",
            "change": "admit O4 exposure iff physical telemetry total_attachment_cycles is positive",
            "simulation_region_force_peel_or_severity_parameters_changed": False,
            "sensor_feature_model_threshold_or_analysis_changed": False,
        },
        "source_hashes": source_hashes,
    }
    _write_json(design_path, design)

    by_scene: dict[str, list[dict[str, Any]]] = {}
    mapping: list[dict[str, Any]] = []
    for global_index, (scene_id, original_id, attrition, rows) in enumerate(
        source_pairs
    ):
        fresh = _fresh_pair(
            rows,
            scene_id=scene_id,
            original_pair_id=original_id,
            global_index=global_index,
            attrition=attrition,
            cohort_id=cohort_id,
            seed_base=seed_base,
        )
        by_scene.setdefault(scene_id, []).extend(fresh)
        mapping.append(
            {
                "scene_id": scene_id,
                "original_counterfactual_group_id": original_id,
                "replacement_counterfactual_group_id": fresh[0][
                    "counterfactual_group_id"
                ],
                "target_operator": fresh[0]["target_operator"],
                "severity_id": fresh[0]["severity_id"],
                "fresh_seed": fresh[0]["operator_seed"],
                "original_attrition_reason": attrition["reason"],
            }
        )

    protocol_paths: list[Path] = []
    schedule_paths: list[Path] = []
    for scene_id, rows in sorted(by_scene.items()):
        rows.sort(key=lambda row: (row["counterfactual_group_id"], row["condition"]))
        schedule_path = output_root / "schedules" / scene_id / "scale/schedule.jsonl"
        protocol_path = output_root / "schedules" / scene_id / "scale/collection_protocol.json"
        _write_jsonl(schedule_path, rows)
        original_protocol_path = (
            SOURCE
            / "schedules/scenes"
            / scene_id
            / "scale/collection_protocol.json"
        )
        protocol = _json(original_protocol_path)
        groups = sorted({str(row["counterfactual_group_id"]) for row in rows})
        protocol.update(
            {
                "benchmark_id": "kinofail_replenishment_f24",
                "protocol_id": f"replenishment-f24-scale-{scene_id}",
                "confirmatory": False,
                "design_path": str(design_path.relative_to(ROOT)),
                "design_sha256": _sha256(design_path),
                "schedule_path": str(schedule_path.relative_to(ROOT)),
                "schedule_sha256": _sha256(schedule_path),
                "status": "frozen",
            }
        )
        protocol["allowed"] = {
            "conditions": sorted({str(row["condition"]) for row in rows}),
            "counterfactual_group_ids": groups,
            "geometry_profiles": sorted({str(row["geometry_profile"]) for row in rows}),
            "physical_realizations": sorted(
                {str(row["physical_realization"]) for row in rows}
            ),
            "scene_families": [scene_id],
            "severity_ids": sorted({str(row["severity_id"]) for row in rows}),
            "target_operators": sorted({str(row["target_operator"]) for row in rows}),
        }
        protocol["collection_contract"] = {
            **protocol["collection_contract"],
            "counterfactual_pairs": len(groups),
            "physical_episodes": 2 * len(groups),
            "posthoc_supplementary_replenishment": True,
            "replacement_of_replacement_permitted": False,
        }
        _write_json(protocol_path, protocol)
        schedule_paths.append(schedule_path)
        protocol_paths.append(protocol_path)

    mapping_path = output_root / "design/replacement_mapping.jsonl"
    _write_jsonl(mapping_path, mapping)
    manifest = {
        "schema_version": "kinofail.f24-replenishment-freeze.v1",
        "created_utc": created,
        "status": "sealed_before_collection",
        "passed": True,
        "smoke": smoke,
        "cohort_id": cohort_id,
        "model_prediction_or_score_read": False,
        "posthoc_supplementary_replenishment": True,
        "source_failed_pairs": len(source_pairs),
        "scheduled_pairs": len(mapping),
        "scheduled_episodes": 2 * len(mapping),
        "design": str(design_path.relative_to(ROOT)),
        "design_sha256": _sha256(design_path),
        "mapping": str(mapping_path.relative_to(ROOT)),
        "mapping_sha256": _sha256(mapping_path),
        "scene_registry": str((SOURCE / "scene_registry.json").relative_to(ROOT)),
        "scene_registry_sha256": _sha256(SOURCE / "scene_registry.json"),
        "material_lock": "outputs/assets/terrain_pbr_confirmatory_v2/terrain_assets.lock.json",
        "material_lock_sha256": _sha256(
            ROOT / "outputs/assets/terrain_pbr_confirmatory_v2/terrain_assets.lock.json"
        ),
        "schedules": [
            {"path": str(path.relative_to(ROOT)), "sha256": _sha256(path)}
            for path in schedule_paths
        ],
        "protocols": [
            {"path": str(path.relative_to(ROOT)), "sha256": _sha256(path)}
            for path in protocol_paths
        ],
        "execution_artifacts": [
            {
                "path": str(path.relative_to(ROOT)),
                "sha256": _sha256(path),
            }
            for path in (
                ROOT / "scripts/isaac_collect_kinofail_replenishment_f24_v1.py",
                ROOT / "scripts/run_kinofail_replenishment_f24_worker.py",
                ROOT / "scripts/run_kinofail_replenishment_f24_supervisor.py",
            )
        ],
    }
    manifest_path = output_root / "freeze_manifest.json"
    _write_json(manifest_path, manifest)
    return manifest


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output-root", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--smoke", action="store_true")
    parser.add_argument("--cohort-id", default="F24-v1")
    parser.add_argument("--seed-base", type=int, default=DEFAULT_SEED_BASE)
    args = parser.parse_args()
    output = args.output_root.resolve()
    if output.exists() and any(output.iterdir()):
        raise RuntimeError(f"refusing to overwrite non-empty F24 root: {output}")
    manifest = build(
        output,
        smoke=bool(args.smoke),
        cohort_id=str(args.cohort_id),
        seed_base=int(args.seed_base),
    )
    print(json.dumps(manifest, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
