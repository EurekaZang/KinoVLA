#!/usr/bin/env python3
"""Freeze fresh, case-complete replacements for all invalid T3 cases.

The original reconfirmation contains 91 invalid T3 cases because at least one
of the two source physics pairs is absent.  F27 replaces both source pairs for
every invalid case, preserving scene/material/operator/severity strata while
using one fresh physics seed shared across the replacement case.  Selection
depends only on missing runtime artifacts and is sealed before acquisition or
prediction.
"""

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

from scripts.prepare_kinofail_confirmatory_valid_conflict_design_v1 import (
    _validate_t3,
)


SOURCE = ROOT / "outputs/kinofail_reconfirmation_v2"
OUTPUT = ROOT / "outputs/kinofail_t3_replenishment_f27"
CORPUS = Path("/data/eureka/KinoVLA/outputs/kinofail_t3_replenishment_f27/corpus")


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def read_json(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise TypeError(path)
    return value


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    rows = [
        json.loads(line)
        for line in path.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]
    if any(not isinstance(row, dict) for row in rows):
        raise TypeError(path)
    return rows


def write_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(value, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )


def write_jsonl(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        "".join(
            json.dumps(row, sort_keys=True, separators=(",", ":")) + "\n"
            for row in rows
        ),
        encoding="utf-8",
    )


def stable_id(prefix: str, *values: object) -> str:
    payload = "|".join(str(value) for value in values).encode("utf-8")
    return f"{prefix}_{hashlib.sha256(payload).hexdigest()[:20]}"


def replace_path_id(value: str, old: str, new: str) -> str:
    if old not in value:
        raise RuntimeError(f"source pair id absent from required path: {value}")
    return value.replace(old, new)


def main() -> int:
    if OUTPUT.exists() or CORPUS.exists():
        raise FileExistsError("refusing to overwrite F27 design or corpus")
    prediction_roots = [
        ROOT / "outputs/eval/unified_moe_v3_reconfirmation_v2_replenished_f26"
    ]
    forbidden = [
        path
        for root in prediction_roots
        if root.exists()
        for path in root.rglob("*")
        if path.is_file() and "prediction" in path.name.lower()
    ]
    if forbidden:
        raise RuntimeError(f"prediction exists before F27 freeze: {forbidden[:3]}")

    design_dir = SOURCE / "schedules/c2_t3"
    capsule_root = SOURCE / "conflict_capsules_ext4"
    _, _, valid_ids, invalid = _validate_t3(
        design_dir=design_dir,
        corpus=capsule_root,
    )
    if len(valid_ids) != 1_409 or len(invalid) != 91:
        raise RuntimeError(
            f"unexpected pre-F27 T3 validity: valid={len(valid_ids)} invalid={len(invalid)}"
        )
    invalid_ids = {str(row["case_id"]) for row in invalid}
    source_cases = {
        str(row["case_id"]): row
        for row in read_jsonl(design_dir / "case_schedule.jsonl")
    }
    source_rows: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in read_jsonl(design_dir / "schedule.jsonl"):
        source_rows[str(row["counterfactual_group_id"])].append(row)

    schedules_by_scene: dict[str, list[dict[str, Any]]] = defaultdict(list)
    mappings: list[dict[str, Any]] = []
    replacement_cases: list[dict[str, Any]] = []
    for index, case_id in enumerate(sorted(invalid_ids)):
        source_case = source_cases[case_id]
        scene_id = str(source_case["scene_id"])
        fresh_seed = 1_990_000_000 + index
        replacement_case_id = stable_id("c2t3_f27", case_id, fresh_seed)
        texture_group = stable_id("ts_f27", case_id, fresh_seed)
        appearance_ids = {
            view: stable_id("app_f27", case_id, fresh_seed, view)
            for view in ("primary", "swap_01", "swap_02")
        }
        group_map: dict[str, str] = {}
        for operator, key in (
            ("O7_visual_remap", "o7_source_physics_group_id"),
            ("O8_invisible_collider", "o8_source_physics_group_id"),
        ):
            original_group = str(source_case[key])
            replacement_group = stable_id(
                "cf_f27", case_id, operator, fresh_seed
            )
            group_map[original_group] = replacement_group
            rows = source_rows[original_group]
            if len(rows) != 2:
                raise RuntimeError(f"invalid source T3 pair: {original_group}")
            for source_row in rows:
                row = json.loads(json.dumps(source_row))
                condition = str(row["condition"])
                row.update(
                    {
                        "benchmark_id": "kinofail_t3_replenishment_f27",
                        "case_id": replacement_case_id,
                        "case_seed": fresh_seed,
                        "counterfactual_group_id": replacement_group,
                        "episode_id": f"{replacement_group}_{'nominal' if condition == 'nominal_counterfactual' else 'anomaly'}",
                        "operator_seed": fresh_seed,
                        "physical_seed": fresh_seed,
                        "runtime_seed": fresh_seed,
                        "design_mode": "posthoc_fixed_case_replenishment",
                        "split": "supplementary_replenishment",
                        "texture_swap_group_id": texture_group,
                    }
                )
                row["physical_nuisance"]["physics_seed"] = fresh_seed
                row["appearance_seed"] = fresh_seed
                for view in row["appearance_views"]:
                    view_id = str(view["appearance_view_id"])
                    view["appearance_seed"] = fresh_seed
                    view["appearance_id"] = appearance_ids[view_id]
                primary = next(
                    view
                    for view in row["appearance_views"]
                    if view["appearance_view_id"] == "primary"
                )
                row["appearance_id"] = primary["appearance_id"]
                required = row["required_outputs"]
                for name in ("depth_optional", "episode_manifest", "proprio", "rgb", "telemetry"):
                    required[name] = replace_path_id(
                        str(required[name]), original_group, replacement_group
                    )
                required["rgb_views"] = {
                    key_view: replace_path_id(
                        str(value), original_group, replacement_group
                    )
                    for key_view, value in required["rgb_views"].items()
                }
                row["replenishment"] = {
                    "protocol": "F27-v1",
                    "fixed_before_collection": True,
                    "replaces_case_id": case_id,
                    "replaces_counterfactual_group_id": original_group,
                    "both_t3_source_groups_recollected": True,
                    "fresh_case_shared_physics_seed": True,
                    "original_scene_material_operator_severity_preserved": True,
                    "result_dependent_retry": False,
                }
                schedules_by_scene[scene_id].append(row)
        replacement_case = json.loads(json.dumps(source_case))
        replacement_case.update(
            {
                "benchmark_id": "kinofail_t3_replenishment_f27",
                "case_id": replacement_case_id,
                "case_seed": fresh_seed,
                "o7_source_physics_group_id": group_map[
                    str(source_case["o7_source_physics_group_id"])
                ],
                "o8_source_physics_group_id": group_map[
                    str(source_case["o8_source_physics_group_id"])
                ],
                "split": "supplementary_replenishment",
            }
        )
        replacement_case["physical_nuisance"]["physics_seed"] = fresh_seed
        replacement_cases.append(replacement_case)
        mappings.append(
            {
                "original_case_id": case_id,
                "replacement_case_id": replacement_case_id,
                "scene_id": scene_id,
                "fresh_seed": fresh_seed,
                "severity_id": str(source_case["severity_id"]),
                "cluster_material": str(source_case["cluster_material"]),
                "original_o7_group_id": str(
                    source_case["o7_source_physics_group_id"]
                ),
                "replacement_o7_group_id": replacement_case[
                    "o7_source_physics_group_id"
                ],
                "original_o8_group_id": str(
                    source_case["o8_source_physics_group_id"]
                ),
                "replacement_o8_group_id": replacement_case[
                    "o8_source_physics_group_id"
                ],
            }
        )

    if len(replacement_cases) != 91 or sum(
        len(rows) for rows in schedules_by_scene.values()
    ) != 364:
        raise RuntimeError("F27 schedule size mismatch")
    mapping_path = OUTPUT / "design/replacement_cases.jsonl"
    case_path = OUTPUT / "design/case_schedule.jsonl"
    write_jsonl(mapping_path, mappings)
    write_jsonl(case_path, replacement_cases)

    source_protocol_root = SOURCE / "schedules/scenes"
    schedule_artifacts = []
    protocol_artifacts = []
    for scene_id in sorted(schedules_by_scene):
        rows = sorted(
            schedules_by_scene[scene_id],
            key=lambda row: (str(row["counterfactual_group_id"]), str(row["condition"])),
        )
        schedule_path = OUTPUT / "schedules" / scene_id / "c2_t3/schedule.jsonl"
        write_jsonl(schedule_path, rows)
        source_protocol_path = (
            source_protocol_root / scene_id / "c2_t3/collection_protocol.json"
        )
        protocol = read_json(source_protocol_path)
        pair_ids = sorted({str(row["counterfactual_group_id"]) for row in rows})
        protocol.update(
            {
                "benchmark_id": "kinofail_t3_replenishment_f27",
                "protocol_id": f"t3-replenishment-f27-{scene_id}",
                "schedule_path": str(schedule_path.relative_to(ROOT)),
                "schedule_sha256": sha256(schedule_path),
                "confirmatory": False,
                "posthoc_supplementary_replenishment": True,
                "status": "frozen",
            }
        )
        protocol["allowed"]["counterfactual_group_ids"] = pair_ids
        protocol["collection_contract"].update(
            {
                "counterfactual_pairs": len(pair_ids),
                "physical_episodes": 2 * len(pair_ids),
                "replacement_cases": len(pair_ids) // 2,
                "both_source_pairs_per_case_recollected": True,
            }
        )
        protocol["f27_provenance"] = {
            "source_protocol": str(source_protocol_path.relative_to(ROOT)),
            "source_protocol_sha256": sha256(source_protocol_path),
            "selection": "all 91 artifact-incomplete T3 cases",
            "model_prediction_or_score_read": False,
        }
        protocol_path = OUTPUT / "protocols" / scene_id / "c2_t3.json"
        write_json(protocol_path, protocol)
        schedule_artifacts.append(
            {"path": str(schedule_path.relative_to(ROOT)), "sha256": sha256(schedule_path)}
        )
        protocol_artifacts.append(
            {"path": str(protocol_path.relative_to(ROOT)), "sha256": sha256(protocol_path)}
        )

    dependencies = [
        ROOT / "scripts/prepare_kinofail_t3_replenishment_f27.py",
        ROOT / "scripts/run_kinofail_t3_replenishment_f27.py",
        ROOT / "scripts/isaac_collect_kinofail_confirmatory_pair_v9.py",
        ROOT / "scripts/isaac_collect_kinofail_confirmatory_pair_v8.py",
        ROOT / "kino_vla/sim/isaac_policy_backend.py",
    ]
    freeze = {
        "schema_version": "kinofail.t3-replenishment-f27-freeze.v1",
        "created_utc": datetime.now(UTC).isoformat(),
        "status": "sealed_before_collection",
        "passed": True,
        "posthoc_supplementary_replenishment": True,
        "original_confirmation_attrition_must_be_reported": True,
        "model_prediction_feature_or_score_read": False,
        "selection_basis": "missing runtime artifacts only",
        "result_dependent_retry_permitted": False,
        "cases": 91,
        "counterfactual_pairs": 182,
        "physical_episodes": 364,
        "minimum_complete_replacement_cases_for_strictly_below_five_percent": 17,
        "mapping": str(mapping_path.relative_to(ROOT)),
        "mapping_sha256": sha256(mapping_path),
        "case_schedule": str(case_path.relative_to(ROOT)),
        "case_schedule_sha256": sha256(case_path),
        "schedules": schedule_artifacts,
        "protocols": protocol_artifacts,
        "corpus_root": str(CORPUS),
        "scene_registry": "outputs/kinofail_reconfirmation_v2/scene_registry.json",
        "scene_registry_sha256": sha256(SOURCE / "scene_registry.json"),
        "material_lock": "outputs/assets/terrain_pbr_confirmatory_v2/terrain_assets.lock.json",
        "material_lock_sha256": sha256(
            ROOT / "outputs/assets/terrain_pbr_confirmatory_v2/terrain_assets.lock.json"
        ),
        "dependencies": [
            {"path": str(path.relative_to(ROOT)), "sha256": sha256(path)}
            for path in dependencies
        ],
    }
    freeze_path = OUTPUT / "freeze_manifest.json"
    write_json(freeze_path, freeze)
    (OUTPUT / "freeze_manifest.sha256").write_text(
        f"{sha256(freeze_path)}  {freeze_path.name}\n", encoding="utf-8"
    )
    print(json.dumps(freeze, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
