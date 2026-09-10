#!/usr/bin/env python3
"""Freeze a model-blind replacement schedule for one operationally lost scene.

The replacement cardinality is fixed by structural file completeness only.
No model prediction, score, or task outcome is read.  Structurally complete
original pairs remain in the evaluation schedule; every incomplete original
pair is replaced one-for-one by a new deterministic physical replicate.
"""

from __future__ import annotations

import copy
import hashlib
import json
import os
from datetime import UTC, datetime
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parents[1]
DESIGN = ROOT / "outputs/kinofail_kino_v4_all191_scale_t2_extension_v2"
SCENE = "kino4c_production_020"
ORIGINAL_SCENE_SCHEDULE = (
    DESIGN / "schedules/scenes" / SCENE / "scale/schedule.jsonl"
)
ORIGINAL_GLOBAL_SCHEDULE = DESIGN / "schedules/global/scale_schedule.jsonl"
ORIGINAL_PROTOCOL = (
    DESIGN / "schedules/scenes" / SCENE / "scale/collection_protocol.json"
)
REGISTRY = DESIGN / "design/scene_registry.json"
CORPUS = Path(
    "/data/eureka/kinofail_kino_v4_all191_scale_t2_extension_v2/corpus"
)
INCIDENT = ROOT / (
    "outputs/freeze/kino_v4_all191_process_group_reclamation_v1/"
    "incident_and_policy_manifest.json"
)
PAIR_RUNNER = ROOT / "scripts/run_kinofail_reconfirmation_pair_partition_v2.py"
OUT = DESIGN / "schedules/operational_supplement_v1"
SCHEDULE = OUT / "scenes" / SCENE / "scale/schedule.jsonl"
PROTOCOL = OUT / "scenes" / SCENE / "scale/collection_protocol.json"
EVALUATION_SCHEDULE = OUT / "global/evaluation_scale_schedule.jsonl"
REPLACEMENT_MAP = OUT / "replacement_map.json"
FREEZE = ROOT / (
    "outputs/freeze/kino_v4_all191_operational_supplement_v1/"
    "freeze_manifest.json"
)


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


def rows(path: Path) -> list[dict[str, Any]]:
    return [
        json.loads(line)
        for line in path.read_text(encoding="utf-8").splitlines()
        if line
    ]


def atomic_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(
        json.dumps(value, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    os.replace(temporary, path)


def atomic_jsonl(path: Path, values: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(
        "".join(json.dumps(value, sort_keys=True) + "\n" for value in values),
        encoding="utf-8",
    )
    os.replace(temporary, path)


def stable_hex(*parts: str, length: int = 20) -> str:
    return hashlib.sha256("::".join(parts).encode()).hexdigest()[:length]


def stable_seed(old_pair_id: str) -> int:
    return int(stable_hex("kino-v4-all191-supplement-v1", old_pair_id, length=8), 16)


def complete_pair_ids() -> set[str]:
    complete = set()
    for path in (CORPUS / SCENE / "pair_summaries").glob("*.json"):
        value = load(path)
        pair_id = str(value.get("counterfactual_group_id"))
        results = value.get("results")
        if (
            isinstance(results, list)
            and len(results) == 2
            and {str(item.get("condition")) for item in results}
            == {"nominal_counterfactual", "anomaly"}
            and all(
                Path(str(item.get("manifest", ""))).is_file()
                for item in results
            )
        ):
            complete.add(pair_id)
    return complete


def replacement_row(
    source: dict[str, Any], old_pair_id: str, new_pair_id: str
) -> dict[str, Any]:
    value = copy.deepcopy(source)
    condition = str(value["condition"])
    new_seed = stable_seed(old_pair_id)
    old_episode_id = str(value["episode_id"])
    new_episode_id = f"{new_pair_id}_{condition}"
    value.update(
        {
            "counterfactual_group_id": new_pair_id,
            "episode_id": new_episode_id,
            "operator_seed": new_seed,
            "physical_seed": new_seed,
            "runtime_seed": (new_seed ^ 0x5A17C3D1) & 0xFFFFFFFF,
            "replicate_index": 2,
            "split": "all191_scale_t2_operational_supplement_v1",
            "design_mode": "sealed_model_blind_operational_supplement",
            "supersedes_operationally_incomplete_pair_id": old_pair_id,
            "artifact_state": "planned",
            "evaluation_eligible": False,
            "texture_swap_group_id": "ts_"
            + stable_hex(new_pair_id, condition),
        }
    )
    nuisance = value.get("physical_nuisance")
    if isinstance(nuisance, dict):
        nuisance["physics_seed"] = new_seed
    required = value.get("required_outputs")
    if not isinstance(required, dict):
        raise RuntimeError(f"missing outputs for {old_pair_id}")

    def replace_path(item: Any) -> Any:
        if isinstance(item, str):
            return item.replace(old_episode_id, new_episode_id)
        if isinstance(item, dict):
            return {key: replace_path(child) for key, child in item.items()}
        return item

    value["required_outputs"] = replace_path(required)
    return value


def main() -> int:
    for path in (
        ORIGINAL_SCENE_SCHEDULE,
        ORIGINAL_GLOBAL_SCHEDULE,
        ORIGINAL_PROTOCOL,
        REGISTRY,
        INCIDENT,
        PAIR_RUNNER,
    ):
        if not path.is_file():
            raise FileNotFoundError(path)
    incident = load(INCIDENT)
    if (
        incident.get("status")
        != "sealed_before_process_reclamation_and_resumption"
        or incident.get("incident", {}).get("active_scene") != SCENE
        or incident.get("blinding", {}).get("model_predictions_read") is not False
    ):
        raise RuntimeError("process-reclamation incident seal is invalid")

    original_scene = rows(ORIGINAL_SCENE_SCHEDULE)
    pair_rows: dict[str, list[dict[str, Any]]] = {}
    for value in original_scene:
        pair_rows.setdefault(str(value["counterfactual_group_id"]), []).append(value)
    if len(pair_rows) != 22 or any(len(values) != 2 for values in pair_rows.values()):
        raise RuntimeError("expected 22 two-sided original pairs")
    complete = complete_pair_ids() & set(pair_rows)
    if len(complete) != 3:
        raise RuntimeError(f"frozen incident expected 3 complete original pairs, got {len(complete)}")
    incomplete = sorted(set(pair_rows) - complete)
    if len(incomplete) != 19:
        raise RuntimeError("replacement cardinality must be 19")

    mapping = {
        old: "cf_" + stable_hex("replacement", SCENE, old)
        for old in incomplete
    }
    if set(mapping.values()) & set(pair_rows):
        raise RuntimeError("replacement pair ID collision")
    supplemental_rows = [
        replacement_row(value, old, mapping[old])
        for old in incomplete
        for value in pair_rows[old]
    ]
    atomic_jsonl(SCHEDULE, supplemental_rows)

    protocol = copy.deepcopy(load(ORIGINAL_PROTOCOL))
    protocol["protocol_id"] = (
        "kinofail-kino-v4-all191-operational-supplement-v1-" + SCENE
    )
    protocol["schedule_sha256"] = sha256(SCHEDULE)
    protocol["allowed"]["counterfactual_group_ids"] = sorted(mapping.values())
    protocol["collection_contract"]["counterfactual_pairs"] = len(mapping)
    protocol["collection_contract"]["physical_episodes"] = 2 * len(mapping)
    protocol["collection_contract"]["result_dependent_retry_permitted"] = False
    protocol["operational_supplement"] = {
        "selection_basis": "structural_file_completeness_only",
        "one_for_one_replacement": True,
        "new_pair_ids": True,
        "new_physical_seeds": True,
        "model_predictions_read": False,
        "method_scores_read": False,
        "original_attrition_markers_preserved": True,
    }
    atomic_json(PROTOCOL, protocol)

    global_rows = rows(ORIGINAL_GLOBAL_SCHEDULE)
    evaluation_rows = []
    replaced_conditions = 0
    for value in global_rows:
        old = str(value["counterfactual_group_id"])
        if old in mapping:
            evaluation_rows.append(replacement_row(value, old, mapping[old]))
            replaced_conditions += 1
        else:
            evaluation_rows.append(value)
    if replaced_conditions != 38 or len(evaluation_rows) != len(global_rows):
        raise RuntimeError("global one-for-one replacement failed")
    atomic_jsonl(EVALUATION_SCHEDULE, evaluation_rows)

    replacement_record = {
        "schema_version": "kinofail.kino-v4-operational-replacement-map.v1",
        "status": "frozen_before_supplemental_acquisition",
        "scene_id": SCENE,
        "complete_original_pair_ids_retained": sorted(complete),
        "incomplete_original_pair_ids_replaced": incomplete,
        "old_to_new_pair_id": mapping,
        "replacement_pair_count": len(mapping),
        "evaluation_pair_count_unchanged": True,
        "model_predictions_read": False,
        "method_scores_read": False,
        "result_dependent_retry_permitted": False,
    }
    atomic_json(REPLACEMENT_MAP, replacement_record)

    evaluation = rows(EVALUATION_SCHEDULE)
    scene_eval = [value for value in evaluation if value["scene_id"] == SCENE]
    eval_pairs = {str(value["counterfactual_group_id"]) for value in scene_eval}
    operators = {str(value["target_operator"]) for value in scene_eval}
    severities = {
        (str(value["target_operator"]), str(value["severity_id"]))
        for value in scene_eval
    }
    if len(eval_pairs) != 22 or len(operators) != 11 or len(severities) != 22:
        raise RuntimeError("recovered evaluation design lost coverage")

    freeze = {
        "schema_version": "kinofail.kino-v4-operational-supplement-freeze.v1",
        "status": "frozen_before_supplemental_acquisition",
        "passed": True,
        "sealed_utc": datetime.now(UTC).isoformat(),
        "scene_id": SCENE,
        "selection_basis": "structural_file_completeness_only",
        "complete_original_pairs": len(complete),
        "one_for_one_replacement_pairs": len(mapping),
        "final_evaluation_pairs_for_scene": len(eval_pairs),
        "final_operator_count_for_scene": len(operators),
        "final_operator_by_severity_cells_for_scene": len(severities),
        "scientific_contract": {
            "operator_set_changed": False,
            "severity_set_changed": False,
            "scene_or_material_changed": False,
            "physics_parameter_values_changed": False,
            "evaluation_scene_weight_changed": False,
            "new_deterministic_physical_replicates": True,
            "original_attrition_markers_preserved": True,
            "result_dependent_retry_enabled": False,
        },
        "blinding": {
            "model_predictions_read": False,
            "method_scores_read": False,
            "replacement_selection_uses_task_outcomes": False,
        },
        "artifacts": {
            "incident": str(INCIDENT),
            "incident_sha256": sha256(INCIDENT),
            "original_scene_schedule": str(ORIGINAL_SCENE_SCHEDULE),
            "original_scene_schedule_sha256": sha256(ORIGINAL_SCENE_SCHEDULE),
            "supplemental_schedule": str(SCHEDULE),
            "supplemental_schedule_sha256": sha256(SCHEDULE),
            "supplemental_protocol": str(PROTOCOL),
            "supplemental_protocol_sha256": sha256(PROTOCOL),
            "evaluation_scale_schedule": str(EVALUATION_SCHEDULE),
            "evaluation_scale_schedule_sha256": sha256(EVALUATION_SCHEDULE),
            "replacement_map": str(REPLACEMENT_MAP),
            "replacement_map_sha256": sha256(REPLACEMENT_MAP),
            "pair_runner": str(PAIR_RUNNER),
            "pair_runner_sha256": sha256(PAIR_RUNNER),
        },
    }
    atomic_json(FREEZE, freeze)
    print(FREEZE)
    print(sha256(FREEZE))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
