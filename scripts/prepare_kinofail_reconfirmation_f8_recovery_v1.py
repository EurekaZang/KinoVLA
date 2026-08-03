#!/usr/bin/env python3
"""Seal the model-blind F8 storage and reboot recovery disposition.

This program reads only frozen schedules, acquisition presence metadata, and
append-only launcher audits.  It never reads features, labels from generated
observations, predictions, scores, or model checkpoints.
"""

from __future__ import annotations

import hashlib
import json
import os
from collections import defaultdict
from datetime import UTC, datetime
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parents[1]
SCENE_ID = "confirm_v2_life_scene_00"
SCHEDULE = (
    ROOT
    / "outputs/kinofail_reconfirmation_v2/schedules/scenes"
    / SCENE_ID
    / "scale/schedule.jsonl"
)
SOURCE_CORPUS = Path(
    "/media/eureka/FC28565528560ED0/tmp/"
    "KinoVLA_reconfirmation_v2/corpus"
) / SCENE_ID
DESTINATION_CORPUS = (
    ROOT / "outputs/kinofail_reconfirmation_v2/corpus_ext4" / SCENE_ID
)
THROTTLE_AUDIT = (
    ROOT
    / "outputs/kinofail_reconfirmation_v2/operational_throttle/"
    "first_scene_three_process_transition.json"
)
F3 = (
    ROOT
    / "outputs/freeze/"
    "unified_moe_v3_reconfirmation_f3_nuisance_contract_amendment1/"
    "amendment_manifest.json"
)
F6 = (
    ROOT
    / "outputs/freeze/"
    "unified_moe_v3_reconfirmation_f6_liveness_watchdog_amendment1/"
    "amendment_manifest.json"
)
F7 = (
    ROOT
    / "outputs/freeze/"
    "unified_moe_v3_reconfirmation_f7_stable_concurrency_amendment1/"
    "amendment_manifest.json"
)
COLLECTOR = (
    ROOT / "scripts/isaac_collect_kinofail_confirmatory_pair_v7.py"
)
RUNNER = ROOT / "scripts/run_kinofail_reconfirmation_pair_partition_v2.py"
PIPELINE = ROOT / "scripts/run_kinofail_reconfirmation_scene_pipeline_v2.py"
WATCHDOG = ROOT / "scripts/watch_kinofail_reconfirmation_collectors_v2.py"
OUT = (
    ROOT
    / "outputs/freeze/"
    "unified_moe_v3_reconfirmation_f8_ext4_recovery_amendment1"
)
STATE_PATH = OUT / "recovery_state.json"
MANIFEST_PATH = OUT / "amendment_manifest.json"


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _json(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise TypeError(f"expected JSON object: {path}")
    return value


def _jsonl(path: Path) -> list[dict[str, Any]]:
    return [
        json.loads(line)
        for line in path.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]


def _write_json(path: Path, value: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(
        json.dumps(value, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    os.replace(temporary, path)


def _list_sha256(values: list[str]) -> str:
    text = "".join(f"{value}\n" for value in values)
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def _inventory(root: Path) -> dict[str, Any]:
    digest = hashlib.sha256()
    files = 0
    bytes_total = 0
    for path in sorted(item for item in root.rglob("*") if item.is_file()):
        relative = path.relative_to(root).as_posix()
        size = path.stat().st_size
        content_sha256 = _sha256(path)
        digest.update(
            f"{relative}\0{size}\0{content_sha256}\n".encode("utf-8")
        )
        files += 1
        bytes_total += size
    return {
        "file_count": files,
        "regular_file_bytes": bytes_total,
        "relative_path_size_content_inventory_sha256": digest.hexdigest(),
    }


def _physical_observation_files(
    corpus: Path,
    rows: list[dict[str, Any]],
) -> list[str]:
    observed: list[str] = []
    for row in rows:
        required = row["required_outputs"]
        candidates = [
            corpus / required["episode_manifest"],
            corpus / required["proprio"],
            corpus / required["telemetry"],
        ]
        for candidate in candidates:
            if candidate.is_file():
                observed.append(candidate.relative_to(corpus).as_posix())
        for key in ("rgb", "depth_optional"):
            directory = corpus / required[key]
            if directory.is_dir():
                observed.extend(
                    path.relative_to(corpus).as_posix()
                    for path in sorted(directory.rglob("*"))
                    if path.is_file()
                )
        for relative in required.get("rgb_views", {}).values():
            directory = corpus / relative
            if directory.is_dir():
                observed.extend(
                    path.relative_to(corpus).as_posix()
                    for path in sorted(directory.rglob("*"))
                    if path.is_file()
                )
    return sorted(set(observed))


def main() -> int:
    required = [
        SCHEDULE,
        SOURCE_CORPUS,
        DESTINATION_CORPUS,
        THROTTLE_AUDIT,
        F3,
        F6,
        F7,
        COLLECTOR,
        RUNNER,
        PIPELINE,
        WATCHDOG,
    ]
    missing = [str(path) for path in required if not path.exists()]
    if missing:
        raise FileNotFoundError(missing)
    if MANIFEST_PATH.exists() or STATE_PATH.exists():
        raise RuntimeError("F8 is already sealed; refusing to overwrite it")
    if not os.statvfs(SOURCE_CORPUS).f_flag & os.ST_RDONLY:
        raise RuntimeError("the dirty NTFS source must remain mounted read-only")

    prediction_candidates = [
        ROOT
        / "outputs/eval/unified_moe_v3_reconfirmation_v2/"
        "predictions.jsonl",
        ROOT
        / "outputs/eval/unified_moe_v3_reconfirmation_v2/"
        "confirmatory_report.json",
        ROOT
        / "outputs/eval/unified_moe_v3_reconfirmation_v2/"
        "confirmatory_score.json",
    ]
    if any(path.exists() for path in prediction_candidates):
        raise RuntimeError(
            "prediction or score artifacts exist; F8 can no longer be sealed"
        )
    feature_root = (
        ROOT
        / "outputs/eval/unified_moe_v3_reconfirmation_v2/shards"
        / SCENE_ID
    )
    if feature_root.exists() and any(
        path.is_file() for path in feature_root.rglob("*")
    ):
        raise RuntimeError(
            "model-blind shard features exist; F8 can no longer be sealed"
        )

    rows = _jsonl(SCHEDULE)
    pairs: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        pairs[str(row["counterfactual_group_id"])].append(row)
    invalid = [
        pair_id
        for pair_id, pair_rows in pairs.items()
        if len(pair_rows) != 2
        or {str(row["condition"]) for row in pair_rows}
        != {"nominal_counterfactual", "anomaly"}
    ]
    if len(rows) != 704 or len(pairs) != 352 or invalid:
        raise RuntimeError("unexpected frozen first-scene scale schedule")

    latest_attempt: dict[str, dict[str, Any]] = {}
    audit_paths = sorted(
        (DESTINATION_CORPUS / "launcher_audits").glob(
            "scale_f3v7_partition_*_of_4.json"
        )
    )
    if len(audit_paths) != 4:
        raise RuntimeError("expected four F3 launcher audits")
    audit_hashes = {}
    for audit_path in audit_paths:
        audit_hashes[str(audit_path.relative_to(ROOT))] = _sha256(audit_path)
        for attempt in _json(audit_path).get("attempts", []):
            pair_id = str(attempt["counterfactual_group_id"])
            previous = latest_attempt.get(pair_id)
            if previous is None or str(attempt.get("started_utc", "")) > str(
                previous.get("started_utc", "")
            ):
                latest_attempt[pair_id] = attempt

    throttle = _json(THROTTLE_AUDIT)
    interrupted_pair = str(throttle["interrupted_pair_id"])
    complete_summary: list[str] = []
    eligible_summary: list[str] = []
    qa_failed_summary: list[str] = []
    terminal_without_summary: list[str] = []
    started_without_summary: list[str] = []
    never_attempted: list[str] = []
    observation_counts: dict[str, int] = {}

    for pair_id in sorted(pairs):
        summary_path = (
            DESTINATION_CORPUS / "pair_summaries" / f"{pair_id}.json"
        )
        observations = _physical_observation_files(
            DESTINATION_CORPUS,
            pairs[pair_id],
        )
        observation_counts[pair_id] = len(observations)
        if summary_path.is_file():
            complete_summary.append(pair_id)
            if _json(summary_path).get("passed") is True:
                eligible_summary.append(pair_id)
            else:
                qa_failed_summary.append(pair_id)
            continue
        attempt = latest_attempt.get(pair_id)
        if attempt is None:
            never_attempted.append(pair_id)
        elif attempt.get("state") == "terminal":
            terminal_without_summary.append(pair_id)
        elif attempt.get("state") == "started":
            started_without_summary.append(pair_id)
        else:
            raise RuntimeError(f"unexpected attempt state for {pair_id}")

    authorized_reexecution = sorted(
        pair_id
        for pair_id in started_without_summary
        if pair_id != interrupted_pair and observation_counts[pair_id] == 0
    )
    if (
        len(complete_summary) != 135
        or len(eligible_summary) != 121
        or len(qa_failed_summary) != 14
        or len(terminal_without_summary) != 16
        or len(started_without_summary) != 4
        or len(never_attempted) != 197
        or len(authorized_reexecution) != 3
        or interrupted_pair not in started_without_summary
    ):
        raise RuntimeError("first-scene recovery classification drifted")

    infrastructure_attrition = sorted(
        set(terminal_without_summary) | {interrupted_pair}
    )
    retained_attrition = sorted(
        set(qa_failed_summary) | set(infrastructure_attrition)
    )
    run_pair_ids = sorted(set(never_attempted) | set(authorized_reexecution))
    if len(infrastructure_attrition) != 17:
        raise RuntimeError("unexpected infrastructure attrition count")
    if len(retained_attrition) != 31 or len(run_pair_ids) != 200:
        raise RuntimeError("unexpected F8 recovery disposition")
    if (
        set(complete_summary)
        | set(infrastructure_attrition)
        | set(run_pair_ids)
    ) != set(pairs):
        raise RuntimeError("F8 disposition does not partition the schedule")
    if (
        set(complete_summary)
        & set(infrastructure_attrition)
        or set(complete_summary) & set(run_pair_ids)
        or set(infrastructure_attrition) & set(run_pair_ids)
    ):
        raise RuntimeError("F8 disposition categories overlap")

    source_inventory = _inventory(SOURCE_CORPUS)
    destination_inventory = _inventory(DESTINATION_CORPUS)
    if source_inventory != destination_inventory:
        raise RuntimeError("ext4 corpus copy differs from read-only NTFS source")

    created_utc = datetime.now(UTC).isoformat()
    state = {
        "schema_version": "kinofail.reconfirmation-f8-recovery-state.v1",
        "created_utc": created_utc,
        "scene_id": SCENE_ID,
        "battery": "scale",
        "schedule": str(SCHEDULE.relative_to(ROOT)),
        "schedule_sha256": _sha256(SCHEDULE),
        "planned_pairs": len(pairs),
        "planned_rows": len(rows),
        "presence_metadata_only": True,
        "labels_features_predictions_scores_or_checkpoints_read": False,
        "f3_launcher_audit_sha256": audit_hashes,
        "source_inventory": source_inventory,
        "destination_inventory": destination_inventory,
        "complete_summary_pair_ids": complete_summary,
        "complete_summary_pair_ids_sha256": _list_sha256(complete_summary),
        "eligible_summary_pair_ids": eligible_summary,
        "eligible_summary_pair_ids_sha256": _list_sha256(eligible_summary),
        "qa_failed_complete_pair_ids": qa_failed_summary,
        "qa_failed_complete_pair_ids_sha256": _list_sha256(
            qa_failed_summary
        ),
        "terminal_without_summary_pair_ids": terminal_without_summary,
        "terminal_without_summary_pair_ids_sha256": _list_sha256(
            terminal_without_summary
        ),
        "f7_interrupted_pair_id": interrupted_pair,
        "infrastructure_attrition_pair_ids": infrastructure_attrition,
        "infrastructure_attrition_pair_ids_sha256": _list_sha256(
            infrastructure_attrition
        ),
        "retained_attrition_pair_ids": retained_attrition,
        "retained_attrition_pair_ids_sha256": _list_sha256(
            retained_attrition
        ),
        "never_attempted_pair_ids": never_attempted,
        "never_attempted_pair_ids_sha256": _list_sha256(never_attempted),
        "authorized_reexecution_pair_ids": authorized_reexecution,
        "authorized_reexecution_pair_ids_sha256": _list_sha256(
            authorized_reexecution
        ),
        "f8_run_pair_ids": run_pair_ids,
        "f8_run_pair_ids_sha256": _list_sha256(run_pair_ids),
        "counts": {
            "complete_summary": len(complete_summary),
            "eligible_summary": len(eligible_summary),
            "qa_failed_complete_summary": len(qa_failed_summary),
            "terminal_without_summary": len(terminal_without_summary),
            "f7_interrupted_without_retry": 1,
            "infrastructure_attrition": len(infrastructure_attrition),
            "retained_attrition_total": len(retained_attrition),
            "never_attempted": len(never_attempted),
            "authorized_zero_observation_reexecution": len(
                authorized_reexecution
            ),
            "f8_run_pairs": len(run_pair_ids),
        },
        "projected_if_all_f8_runs_emit_summaries": {
            "summary_pairs": len(complete_summary) + len(run_pair_ids),
            "summary_fraction": (
                len(complete_summary) + len(run_pair_ids)
            )
            / len(pairs),
            "infrastructure_missing_pairs": len(infrastructure_attrition),
            "infrastructure_missing_fraction": len(infrastructure_attrition)
            / len(pairs),
        },
    }
    _write_json(STATE_PATH, state)

    manifest = {
        "schema_version": (
            "kinofail.reconfirmation-f8-ext4-recovery-amendment.v1"
        ),
        "status": "sealed_before_f8_recovery_execution",
        "created_utc": created_utc,
        "passed": True,
        "scope": (
            "storage relocation plus exact zero-observation reboot recovery "
            "and future three-process acquisition"
        ),
        "predecessor_f3": str(F3.relative_to(ROOT)),
        "predecessor_f3_sha256": _sha256(F3),
        "predecessor_f6": str(F6.relative_to(ROOT)),
        "predecessor_f6_sha256": _sha256(F6),
        "predecessor_f7": str(F7.relative_to(ROOT)),
        "predecessor_f7_sha256": _sha256(F7),
        "recovery_state": str(STATE_PATH.relative_to(ROOT)),
        "recovery_state_sha256": _sha256(STATE_PATH),
        "source": str(Path(__file__).resolve().relative_to(ROOT)),
        "source_sha256": _sha256(Path(__file__).resolve()),
        "cause": {
            "classification": (
                "unclean_reboot_left_the_ntfs_scratch_volume_dirty_and_"
                "read_only_after_gpu_driver_liveness_failure"
            ),
            "ntfs_source_used_read_only": True,
            "ext4_copy_content_inventory_equal": True,
            "model_predictions_scores_features_or_checkpoints_used": False,
            "physical_endpoint_values_used_for_recovery_selection": False,
            "selection_inputs": (
                "launcher state and presence of sensor/manifest files only"
            ),
        },
        "storage": {
            "source": str(SOURCE_CORPUS),
            "destination": str(DESTINATION_CORPUS),
            "source_read_only": True,
            "source_inventory": source_inventory,
            "destination_inventory": destination_inventory,
            "future_scratch_root": str(
                DESTINATION_CORPUS.parent.relative_to(ROOT)
            ),
        },
        "recovery": {
            "collector": str(COLLECTOR.relative_to(ROOT)),
            "collector_sha256": _sha256(COLLECTOR),
            "runner": str(RUNNER.relative_to(ROOT)),
            "runner_sha256": _sha256(RUNNER),
            "pipeline": str(PIPELINE.relative_to(ROOT)),
            "pipeline_sha256": _sha256(PIPELINE),
            "watchdog": str(WATCHDOG.relative_to(ROOT)),
            "watchdog_sha256": _sha256(WATCHDOG),
            "deterministic_partition_count": 4,
            "maximum_concurrent_isaac_processes": 3,
            "authorized_reexecution_pair_ids": authorized_reexecution,
            "authorized_reexecution_pair_ids_sha256": _list_sha256(
                authorized_reexecution
            ),
            "authorized_reexecution_rule": (
                "F3 audit state was started, no physical observation file "
                "exists, process ended only because the host rebooted, and "
                "the F7 deliberately interrupted partition-3 pair is excluded"
            ),
            "never_attempted_pair_count": len(never_attempted),
            "never_attempted_pair_ids_sha256": _list_sha256(
                never_attempted
            ),
            "retained_attrition_pair_ids": retained_attrition,
            "retained_attrition_pair_ids_sha256": _list_sha256(
                retained_attrition
            ),
            "result_dependent_retry_permitted": False,
            "any_f8_terminal_attempt_retried": False,
        },
        "timing_evidence": {
            "scale_features_existed_at_seal": False,
            "reconfirmation_predictions_existed_at_seal": False,
            "reconfirmation_scores_existed_at_seal": False,
            "some_model_blind_physical_acquisition_existed": True,
        },
        "scientific_contract": {
            "scene_material_seed_operator_or_parameter_changed": False,
            "simulation_logic_changed": False,
            "sensor_or_feature_logic_changed": False,
            "model_or_route_changed": False,
            "threshold_or_analysis_changed": False,
            "result_dependent_retry_enabled": False,
            "existing_observation_or_summary_modified": False,
        },
    }
    _write_json(MANIFEST_PATH, manifest)
    print(
        json.dumps(
            {
                "state": str(STATE_PATH),
                "state_sha256": _sha256(STATE_PATH),
                "manifest": str(MANIFEST_PATH),
                "manifest_sha256": _sha256(MANIFEST_PATH),
                "counts": state["counts"],
            },
            indent=2,
            sort_keys=True,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
