#!/usr/bin/env python3
"""Merge model-blind shards, infer once, and score the reconfirmation once."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import subprocess
import sys
import time
from datetime import UTC, datetime
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parents[1]
PYTHON = Path("/home/eureka/miniconda3/envs/kinovla/bin/python")
REGISTRY = ROOT / "outputs/kinofail_reconfirmation_v2/scene_registry.json"
SCHEDULE_ROOT = ROOT / "outputs/kinofail_reconfirmation_v2/schedules"
EVAL_ROOT = ROOT / "outputs/eval/unified_moe_v3_reconfirmation_v2"
SHARD_ROOT = EVAL_ROOT / "shards"
CAPSULE_ROOT = (
    ROOT / "outputs/kinofail_reconfirmation_v2/conflict_capsules_ext4"
)
RECEIPT_ROOT = ROOT / "outputs/kinofail_reconfirmation_v2/prune_receipts"
PIPELINE_ROOT = ROOT / "outputs/kinofail_reconfirmation_v2/orchestration"
F0 = (
    ROOT
    / "outputs/freeze/"
    "unified_moe_v3_reconfirmation_f0_transitive_amendment1/"
    "freeze_manifest.json"
)
F1 = (
    ROOT
    / "outputs/freeze/unified_moe_v3_reconfirmation_f1/"
    "seal_manifest.json"
)
F2 = (
    ROOT
    / "outputs/freeze/"
    "unified_moe_v3_reconfirmation_f2_collector_provenance_amendment1/"
    "amendment_manifest.json"
)
F3 = (
    ROOT
    / "outputs/freeze/"
    "unified_moe_v3_reconfirmation_f3_nuisance_contract_amendment1/"
    "amendment_manifest.json"
)
F8 = (
    ROOT
    / "outputs/freeze/"
    "unified_moe_v3_reconfirmation_f8_ext4_recovery_amendment1/"
    "amendment_manifest.json"
)
F9 = (
    ROOT
    / "outputs/freeze/"
    "unified_moe_v3_reconfirmation_f9_global_attrition_amendment1/"
    "amendment_manifest.json"
)
F10 = (
    ROOT
    / "outputs/freeze/"
    "unified_moe_v3_reconfirmation_f10_t3_scene_source_amendment1/"
    "amendment_manifest.json"
)
F11 = (
    ROOT
    / "outputs/freeze/"
    "unified_moe_v3_reconfirmation_f11_global_capsule_attrition_amendment1/"
    "amendment_manifest.json"
)
F12 = (
    ROOT
    / "outputs/freeze/"
    "unified_moe_v3_reconfirmation_f12_process_local_rtx_texture_amendment1/"
    "amendment_manifest.json"
)
F13 = (
    ROOT
    / "outputs/freeze/"
    "unified_moe_v3_reconfirmation_f13_launcher_eligibility_amendment1/"
    "amendment_manifest.json"
)
F14 = (
    ROOT
    / "outputs/freeze/"
    "unified_moe_v3_reconfirmation_f14_t2_zero_observation_liveness_amendment1/"
    "amendment_manifest.json"
)
F14_RECOVERY_AUDIT = (
    ROOT
    / "outputs/kinofail_reconfirmation_v2/corpus_ext4/"
    "confirm_v2_life_scene_01/c2_t2/launcher_audits/"
    "confirm_v2_life_scene_01_f14_resume.json"
)
F15 = (
    ROOT
    / "outputs/freeze/"
    "unified_moe_v3_reconfirmation_f15_runner_supersession_amendment1/"
    "amendment_manifest.json"
)
F15_RECOVERY_AUDIT = (
    ROOT
    / "outputs/kinofail_reconfirmation_v2/orchestration/"
    "confirm_v2_life_scene_01/f15_partition_recovery.json"
)
FRESHNESS = SCHEDULE_ROOT / "freshness_audit.json"
F13_VALIDATION_ROOT = (
    ROOT / "outputs/kinofail_reconfirmation_v2/f13_validations"
)

if str(ROOT / "scripts") not in sys.path:
    sys.path.insert(0, str(ROOT / "scripts"))
from seal_and_prune_kinofail_confirmatory_shard_f13 import (  # noqa: E402
    validate_scene as _validate_f13_scene,
)


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


def _scenes() -> list[str]:
    scenes = [str(row["scene_id"]) for row in _json(REGISTRY)["scenes"]]
    if len(scenes) != 30 or len(set(scenes)) != 30:
        raise RuntimeError("reconfirmation registry must contain 30 scenes")
    return sorted(scenes)


def _active_collectors() -> int:
    count = 0
    for entry in Path("/proc").iterdir():
        if not entry.name.isdigit():
            continue
        try:
            command = (
                (entry / "cmdline")
                .read_bytes()
                .replace(b"\0", b" ")
                .decode("utf-8", errors="replace")
            )
        except (FileNotFoundError, PermissionError, ProcessLookupError):
            continue
        if "isaac_collect_kinofail_confirmatory_" in command:
            count += 1
    return count


def _complete(scenes: list[str]) -> bool:
    for scene in scenes:
        state_path = PIPELINE_ROOT / scene / "pipeline_state.json"
        if state_path.is_file():
            state = _json(state_path)
            if state.get("status") == "terminal_failure":
                raise RuntimeError(
                    f"scene pipeline failed before finalization: {scene}: "
                    f"{state.get('error')}"
                )
        receipt = RECEIPT_ROOT / scene / "completed.json"
        capsule = CAPSULE_ROOT / scene / "capsule_manifest.json"
        if not receipt.is_file() or not capsule.is_file():
            return False
        if (
            _json(receipt).get("state") != "completed"
            or _json(capsule).get("passed") is not True
        ):
            return False
        validation = (
            F13_VALIDATION_ROOT / scene / "validation.json"
        )
        if not validation.is_file():
            _validate_f13_scene(
                scene_id=scene,
                derived_root=SHARD_ROOT,
                receipt_path=receipt,
                capsule_path=capsule,
                output_path=validation,
            )
        if _json(validation).get("passed") is not True:
            raise RuntimeError(f"F13 scene validation failed: {scene}")
    return True


def _wait(scenes: list[str], poll_seconds: int) -> None:
    last_report = 0.0
    while True:
        complete = _complete(scenes)
        active = _active_collectors()
        if complete and active == 0:
            return
        now = time.monotonic()
        if now - last_report >= 600:
            print(
                json.dumps(
                    {
                        "stage": "waiting_for_30_model_blind_scene_shards",
                        "complete_receipts": sum(
                            (
                                RECEIPT_ROOT / scene / "completed.json"
                            ).is_file()
                            for scene in scenes
                        ),
                        "complete_capsules": sum(
                            (
                                CAPSULE_ROOT
                                / scene
                                / "capsule_manifest.json"
                            ).is_file()
                            for scene in scenes
                        ),
                        "complete_f13_validations": sum(
                            (
                                F13_VALIDATION_ROOT
                                / scene
                                / "validation.json"
                            ).is_file()
                            for scene in scenes
                        ),
                        "active_collectors": active,
                    },
                    sort_keys=True,
                ),
                flush=True,
            )
            last_report = now
        time.sleep(poll_seconds)


def _global_snapshot_attrition_gate(
    scenes: list[str],
) -> dict[str, Any]:
    counts = {
        "scale": {"planned_pairs": 0, "attrited_pairs": 0},
        "t3": {"planned_pairs": 0, "attrited_pairs": 0},
    }
    per_scene = []
    for scene in scenes:
        shard = _json(
            SCHEDULE_ROOT / "scenes" / scene / "shard_manifest.json"
        )
        validation = _json(
            F13_VALIDATION_ROOT / scene / "validation.json"
        )
        row = {"scene_id": scene}
        for battery, planned_key in (
            ("scale", "scale_pairs"),
            ("t3", "t3_pairs"),
        ):
            planned = int(shard["counts"][planned_key])
            validation_key = "c2_t3" if battery == "t3" else battery
            attrited = int(
                validation["batteries"][validation_key][
                    "attrited_pairs"
                ]
            )
            if attrited < 0 or attrited > planned:
                raise RuntimeError(
                    f"invalid {battery} attrition accounting for {scene}"
                )
            counts[battery]["planned_pairs"] += planned
            counts[battery]["attrited_pairs"] += attrited
            row[battery] = {
                "planned_pairs": planned,
                "attrited_pairs": attrited,
                "retained_pairs": planned - attrited,
            }
        per_scene.append(row)
    overall_planned = sum(
        row["planned_pairs"] for row in counts.values()
    )
    overall_attrited = sum(
        row["attrited_pairs"] for row in counts.values()
    )
    checks = {}
    for battery, row in counts.items():
        row["attrited_fraction"] = (
            row["attrited_pairs"] / row["planned_pairs"]
        )
        checks[f"{battery}_at_or_below_five_percent"] = (
            row["attrited_pairs"]
            <= int(0.05 * row["planned_pairs"])
        )
    checks["overall_at_or_below_five_percent"] = (
        overall_attrited <= int(0.05 * overall_planned)
    )
    result = {
        "schema_version": (
            "kinofail.reconfirmation-global-snapshot-attrition.v1"
        ),
        "passed": all(checks.values()),
        "threshold": 0.05,
        "threshold_scope": "per_battery_and_overall_not_per_scene",
        "counts": {
            **counts,
            "overall": {
                "planned_pairs": overall_planned,
                "attrited_pairs": overall_attrited,
                "attrited_fraction": (
                    overall_attrited / overall_planned
                ),
            },
        },
        "checks": checks,
        "per_scene": per_scene,
    }
    if result["passed"] is not True:
        raise RuntimeError(
            "frozen battery-level or overall attrition gate failed"
        )
    return result


def _run(command: list[str], name: str) -> None:
    log_root = EVAL_ROOT / "finalization_logs"
    log_root.mkdir(parents=True, exist_ok=True)
    log_path = log_root / f"{name}.log"
    environment = os.environ.copy()
    environment["HF_HUB_OFFLINE"] = "1"
    with log_path.open("x", encoding="utf-8") as log:
        completed = subprocess.run(
            command,
            cwd=ROOT,
            env=environment,
            stdout=log,
            stderr=subprocess.STDOUT,
            check=False,
        )
    if completed.returncode != 0:
        tail = log_path.read_text(encoding="utf-8", errors="replace")[-8000:]
        raise RuntimeError(
            f"reconfirmation finalization failed at {name}: "
            f"{completed.returncode}\n{tail}"
        )


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--poll-seconds", type=int, default=30)
    args = parser.parse_args()
    if args.poll_seconds < 5:
        raise ValueError("poll interval must be at least five seconds")
    final_audit_path = EVAL_ROOT / "finalization_audit.json"
    if final_audit_path.exists():
        raise FileExistsError(final_audit_path)
    for path in (
        F0, F1, F2, F3, F8, F9, F10, F11, F12, F13, F14, F15, FRESHNESS
    ):
        if not path.is_file():
            raise FileNotFoundError(path)
    f10 = _json(F10)
    if (
        f10.get("passed") is not True
        or f10.get("schema_version")
        != (
            "kinofail.reconfirmation-f10-t3-scene-source-"
            "provenance-amendment.v1"
        )
        or f10.get("status")
        != "sealed_before_resumption_of_t3_acquisition"
    ):
        raise RuntimeError("F10 provenance amendment is invalid")
    f13 = _json(F13)
    f11 = _json(F11)
    f11_correction = f11.get("correction", {})
    capsule_builder = ROOT / str(
        f11_correction.get("capsule_builder", "")
    )
    if (
        f11.get("passed") is not True
        or f11.get("schema_version")
        != (
            "kinofail.reconfirmation-f11-global-capsule-"
            "attrition-amendment.v1"
        )
        or f11.get("status")
        != "sealed_before_resumption_of_physical_acquisition"
        or not capsule_builder.is_file()
        or f11_correction.get("capsule_builder_sha256")
        != f13.get("correction", {}).get(
            "predecessor_capsule_builder_sha256"
        )
        or f11_correction.get("global_gate_preserved") is not True
    ):
        raise RuntimeError("F11 global attrition amendment is invalid")
    f12 = _json(F12)
    f12_correction = f12.get("correction", {})
    backend = ROOT / str(f12_correction.get("backend", ""))
    if (
        f12.get("passed") is not True
        or f12.get("schema_version")
        != (
            "kinofail.reconfirmation-f12-process-local-rtx-"
            "texture-amendment.v1"
        )
        or f12.get("status")
        != "sealed_before_resumption_after_rtx_texture_race"
        or not backend.is_file()
        or f12_correction.get("backend_sha256") != _sha256(backend)
        or f12_correction.get("process_local_namespace") is not True
    ):
        raise RuntimeError(
            "F12 process-local RTX texture amendment is invalid"
        )
    f13_correction = f13.get("correction", {})
    helper = ROOT / str(f13_correction.get("ledger_helper", ""))
    snapshot_entrypoint = ROOT / str(
        f13_correction.get("snapshot_entrypoint", "")
    )
    capsule_builder_f13 = ROOT / str(
        f13_correction.get("capsule_builder", "")
    )
    f14 = _json(F14)
    f14_recovery = f14.get("recovery", {})
    f14_script = ROOT / str(f14_recovery.get("recovery_script", ""))
    f15 = _json(F15)
    f15_correction = f15.get("correction", {})
    f15_runner = ROOT / str(f15_correction.get("runner", ""))
    f15_script = ROOT / str(
        f15_correction.get("recovery_script", "")
    )
    if (
        f13.get("passed") is not True
        or f13.get("schema_version")
        != (
            "kinofail.reconfirmation-f13-launcher-eligibility-"
            "amendment.v1"
        )
        or f13.get("status")
        != "sealed_before_resumption_of_scene_01_post_acquisition"
        or f13.get("predecessor_f12_sha256") != _sha256(F12)
        or f13_correction.get("predecessor_finalizer_sha256")
        != f12_correction.get("finalizer_sha256")
        or f14.get("passed") is not True
        or f14.get("schema_version")
        != (
            "kinofail.reconfirmation-f14-t2-zero-observation-"
            "liveness-amendment.v1"
        )
        or f14.get("status")
        != "sealed_before_termination_and_zero_observation_resume"
        or f14.get("predecessor_f13_sha256") != _sha256(F13)
        or f14_recovery.get("predecessor_finalizer_sha256")
        != f13_correction.get("finalizer_sha256")
        or f14_recovery.get("finalizer_sha256")
        != f15_correction.get("predecessor_finalizer_sha256")
        or f15.get("passed") is not True
        or f15.get("schema_version")
        != (
            "kinofail.reconfirmation-f15-runner-"
            "supersession-amendment.v1"
        )
        or f15.get("status")
        != "sealed_before_scene01_unstarted_partition_launch"
        or f15.get("predecessor_f14_sha256") != _sha256(F14)
        or f15_correction.get("finalizer_sha256")
        != _sha256(Path(__file__).resolve())
        or not f15_runner.is_file()
        or f15_correction.get("runner_sha256")
        != _sha256(f15_runner)
        or not f15_script.is_file()
        or f15_correction.get("recovery_script_sha256")
        != _sha256(f15_script)
        or not f14_script.is_file()
        or f14_recovery.get("recovery_script_sha256")
        != _sha256(f14_script)
        or not helper.is_file()
        or f13_correction.get("ledger_helper_sha256") != _sha256(helper)
        or not snapshot_entrypoint.is_file()
        or f13_correction.get("snapshot_entrypoint_sha256")
        != _sha256(snapshot_entrypoint)
        or not capsule_builder_f13.is_file()
        or f13_correction.get("capsule_builder_sha256")
        != _sha256(capsule_builder_f13)
    ):
        raise RuntimeError("F13/F14 operational amendment chain is invalid")
    f14_audit = _json(F14_RECOVERY_AUDIT)
    if (
        f14_audit.get("state") != "terminal"
        or f14_audit.get("passed") is not True
        or f14_audit.get("summary_exists") is not True
        or f14_audit.get("all_scheduled_cases_accounted") is not True
        or int(f14_audit.get("summary_case_count", -1)) != 50
        or f14_audit.get("sealed_manifest_hashes_preserved") is not True
        or f14_audit.get("existing_manifest_reexecution_permitted")
        is not False
        or f14_audit.get("result_dependent_retry_or_selection") is not False
        or f14_audit.get(
            "model_feature_label_outcome_prediction_or_score_read"
        )
        is not False
        or f14_audit.get("operational_amendment_sha256")
        != _sha256(F14)
    ):
        raise RuntimeError("F14 T2 liveness recovery audit is invalid")
    f15_audit = _json(F15_RECOVERY_AUDIT)
    if (
        f15_audit.get("state") != "terminal"
        or f15_audit.get("passed") is not True
        or f15_audit.get("scene_parent_resumed") is not True
        or f15_audit.get("partitions_2_and_3_had_launcher_attempts_before_f15")
        is not False
        or f15_audit.get("existing_pair_reexecution_permitted") is not False
        or f15_audit.get("result_dependent_retry_or_selection") is not False
        or f15_audit.get(
            "model_feature_label_outcome_prediction_or_score_read"
        )
        is not False
        or int(
            f15_audit.get(
                "maximum_concurrent_pair_runners_observed", 99
            )
        )
        > 3
        or f15_audit.get("accounting", {}).get(
            "schedule_partitioned_exactly_once"
        )
        is not True
        or int(
            f15_audit.get("accounting", {}).get(
                "unique_terminal_pair_ids", -1
            )
        )
        != 352
        or f15_audit.get("operational_amendment_sha256")
        != _sha256(F15)
    ):
        raise RuntimeError(
            "F15 never-started partition recovery audit is invalid"
        )
    if list(EVAL_ROOT.glob("**/*prediction*")):
        raise RuntimeError("prediction artifact exists before finalization")
    scenes = _scenes()
    _wait(scenes, args.poll_seconds)
    global_attrition = _global_snapshot_attrition_gate(scenes)

    conflict = EVAL_ROOT / "conflict"
    merged_t2 = conflict / "t2_merged"
    merge = [
        str(PYTHON),
        "scripts/merge_kinofail_confirmatory_t2_features_v1.py",
    ]
    for scene in scenes:
        merge.extend(
            [
                "--scene-feature-dir",
                str(SHARD_ROOT / scene / "c2_t2/features"),
                "--scene-corpus-dir",
                str(CAPSULE_ROOT / scene / "c2_t2"),
            ]
        )
    merge.extend(
        [
            "--t2-schedule",
            str(SCHEDULE_ROOT / "c2_t2/schedule.jsonl"),
            "--output",
            str(merged_t2),
        ]
    )
    _run(merge, "01_merge_t2_capsules")

    valid_design = conflict / "valid_design"
    _run(
        [
            str(PYTHON),
            "scripts/prepare_kinofail_confirmatory_valid_conflict_design_v1.py",
            "--t2-schedule",
            str(SCHEDULE_ROOT / "c2_t2/schedule.jsonl"),
            "--t2-features",
            str(merged_t2),
            "--t3-design",
            str(SCHEDULE_ROOT / "c2_t3"),
            "--t3-corpus",
            str(CAPSULE_ROOT),
            "--output",
            str(valid_design),
        ],
        "02_valid_conflict_design",
    )

    base_features = conflict / "c2_base"
    _run(
        [
            str(PYTHON),
            "scripts/build_kinofail_confirmatory_c2_base_features_v1.py",
            "--c1-features",
            str(merged_t2),
            "--c1-corpus",
            str(CAPSULE_ROOT),
            "--t3-design",
            str(valid_design),
            "--t3-corpus",
            str(CAPSULE_ROOT),
            "--output",
            str(base_features),
            "--batch-size",
            "64",
        ],
        "03_conflict_base_features",
    )

    conflict_features = conflict / "features"
    v5 = [
        str(PYTHON),
        "scripts/build_kinofail_realistic_c2_v5_features.py",
        "--base-features",
        str(base_features),
    ]
    for scene in scenes:
        v5.extend(
            ["--t2-corpus", str(CAPSULE_ROOT / scene / "c2_t2")]
        )
    v5.extend(
        [
            "--t3-corpus",
            str(CAPSULE_ROOT),
            "--output",
            str(conflict_features),
        ]
    )
    _run(v5, "04_conflict_v5_features")

    inputs = EVAL_ROOT / "evaluation_inputs"
    build_inputs = [
        str(PYTHON),
        "scripts/build_kinofail_confirmatory_evaluation_bundle_inputs_v1.py",
        "--scale-schedule",
        str(SCHEDULE_ROOT / "scale_schedule.jsonl"),
    ]
    for scene in scenes:
        build_inputs.extend(
            [
                "--scale-snapshot-dir",
                str(SHARD_ROOT / scene / "scale/snapshots"),
                "--scale-unified-dir",
                str(SHARD_ROOT / scene / "scale/unified_features"),
            ]
        )
    build_inputs.extend(
        [
            "--conflict-schedule",
            str(SCHEDULE_ROOT / "conflict_schedule.jsonl"),
            "--conflict-feature-dir",
            str(conflict_features),
            "--output",
            str(inputs),
        ]
    )
    _run(build_inputs, "05_evaluation_inputs")

    blind_bundle = EVAL_ROOT / "blind_bundle"
    _run(
        [
            str(PYTHON),
            "scripts/assemble_kinofail_confirmatory_blind_bundle_v1.py",
            "--f1-manifest",
            str(F1),
            "--feature-shard",
            str(inputs / "scale_features.npz"),
            "--feature-shard",
            str(inputs / "conflict_features.npz"),
            "--truth-ledger",
            str(inputs / "scale_truth.jsonl"),
            "--truth-ledger",
            str(inputs / "conflict_truth.jsonl"),
            "--out",
            str(blind_bundle),
        ],
        "06_blind_bundle",
    )

    predictions = EVAL_ROOT / "blind_predictions"
    _run(
        [
            str(PYTHON),
            "scripts/predict_kinofail_unified_moe_v3_blind.py",
            "--f0-manifest",
            str(F0),
            "--f1-manifest",
            str(F1),
            "--freshness-audit",
            str(FRESHNESS),
            "--features",
            str(blind_bundle / "blind_features.npz"),
            "--output-dir",
            str(predictions),
        ],
        "07_blind_prediction_once",
    )
    prediction_manifest = _json(
        predictions / "prediction_manifest.json"
    )
    if (
        prediction_manifest.get("status") != "blind_predictions_sealed"
        or prediction_manifest.get("labels_or_outcomes_read") is not False
        or prediction_manifest.get("fit_or_refit_called") is not False
    ):
        raise RuntimeError("blind prediction contract failed")

    protocol = EVAL_ROOT / "scoring_protocol.json"
    _run(
        [
            str(PYTHON),
            "scripts/build_kinofail_confirmatory_scoring_protocol_v1.py",
            "--blind-predictions",
            str(predictions / "blind_predictions.jsonl"),
            "--truth-key",
            str(blind_bundle / "truth_key.jsonl"),
            "--f0-manifest",
            str(F0),
            "--f1-manifest",
            str(F1),
            "--feature-manifest",
            str(blind_bundle / "bundle_manifest.json"),
            "--out",
            str(protocol),
        ],
        "08_seal_scoring_protocol",
    )

    report = EVAL_ROOT / "confirmatory_report.json"
    _run(
        [
            str(PYTHON),
            "scripts/score_kinofail_unified_moe_v3_confirmatory.py",
            "--protocol",
            str(protocol),
            "--blind-predictions",
            str(predictions / "blind_predictions.jsonl"),
            "--truth-key",
            str(blind_bundle / "truth_key.jsonl"),
            "--out",
            str(report),
        ],
        "09_score_once",
    )
    scored = _json(report)
    if scored.get("confirmatory_protocol_valid") is not True:
        raise RuntimeError("one-shot scoring protocol was invalid")

    audit = {
        "schema_version": "kinofail.reconfirmation-finalization.v2",
        "created_utc": datetime.now(UTC).isoformat(),
        "passed": True,
        "score_once": True,
        "report_regardless_of_outcome": True,
        "passed_all_preregistered_gates": scored.get(
            "passed_all_preregistered_gates"
        ),
        "model_or_endpoint_outcomes_used_before_blind_prediction": False,
        "source_sha256": {
            "f0": _sha256(F0),
            "f1": _sha256(F1),
            "f2_operational_amendment": _sha256(F2),
            "f3_operational_amendment": _sha256(F3),
            "f8_operational_amendment": _sha256(F8),
            "f9_global_attrition_amendment": _sha256(F9),
            "f10_scene_source_amendment": _sha256(F10),
            "f11_capsule_attrition_amendment": _sha256(F11),
            "f12_process_local_texture_amendment": _sha256(F12),
            "f13_launcher_eligibility_amendment": _sha256(F13),
            "f14_t2_liveness_amendment": _sha256(F14),
            "f14_t2_liveness_recovery_audit": _sha256(
                F14_RECOVERY_AUDIT
            ),
            "f15_runner_supersession_amendment": _sha256(F15),
            "f15_partition_recovery_audit": _sha256(
                F15_RECOVERY_AUDIT
            ),
            "freshness": _sha256(FRESHNESS),
            "conflict_features": _sha256(
                conflict_features / "feature_manifest.json"
            ),
            "evaluation_inputs": _sha256(inputs / "audit.json"),
            "blind_bundle": _sha256(blind_bundle / "bundle_manifest.json"),
            "blind_predictions": _sha256(
                predictions / "prediction_manifest.json"
            ),
            "scoring_protocol": _sha256(protocol),
            "confirmatory_report": _sha256(report),
        },
        "counts": {
            "scenes": len(scenes),
            "capsules": len(scenes),
            "prediction_rows": prediction_manifest.get(
                "artifact", {}
            ).get("rows"),
        },
        "global_snapshot_attrition": global_attrition,
        "f13_scene_validations": {
            scene: _sha256(
                F13_VALIDATION_ROOT / scene / "validation.json"
            )
            for scene in scenes
        },
    }
    final_audit_path.write_text(
        json.dumps(audit, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    print(json.dumps(audit, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
