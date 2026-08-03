#!/usr/bin/env python3
"""One-shot model-blind recovery of production-scene-00 T2 liveness.

The recovery preserves the contiguous prefix of sealed passing manifests
byte-for-byte, permits re-entry only into the next scheduled case when its
directory contains zero files, and continues all never-attempted cases through
the frozen collector's existing ``--resume`` path.  The scene pipeline is
paused while recovery runs so that T3 cannot overlap the replacement T2 Isaac
process.
"""

from __future__ import annotations

import argparse
import json
import os
import signal
import subprocess
import sys
import time
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from scripts.recover_kinofail_reconfirmation_t2_f14 import (
    _process_age_seconds,
    _process_state,
    read_json,
    read_jsonl,
    sha256,
    terminate_exact_processes,
    write_json,
)


SCENE = "confirm_v2_production_scene_00"
F15 = (
    ROOT
    / "outputs/freeze/"
    "unified_moe_v3_reconfirmation_f15_runner_supersession_amendment1/"
    "amendment_manifest.json"
)
F16 = (
    ROOT
    / "outputs/freeze/"
    "unified_moe_v3_reconfirmation_f16_t2_production00_liveness_amendment1/"
    "amendment_manifest.json"
)
SCHEDULE = (
    ROOT
    / "outputs/kinofail_reconfirmation_v2/schedules/c2_t2/schedule.jsonl"
)
REGISTRY = ROOT / "outputs/kinofail_reconfirmation_v2/scene_registry.json"
ASSET_LOCK = (
    ROOT
    / "outputs/assets/terrain_pbr_confirmatory_v2/terrain_assets.lock.json"
)
PROTOCOL = (
    ROOT
    / "outputs/kinofail_reconfirmation_v2/schedules/c2_t2/"
    "collection_protocol.json"
)
OUT = (
    ROOT
    / "outputs/kinofail_reconfirmation_v2/corpus_ext4"
    / SCENE
    / "c2_t2"
)
COLLECTOR = ROOT / "scripts/isaac_collect_kinofail_confirmatory_t2_v1.py"
IMPLEMENTATION = (
    ROOT / "scripts/isaac_collect_kinofail_realistic_c1_causal_v1.py"
)
SCENE_PIPELINE = (
    ROOT / "scripts/run_kinofail_reconfirmation_scene_pipeline_v2.py"
)
SOURCE_AUDIT = OUT / "launcher_audits" / f"{SCENE}.json"
SOURCE_LOG = OUT / "launcher_logs" / f"{SCENE}.log"
PIPELINE_STATE = (
    ROOT
    / "outputs/kinofail_reconfirmation_v2/orchestration"
    / SCENE
    / "pipeline_state.json"
)
RECOVERY_AUDIT = OUT / "launcher_audits" / f"{SCENE}_f16_resume.json"
RECOVERY_LOG = OUT / "launcher_logs" / f"{SCENE}_f16_resume.log"
ISAACLAB = Path("/home/eureka/IsaacLab-v2.3.0/isaaclab.sh")
EXPERIENCE = Path(
    "/home/eureka/IsaacLab-v2.3.0/apps/"
    "isaaclab.python.headless.rendering.kit"
)
CONDA_PREFIX = Path("/home/eureka/miniconda3/envs/kinovla")
ISAACLAB_PYTHONPATH = (
    "/home/eureka/IsaacLab-v2.3.0/source/isaaclab:"
    "/home/eureka/IsaacLab-v2.3.0/source/isaaclab_tasks:"
    "/home/eureka/IsaacLab-v2.3.0/source/isaaclab_assets:"
    "/home/eureka/IsaacLab-v2.3.0/source/isaaclab_rl:"
    "/home/eureka/KinoVLA"
)
LIVENESS_DEADLINE_SECONDS = 600
EXPECTED_CASE_COUNT = 50
EXPECTED_SEALED_PREFIX = 30


def _cmdline(pid: int) -> list[str]:
    try:
        return [
            token.decode("utf-8", errors="replace")
            for token in (
                Path("/proc") / str(pid) / "cmdline"
            ).read_bytes().split(b"\0")
            if token
        ]
    except (FileNotFoundError, PermissionError, ProcessLookupError):
        return []


def _matching_processes(
    *,
    required_tokens: tuple[str, ...],
) -> list[dict[str, Any]]:
    found = []
    for entry in Path("/proc").iterdir():
        if not entry.name.isdigit():
            continue
        pid = int(entry.name)
        tokens = _cmdline(pid)
        joined = " ".join(tokens)
        if not tokens or not all(token in joined for token in required_tokens):
            continue
        found.append(
            {
                "pid": pid,
                "state": _process_state(pid),
                "age_seconds": _process_age_seconds(pid),
                "executable": tokens[0],
                "command": tokens,
            }
        )
    return sorted(found, key=lambda row: int(row["pid"]))


def t2_processes() -> list[dict[str, Any]]:
    return _matching_processes(
        required_tokens=(str(COLLECTOR), "--scene", SCENE)
    )


def scene_pipeline_processes() -> list[dict[str, Any]]:
    return _matching_processes(
        required_tokens=(str(SCENE_PIPELINE), "--scene", SCENE)
    )


def source_launcher_processes() -> list[dict[str, Any]]:
    runner = ROOT / "scripts/run_kinofail_reconfirmation_t2_scene_v2.py"
    return _matching_processes(
        required_tokens=(str(runner), "--scene", SCENE)
    )


def scene_rows() -> list[dict[str, Any]]:
    return [
        row
        for row in read_jsonl(SCHEDULE)
        if str(row["scene_cluster"]) == SCENE
    ]


def case_recovery_state(
    *,
    rows: list[dict[str, Any]],
    out: Path,
) -> dict[str, Any]:
    case_ids = [str(row["case_id"]) for row in rows]
    if len(case_ids) != len(set(case_ids)):
        raise RuntimeError("T2 schedule repeats a case_id")
    sealed: list[dict[str, Any]] = []
    zero_observation: list[str] = []
    never_attempted: list[str] = []
    first_unsealed_index: int | None = None
    for index, case_id in enumerate(case_ids):
        case_dir = out / case_id
        manifest = case_dir / "manifest.json"
        if manifest.is_file():
            value = read_json(manifest)
            if (
                value.get("case_id") != case_id
                or value.get("scene_cluster") != SCENE
                or value.get("passed") is not True
            ):
                raise RuntimeError(
                    "F16 forbids replay of an existing failed or mismatched "
                    f"T2 manifest: {manifest}"
                )
            if first_unsealed_index is not None:
                raise RuntimeError(
                    "sealed T2 manifests are not a contiguous prefix"
                )
            sealed.append(
                {
                    "case_id": case_id,
                    "manifest": str(manifest),
                    "manifest_sha256": sha256(manifest),
                }
            )
            continue
        if first_unsealed_index is None:
            first_unsealed_index = index
        if case_dir.exists():
            files = [path for path in case_dir.rglob("*") if path.is_file()]
            if files:
                raise RuntimeError(
                    "F16 zero-observation case contains artifact files: "
                    f"{case_id}"
                )
            zero_observation.append(case_id)
        else:
            never_attempted.append(case_id)
    if len(case_ids) != EXPECTED_CASE_COUNT:
        raise RuntimeError("unexpected T2 scene case count")
    if len(sealed) != EXPECTED_SEALED_PREFIX:
        raise RuntimeError("unexpected sealed T2 prefix length")
    if len(zero_observation) != 1:
        raise RuntimeError(
            "F16 requires exactly one started zero-observation T2 case"
        )
    if first_unsealed_index != len(sealed):
        raise RuntimeError("F16 T2 prefix accounting is inconsistent")
    if zero_observation[0] != case_ids[len(sealed)]:
        raise RuntimeError(
            "F16 zero-observation case is not the next scheduled case"
        )
    if (
        len(sealed) + len(zero_observation) + len(never_attempted)
        != len(case_ids)
    ):
        raise RuntimeError("F16 T2 cases do not partition the schedule")
    return {
        "scheduled_case_count": len(case_ids),
        "sealed_manifest_count": len(sealed),
        "sealed_manifests": sealed,
        "zero_observation_case_ids": zero_observation,
        "never_attempted_case_ids": never_attempted,
        "existing_failed_manifest_count": 0,
        "case_ids_partition_schedule": True,
    }


def _prediction_files() -> list[str]:
    root = ROOT / "outputs/eval/unified_moe_v3_reconfirmation_v2"
    values = []
    for path in root.rglob("*"):
        if path.is_file() and (
            "prediction" in path.name.lower()
            or "score" in path.name.lower()
            or path.name == "confirmatory_report.json"
        ):
            values.append(str(path))
    return sorted(values)


def _preflight() -> dict[str, Any]:
    for path in (
        F15,
        SCHEDULE,
        REGISTRY,
        ASSET_LOCK,
        PROTOCOL,
        COLLECTOR,
        IMPLEMENTATION,
        SCENE_PIPELINE,
        SOURCE_AUDIT,
        SOURCE_LOG,
        PIPELINE_STATE,
        ISAACLAB,
        EXPERIENCE,
        CONDA_PREFIX / "bin/python",
    ):
        if not path.is_file():
            raise FileNotFoundError(path)
    if _prediction_files():
        raise RuntimeError("prediction or score exists before F16 recovery")
    audit = read_json(SOURCE_AUDIT)
    if (
        audit.get("state") != "started"
        or audit.get("retry_authorized") is not False
        or audit.get("summary_exists") is not None
    ):
        raise RuntimeError("F16 source T2 launch is not an interrupted start")
    state = case_recovery_state(rows=scene_rows(), out=OUT)
    processes = t2_processes()
    pipelines = scene_pipeline_processes()
    if not processes:
        raise RuntimeError("F16 T2 collector process is absent")
    if len(pipelines) != 1:
        raise RuntimeError("F16 requires exactly one active scene pipeline")
    inactive_seconds = max(0.0, time.time() - SOURCE_LOG.stat().st_mtime)
    maximum_age = max(float(row["age_seconds"]) for row in processes)
    if (
        inactive_seconds < LIVENESS_DEADLINE_SECONDS
        or maximum_age < LIVENESS_DEADLINE_SECONDS
    ):
        raise RuntimeError("F16 fixed liveness deadline has not elapsed")
    return {
        "recovery_state": state,
        "processes": processes,
        "pipelines": pipelines,
        "source_launcher_processes": source_launcher_processes(),
        "inactive_seconds": inactive_seconds,
        "maximum_age": maximum_age,
    }


def _amendment(preflight: dict[str, Any]) -> dict[str, Any]:
    recovery_state = preflight["recovery_state"]
    return {
        "schema_version": (
            "kinofail.reconfirmation-f16-t2-production00-"
            "liveness-amendment.v1"
        ),
        "created_utc": datetime.now(UTC).isoformat(),
        "status": "sealed_before_termination_and_zero_observation_resume",
        "passed": True,
        "predecessor_f15": str(F15.relative_to(ROOT)),
        "predecessor_f15_sha256": sha256(F15),
        "scope": (
            "pause only production-scene-00 orchestration; terminate only its "
            "fixed-deadline T2 collector; preserve 30 sealed manifests "
            "byte-for-byte; resume exactly one zero-file case and 19 "
            "never-attempted cases through the frozen collector"
        ),
        "evidence": {
            "source_launcher_audit": str(SOURCE_AUDIT),
            "source_launcher_audit_sha256_before_termination": sha256(
                SOURCE_AUDIT
            ),
            "source_runtime_log": str(SOURCE_LOG),
            "source_runtime_log_sha256_before_termination": sha256(SOURCE_LOG),
            "source_runtime_log_bytes": SOURCE_LOG.stat().st_size,
            "source_runtime_log_last_modified_utc": datetime.fromtimestamp(
                SOURCE_LOG.stat().st_mtime, tz=UTC
            ).isoformat(),
            "source_pipeline_state": str(PIPELINE_STATE),
            "source_pipeline_state_sha256_before_pause": sha256(PIPELINE_STATE),
            "model_blind_feature_files_for_scene_existed_at_seal": False,
            "reconfirmation_prediction_files_existed_at_seal": False,
            "reconfirmation_score_files_existed_at_seal": False,
        },
        "incident": {
            "classification": (
                "native_isaac_t2_process_liveness_failure_after_a_"
                "contiguous_sealed_case_prefix"
            ),
            "scene_id": SCENE,
            "fixed_liveness_deadline_seconds": LIVENESS_DEADLINE_SECONDS,
            "log_inactivity_seconds_at_preseal_audit": preflight[
                "inactive_seconds"
            ],
            "maximum_process_age_seconds_at_preseal_audit": preflight[
                "maximum_age"
            ],
            "identified_from": (
                "process age, launcher state, log inactivity, frozen schedule "
                "order, and artifact-file existence only"
            ),
            "model_feature_label_outcome_prediction_or_score_used": False,
            "observed_t2_processes": preflight["processes"],
            "observed_scene_pipeline_processes": preflight["pipelines"],
            **recovery_state,
        },
        "recovery": {
            "recovery_script": str(Path(__file__).resolve().relative_to(ROOT)),
            "recovery_script_sha256": sha256(Path(__file__).resolve()),
            "collector": str(COLLECTOR.relative_to(ROOT)),
            "collector_sha256": sha256(COLLECTOR),
            "collector_implementation": str(IMPLEMENTATION.relative_to(ROOT)),
            "collector_implementation_sha256": sha256(IMPLEMENTATION),
            "schedule": str(SCHEDULE.relative_to(ROOT)),
            "schedule_sha256": sha256(SCHEDULE),
            "protocol": str(PROTOCOL.relative_to(ROOT)),
            "protocol_sha256": sha256(PROTOCOL),
            "scene_pipeline": str(SCENE_PIPELINE.relative_to(ROOT)),
            "scene_pipeline_sha256": sha256(SCENE_PIPELINE),
            "uses_existing_frozen_collector_resume_path": True,
            "scene_pipeline_paused_until_resume_passes": True,
        },
        "retry_disposition": {
            "existing_manifest_case_ids_reexecuted": [],
            "existing_manifest_reexecution_permitted": False,
            "maximum_resume_launches_for_this_incident": 1,
            "zero_observation_case_reexecution_permitted": recovery_state[
                "zero_observation_case_ids"
            ],
            "never_attempted_cases_are_not_retries": True,
            "result_dependent_retry_permitted": False,
        },
        "scientific_contract": {
            "architecture_checkpoint_or_route_changed": False,
            "existing_manifest_or_observation_modified": False,
            "feature_definition_or_sample_selection_changed": False,
            "operator_parameter_or_physics_changed": False,
            "scene_material_seed_or_schedule_changed": False,
            "sensor_or_render_logic_changed": False,
            "statistical_analysis_or_threshold_changed": False,
            "result_dependent_retry_enabled": False,
        },
    }


def _validate_amendment(
    amendment: dict[str, Any],
    preflight: dict[str, Any],
) -> None:
    recovery = amendment.get("recovery", {})
    incident = amendment.get("incident", {})
    if (
        amendment.get("passed") is not True
        or amendment.get("schema_version")
        != (
            "kinofail.reconfirmation-f16-t2-production00-"
            "liveness-amendment.v1"
        )
        or amendment.get("status")
        != "sealed_before_termination_and_zero_observation_resume"
        or amendment.get("predecessor_f15_sha256") != sha256(F15)
        or recovery.get("recovery_script_sha256")
        != sha256(Path(__file__).resolve())
        or recovery.get("collector_sha256") != sha256(COLLECTOR)
        or recovery.get("collector_implementation_sha256")
        != sha256(IMPLEMENTATION)
        or recovery.get("schedule_sha256") != sha256(SCHEDULE)
        or recovery.get("protocol_sha256") != sha256(PROTOCOL)
        or incident.get("sealed_manifests")
        != preflight["recovery_state"]["sealed_manifests"]
        or incident.get("zero_observation_case_ids")
        != preflight["recovery_state"]["zero_observation_case_ids"]
        or incident.get("never_attempted_case_ids")
        != preflight["recovery_state"]["never_attempted_case_ids"]
    ):
        raise RuntimeError("invalid or state-mismatched F16 amendment")


def _pause_pipelines(
    pipelines: list[dict[str, Any]],
) -> list[int]:
    stopped = []
    for row in pipelines:
        pid = int(row["pid"])
        os.kill(pid, signal.SIGSTOP)
        stopped.append(pid)
    deadline = time.monotonic() + 10.0
    while time.monotonic() < deadline:
        if all(_process_state(pid) == "T" for pid in stopped):
            return stopped
        time.sleep(0.1)
    raise RuntimeError("scene pipeline did not enter stopped state")


def _resume_pipelines(pids: list[int]) -> list[int]:
    resumed = []
    for pid in pids:
        if _process_state(pid) not in {None, "Z"}:
            os.kill(pid, signal.SIGCONT)
            resumed.append(pid)
    return resumed


def _wait_for_source_launcher_exit(timeout_seconds: int = 30) -> None:
    deadline = time.monotonic() + timeout_seconds
    while time.monotonic() < deadline:
        if not source_launcher_processes():
            return
        time.sleep(0.5)
    raise RuntimeError("source T2 launcher did not exit after collector stop")


def _run_resume() -> subprocess.CompletedProcess[Any]:
    environment = os.environ.copy()
    environment["CONDA_PREFIX"] = str(CONDA_PREFIX)
    environment["PATH"] = (
        f"{CONDA_PREFIX / 'bin'}:{environment.get('PATH', '')}"
    )
    environment["OMNI_KIT_ACCEPT_EULA"] = "YES"
    environment["PYTHONPATH"] = ISAACLAB_PYTHONPATH
    command = [
        str(ISAACLAB),
        "-p",
        str(COLLECTOR),
        "--schedule",
        str(SCHEDULE),
        "--scene-registry",
        str(REGISTRY),
        "--asset-lock",
        str(ASSET_LOCK),
        "--protocol",
        str(PROTOCOL),
        "--out",
        str(OUT),
        "--scene",
        SCENE,
        "--resume",
        "--headless",
        "--enable_cameras",
        "--experience",
        str(EXPERIENCE),
    ]
    with RECOVERY_LOG.open("x", encoding="utf-8") as log:
        return subprocess.run(
            command,
            cwd=ROOT,
            env=environment,
            stdout=log,
            stderr=subprocess.STDOUT,
            check=False,
        )


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--preflight", action="store_true")
    args = parser.parse_args()
    preflight = _preflight()
    if args.preflight:
        print(json.dumps(preflight, indent=2, sort_keys=True))
        return 0
    if RECOVERY_AUDIT.exists() or RECOVERY_LOG.exists():
        raise FileExistsError(
            RECOVERY_AUDIT if RECOVERY_AUDIT.exists() else RECOVERY_LOG
        )
    if F16.exists():
        amendment = read_json(F16)
    else:
        amendment = _amendment(preflight)
        write_json(F16, amendment)
    _validate_amendment(amendment, preflight)
    if (
        sha256(SOURCE_AUDIT)
        != amendment["evidence"][
            "source_launcher_audit_sha256_before_termination"
        ]
        or sha256(SOURCE_LOG)
        != amendment["evidence"][
            "source_runtime_log_sha256_before_termination"
        ]
    ):
        raise RuntimeError("F16 source evidence changed after seal")

    stopped_pipelines = _pause_pipelines(preflight["pipelines"])
    termination = terminate_exact_processes(preflight["processes"])
    _wait_for_source_launcher_exit()
    started = {
        "schema_version": (
            "kinofail.reconfirmation-f16-t2-production00-resume.v1"
        ),
        "state": "started",
        "started_utc": datetime.now(UTC).isoformat(),
        "scene_id": SCENE,
        "operational_amendment": str(F16),
        "operational_amendment_sha256": sha256(F16),
        "source_launcher_audit": str(SOURCE_AUDIT),
        "source_runtime_log": str(SOURCE_LOG),
        "source_log_inactive_seconds_at_action": preflight[
            "inactive_seconds"
        ],
        "maximum_process_age_seconds_at_action": preflight["maximum_age"],
        "fixed_liveness_deadline_seconds": LIVENESS_DEADLINE_SECONDS,
        "scene_pipeline_pids_stopped": stopped_pipelines,
        "termination": termination,
        "recovery_state": preflight["recovery_state"],
        "existing_manifest_reexecution_permitted": False,
        "zero_observation_reexecution_permitted": True,
        "result_dependent_retry_or_selection": False,
        "model_feature_label_outcome_prediction_or_score_read": False,
    }
    write_json(RECOVERY_AUDIT, started)
    completed = _run_resume()

    summary_path = OUT / "scene_summaries" / f"{SCENE}.json"
    summary = read_json(summary_path) if summary_path.is_file() else {}
    rows = scene_rows()
    final_manifests = [
        OUT / str(row["case_id"]) / "manifest.json" for row in rows
    ]
    all_scheduled_cases_accounted = (
        all(path.is_file() for path in final_manifests)
        and int(summary.get("case_count", -1)) == len(rows)
        and len(summary.get("results", [])) == len(rows)
    )
    passed = summary.get("passed") is True and all_scheduled_cases_accounted
    resumed_pipelines = _resume_pipelines(stopped_pipelines) if passed else []
    terminal = {
        **started,
        "state": "terminal",
        "completed_utc": datetime.now(UTC).isoformat(),
        "returncode": int(completed.returncode),
        "summary_exists": summary_path.is_file(),
        "passed": passed,
        "summary": str(summary_path),
        "summary_sha256": (
            sha256(summary_path) if summary_path.is_file() else None
        ),
        "summary_case_count": summary.get("case_count"),
        "all_scheduled_cases_accounted": all_scheduled_cases_accounted,
        "final_manifest_sha256": {
            path.parent.name: sha256(path)
            for path in final_manifests
            if path.is_file()
        },
        "resume_log": str(RECOVERY_LOG),
        "resume_log_sha256": sha256(RECOVERY_LOG),
        "source_launcher_audit_sha256_after_termination": sha256(
            SOURCE_AUDIT
        ),
        "sealed_manifest_hashes_preserved": all(
            sha256(Path(row["manifest"])) == row["manifest_sha256"]
            for row in preflight["recovery_state"]["sealed_manifests"]
        ),
        "scene_pipeline_pids_resumed": resumed_pipelines,
    }
    write_json(RECOVERY_AUDIT, terminal)
    print(json.dumps(terminal, indent=2, sort_keys=True))
    return 0 if passed else 2


if __name__ == "__main__":
    raise SystemExit(main())
