#!/usr/bin/env python3
"""Seal F21 before terminating the scene05 T2 liveness failure."""

from __future__ import annotations

import hashlib
import json
import os
import sys
import time
from datetime import UTC, datetime
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
SCENE = "confirm_v2_production_scene_05"
OUT = (
    Path("/data/eureka/KinoVLA")
    / "outputs/kinofail_reconfirmation_v2/corpus_ext4"
    / SCENE
    / "c2_t2"
)
SCHEDULE = (
    ROOT
    / "outputs/kinofail_reconfirmation_v2/schedules/c2_t2/schedule.jsonl"
)
SOURCE_AUDIT = OUT / "launcher_audits" / f"{SCENE}.json"
SOURCE_LOG = OUT / "launcher_logs" / f"{SCENE}.log"
PIPELINE_STATE = (
    ROOT
    / "outputs/kinofail_reconfirmation_v2/orchestration"
    / SCENE
    / "pipeline_state.json"
)
F20 = (
    ROOT
    / "outputs/freeze/"
    "unified_moe_v3_reconfirmation_f20_finalizer_reference_amendment1/"
    "amendment_manifest.json"
)
F21 = (
    ROOT
    / "outputs/freeze/"
    "unified_moe_v3_reconfirmation_f21_atomic_t2_amendment1/"
    "amendment_manifest.json"
)
LIVENESS_DEADLINE_SECONDS = 600


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
    return [
        json.loads(line)
        for line in path.read_text(encoding="utf-8").splitlines()
        if line
    ]


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


def matching_processes(*required: str) -> list[dict[str, Any]]:
    found = []
    for entry in Path("/proc").iterdir():
        if not entry.name.isdigit():
            continue
        tokens = _cmdline(int(entry.name))
        joined = " ".join(tokens)
        if not tokens or not all(value in joined for value in required):
            continue
        try:
            state = (entry / "stat").read_text().split()[2]
        except (FileNotFoundError, ProcessLookupError):
            continue
        found.append(
            {
                "pid": int(entry.name),
                "state": state,
                "command": tokens,
            }
        )
    return sorted(found, key=lambda row: int(row["pid"]))


def scheduled_rows() -> list[dict[str, Any]]:
    rows = [
        row
        for row in read_jsonl(SCHEDULE)
        if str(row["scene_cluster"]) == SCENE
    ]
    if len(rows) != 50:
        raise RuntimeError("unexpected scene05 T2 schedule size")
    return rows


def incident_case_state() -> dict[str, Any]:
    rows = scheduled_rows()
    sealed = []
    empty = []
    never = []
    first_unsealed: int | None = None
    for index, row in enumerate(rows):
        case_id = str(row["case_id"])
        case_dir = OUT / case_id
        manifest_path = case_dir / "manifest.json"
        if manifest_path.is_file():
            manifest = read_json(manifest_path)
            if (
                manifest.get("case_id") != case_id
                or manifest.get("scene_cluster") != SCENE
                or manifest.get("passed") is not True
                or first_unsealed is not None
            ):
                raise RuntimeError("scene05 sealed T2 prefix is invalid")
            sealed.append(
                {
                    "case_id": case_id,
                    "manifest": str(manifest_path),
                    "manifest_sha256": sha256(manifest_path),
                }
            )
            continue
        if first_unsealed is None:
            first_unsealed = index
        if case_dir.exists():
            files = [path for path in case_dir.rglob("*") if path.is_file()]
            if files:
                raise RuntimeError("scene05 next T2 case is not zero-observation")
            empty.append(case_id)
        else:
            never.append(case_id)
    if (
        len(sealed) != 47
        or empty != [str(rows[47]["case_id"])]
        or never != [str(rows[48]["case_id"]), str(rows[49]["case_id"])]
        or first_unsealed != 47
    ):
        raise RuntimeError("unexpected scene05 F21 incident partition")
    return {
        "scheduled_case_count": 50,
        "sealed_manifest_count": len(sealed),
        "sealed_manifests": sealed,
        "empty_zero_observation_case_ids": empty,
        "never_attempted_case_ids": never,
        "case_ids_partition_schedule": True,
    }


def authenticated_inventory(
    state: dict[str, Any],
) -> tuple[list[dict[str, Any]], str]:
    rows = []
    for sealed in state["sealed_manifests"]:
        case_dir = OUT / sealed["case_id"]
        for path in sorted(case_dir.rglob("*")):
            if path.is_file():
                rows.append(
                    {
                        "path": str(path.relative_to(OUT)),
                        "bytes": path.stat().st_size,
                        "sha256": sha256(path),
                    }
                )
    digest = hashlib.sha256()
    for row in rows:
        digest.update(
            (
                f"{row['path']}\\0{row['bytes']}\\0{row['sha256']}\\n"
            ).encode("utf-8")
        )
    return rows, digest.hexdigest()


def _prediction_files() -> list[str]:
    root = ROOT / "outputs/eval/unified_moe_v3_reconfirmation_v2"
    return sorted(
        str(path)
        for path in root.rglob("*")
        if path.is_file()
        and (
            "prediction" in path.name.lower()
            or "score" in path.name.lower()
            or path.name == "confirmatory_report.json"
        )
    )


def main() -> int:
    from scripts import run_kinofail_reconfirmation_t2_atomic_f21 as runner

    paths = {
        "case_collector_f21": (
            ROOT
            / "scripts/isaac_collect_kinofail_confirmatory_t2_case_f21.py"
        ),
        "atomic_t2_runner_f21": Path(runner.__file__).resolve(),
        "scene_pipeline_v5": (
            ROOT / "scripts/run_kinofail_reconfirmation_scene_pipeline_v5.py"
        ),
        "all_scenes_v5": (
            ROOT
            / "scripts/run_kinofail_reconfirmation_all_scene_pipelines_v5.py"
        ),
        "finalizer_v7": (
            ROOT / "scripts/finalize_kinofail_reconfirmation_v7.py"
        ),
        "scene05_recovery_f21": (
            ROOT
            / "scripts/recover_kinofail_reconfirmation_scene05_f21.py"
        ),
        "sealer_v1": Path(__file__).resolve(),
        "frozen_t2_collector": (
            ROOT / "scripts/isaac_collect_kinofail_confirmatory_t2_v1.py"
        ),
    }
    for path in (
        *paths.values(),
        F20,
        SOURCE_AUDIT,
        SOURCE_LOG,
        PIPELINE_STATE,
        SCHEDULE,
    ):
        if not path.is_file():
            raise FileNotFoundError(path)
    if F21.exists():
        raise FileExistsError(F21)
    if _prediction_files():
        raise RuntimeError("prediction or score exists before F21 seal")
    source = read_json(SOURCE_AUDIT)
    if source.get("state") != "started" or source.get("retry_authorized") is not False:
        raise RuntimeError("scene05 source T2 audit is not an interrupted start")
    inactivity = time.time() - SOURCE_LOG.stat().st_mtime
    if inactivity < LIVENESS_DEADLINE_SECONDS:
        raise RuntimeError("F21 liveness deadline has not elapsed")
    processes = matching_processes(
        "isaac_collect_kinofail_confirmatory_t2_v1.py",
        "--scene",
        SCENE,
    )
    if not processes:
        raise RuntimeError("scene05 stuck T2 process is absent")
    state = incident_case_state()
    inventory, inventory_sha = authenticated_inventory(state)
    manifest = {
        "schema_version": (
            "kinofail.reconfirmation-f21-atomic-t2-amendment.v1"
        ),
        "created_utc": datetime.now(UTC).isoformat(),
        "status": (
            "sealed_during_scene05_t2_liveness_failure_before_termination"
        ),
        "passed": True,
        "predecessor_f20": str(F20.relative_to(ROOT)),
        "predecessor_f20_sha256": sha256(F20),
        "scope": (
            "replace the long-lived 50-case T2 Isaac process with one fresh "
            "process per frozen case; write each attempt to same-filesystem "
            "staging; atomically admit only a validated manifest; hard-timeout "
            "non-terminal operational attempts without replaying admitted cases"
        ),
        "incident": {
            "classification": (
                "recurrent_long_lived_isaac_t2_liveness_failure"
            ),
            "scene_id": SCENE,
            "source_launcher_audit": str(SOURCE_AUDIT),
            "source_launcher_audit_sha256": sha256(SOURCE_AUDIT),
            "source_runtime_log": str(SOURCE_LOG),
            "source_runtime_log_sha256": sha256(SOURCE_LOG),
            "source_runtime_log_bytes": SOURCE_LOG.stat().st_size,
            "source_log_inactivity_seconds": inactivity,
            "source_pipeline_state": str(PIPELINE_STATE),
            "source_pipeline_state_sha256": sha256(PIPELINE_STATE),
            "observed_t2_processes": processes,
            "identified_from": (
                "process liveness, frozen schedule order, log inactivity, "
                "and artifact existence only"
            ),
            "model_feature_prediction_score_or_label_used": False,
            **state,
        },
        "evidence": {
            "authenticated_case_file_count": len(inventory),
            "authenticated_case_bytes": sum(
                int(row["bytes"]) for row in inventory
            ),
            "authenticated_relative_size_content_inventory_sha256": (
                inventory_sha
            ),
            "reconfirmation_prediction_files_existed_at_seal": False,
            "reconfirmation_score_files_existed_at_seal": False,
        },
        "correction": {
            **{
                f"{name}_sha256": sha256(path)
                for name, path in paths.items()
            },
            "fresh_isaac_process_per_t2_case": True,
            "atomic_case_commit": True,
            "same_filesystem_staging": True,
            "case_hard_timeout_seconds": (
                runner.CASE_HARD_TIMEOUT_SECONDS
            ),
            "post_manifest_exit_grace_seconds": (
                runner.POST_MANIFEST_EXIT_GRACE_SECONDS
            ),
            "maximum_operational_attempts_per_case": (
                runner.MAX_OPERATIONAL_ATTEMPTS_PER_CASE
            ),
            "shared_three_slot_broker_preserved": True,
            "frozen_collector_loaded_without_source_modification": True,
        },
        "retry_disposition": {
            "existing_completed_case_reexecution_permitted": False,
            "failed_admitted_case_reexecution_permitted": False,
            "non_admitted_staging_attempt_may_restart": True,
            "zero_observation_legacy_case_may_enter_once": (
                state["empty_zero_observation_case_ids"]
            ),
            "never_attempted_cases_are_not_retries": True,
            "result_dependent_retry_permitted": False,
        },
        "scientific_contract": {
            "schedule_protocol_or_case_membership_changed": False,
            "physics_sensor_render_or_seed_changed": False,
            "existing_completed_case_reexecuted": False,
            "feature_model_route_threshold_or_analysis_changed": False,
            "result_dependent_retry_enabled": False,
            "operator_scene_material_or_camera_changed": False,
        },
    }
    F21.parent.mkdir(parents=True, exist_ok=True)
    temporary = F21.with_suffix(".json.tmp")
    temporary.write_text(
        json.dumps(manifest, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    os.replace(temporary, F21)
    print(json.dumps(manifest, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
