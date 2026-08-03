#!/usr/bin/env python3
"""F21 unattended T2 runner: fresh process, staging, and atomic commit per case."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import shutil
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

from scripts.kinofail_reconfirmation_slot_pool_v1 import (
    AUTHKEY_ENV,
    CAPACITY_ENV,
    HOST_ENV,
    PORT_ENV,
    connect_slot_pool,
)


F21 = (
    ROOT
    / "outputs/freeze/"
    "unified_moe_v3_reconfirmation_f21_atomic_t2_amendment1/"
    "amendment_manifest.json"
)
FROZEN_COLLECTOR = (
    ROOT / "scripts/isaac_collect_kinofail_confirmatory_t2_v1.py"
)
CASE_COLLECTOR = (
    ROOT / "scripts/isaac_collect_kinofail_confirmatory_t2_case_f21.py"
)
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
CASE_ENV = "KINOVLA_F21_T2_CASE_ID"
POLL_SECONDS = 15
CASE_HARD_TIMEOUT_SECONDS = 300
POST_MANIFEST_EXIT_GRACE_SECONDS = 60
MAX_OPERATIONAL_ATTEMPTS_PER_CASE = 3
EXPECTED_CASES_PER_SCENE = 50


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
    values = [
        json.loads(line)
        for line in path.read_text(encoding="utf-8").splitlines()
        if line
    ]
    if not all(isinstance(value, dict) for value in values):
        raise TypeError(path)
    return values


def write_json(path: Path, value: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(
        json.dumps(value, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    os.replace(temporary, path)


def _validate_f21() -> dict[str, Any]:
    amendment = read_json(F21)
    correction = amendment.get("correction", {})
    scientific = amendment.get("scientific_contract", {})
    paths = {
        "case_collector_f21": CASE_COLLECTOR,
        "atomic_t2_runner_f21": Path(__file__).resolve(),
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
        "sealer_v1": ROOT / "scripts/seal_kinofail_reconfirmation_f21.py",
        "frozen_t2_collector": FROZEN_COLLECTOR,
    }
    if (
        amendment.get("schema_version")
        != "kinofail.reconfirmation-f21-atomic-t2-amendment.v1"
        or amendment.get("status")
        != "sealed_during_scene05_t2_liveness_failure_before_termination"
        or amendment.get("passed") is not True
        or any(
            not path.is_file()
            or correction.get(f"{name}_sha256") != sha256(path)
            for name, path in paths.items()
        )
        or correction.get("fresh_isaac_process_per_t2_case") is not True
        or correction.get("atomic_case_commit") is not True
        or correction.get("case_hard_timeout_seconds")
        != CASE_HARD_TIMEOUT_SECONDS
        or correction.get("maximum_operational_attempts_per_case")
        != MAX_OPERATIONAL_ATTEMPTS_PER_CASE
        or correction.get("shared_three_slot_broker_preserved") is not True
        or scientific.get("schedule_protocol_or_case_membership_changed")
        is not False
        or scientific.get("physics_sensor_render_or_seed_changed") is not False
        or scientific.get("existing_completed_case_reexecuted") is not False
        or scientific.get("feature_model_route_threshold_or_analysis_changed")
        is not False
        or scientific.get("result_dependent_retry_enabled") is not False
    ):
        raise RuntimeError("F21 atomic-T2 amendment is invalid")
    return amendment


def _prediction_files() -> list[str]:
    root = ROOT / "outputs/eval/unified_moe_v3_reconfirmation_v2"
    if not root.exists():
        return []
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


def scene_rows(schedule: Path, scene: str) -> list[dict[str, Any]]:
    rows = [
        row
        for row in read_jsonl(schedule)
        if str(row["scene_cluster"]) == scene
    ]
    if len(rows) != EXPECTED_CASES_PER_SCENE:
        raise RuntimeError(
            f"F21 expected {EXPECTED_CASES_PER_SCENE} T2 cases for {scene}"
        )
    case_ids = [str(row["case_id"]) for row in rows]
    if len(case_ids) != len(set(case_ids)):
        raise RuntimeError("F21 T2 schedule repeats a case_id")
    return rows


def validate_case_manifest(
    manifest_path: Path,
    *,
    case_id: str,
    scene: str,
) -> dict[str, Any]:
    manifest = read_json(manifest_path)
    if (
        str(manifest.get("case_id")) != case_id
        or str(manifest.get("scene_cluster")) != scene
    ):
        raise RuntimeError(f"mismatched T2 manifest: {manifest_path}")
    artifacts = manifest.get("artifacts", {})
    observables = artifacts.get("observables", {})
    observables_path = manifest_path.parent / str(observables.get("path", ""))
    if (
        not observables_path.is_file()
        or observables.get("sha256") != sha256(observables_path)
    ):
        raise RuntimeError(f"invalid T2 observables: {manifest_path}")
    image_hashes = artifacts.get("image_sha256", {})
    if (
        not isinstance(image_hashes, dict)
        or len(image_hashes) != 30
        or any(
            not (manifest_path.parent / relative).is_file()
            or sha256(manifest_path.parent / relative) != digest
            for relative, digest in image_hashes.items()
        )
    ):
        raise RuntimeError(f"invalid T2 image inventory: {manifest_path}")
    return manifest


def corpus_case_state(
    *,
    rows: list[dict[str, Any]],
    out: Path,
    scene: str,
) -> dict[str, Any]:
    sealed = []
    empty = []
    never = []
    first_unsealed: int | None = None
    for index, row in enumerate(rows):
        case_id = str(row["case_id"])
        case_dir = out / case_id
        manifest_path = case_dir / "manifest.json"
        if manifest_path.is_file():
            manifest = validate_case_manifest(
                manifest_path,
                case_id=case_id,
                scene=scene,
            )
            if manifest.get("passed") is not True:
                raise RuntimeError(
                    "F21 never retries an admitted failed T2 case: "
                    f"{manifest_path}"
                )
            if first_unsealed is not None:
                raise RuntimeError(
                    "F21 requires completed T2 manifests to be a prefix"
                )
            sealed.append(
                {
                    "case_id": case_id,
                    "manifest": str(manifest_path),
                    "sha256": sha256(manifest_path),
                }
            )
            continue
        if first_unsealed is None:
            first_unsealed = index
        if case_dir.exists():
            files = [path for path in case_dir.rglob("*") if path.is_file()]
            if files:
                raise RuntimeError(
                    "F21 refuses a non-atomic partial corpus case: "
                    f"{case_id}"
                )
            empty.append(case_id)
        else:
            never.append(case_id)
    if len(empty) > 1:
        raise RuntimeError("F21 found multiple empty corpus case directories")
    if empty and empty[0] != str(rows[len(sealed)]["case_id"]):
        raise RuntimeError("F21 empty case is not the next scheduled case")
    if first_unsealed not in {None, len(sealed)}:
        raise RuntimeError("F21 T2 prefix accounting is inconsistent")
    return {
        "scheduled_case_count": len(rows),
        "sealed_manifest_count": len(sealed),
        "sealed_manifests": sealed,
        "empty_zero_observation_case_ids": empty,
        "never_attempted_case_ids": never,
        "case_ids_partition_schedule": (
            len(sealed) + len(empty) + len(never) == len(rows)
        ),
    }


def _bounded_inventory(root: Path) -> list[dict[str, Any]]:
    if not root.exists():
        return []
    return [
        {
            "path": str(path.relative_to(root)),
            "bytes": path.stat().st_size,
            "sha256": sha256(path),
        }
        for path in sorted(root.rglob("*"))
        if path.is_file()
    ]


def _terminate_group(process: subprocess.Popen[Any]) -> dict[str, Any]:
    action: dict[str, Any] = {"pid": process.pid, "sigterm": False, "sigkill": False}
    if process.poll() is not None:
        action["returncode"] = int(process.returncode)
        return action
    try:
        os.killpg(process.pid, signal.SIGTERM)
        action["sigterm"] = True
    except ProcessLookupError:
        pass
    try:
        process.wait(timeout=15)
    except subprocess.TimeoutExpired:
        try:
            os.killpg(process.pid, signal.SIGKILL)
            action["sigkill"] = True
        except ProcessLookupError:
            pass
        process.wait(timeout=15)
    action["returncode"] = int(process.returncode)
    return action


def _safe_remove_staging(path: Path, *, staging_root: Path) -> None:
    resolved = path.resolve()
    root = staging_root.resolve()
    if resolved == root or root not in resolved.parents:
        raise RuntimeError(f"refusing unbounded staging cleanup: {path}")
    if path.exists():
        shutil.rmtree(path)


def _commit_case(
    *,
    staged_case: Path,
    final_case: Path,
    case_id: str,
    scene: str,
) -> dict[str, Any]:
    manifest_path = staged_case / "manifest.json"
    manifest = validate_case_manifest(
        manifest_path,
        case_id=case_id,
        scene=scene,
    )
    source_manifest_sha256 = sha256(manifest_path)
    if final_case.exists():
        files = [path for path in final_case.rglob("*") if path.is_file()]
        if files:
            raise RuntimeError(f"F21 refuses to replace corpus case: {final_case}")
        final_case.rmdir()
    os.replace(staged_case, final_case)
    committed_manifest = final_case / "manifest.json"
    if sha256(committed_manifest) != source_manifest_sha256:
        raise RuntimeError("F21 atomic case commit changed the manifest")
    return manifest


def _result_from_manifest(
    manifest: dict[str, Any],
    *,
    manifest_path: Path,
    resumed: bool,
) -> dict[str, Any]:
    samples = manifest.get("samples", [])
    shared = {
        str(row.get("shared_proprio_sha256"))
        for row in samples
        if row.get("shared_proprio_sha256")
    }
    row: dict[str, Any] = {
        "case_id": manifest["case_id"],
        "scene_cluster": manifest["scene_cluster"],
        "passed": manifest.get("passed") is True,
        "manifest": str(manifest_path),
    }
    if len(shared) == 1:
        row["shared_proprio_sha256"] = next(iter(shared))
    if manifest.get("checks", {}).get(
        "render_did_not_advance_or_perturb_physics"
    ) is True:
        row["render_state_max_abs_delta"] = 0.0
    if resumed:
        row["resumed"] = True
    return row


def _slot_connection(
    *, standalone_recovery: bool
) -> tuple[Any | None, Any | None]:
    present = all(
        os.environ.get(key)
        for key in (HOST_ENV, PORT_ENV, AUTHKEY_ENV, CAPACITY_ENV)
    )
    if present:
        return connect_slot_pool()
    if standalone_recovery:
        return None, None
    raise RuntimeError(
        "F21 production T2 runner requires the shared slot broker"
    )


def _collector_command(
    *,
    schedule: Path,
    registry: Path,
    asset_lock: Path,
    protocol: Path,
    stage_out: Path,
    scene: str,
) -> list[str]:
    return [
        str(ISAACLAB),
        "-p",
        str(CASE_COLLECTOR),
        "--schedule",
        str(schedule),
        "--scene-registry",
        str(registry),
        "--asset-lock",
        str(asset_lock),
        "--protocol",
        str(protocol),
        "--out",
        str(stage_out),
        "--scene",
        scene,
        "--headless",
        "--enable_cameras",
        "--experience",
        str(EXPERIENCE),
    ]


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--schedule", type=Path, required=True)
    parser.add_argument("--scene-registry", type=Path, required=True)
    parser.add_argument("--asset-lock", type=Path, required=True)
    parser.add_argument("--protocol", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--scene", required=True)
    parser.add_argument("--standalone-recovery", action="store_true")
    args = parser.parse_args()

    amendment = _validate_f21()
    schedule = args.schedule.resolve(strict=True)
    registry = args.scene_registry.resolve(strict=True)
    asset_lock = args.asset_lock.resolve(strict=True)
    protocol = args.protocol.resolve(strict=True)
    out = args.out.resolve(strict=False)
    out.mkdir(parents=True, exist_ok=True)
    out = out.resolve(strict=True)
    if _prediction_files():
        raise RuntimeError("prediction or score exists before F21 T2 collection")
    frozen_protocol = read_json(protocol)
    for key, actual in (
        ("schedule_sha256", sha256(schedule)),
        ("scene_registry_sha256", sha256(registry)),
        ("asset_lock_sha256", sha256(asset_lock)),
        ("collector_sha256", sha256(FROZEN_COLLECTOR)),
    ):
        if frozen_protocol.get(key) != actual:
            raise RuntimeError(f"frozen T2 protocol {key} mismatch")
    rows = scene_rows(schedule, args.scene)
    initial = corpus_case_state(rows=rows, out=out, scene=args.scene)
    initial_hashes = {
        row["case_id"]: row["sha256"] for row in initial["sealed_manifests"]
    }

    audit_dir = out / "launcher_audits"
    log_dir = out / "launcher_logs/f21_atomic"
    audit_path = audit_dir / f"{args.scene}_f21_atomic.json"
    if audit_path.is_file():
        previous = read_json(audit_path)
        summary_path = out / "scene_summaries" / f"{args.scene}.json"
        if (
            previous.get("state") == "terminal"
            and previous.get("passed") is True
            and summary_path.is_file()
            and read_json(summary_path).get("passed") is True
        ):
            print(json.dumps(previous, indent=2, sort_keys=True))
            return 0
        if previous.get("state") not in {"started", "recovering"}:
            raise RuntimeError("existing F21 T2 audit is not resumable")

    staging_root = (
        out.parent.parent / ".f21_t2_staging" / args.scene
    ).resolve()
    staging_root.mkdir(parents=True, exist_ok=True)
    audit: dict[str, Any] = {
        "schema_version": "kinofail.reconfirmation-t2-atomic-f21.v1",
        "state": "started",
        "started_utc": datetime.now(UTC).isoformat(),
        "scene_id": args.scene,
        "operational_amendment": str(F21),
        "operational_amendment_sha256": sha256(F21),
        "amendment_status": amendment["status"],
        "schedule": str(schedule),
        "schedule_sha256": sha256(schedule),
        "protocol": str(protocol),
        "protocol_sha256": sha256(protocol),
        "frozen_collector": str(FROZEN_COLLECTOR),
        "frozen_collector_sha256": sha256(FROZEN_COLLECTOR),
        "case_collector": str(CASE_COLLECTOR),
        "case_collector_sha256": sha256(CASE_COLLECTOR),
        "fresh_isaac_process_per_case": True,
        "atomic_case_commit": True,
        "case_hard_timeout_seconds": CASE_HARD_TIMEOUT_SECONDS,
        "post_manifest_exit_grace_seconds": (
            POST_MANIFEST_EXIT_GRACE_SECONDS
        ),
        "maximum_operational_attempts_per_case": (
            MAX_OPERATIONAL_ATTEMPTS_PER_CASE
        ),
        "initial_case_state": initial,
        "attempts": [],
        "model_feature_prediction_score_or_label_read": False,
        "result_dependent_retry_or_selection": False,
        "scientific_content_changed": False,
    }
    write_json(audit_path, audit)
    manager, slot_pool = _slot_connection(
        standalone_recovery=args.standalone_recovery
    )
    _ = manager
    results: list[dict[str, Any]] = []
    try:
        for case_index, row in enumerate(rows):
            case_id = str(row["case_id"])
            final_case = out / case_id
            manifest_path = final_case / "manifest.json"
            if manifest_path.is_file():
                manifest = validate_case_manifest(
                    manifest_path,
                    case_id=case_id,
                    scene=args.scene,
                )
                if manifest.get("passed") is not True:
                    raise RuntimeError(
                        "F21 refuses to retry an admitted failed case"
                    )
                results.append(
                    _result_from_manifest(
                        manifest,
                        manifest_path=manifest_path,
                        resumed=True,
                    )
                )
                continue

            committed = False
            for attempt_index in range(
                1, MAX_OPERATIONAL_ATTEMPTS_PER_CASE + 1
            ):
                attempt_root = (
                    staging_root
                    / case_id
                    / f"attempt_{attempt_index:02d}"
                )
                if attempt_root.exists():
                    _safe_remove_staging(
                        attempt_root, staging_root=staging_root
                    )
                stage_out = attempt_root / "out"
                stage_out.mkdir(parents=True)
                log_path = (
                    log_dir / f"{case_index:02d}_{case_id}_a{attempt_index}.log"
                )
                log_path.parent.mkdir(parents=True, exist_ok=True)
                command = _collector_command(
                    schedule=schedule,
                    registry=registry,
                    asset_lock=asset_lock,
                    protocol=protocol,
                    stage_out=stage_out,
                    scene=args.scene,
                )
                ticket = int(slot_pool.acquire()) if slot_pool is not None else None
                started_mono = time.monotonic()
                attempt: dict[str, Any] = {
                    "case_id": case_id,
                    "case_index": case_index,
                    "attempt_index": attempt_index,
                    "state": "started",
                    "started_utc": datetime.now(UTC).isoformat(),
                    "command": command,
                    "log": str(log_path),
                    "slot_ticket": ticket,
                    "staging_root": str(attempt_root),
                }
                audit["state"] = "recovering"
                audit["active_case_id"] = case_id
                audit["active_case_index"] = case_index
                audit["active_attempt_index"] = attempt_index
                audit["heartbeat_utc"] = datetime.now(UTC).isoformat()
                audit["attempts"].append(attempt)
                write_json(audit_path, audit)
                environment = os.environ.copy()
                environment.update(
                    {
                        "CONDA_PREFIX": str(CONDA_PREFIX),
                        "PATH": (
                            f"{CONDA_PREFIX / 'bin'}:"
                            f"{environment.get('PATH', '')}"
                        ),
                        "OMNI_KIT_ACCEPT_EULA": "YES",
                        "PYTHONPATH": ISAACLAB_PYTHONPATH,
                        CASE_ENV: case_id,
                    }
                )
                try:
                    with log_path.open("x", encoding="utf-8") as log:
                        process = subprocess.Popen(
                            command,
                            cwd=ROOT,
                            env=environment,
                            stdout=log,
                            stderr=subprocess.STDOUT,
                            start_new_session=True,
                        )
                        stage_manifest = (
                            stage_out / case_id / "manifest.json"
                        )
                        manifest_seen_mono: float | None = None
                        termination_reason: str | None = None
                        while process.poll() is None:
                            now = time.monotonic()
                            if (
                                stage_manifest.is_file()
                                and manifest_seen_mono is None
                            ):
                                manifest_seen_mono = now
                            if (
                                now - started_mono
                                >= CASE_HARD_TIMEOUT_SECONDS
                            ):
                                termination_reason = "case_hard_timeout"
                                break
                            if (
                                manifest_seen_mono is not None
                                and now - manifest_seen_mono
                                >= POST_MANIFEST_EXIT_GRACE_SECONDS
                            ):
                                termination_reason = (
                                    "post_manifest_close_timeout"
                                )
                                break
                            audit["heartbeat_utc"] = (
                                datetime.now(UTC).isoformat()
                            )
                            audit["active_process_pid"] = process.pid
                            audit["active_elapsed_seconds"] = (
                                now - started_mono
                            )
                            audit["active_manifest_exists"] = (
                                stage_manifest.is_file()
                            )
                            write_json(audit_path, audit)
                            time.sleep(POLL_SECONDS)
                        termination = (
                            _terminate_group(process)
                            if process.poll() is None
                            else {
                                "pid": process.pid,
                                "returncode": int(process.returncode),
                                "sigterm": False,
                                "sigkill": False,
                            }
                        )
                finally:
                    if slot_pool is not None and ticket is not None:
                        slot_pool.release(ticket)
                attempt["completed_utc"] = datetime.now(UTC).isoformat()
                attempt["elapsed_seconds"] = time.monotonic() - started_mono
                attempt["termination_reason"] = termination_reason
                attempt["termination"] = termination
                staged_case = stage_out / case_id
                if stage_manifest.is_file():
                    manifest = validate_case_manifest(
                        stage_manifest,
                        case_id=case_id,
                        scene=args.scene,
                    )
                    attempt["manifest_passed"] = (
                        manifest.get("passed") is True
                    )
                    attempt["manifest_sha256"] = sha256(stage_manifest)
                    _commit_case(
                        staged_case=staged_case,
                        final_case=final_case,
                        case_id=case_id,
                        scene=args.scene,
                    )
                    committed_manifest = final_case / "manifest.json"
                    results.append(
                        _result_from_manifest(
                            manifest,
                            manifest_path=committed_manifest,
                            resumed=False,
                        )
                    )
                    attempt["state"] = "committed"
                    committed = manifest.get("passed") is True
                    _safe_remove_staging(
                        attempt_root, staging_root=staging_root
                    )
                    write_json(audit_path, audit)
                    if not committed:
                        raise RuntimeError(
                            "F21 retained a terminal failed T2 case without retry"
                        )
                    break
                attempt["state"] = "operational_failure_without_manifest"
                attempt["staging_inventory"] = _bounded_inventory(attempt_root)
                _safe_remove_staging(
                    attempt_root, staging_root=staging_root
                )
                write_json(audit_path, audit)
            if not committed:
                raise RuntimeError(
                    f"F21 exhausted operational attempts for {case_id}"
                )

        summary = {
            "schema_version": "kinofail.realistic-c1-causal-scene.v1",
            "scene_cluster": args.scene,
            "passed": (
                len(results) == len(rows)
                and all(result.get("passed") is True for result in results)
            ),
            "case_count": len(results),
            "results": results,
        }
        summary_path = out / "scene_summaries" / f"{args.scene}.json"
        write_json(summary_path, summary)
        final_state = corpus_case_state(
            rows=rows, out=out, scene=args.scene
        )
        preserved = all(
            (out / case_id / "manifest.json").is_file()
            and sha256(out / case_id / "manifest.json") == digest
            for case_id, digest in initial_hashes.items()
        )
        passed = (
            summary["passed"] is True
            and final_state["sealed_manifest_count"] == len(rows)
            and preserved
        )
        audit.update(
            {
                "state": "terminal",
                "completed_utc": datetime.now(UTC).isoformat(),
                "passed": passed,
                "summary": str(summary_path),
                "summary_sha256": sha256(summary_path),
                "final_case_state": final_state,
                "initial_manifest_hashes_preserved": preserved,
                "active_case_id": None,
                "active_process_pid": None,
                "heartbeat_utc": datetime.now(UTC).isoformat(),
            }
        )
        write_json(audit_path, audit)
        print(json.dumps(audit, indent=2, sort_keys=True))
        return 0 if passed else 2
    except BaseException as error:
        audit.update(
            {
                "state": "terminal_failure",
                "failed_utc": datetime.now(UTC).isoformat(),
                "passed": False,
                "error": f"{type(error).__name__}: {error}",
                "heartbeat_utc": datetime.now(UTC).isoformat(),
            }
        )
        write_json(audit_path, audit)
        raise


if __name__ == "__main__":
    raise SystemExit(main())
