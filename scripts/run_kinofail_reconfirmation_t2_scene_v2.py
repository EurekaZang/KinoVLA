#!/usr/bin/env python3
"""Launch one frozen T2 scene once, with a pre-execution audit record."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import signal
import subprocess
import time
from datetime import UTC, datetime
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parents[1]
ISAACLAB = Path("/home/eureka/IsaacLab-v2.3.0/isaaclab.sh")
EXPERIENCE = Path(
    "/home/eureka/IsaacLab-v2.3.0/apps/"
    "isaaclab.python.headless.rendering.kit"
)
CONDA_PREFIX = Path("/home/eureka/miniconda3/envs/kinovla")
DEFAULT_COLLECTOR = (
    ROOT / "scripts/isaac_collect_kinofail_confirmatory_t2_v1.py"
)
ISAACLAB_PYTHONPATH = (
    "/home/eureka/IsaacLab-v2.3.0/source/isaaclab:"
    "/home/eureka/IsaacLab-v2.3.0/source/isaaclab_tasks:"
    "/home/eureka/IsaacLab-v2.3.0/source/isaaclab_assets:"
    "/home/eureka/IsaacLab-v2.3.0/source/isaaclab_rl:"
    "/home/eureka/KinoVLA"
)
DEFAULT_TIMEOUT_S = 900.0
TERM_GRACE_S = 15.0


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


def _write_json(path: Path, value: dict[str, Any]) -> None:
    temporary = path.with_suffix(".tmp")
    temporary.write_text(
        json.dumps(value, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    os.replace(temporary, path)


def _process_group_exists(pgid: int) -> bool:
    try:
        os.killpg(pgid, 0)
    except ProcessLookupError:
        return False
    return True


def _terminate_process_group(
    process: subprocess.Popen[Any], pgid: int, *, grace_s: float = TERM_GRACE_S
) -> None:
    """Reclaim the whole isolated Isaac group, including orphaned renderers."""
    if _process_group_exists(pgid):
        os.killpg(pgid, signal.SIGTERM)
        deadline = time.monotonic() + grace_s
        while time.monotonic() < deadline and _process_group_exists(pgid):
            time.sleep(0.1)
    if _process_group_exists(pgid):
        os.killpg(pgid, signal.SIGKILL)
    try:
        process.wait(timeout=grace_s)
    except subprocess.TimeoutExpired:
        pass
    if _process_group_exists(pgid):
        raise RuntimeError(f"T2 Isaac process group {pgid} survived reclamation")


def _validate_operational_policy(
    path: Path,
    *,
    schedule: Path,
    registry: Path,
    asset_lock: Path,
    protocol: Path,
    collector: Path,
) -> dict[str, Any]:
    policy = _json(path)
    expected = {
        "schedule_sha256": _sha256(schedule),
        "scene_registry_sha256": _sha256(registry),
        "asset_lock_sha256": _sha256(asset_lock),
        "protocol_sha256": _sha256(protocol),
        "collector_sha256": _sha256(collector),
        "t2_runner_sha256": _sha256(Path(__file__).resolve()),
    }
    if (
        policy.get("schema_version")
        != "kinofail.kino-v4-all191-t2-operational-timeout-policy.v1"
        or policy.get("status")
        != "sealed_before_process_termination_and_campaign_resumption"
        or policy.get("model_predictions_read") is not False
        or policy.get("method_scores_read") is not False
        or policy.get("result_dependent_retry_permitted") is not False
        or policy.get("scientific_content_changed") is not False
        or policy.get("retry_trigger") != "collector_wall_timeout_only"
        or int(policy.get("maximum_attempts_per_scene", -1)) != 2
    ):
        raise RuntimeError("invalid frozen T2 operational timeout policy")
    for key, value in expected.items():
        if policy.get("artifacts", {}).get(key) != value:
            raise RuntimeError(f"T2 operational policy {key} mismatch")
    return policy


def _normalise_resumed_summary(
    summary_path: Path,
    scene_rows: list[dict[str, Any]],
    out: Path,
) -> None:
    """Restore manifest pointers omitted by the frozen collector's resume path."""
    if not summary_path.is_file():
        return
    summary = _json(summary_path)
    results = summary.get("results")
    if not isinstance(results, list):
        return
    expected = {str(row["case_id"]) for row in scene_rows}
    if {str(row.get("case_id")) for row in results} != expected:
        return
    changed = False
    for result in results:
        case_id = str(result["case_id"])
        manifest_path = out / case_id / "manifest.json"
        if not manifest_path.is_file():
            return
        manifest = _json(manifest_path)
        if manifest.get("passed") is not True:
            return
        if not result.get("manifest"):
            result["manifest"] = str(manifest_path)
            changed = True
    if changed:
        summary["resume_metadata_normalised"] = True
        _write_json(summary_path, summary)


def _launch_once(
    command: list[str],
    *,
    environment: dict[str, str],
    log_path: Path,
    timeout_s: float,
) -> dict[str, Any]:
    started = datetime.now(UTC).isoformat()
    timed_out = False
    with log_path.open("x", encoding="utf-8") as log:
        process = subprocess.Popen(
            command,
            cwd=ROOT,
            env=environment,
            stdout=log,
            stderr=subprocess.STDOUT,
            start_new_session=True,
        )
        pgid = process.pid
        try:
            returncode = int(process.wait(timeout=timeout_s))
        except subprocess.TimeoutExpired:
            timed_out = True
            _terminate_process_group(process, pgid)
            returncode = 124
        else:
            # A shell leader can exit before Kit descendants. Always reclaim
            # the original isolated group before the next Isaac launch.
            _terminate_process_group(process, pgid)
    return {
        "started_utc": started,
        "completed_utc": datetime.now(UTC).isoformat(),
        "returncode": returncode,
        "timed_out": timed_out,
        "timeout_s": timeout_s,
        "process_group_id": pgid,
        "log": str(log_path),
        "log_sha256": _sha256(log_path),
    }


def _operationally_complete(
    summary_path: Path, scene_rows: list[dict[str, Any]]
) -> bool:
    """Separate terminal artifact completion from scientific case validity."""
    if not summary_path.is_file():
        return False
    summary = _json(summary_path)
    results = summary.get("results")
    expected = {str(row["case_id"]) for row in scene_rows}
    if (
        not isinstance(results, list)
        or int(summary.get("case_count", -1)) != len(scene_rows)
        or len(results) != len(scene_rows)
        or {str(row.get("case_id")) for row in results} != expected
    ):
        return False
    return all(
        Path(str(row.get("manifest", ""))).is_file()
        for row in results
    )


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--schedule", type=Path, required=True)
    parser.add_argument("--scene-registry", type=Path, required=True)
    parser.add_argument("--asset-lock", type=Path, required=True)
    parser.add_argument("--protocol", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--scene", required=True)
    parser.add_argument("--collector", type=Path, default=DEFAULT_COLLECTOR)
    parser.add_argument("--operational-policy", type=Path, required=True)
    args = parser.parse_args()

    schedule = args.schedule.resolve()
    registry = args.scene_registry.resolve()
    asset_lock = args.asset_lock.resolve()
    protocol_path = args.protocol.resolve()
    collector = args.collector.resolve()
    operational_policy_path = args.operational_policy.resolve()
    out = args.out.resolve()
    for path in (
        schedule,
        registry,
        asset_lock,
        protocol_path,
        collector,
        operational_policy_path,
        ISAACLAB,
        EXPERIENCE,
        CONDA_PREFIX / "bin/python",
    ):
        if not path.is_file():
            raise FileNotFoundError(path)

    operational_policy = _validate_operational_policy(
        operational_policy_path,
        schedule=schedule,
        registry=registry,
        asset_lock=asset_lock,
        protocol=protocol_path,
        collector=collector,
    )
    timeout_s = float(operational_policy["per_attempt_wall_timeout_s"])
    if timeout_s != DEFAULT_TIMEOUT_S:
        raise RuntimeError("unexpected frozen T2 wall timeout")

    protocol = _json(protocol_path)
    if protocol.get("status") != "frozen":
        raise RuntimeError("T2 protocol is not frozen")
    for key, actual in (
        ("schedule_sha256", _sha256(schedule)),
        ("scene_registry_sha256", _sha256(registry)),
        ("asset_lock_sha256", _sha256(asset_lock)),
        ("collector_sha256", _sha256(collector)),
    ):
        if protocol.get(key) != actual:
            raise RuntimeError(f"frozen T2 protocol {key} mismatch")
    scene_rows = [
        row
        for row in _jsonl(schedule)
        if str(row["scene_cluster"]) == args.scene
    ]
    if not scene_rows:
        raise RuntimeError(f"no frozen T2 cases for {args.scene}")

    out.mkdir(parents=True, exist_ok=True)
    audit_dir = out / "launcher_audits"
    log_dir = out / "launcher_logs"
    audit_dir.mkdir(parents=True, exist_ok=True)
    log_dir.mkdir(parents=True, exist_ok=True)
    audit_path = audit_dir / f"{args.scene}.json"
    summary_path = out / "scene_summaries" / f"{args.scene}.json"
    if audit_path.is_file():
        audit = _json(audit_path)
        if (
            audit.get("schedule_sha256") != _sha256(schedule)
            or audit.get("scene_id") != args.scene
        ):
            raise RuntimeError("existing T2 launch audit belongs to another design")
        if _operationally_complete(summary_path, scene_rows):
            return 0
        incident_scene = str(
            operational_policy.get("trigger_incident", {}).get("scene_id", "")
        )
        if (
            args.scene != incident_scene
            or audit.get("state") not in {"started", "terminal"}
        ):
            return 2

    log_path = log_dir / f"{args.scene}.log"
    prior_audit_sha256 = _sha256(audit_path) if audit_path.is_file() else None
    audit = {
        "schema_version": "kinofail.reconfirmation-t2-launch.v2",
        "state": "started",
        "started_utc": datetime.now(UTC).isoformat(),
        "retry_authorized": prior_audit_sha256 is not None,
        "retry_trigger": (
            "sealed_preexisting_wall_timeout_incident"
            if prior_audit_sha256 is not None
            else None
        ),
        "prior_audit_sha256": prior_audit_sha256,
        "scientific_content_changed": False,
        "scene_id": args.scene,
        "case_count": len(scene_rows),
        "schedule": str(schedule),
        "schedule_sha256": _sha256(schedule),
        "protocol": str(protocol_path),
        "protocol_sha256": _sha256(protocol_path),
        "scene_registry_sha256": _sha256(registry),
        "asset_lock_sha256": _sha256(asset_lock),
        "collector_sha256": _sha256(collector),
        "operational_policy": str(operational_policy_path),
        "operational_policy_sha256": _sha256(operational_policy_path),
        "log": str(log_path),
    }
    _write_json(audit_path, audit)

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
        str(collector),
        "--schedule",
        str(schedule),
        "--scene-registry",
        str(registry),
        "--asset-lock",
        str(asset_lock),
        "--protocol",
        str(protocol_path),
        "--out",
        str(out),
        "--scene",
        args.scene,
        "--headless",
        "--enable_cameras",
        "--experience",
        str(EXPERIENCE),
    ]
    attempts: list[dict[str, Any]] = []
    first_attempt_index = 1 if prior_audit_sha256 is not None else 0
    for attempt_index in range(first_attempt_index, 2):
        attempt_log_path = log_dir / f"{args.scene}.attempt_{attempt_index}.log"
        attempt_command = list(command)
        if attempt_index > 0:
            attempt_command.append("--resume")
        attempt = _launch_once(
            attempt_command,
            environment=environment,
            log_path=attempt_log_path,
            timeout_s=timeout_s,
        )
        attempt["attempt_index"] = attempt_index
        attempt["resume"] = attempt_index > 0
        attempts.append(attempt)
        _normalise_resumed_summary(summary_path, scene_rows, out)
        if _operationally_complete(summary_path, scene_rows):
            break
        # Only a wall timeout, never a return code or validity outcome,
        # authorizes the single pre-frozen operational resume attempt.
        if not attempt["timed_out"]:
            break
    summary = _json(summary_path) if summary_path.is_file() else {}
    operationally_complete = (
        bool(attempts)
        and int(attempts[-1]["returncode"]) in {0, 2}
        and _operationally_complete(summary_path, scene_rows)
    )
    terminal = {
        **audit,
        "state": "terminal",
        "completed_utc": datetime.now(UTC).isoformat(),
        "returncode": int(attempts[-1]["returncode"]) if attempts else 2,
        "attempts": attempts,
        "wall_timeout_observed": any(row["timed_out"] for row in attempts),
        "summary_exists": summary_path.is_file(),
        "passed": summary.get("passed") is True,
        "operationally_complete": operationally_complete,
        "scientific_validity_reclassified": False,
        "global_model_blind_validity_seal_still_required": True,
    }
    _write_json(audit_path, terminal)
    print(json.dumps(terminal, indent=2, sort_keys=True))
    return 0 if operationally_complete else 2


if __name__ == "__main__":
    raise SystemExit(main())
