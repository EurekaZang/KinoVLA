#!/usr/bin/env python3
"""One-shot, model-blind recovery of the scene-01 T2 liveness incident.

F14 permits a new Isaac process only after a fixed liveness deadline and only
when every case with a manifest is preserved verbatim.  A case may be entered
again only when its case directory contains no files, i.e. when the interrupted
process produced no physical observation for that case.  The collector's
``--resume`` path then skips every sealed passing manifest and continues the
zero-observation case plus cases that were never attempted.
"""

from __future__ import annotations

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
SCENE = "confirm_v2_life_scene_01"
F14 = (
    ROOT
    / "outputs/freeze/"
    "unified_moe_v3_reconfirmation_f14_t2_zero_observation_liveness_amendment1/"
    "amendment_manifest.json"
)
SCHEDULE = (
    ROOT
    / "outputs/kinofail_reconfirmation_v2/schedules/c2_t2/schedule.jsonl"
)
REGISTRY = (
    ROOT / "outputs/kinofail_reconfirmation_v2/scene_registry.json"
)
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
COLLECTOR = (
    ROOT / "scripts/isaac_collect_kinofail_confirmatory_t2_v1.py"
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


def case_recovery_state(
    *,
    scene_rows: list[dict[str, Any]],
    out: Path,
) -> dict[str, Any]:
    case_ids = [str(row["case_id"]) for row in scene_rows]
    if len(case_ids) != len(set(case_ids)):
        raise RuntimeError("T2 schedule repeats a case_id")
    sealed = []
    zero_observation = []
    never_attempted = []
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
                    "F14 forbids replay of an existing failed or mismatched "
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
                    "F14 zero-observation case contains artifact files: "
                    f"{case_id}"
                )
            zero_observation.append(case_id)
        else:
            never_attempted.append(case_id)
    if len(zero_observation) != 1:
        raise RuntimeError(
            "F14 requires exactly one started zero-observation T2 case"
        )
    if first_unsealed_index != len(sealed):
        raise RuntimeError("F14 T2 prefix accounting is inconsistent")
    if zero_observation[0] != case_ids[len(sealed)]:
        raise RuntimeError(
            "F14 zero-observation case is not the next scheduled case"
        )
    if (
        len(sealed) + len(zero_observation) + len(never_attempted)
        != len(case_ids)
    ):
        raise RuntimeError("F14 T2 cases do not partition the schedule")
    return {
        "scheduled_case_count": len(case_ids),
        "sealed_manifest_count": len(sealed),
        "sealed_manifests": sealed,
        "zero_observation_case_ids": zero_observation,
        "never_attempted_case_ids": never_attempted,
        "existing_failed_manifest_count": 0,
        "case_ids_partition_schedule": True,
    }


def _process_age_seconds(pid: int) -> float:
    stat = (Path("/proc") / str(pid) / "stat").read_text().split()
    start_ticks = int(stat[21])
    clock_ticks = int(os.sysconf(os.sysconf_names["SC_CLK_TCK"]))
    uptime = float(Path("/proc/uptime").read_text().split()[0])
    return max(0.0, uptime - start_ticks / clock_ticks)


def _process_state(pid: int) -> str | None:
    try:
        return (Path("/proc") / str(pid) / "stat").read_text().split()[2]
    except (FileNotFoundError, PermissionError, ProcessLookupError):
        return None


def t2_processes() -> list[dict[str, Any]]:
    found = []
    for entry in Path("/proc").iterdir():
        if not entry.name.isdigit():
            continue
        try:
            tokens = [
                token.decode("utf-8", errors="replace")
                for token in (entry / "cmdline").read_bytes().split(b"\0")
                if token
            ]
        except (FileNotFoundError, PermissionError, ProcessLookupError):
            continue
        joined = " ".join(tokens)
        if (
            str(COLLECTOR) not in joined
            or "--scene" not in tokens
            or SCENE not in tokens
        ):
            continue
        pid = int(entry.name)
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


def terminate_exact_processes(
    processes: list[dict[str, Any]],
    *,
    grace_seconds: int = 15,
) -> dict[str, Any]:
    targets = [
        int(process["pid"])
        for process in processes
        if process.get("state") not in {None, "Z"}
    ]
    term_sent = []
    for pid in targets:
        try:
            os.kill(pid, signal.SIGTERM)
            term_sent.append(pid)
        except ProcessLookupError:
            pass
    deadline = time.monotonic() + grace_seconds
    while time.monotonic() < deadline:
        if not any(_process_state(pid) not in {None, "Z"} for pid in targets):
            break
        time.sleep(1)
    kill_sent = []
    for pid in targets:
        if _process_state(pid) not in {None, "Z"}:
            try:
                os.kill(pid, signal.SIGKILL)
                kill_sent.append(pid)
            except ProcessLookupError:
                pass
    return {
        "observed_processes": processes,
        "sigterm_sent": term_sent,
        "sigkill_sent": kill_sent,
    }


def _prediction_files() -> list[str]:
    root = (
        ROOT / "outputs/eval/unified_moe_v3_reconfirmation_v2"
    )
    values = []
    for path in root.rglob("*"):
        if path.is_file() and (
            "prediction" in path.name.lower()
            or "score" in path.name.lower()
            or path.name == "confirmatory_report.json"
        ):
            values.append(str(path))
    return sorted(values)


def main() -> int:
    if not F14.is_file():
        raise FileNotFoundError(F14)
    amendment = read_json(F14)
    recovery = amendment.get("recovery", {})
    if (
        amendment.get("passed") is not True
        or amendment.get("schema_version")
        != (
            "kinofail.reconfirmation-f14-t2-zero-observation-"
            "liveness-amendment.v1"
        )
        or amendment.get("status")
        != "sealed_before_termination_and_zero_observation_resume"
        or recovery.get("recovery_script_sha256")
        != sha256(Path(__file__).resolve())
        or recovery.get("collector_sha256") != sha256(COLLECTOR)
        or recovery.get("schedule_sha256") != sha256(SCHEDULE)
        or recovery.get("protocol_sha256") != sha256(PROTOCOL)
    ):
        raise RuntimeError("invalid F14 amendment")
    if _prediction_files():
        raise RuntimeError("prediction or score exists before F14 recovery")
    launcher_audit = (
        OUT / "launcher_audits" / f"{SCENE}.json"
    )
    original = read_json(launcher_audit)
    if (
        original.get("state") != "started"
        or original.get("retry_authorized") is not False
        or original.get("summary_exists") is not None
    ):
        raise RuntimeError("F14 source T2 launch is not an interrupted start")
    scene_rows = [
        row
        for row in read_jsonl(SCHEDULE)
        if str(row["scene_cluster"]) == SCENE
    ]
    recovery_state = case_recovery_state(
        scene_rows=scene_rows,
        out=OUT,
    )
    expected = amendment["incident"]
    if (
        recovery_state["sealed_manifest_count"]
        != int(expected["sealed_manifest_count"])
        or recovery_state["sealed_manifests"]
        != expected["sealed_manifests"]
        or recovery_state["zero_observation_case_ids"]
        != expected["zero_observation_case_ids"]
        or recovery_state["never_attempted_case_ids"]
        != expected["never_attempted_case_ids"]
    ):
        raise RuntimeError("F14 recovery state differs from sealed evidence")
    log_path = Path(str(original["log"]))
    if (
        sha256(launcher_audit)
        != amendment["evidence"][
            "source_launcher_audit_sha256_before_termination"
        ]
        or sha256(log_path)
        != amendment["evidence"][
            "source_runtime_log_sha256_before_termination"
        ]
    ):
        raise RuntimeError("F14 source evidence changed after seal")
    inactive_seconds = max(0.0, time.time() - log_path.stat().st_mtime)
    processes = t2_processes()
    if not processes:
        raise RuntimeError("F14 T2 collector process is absent")
    maximum_age = max(float(row["age_seconds"]) for row in processes)
    if (
        inactive_seconds < LIVENESS_DEADLINE_SECONDS
        or maximum_age < LIVENESS_DEADLINE_SECONDS
    ):
        raise RuntimeError("F14 fixed liveness deadline has not elapsed")

    audit_path = (
        OUT / "launcher_audits" / f"{SCENE}_f14_resume.json"
    )
    resume_log = (
        OUT / "launcher_logs" / f"{SCENE}_f14_resume.log"
    )
    if audit_path.exists() or resume_log.exists():
        raise FileExistsError(audit_path if audit_path.exists() else resume_log)
    action = terminate_exact_processes(processes)
    audit = {
        "schema_version": (
            "kinofail.reconfirmation-f14-t2-zero-observation-resume.v1"
        ),
        "state": "started",
        "started_utc": datetime.now(UTC).isoformat(),
        "scene_id": SCENE,
        "operational_amendment": str(F14),
        "operational_amendment_sha256": sha256(F14),
        "source_launcher_audit": str(launcher_audit),
        "source_launcher_audit_sha256_before_termination": (
            amendment["evidence"][
                "source_launcher_audit_sha256_before_termination"
            ]
        ),
        "source_runtime_log": str(log_path),
        "source_runtime_log_sha256_before_termination": (
            amendment["evidence"][
                "source_runtime_log_sha256_before_termination"
            ]
        ),
        "source_log_inactive_seconds_at_action": inactive_seconds,
        "maximum_process_age_seconds_at_action": maximum_age,
        "fixed_liveness_deadline_seconds": LIVENESS_DEADLINE_SECONDS,
        "termination": action,
        "recovery_state": recovery_state,
        "existing_manifest_reexecution_permitted": False,
        "zero_observation_reexecution_permitted": True,
        "result_dependent_retry_or_selection": False,
        "model_feature_label_outcome_prediction_or_score_read": False,
    }
    write_json(audit_path, audit)

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
    with resume_log.open("x", encoding="utf-8") as log:
        completed = subprocess.run(
            command,
            cwd=ROOT,
            env=environment,
            stdout=log,
            stderr=subprocess.STDOUT,
            check=False,
        )
    summary_path = OUT / "scene_summaries" / f"{SCENE}.json"
    summary = read_json(summary_path) if summary_path.is_file() else {}
    final_manifests = [
        OUT / str(row["case_id"]) / "manifest.json"
        for row in scene_rows
    ]
    all_scheduled_cases_accounted = (
        all(path.is_file() for path in final_manifests)
        and int(summary.get("case_count", -1)) == len(scene_rows)
        and len(summary.get("results", [])) == len(scene_rows)
    )
    terminal = {
        **audit,
        "state": "terminal",
        "completed_utc": datetime.now(UTC).isoformat(),
        "returncode": int(completed.returncode),
        "summary_exists": summary_path.is_file(),
        "passed": summary.get("passed") is True,
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
        "resume_log": str(resume_log),
        "resume_log_sha256": sha256(resume_log),
        "source_launcher_audit_sha256_after_termination": (
            sha256(launcher_audit)
        ),
        "sealed_manifest_hashes_preserved": all(
            sha256(Path(row["manifest"])) == row["manifest_sha256"]
            for row in recovery_state["sealed_manifests"]
        ),
    }
    write_json(audit_path, terminal)
    print(json.dumps(terminal, indent=2, sort_keys=True))
    return 0 if terminal["passed"] and all_scheduled_cases_accounted else 2


if __name__ == "__main__":
    raise SystemExit(main())
