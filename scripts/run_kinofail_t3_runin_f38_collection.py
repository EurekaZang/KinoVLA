#!/usr/bin/env python3
"""Write-once, resumable collector for sealed F38 T3 run-in cohorts."""

from __future__ import annotations

import argparse
import concurrent.futures
import json
import os
import signal
import subprocess
import sys
import threading
import time
from datetime import UTC, datetime
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from kino_vla.eval.c2_temporal_v5 import geometry_aligned_invariant_summary  # noqa: E402


ISAACLAB = Path("/home/eureka/IsaacLab-v2.3.0/isaaclab.sh")
EXPERIENCE = Path("/home/eureka/IsaacLab-v2.3.0/apps/isaaclab.python.headless.rendering.kit")
CONDA = Path("/home/eureka/miniconda3/envs/kinovla")
PYTHONPATH = ":".join(
    (
        "/home/eureka/IsaacLab-v2.3.0/source/isaaclab",
        "/home/eureka/IsaacLab-v2.3.0/source/isaaclab_tasks",
        "/home/eureka/IsaacLab-v2.3.0/source/isaaclab_assets",
        "/home/eureka/IsaacLab-v2.3.0/source/isaaclab_rl",
        str(ROOT),
    )
)


def sha256(path: Path) -> str:
    import hashlib

    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def read_json(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text())
    if not isinstance(value, dict):
        raise TypeError(path)
    return value


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    return [json.loads(line) for line in path.read_text().splitlines() if line]


def atomic_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n")
    os.replace(temporary, path)


def pid_matches(pid: int, pair_id: str, collector: Path) -> bool:
    path = Path(f"/proc/{pid}/cmdline")
    if not path.is_file():
        return False
    try:
        command = path.read_bytes().replace(b"\0", b" ").decode(errors="replace")
    except OSError:
        return False
    return pair_id in command and collector.name in command


def terminate(pid: int) -> None:
    try:
        os.killpg(pid, signal.SIGTERM)
    except ProcessLookupError:
        return
    deadline = time.monotonic() + 15
    while time.monotonic() < deadline:
        try:
            os.killpg(pid, 0)
        except ProcessLookupError:
            return
        time.sleep(0.25)
    try:
        os.killpg(pid, signal.SIGKILL)
    except ProcessLookupError:
        pass


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--seal", type=Path, required=True)
    parser.add_argument("--max-workers", type=int)
    parser.add_argument("--timeout-s", type=int, default=900)
    args = parser.parse_args()
    seal_path = args.seal.resolve()
    seal = read_json(seal_path)
    sidecar = seal_path.with_name("seal_manifest.sha256")
    if (
        not sidecar.is_file()
        or sidecar.read_text().split()[0] != sha256(seal_path)
        or seal.get("passed") is not True
        or seal.get("model_prediction_truth_key_or_score_read") is not False
        or seal.get("result_dependent_retry") is not False
    ):
        raise RuntimeError("invalid F38 collection seal")
    selected_path = ROOT / str(seal["selected_cases"])
    registry = ROOT / str(seal["scene_registry"])
    asset_lock = ROOT / str(seal["material_lock"])
    collector = ROOT / str(seal["collector"])
    runner = Path(__file__).resolve()
    for path, expected in (
        (selected_path, seal["selected_cases_sha256"]),
        (registry, seal["scene_registry_sha256"]),
        (asset_lock, seal["material_lock_sha256"]),
        (collector, seal["collector_sha256"]),
        (runner, seal["runner_sha256"]),
        (ROOT / "kino_vla/eval/c2_temporal_v5.py", seal["v5_feature_function_sha256"]),
    ):
        if not path.is_file() or sha256(path) != expected:
            raise RuntimeError(f"F38 sealed dependency drift: {path}")
    for section in ("schedules", "protocols"):
        for relative, expected in seal.get("source_sha256", {}).get(section, {}).items():
            path = ROOT / relative
            if not path.is_file() or sha256(path) != expected:
                raise RuntimeError(f"F38 sealed {section} drift: {path}")
    selected = read_jsonl(selected_path)
    tasks = [
        {**row, "pair_id": pair_id}
        for row in selected
        for pair_id in row["pair_ids"]
    ]
    if len(tasks) != int(seal["counts"]["pairs_to_collect"]):
        raise RuntimeError("F38 task count differs from seal")
    workers = int(args.max_workers or seal["maximum_concurrent_isaac_processes"])
    if workers != int(seal["maximum_concurrent_isaac_processes"]):
        raise ValueError("worker count differs from frozen F38 seal")
    output = ROOT / str(seal["output_root"])
    corpus = Path(str(seal["corpus_root"]))
    output.mkdir(parents=True, exist_ok=True)
    corpus.mkdir(parents=True, exist_ok=True)
    attempts = output / "attempts"
    logs = output / "logs"
    attempts.mkdir(exist_ok=True)
    logs.mkdir(exist_ok=True)
    environment = {
        **os.environ,
        "CONDA_PREFIX": str(CONDA),
        "PATH": f"{CONDA / 'bin'}:{os.environ.get('PATH', '')}",
        "OMNI_KIT_ACCEPT_EULA": "YES",
        "PYTHONPATH": PYTHONPATH,
        "TERM": "xterm-256color",
        "HF_HUB_OFFLINE": "1",
    }
    state_lock = threading.Lock()
    completed = 0

    def reconcile(prior: dict[str, Any], task: dict[str, Any], attempt_path: Path) -> dict[str, Any]:
        pair_id = str(task["pair_id"])
        scene = str(task["scene_id"])
        pid = int(prior.get("pid", -1))
        while pid > 0 and pid_matches(pid, pair_id, collector):
            time.sleep(5)
        summary = corpus / scene / "pair_summaries" / f"{pair_id}.json"
        value = read_json(summary) if summary.is_file() else {}
        terminal = {
            **prior,
            "state": "terminal",
            "completed_utc": datetime.now(UTC).isoformat(),
            "reconciled_after_supervisor_restart": True,
            "summary_exists": summary.is_file(),
            "summary_passed": value.get("passed") is True,
            "passed": value.get("passed") is True,
        }
        atomic_json(attempt_path, terminal)
        return terminal

    def run_one(task: dict[str, Any]) -> dict[str, Any]:
        pair_id = str(task["pair_id"])
        scene = str(task["scene_id"])
        attempt_path = attempts / f"{pair_id}.json"
        if attempt_path.exists():
            prior = read_json(attempt_path)
            if prior.get("state") == "terminal":
                return prior
            return reconcile(prior, task, attempt_path)
        log_path = logs / f"{pair_id}.log"
        started = {
            "schema_version": "kinofail.t3-runin-f38-attempt.v1",
            "state": "started",
            "started_utc": datetime.now(UTC).isoformat(),
            "pair_id": pair_id,
            "scene_id": scene,
            "scientific_collection_attempt_index": 1,
            "retry_authorized": False,
            "result_dependent_retry": False,
            "model_prediction_truth_key_or_score_read": False,
            "log": str(log_path),
        }
        atomic_json(attempt_path, started)
        command = [
            str(ISAACLAB), "-p", str(collector),
            "--schedule", str(ROOT / task["schedule"]),
            "--scene-registry", str(registry),
            "--protocol", str(ROOT / task["protocol"]),
            "--asset-lock", str(asset_lock),
            "--corpus-root", str(corpus / scene),
            "--counterfactual-group-id", pair_id,
            "--headless", "--enable_cameras", "--experience", str(EXPERIENCE),
        ]
        with log_path.open("x") as stream:
            process = subprocess.Popen(
                command,
                cwd=ROOT,
                env=environment,
                stdout=stream,
                stderr=subprocess.STDOUT,
                start_new_session=True,
            )
            started["pid"] = process.pid
            atomic_json(attempt_path, started)
            timed_out = False
            try:
                returncode = process.wait(timeout=args.timeout_s)
            except subprocess.TimeoutExpired:
                timed_out = True
                terminate(process.pid)
                returncode = process.wait()
        summary = corpus / scene / "pair_summaries" / f"{pair_id}.json"
        value = read_json(summary) if summary.is_file() else {}
        terminal = {
            **started,
            "state": "terminal",
            "completed_utc": datetime.now(UTC).isoformat(),
            "returncode": int(returncode),
            "timed_out": timed_out,
            "summary_exists": summary.is_file(),
            "summary_passed": value.get("passed") is True,
            "passed": returncode == 0 and value.get("passed") is True,
        }
        atomic_json(attempt_path, terminal)
        print(json.dumps({"pair_id": pair_id, "scene_id": scene, "passed": terminal["passed"], "returncode": returncode}), flush=True)
        return terminal

    def update_state() -> None:
        rows = [read_json(path) for path in attempts.glob("*.json")]
        atomic_json(
            output / "state.json",
            {
                "schema_version": "kinofail.t3-runin-f38-state.v1",
                "updated_utc": datetime.now(UTC).isoformat(),
                "state": "running",
                "attempt_records": len(rows),
                "terminal_pairs": sum(row.get("state") == "terminal" for row in rows),
                "passed_pairs": sum(row.get("passed") is True for row in rows),
                "total_pairs": len(tasks),
                "maximum_concurrent_isaac_processes": workers,
            },
        )

    with concurrent.futures.ThreadPoolExecutor(max_workers=workers) as pool:
        futures = [pool.submit(run_one, task) for task in tasks]
        for future in concurrent.futures.as_completed(futures):
            future.result()
            with state_lock:
                completed += 1
                update_state()

    attempt_rows = {path.stem: read_json(path) for path in attempts.glob("*.json")}
    accepted = []
    rejected = []
    alignments: dict[str, Any] = {}
    schedule_cache = {
        str(case["schedule"]): read_jsonl(ROOT / case["schedule"])
        for case in selected
    }
    for case in selected:
        issues = []
        for pair_id in case["pair_ids"]:
            if attempt_rows.get(str(pair_id), {}).get("passed") is not True:
                issues.append(f"pair_failed:{pair_id}")
                continue
            schedule_rows = schedule_cache[str(case["schedule"])]
            anomaly = next(
                row
                for row in schedule_rows
                if row["counterfactual_group_id"] == pair_id and row["condition"] == "anomaly"
            )
            episode = corpus / case["scene_id"] / Path(anomaly["required_outputs"]["episode_manifest"]).parent
            try:
                _, alignment = geometry_aligned_invariant_summary(episode)
            except (KeyError, OSError, TypeError, ValueError) as exc:
                issues.append(f"temporal_window_failed:{pair_id}:{type(exc).__name__}")
            else:
                if not alignment["window_start_s"] < alignment["encounter_time_s"]:
                    issues.append(f"no_preinteraction_history:{pair_id}")
                alignments[str(anomaly["episode_id"])] = alignment
        row = {**case, "passed": not issues, "issues": issues}
        (accepted if row["passed"] else rejected).append(row)
    planned = int(seal["counts"]["planned_cases"])
    accepted_count = len(accepted)
    attrition = (planned - accepted_count) / planned
    strict = seal.get("strict_all_cases_pass") is True
    checks = {
        "all_pairs_terminal": len(attempt_rows) == len(tasks) and all(row.get("state") == "terminal" for row in attempt_rows.values()),
        "required_case_gate": accepted_count == planned if strict else attrition < float(seal["maximum_case_attrition_rate"]),
        "all_accepted_cases_have_two_temporal_windows": len(alignments) == 2 * accepted_count,
        "no_prediction_truth_key_or_score_read": True,
    }
    audit = {
        "schema_version": str(seal["audit_schema_version"]),
        "created_utc": datetime.now(UTC).isoformat(),
        "passed": all(checks.values()),
        "development_only": seal.get("development_only") is True,
        "counts_as_confirmatory_evidence": seal.get("counts_as_confirmatory_evidence") is True,
        "model_prediction_truth_key_or_score_read": False,
        "result_dependent_selection_or_retry": False,
        "checks": checks,
        "counts": {
            "planned_cases": planned,
            "selected_cases": len(selected),
            "accepted_cases": accepted_count,
            "rejected_or_excluded_cases": planned - accepted_count,
            "pairs": len(tasks),
            "passed_pairs": sum(row.get("passed") is True for row in attempt_rows.values()),
        },
        "case_attrition_rate": attrition,
        "accepted_cases": accepted,
        "rejected_cases": rejected,
        "episode_alignments": alignments,
        "seal_sha256": sha256(seal_path),
    }
    atomic_json(output / "final_audit.json", audit)
    atomic_json(output / "state.json", {"state": "complete" if audit["passed"] else "scientific_gate_failed", "passed": audit["passed"], "updated_utc": datetime.now(UTC).isoformat(), "counts": audit["counts"]})
    print(json.dumps({"passed": audit["passed"], "counts": audit["counts"], "case_attrition_rate": attrition, "audit": str(output / "final_audit.json")}, indent=2, sort_keys=True))
    return 0 if audit["passed"] else 2


if __name__ == "__main__":
    raise SystemExit(main())
