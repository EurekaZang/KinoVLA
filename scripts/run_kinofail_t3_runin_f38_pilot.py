#!/usr/bin/env python3
"""Run and audit the sealed, development-only F38 run-in pilot."""

from __future__ import annotations

import concurrent.futures
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

from kino_vla.eval.c2_temporal_v5 import geometry_aligned_invariant_summary  # noqa: E402
from scripts.seal_kinofail_t3_runin_f38_pilot import sha256  # noqa: E402


ISAACLAB = Path("/home/eureka/IsaacLab-v2.3.0/isaaclab.sh")
EXPERIENCE = Path("/home/eureka/IsaacLab-v2.3.0/apps/isaaclab.python.headless.rendering.kit")
CONDA = Path("/home/eureka/miniconda3/envs/kinovla")
SEAL = ROOT / "outputs/freeze/kinofail_t3_runin_f38_pilot/seal_manifest.json"
OUTPUT = ROOT / "outputs/kinofail_t3_runin_f38_pilot"
PYTHONPATH = ":".join(
    (
        "/home/eureka/IsaacLab-v2.3.0/source/isaaclab",
        "/home/eureka/IsaacLab-v2.3.0/source/isaaclab_tasks",
        "/home/eureka/IsaacLab-v2.3.0/source/isaaclab_assets",
        "/home/eureka/IsaacLab-v2.3.0/source/isaaclab_rl",
        str(ROOT),
    )
)


def read_json(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text())
    if not isinstance(value, dict):
        raise TypeError(path)
    return value


def atomic_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n")
    os.replace(temporary, path)


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
    if OUTPUT.exists():
        raise FileExistsError(OUTPUT)
    seal = read_json(SEAL)
    if (
        SEAL.with_name("seal_manifest.sha256").read_text().split()[0] != sha256(SEAL)
        or seal.get("status") != "sealed_before_development_only_physical_pilot"
        or seal.get("development_only") is not True
        or seal.get("counts_as_confirmatory_evidence") is not False
    ):
        raise RuntimeError("invalid F38 pilot seal")
    for section in ("schedules", "protocols"):
        for row in seal["source_sha256"][section]:
            path = ROOT / row["path"]
            if sha256(path) != row["sha256"]:
                raise RuntimeError(f"pilot dependency drift: {path}")
    for path_string, expected in seal["source_sha256"]["scripts"].items():
        path = ROOT / path_string
        if sha256(path) != expected:
            raise RuntimeError(f"pilot script drift: {path}")
    selected = [
        json.loads(line)
        for line in (ROOT / seal["selected_cases"]).read_text().splitlines()
        if line
    ]
    corpus = Path(seal["corpus_root"])
    OUTPUT.mkdir(parents=True, exist_ok=False)
    corpus.mkdir(parents=True, exist_ok=False)
    logs = OUTPUT / "logs"
    logs.mkdir()
    tasks = [
        {**row, "pair_id": pair_id}
        for row in selected
        for pair_id in row["pair_ids"]
    ]
    environment = {
        **os.environ,
        "CONDA_PREFIX": str(CONDA),
        "PATH": f"{CONDA / 'bin'}:{os.environ.get('PATH', '')}",
        "OMNI_KIT_ACCEPT_EULA": "YES",
        "PYTHONPATH": PYTHONPATH,
        "TERM": "xterm-256color",
        "HF_HUB_OFFLINE": "1",
    }

    def run_one(task: dict[str, Any]) -> dict[str, Any]:
        pair_id = str(task["pair_id"])
        scene = str(task["scene_id"])
        log_path = logs / f"{pair_id}.log"
        command = [
            str(ISAACLAB), "-p",
            str(ROOT / "scripts/isaac_collect_kinofail_confirmatory_t3_runin_f38.py"),
            "--schedule", str(ROOT / task["schedule"]),
            "--scene-registry", str(ROOT / seal["scene_registry"]),
            "--protocol", str(ROOT / task["protocol"]),
            "--asset-lock", str(ROOT / seal["material_lock"]),
            "--corpus-root", str(corpus / scene),
            "--counterfactual-group-id", pair_id,
            "--headless", "--enable_cameras", "--experience", str(EXPERIENCE),
        ]
        started = datetime.now(UTC).isoformat()
        with log_path.open("x") as stream:
            process = subprocess.Popen(
                command,
                cwd=ROOT,
                env=environment,
                stdout=stream,
                stderr=subprocess.STDOUT,
                start_new_session=True,
            )
            try:
                returncode = process.wait(timeout=900)
            except subprocess.TimeoutExpired:
                terminate(process.pid)
                returncode = process.wait()
        summary = corpus / scene / "pair_summaries" / f"{pair_id}.json"
        value = read_json(summary) if summary.is_file() else {}
        result = {
            "pair_id": pair_id,
            "scene_id": scene,
            "started_utc": started,
            "completed_utc": datetime.now(UTC).isoformat(),
            "returncode": int(returncode),
            "summary_exists": summary.is_file(),
            "summary_passed": value.get("passed") is True,
            "passed": returncode == 0 and value.get("passed") is True,
            "log": str(log_path),
        }
        atomic_json(OUTPUT / "attempts" / f"{pair_id}.json", result)
        print(json.dumps(result, sort_keys=True), flush=True)
        return result

    results: list[dict[str, Any]] = []
    with concurrent.futures.ThreadPoolExecutor(max_workers=3) as pool:
        for result in concurrent.futures.as_completed([pool.submit(run_one, task) for task in tasks]):
            results.append(result.result())
            atomic_json(
                OUTPUT / "state.json",
                {
                    "state": "running" if len(results) < len(tasks) else "auditing",
                    "completed_pairs": len(results),
                    "total_pairs": len(tasks),
                    "passed_pairs": sum(row["passed"] for row in results),
                    "updated_utc": datetime.now(UTC).isoformat(),
                },
            )
    alignments = {}
    for task in tasks:
        scene = str(task["scene_id"])
        pair_id = str(task["pair_id"])
        rows = [
            json.loads(line)
            for line in (ROOT / task["schedule"]).read_text().splitlines()
            if line and json.loads(line)["counterfactual_group_id"] == pair_id
        ]
        anomaly = next(row for row in rows if row["condition"] == "anomaly")
        episode = corpus / scene / Path(anomaly["required_outputs"]["episode_manifest"]).parent
        _, alignment = geometry_aligned_invariant_summary(episode)
        alignments[str(anomaly["episode_id"])] = alignment
    checks = {
        "six_pairs_terminal_and_passed": len(results) == 6 and all(row["passed"] for row in results),
        "six_anomaly_windows_extractable": len(alignments) == 6,
        "all_have_preinteraction_history": all(row["window_start_s"] < row["encounter_time_s"] for row in alignments.values()),
        "frozen_end_skew_respected": all(row["end_skew_s"] <= 0.021 for row in alignments.values()),
    }
    audit = {
        "schema_version": "kinofail.t3-runin-f38-pilot-audit.v1",
        "created_utc": datetime.now(UTC).isoformat(),
        "passed": all(checks.values()),
        "development_only": True,
        "counts_as_confirmatory_evidence": False,
        "model_prediction_truth_key_or_score_read": False,
        "checks": checks,
        "counts": {"cases": 3, "pairs": 6, "episodes": 12},
        "alignments": alignments,
        "seal_sha256": sha256(SEAL),
    }
    atomic_json(OUTPUT / "postrun_audit.json", audit)
    atomic_json(OUTPUT / "state.json", {"state": "complete" if audit["passed"] else "failed", "passed": audit["passed"], "updated_utc": datetime.now(UTC).isoformat()})
    print(json.dumps(audit, indent=2, sort_keys=True))
    return 0 if audit["passed"] else 2


if __name__ == "__main__":
    raise SystemExit(main())
