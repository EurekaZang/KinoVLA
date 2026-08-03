#!/usr/bin/env python3
"""Wait for A5 lighting, reduce it immediately, then launch the five-seed scale batch."""

from __future__ import annotations

import argparse
import fcntl
import json
import os
import subprocess
import sys
import time
from datetime import UTC, datetime
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
PYTHON = Path("/home/eureka/miniconda3/envs/kinovla/bin/python")
AUDIT = ROOT / "outputs/eval/realistic_a0_a7_v6/a5_then_scale_v8_orchestration.json"
LOCK = ROOT / "outputs/eval/realistic_a0_a7_v6/a5_then_scale_v8.lock"


def _alive(pid: int) -> bool:
    try:
        os.kill(pid, 0)
        return True
    except ProcessLookupError:
        return False


def _run(args: list[str]) -> dict:
    command = [str(PYTHON), *args]
    print(json.dumps({"running": command}), flush=True)
    completed = subprocess.run(command, cwd=ROOT, check=False)
    return {"command": command, "returncode": completed.returncode}


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--wait-pid", type=int, required=True)
    args = parser.parse_args()
    LOCK.parent.mkdir(parents=True, exist_ok=True)
    lock_handle = LOCK.open("w", encoding="utf-8")
    try:
        fcntl.flock(lock_handle, fcntl.LOCK_EX | fcntl.LOCK_NB)
    except BlockingIOError:
        print(json.dumps({"status": "duplicate_watcher_rejected", "lock": str(LOCK)}))
        return 3
    lock_handle.write(str(os.getpid()) + "\n")
    lock_handle.flush()
    while _alive(args.wait_pid):
        time.sleep(20.0)
    stages = [
        _run([
            "scripts/build_kinofail_realistic_snapshot_dev.py",
            "--protocol", "configs/eval/kinofail_realistic_a5_lighting_snapshot_v1.json",
        ]),
        _run([
            "scripts/extract_kinofail_realistic_features.py",
            "--snapshot-dir", "outputs/eval/realistic_a0_a7_v4/a5_lighting_snapshots",
            "--output-dir", "outputs/eval/realistic_a0_a7_v4/a5_lighting_features",
        ]),
        _run(["scripts/run_kinofail_realistic_a5_lighting_eval_v1.py"]),
        _run(["scripts/run_kinofail_realistic_scale_v8_nuisance_v1.py"]),
        _run(["scripts/run_kinofail_realistic_scale_v8_postprocess_v1.py"]),
        _run([
            "scripts/run_kinofail_realistic_scale_v2.py",
            "--schedule", "outputs/kinofail_realistic/design_a5_realization_v1/schedule.jsonl",
            "--scene-registry", "configs/data/kinofail_realistic_scale_scene_registry_v1.json",
            "--protocol", "configs/data/kinofail_realistic_a5_realization_formal_v2.json",
            "--corpus-root", "outputs/kinofail_realistic/corpus_a5_realization_v1",
            "--collector", "scripts/isaac_collect_kinofail_realistic_a5_realization_v2.py",
            "--audit-name", "full_collection.json",
        ]),
        _run([
            "scripts/build_kinofail_realistic_snapshot_dev.py",
            "--protocol", "configs/eval/kinofail_realistic_a5_realization_snapshot_v1.json",
        ]),
        _run([
            "scripts/extract_kinofail_realistic_features.py",
            "--snapshot-dir", "outputs/eval/realistic_a0_a7_v4/a5_realization_snapshots",
            "--output-dir", "outputs/eval/realistic_a0_a7_v4/a5_realization_features",
        ]),
        _run(["scripts/run_kinofail_realistic_a5_realization_eval_v1.py"]),
    ]
    if (
        ROOT / "outputs/eval/realistic_a0_a7_v4/a5_lighting_heldout.json"
    ).is_file() and (
        ROOT / "outputs/eval/realistic_a0_a7_v4/a5_physical_realization_heldout.json"
    ).is_file():
        stages.append(_run(["scripts/update_kinofail_realistic_a5_with_lighting_v1.py"]))
    stages.append(_run([
        "scripts/audit_kinofail_realistic_a0_a7.py",
        "--contract", "configs/eval/kinofail_realistic_a0_a7_v4.json",
        "--out", "outputs/eval/realistic_a0_a7_v4/readiness_audit.json",
    ]))
    if all((ROOT / value).is_file() for value in (
        "outputs/eval/realistic_a0_a7_v6/a0_certificate.json",
        "outputs/eval/realistic_a0_a7_v6/a5_selective.json",
        "outputs/eval/realistic_a0_a7_v4/a5_lighting_heldout.json",
        "outputs/eval/realistic_a0_a7_v4/a5_physical_realization_heldout.json",
    )):
        stages.append(_run([
            "scripts/rebind_kinofail_realistic_side_evidence_v5.py",
            "--target-root", "outputs/eval/realistic_a0_a7_v6",
        ]))
        stages.append(_run([
            "scripts/assemble_kinofail_realistic_a7_ablation_v1.py",
            "--contract", "configs/eval/kinofail_realistic_a0_a7_v6.json",
            "--evidence-root", "outputs/eval/realistic_a0_a7_v6",
            "--corpus-root", "outputs/kinofail_realistic/corpus_scale_v8_replication_v1",
            "--out", "outputs/eval/realistic_a0_a7_v6/a7_ablation.json",
        ]))
        stages.append(_run([
            "scripts/audit_kinofail_realistic_a0_a7.py",
            "--contract", "configs/eval/kinofail_realistic_a0_a7_v6.json",
            "--out", "outputs/eval/realistic_a0_a7_v6/readiness_audit.json",
        ]))
        stages.append(_run([
            "scripts/report_kinofail_realistic_icra_readiness_v1.py",
            "--contract", "configs/eval/kinofail_realistic_a0_a7_v6.json",
            "--out", "outputs/eval/realistic_a0_a7_v6/icra_benchmark_readiness.json",
        ]))
    AUDIT.parent.mkdir(parents=True, exist_ok=True)
    AUDIT.write_text(json.dumps({
        "schema_version": "kinofail.a5-then-scale-v8-orchestration.v1",
        "created_utc": datetime.now(UTC).isoformat(),
        "waited_for_pid": args.wait_pid,
        "pre_scale_stages": stages,
        "scale_launch_policy": (
            "scale-v8 runs immediately after the already-started lighting arm; no per-case tuning; "
            "A5 negative results do not block scale or postprocessing"
        ),
    }, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
