#!/usr/bin/env python3
"""Record the resource-driven F28 interruption without altering attempts."""

from __future__ import annotations

import hashlib
import json
import subprocess
from datetime import UTC, datetime
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parents[1]
F28 = ROOT / "outputs/kinofail_t3_replenishment_f28"
CORPUS = Path("/data/eureka/KinoVLA/outputs/kinofail_t3_replenishment_f28/corpus")


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


def main() -> int:
    output = F28 / "interruption_audit.json"
    if output.exists():
        raise FileExistsError(output)
    seal_path = F28 / "seal_manifest.json"
    seal = read_json(seal_path)
    sidecar = F28 / "seal_manifest.sha256"
    if sidecar.read_text(encoding="utf-8").split()[0] != sha256(seal_path):
        raise RuntimeError("F28 seal drift")
    process_check = subprocess.run(
        [
            "pgrep",
            "-f",
            "^/home/eureka/miniconda3/envs/kinovla/bin/python /home/eureka/KinoVLA/scripts/isaac_collect_kinofail_confirmatory_pair_v9.py",
        ],
        check=False,
        capture_output=True,
        text=True,
    )
    if process_check.returncode == 0 and process_check.stdout.strip():
        raise RuntimeError("F28 collector is still active")

    attempt_paths = sorted((F28 / "attempts").glob("*.json"))
    attempts = [read_json(path) for path in attempt_paths]
    summary_paths = sorted(CORPUS.rglob("pair_summaries/*.json")) if CORPUS.exists() else []
    summaries = [read_json(path) for path in summary_paths]
    log_paths = sorted((F28 / "logs").glob("*.log"))
    log_text = "\n".join(
        path.read_text(encoding="utf-8", errors="replace") for path in log_paths
    )
    attempted_pairs = sorted(str(row["pair_id"]) for row in attempts)
    audit = {
        "schema_version": "kinofail.t3-replenishment-f28-interruption-audit.v1",
        "created_utc": datetime.now(UTC).isoformat(),
        "status": "sealed_interrupted_resource_oversubscription",
        "passed": True,
        "f28_supervisor_active": False,
        "f28_collector_active": False,
        "cause": "six concurrent RTX Isaac processes exceeded available CUDA memory",
        "cuda_oom_observed": "CUDA error: out of memory" in log_text,
        "zero_norm_quaternion_observed_after_resource_contention": "Found zero norm quaternions" in log_text,
        "model_prediction_feature_or_score_read": False,
        "attempted_pair_count": len(attempts),
        "attempted_pair_ids": attempted_pairs,
        "terminal_attempt_count": sum(row.get("state") == "terminal" for row in attempts),
        "interrupted_started_attempt_count": sum(row.get("state") == "started" for row in attempts),
        "summary_count": len(summaries),
        "passed_summary_count": sum(row.get("passed") is True for row in summaries),
        "accepted_pair_count": 0,
        "attempt_inventory": [
            {"path": str(path.relative_to(ROOT)), "sha256": sha256(path)}
            for path in attempt_paths
        ],
        "log_inventory": [
            {"path": str(path.relative_to(ROOT)), "sha256": sha256(path)}
            for path in log_paths
        ],
        "summary_inventory": [
            {"path": str(path), "sha256": sha256(path)} for path in summary_paths
        ],
        "f28_seal": str(seal_path.relative_to(ROOT)),
        "f28_seal_sha256": sha256(seal_path),
        "next_execution_contract": {
            "exclude_every_f28_attempted_pair": True,
            "exclude_entire_case_if_either_pair_was_attempted": True,
            "reuse_f28_artifacts": False,
            "maximum_concurrent_isaac_processes": 3,
            "selection_may_not_depend_on_pass_fail_outcome": True,
        },
    }
    if (
        len(attempts) != 13
        or audit["passed_summary_count"] != 0
        or not audit["cuda_oom_observed"]
        or not audit["zero_norm_quaternion_observed_after_resource_contention"]
    ):
        raise RuntimeError("unexpected F28 interruption state")
    output.write_text(
        json.dumps(audit, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    print(json.dumps(audit, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
