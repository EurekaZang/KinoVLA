#!/usr/bin/env python3
"""Preserve the model-blind six-process F38 infrastructure failure."""

from __future__ import annotations

import json
import sys
from datetime import UTC, datetime
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from scripts.seal_kinofail_t3_runin_f38_pilot import read_json, read_jsonl, sha256  # noqa: E402


SEAL = ROOT / "outputs/freeze/kinofail_t3_runin_f38_formal/seal_manifest.json"
SELECTED = ROOT / "outputs/freeze/kinofail_t3_runin_f38_formal/selected_cases.jsonl"
RUN = ROOT / "outputs/kinofail_t3_runin_f38_formal"
CORPUS = Path("/data/eureka/KinoVLA/outputs/kinofail_t3_runin_f38_formal/corpus")
OUTPUT = ROOT / "outputs/kinofail_t3_runin_f38_concurrency_failure/audit.json"


def main() -> int:
    if OUTPUT.exists():
        raise FileExistsError(OUTPUT)
    for path in Path("/proc").glob("[0-9]*/cmdline"):
        try:
            command = path.read_bytes().decode(errors="replace")
        except (OSError, PermissionError):
            continue
        if "isaac_collect_kinofail_confirmatory_t3_runin_f38_v3" in command:
            raise RuntimeError("F38 child process is still active")
    seal = read_json(SEAL)
    attempts = {path.stem: read_json(path) for path in sorted((RUN / "attempts").glob("*.json"))}
    selected = read_jsonl(SELECTED)
    attempted = set(attempts)
    touched = [row for row in selected if attempted.intersection(str(value) for value in row["pair_ids"])]
    if (
        seal.get("maximum_concurrent_isaac_processes") != 6
        or len(attempts) != 12
        or len(touched) != 6
        or any(set(str(value) for value in row["pair_ids"]) - attempted for row in touched)
    ):
        raise RuntimeError("unexpected F38 concurrency-failure boundary")
    forbidden = [
        path
        for pattern in ("*prediction*", "*truth*", "*score*")
        for path in RUN.rglob(pattern)
    ]
    if forbidden:
        raise RuntimeError(f"model result artifact exists: {forbidden[:3]}")
    reasons = {"gpu_out_of_memory": 0, "zero_norm_quaternion_after_resource_pressure": 0, "other": 0}
    logs = {}
    for pair_id in sorted(attempts):
        path = RUN / "logs" / f"{pair_id}.log"
        text = path.read_text(errors="replace")
        if "out of memory" in text.lower() or "ERROR_OUT_OF_DEVICE_MEMORY" in text:
            reason = "gpu_out_of_memory"
        elif "zero norm quaternions" in text:
            reason = "zero_norm_quaternion_after_resource_pressure"
        else:
            reason = "other"
        reasons[reason] += 1
        logs[pair_id] = {"reason": reason, "sha256": sha256(path)}
    if reasons["gpu_out_of_memory"] < 1:
        raise RuntimeError("F38 failure is not the diagnosed concurrency OOM")
    summaries = sorted(CORPUS.glob("*/pair_summaries/*.json"))
    audit = {
        "schema_version": "kinofail.t3-runin-f38-concurrency-failure-audit.v1",
        "created_utc": datetime.now(UTC).isoformat(),
        "status": "model_blind_six_process_resource_failure_preserved",
        "passed": True,
        "model_prediction_truth_key_or_score_read": False,
        "result_dependent_retry": False,
        "all_touched_cases_excluded_from_successor": True,
        "counts": {
            "attempted_pairs": len(attempts),
            "touched_complete_cases": len(touched),
            "pair_summaries_written_before_stop": len(summaries),
            "unattempted_selected_cases": len(selected) - len(touched),
        },
        "failure_reasons": reasons,
        "touched_case_ids": sorted(str(row["case_id"]) for row in touched),
        "touched_pair_ids": sorted(attempted),
        "logs": logs,
        "source_sha256": {
            "formal_seal": sha256(SEAL),
            "formal_selected_cases": sha256(SELECTED),
            "attempt_records": {pair_id: sha256(RUN / "attempts" / f"{pair_id}.json") for pair_id in sorted(attempts)},
            "pair_summaries": {str(path): sha256(path) for path in summaries},
        },
        "successor_contract": {
            "maximum_concurrent_isaac_processes": 3,
            "touched_pairs_reexecuted": False,
            "touched_cases_used_as_evidence": False,
            "remaining_case_selection_uses_outcomes": False,
        },
    }
    OUTPUT.parent.mkdir(parents=True, exist_ok=False)
    OUTPUT.write_text(json.dumps(audit, indent=2, sort_keys=True) + "\n")
    print(json.dumps(audit, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
