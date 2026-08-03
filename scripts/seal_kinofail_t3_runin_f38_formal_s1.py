#!/usr/bin/env python3
"""Seal the three-process F38 successor excluding every touched case."""

from __future__ import annotations

import json
import sys
from datetime import UTC, datetime
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from scripts.seal_kinofail_t3_runin_f38_pilot import read_json, read_jsonl, sha256  # noqa: E402


PARENT = ROOT / "outputs/freeze/kinofail_t3_runin_f38_formal/seal_manifest.json"
FAILURE = ROOT / "outputs/kinofail_t3_runin_f38_concurrency_failure/audit.json"
PARENT_SELECTED = ROOT / "outputs/freeze/kinofail_t3_runin_f38_formal/selected_cases.jsonl"
OUTPUT = ROOT / "outputs/freeze/kinofail_t3_runin_f38_formal_s1"
RUN = ROOT / "outputs/kinofail_t3_runin_f38_formal_s1"
CORPUS = Path("/data/eureka/KinoVLA/outputs/kinofail_t3_runin_f38_formal_s1/corpus")


def main() -> int:
    if OUTPUT.exists() or RUN.exists() or CORPUS.exists():
        raise FileExistsError("refusing to overwrite F38 formal successor")
    parent = read_json(PARENT)
    failure = read_json(FAILURE)
    if (
        parent.get("status") != "sealed_before_formal_result_blind_t3_recollection"
        or failure.get("status") != "model_blind_six_process_resource_failure_preserved"
        or failure.get("passed") is not True
        or failure.get("model_prediction_truth_key_or_score_read") is not False
        or failure.get("successor_contract", {}).get("maximum_concurrent_isaac_processes") != 3
    ):
        raise RuntimeError("invalid F38 parent failure boundary")
    touched = set(str(value) for value in failure["touched_case_ids"])
    selected = [row for row in read_jsonl(PARENT_SELECTED) if str(row["case_id"]) not in touched]
    if len(touched) != 6 or len(selected) != 1485 or len({pair for row in selected for pair in row["pair_ids"]}) != 2970:
        raise RuntimeError("F38 successor selection mismatch")
    OUTPUT.mkdir(parents=True, exist_ok=False)
    selected_path = OUTPUT / "selected_cases.jsonl"
    selected_path.write_text("".join(json.dumps(row, sort_keys=True) + "\n" for row in selected))
    seal = {
        **{
            key: value
            for key, value in parent.items()
            if key not in {
                "created_utc", "status", "counts", "selected_cases", "selected_cases_sha256",
                "maximum_concurrent_isaac_processes", "output_root", "corpus_root", "source_sha256",
            }
        },
        "schema_version": "kinofail.t3-runin-f38-formal-s1-seal.v1",
        "created_utc": datetime.now(UTC).isoformat(),
        "status": "sealed_before_three_process_result_blind_successor_collection",
        "passed": True,
        "all_six_process_touched_cases_excluded": True,
        "counts": {
            "planned_cases": 1500,
            "development_excluded_cases": 9,
            "six_process_touched_cases_excluded": 6,
            "selected_formal_cases": 1485,
            "pairs_to_collect": 2970,
            "physical_episodes": 5940,
        },
        "selected_cases": str(selected_path.relative_to(ROOT)),
        "selected_cases_sha256": sha256(selected_path),
        "maximum_concurrent_isaac_processes": 3,
        "output_root": str(RUN.relative_to(ROOT)),
        "corpus_root": str(CORPUS),
        "source_sha256": {
            **parent["source_sha256"],
            "parent_formal_seal": sha256(PARENT),
            "six_process_failure_audit": sha256(FAILURE),
            "parent_selected_cases": sha256(PARENT_SELECTED),
        },
    }
    path = OUTPUT / "seal_manifest.json"
    path.write_text(json.dumps(seal, indent=2, sort_keys=True) + "\n")
    (OUTPUT / "seal_manifest.sha256").write_text(f"{sha256(path)}  {path.name}\n")
    print(json.dumps(seal, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
