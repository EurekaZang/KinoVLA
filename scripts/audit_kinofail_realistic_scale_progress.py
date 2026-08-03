#!/usr/bin/env python3
"""Read-only progress and serious-bug audit for the running realistic scale collection."""

from __future__ import annotations

import argparse
import json
from collections import Counter, defaultdict
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]


def _json(path: Path) -> dict:
    if not path.is_file():
        return {}
    value = json.loads(path.read_text(encoding="utf-8"))
    return value if isinstance(value, dict) else {}


def _classify(row: dict, manifest: dict) -> tuple[str, list[str]]:
    if not manifest:
        return "missing", ["manifest_missing"]
    validation = manifest.get("runtime_validation", {})
    if "passed" not in validation and not validation.get("issues"):
        return "collecting", ["runtime_validation_pending"]
    if validation.get("passed") is True:
        return "accepted", []
    issues = [str(value) for value in validation.get("issues", [])]
    remaining = [
        value for value in issues
        if not value.endswith("appearance_effect_too_small")
        and value != "rgb_spatial_contrast_too_low"
    ]
    if remaining == ["operator_local_qa_failed"] and row["condition"] == "anomaly" and row["target_operator"] == "O4_tether":
        telemetry = manifest.get("operator_readback", {}).get("telemetry", {})
        certified = (
            int(telemetry.get("total_attachment_cycles", 0)) > 0
            and float(telemetry.get("total_applied_force_n", 0.0)) > 0.0
            and float(telemetry.get("total_tangential_work_j", 0.0)) > 0.0
        )
        if certified:
            return "accepted_o4_validator_recovery", issues
    if not remaining and issues:
        return "accepted_appearance_warning", issues
    return "blocking_repair", issues or ["runtime_validation_failed_without_issue"]


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--schedule", type=Path, default=ROOT / "outputs/kinofail_realistic/design_scale_v2/pilot_schedule.jsonl")
    parser.add_argument("--corpus-root", type=Path, default=ROOT / "outputs/kinofail_realistic/corpus_scale_v7")
    args = parser.parse_args()
    rows = [json.loads(line) for line in args.schedule.read_text(encoding="utf-8").splitlines() if line]
    groups = defaultdict(list)
    records = []
    for row in rows:
        groups[row["counterfactual_group_id"]].append(row)
        path = args.corpus_root / row["required_outputs"]["episode_manifest"]
        status, issues = _classify(row, _json(path))
        records.append({
            "episode_id": row["episode_id"],
            "pair_id": row["counterfactual_group_id"],
            "operator": row["target_operator"],
            "condition": row["condition"],
            "status": status,
            "issues": issues,
        })
    pair_status = {}
    for pair_id, pair in groups.items():
        statuses = [next(value["status"] for value in records if value["episode_id"] == row["episode_id"]) for row in pair]
        if "blocking_repair" in statuses:
            pair_status[pair_id] = "blocking_repair"
        elif "missing" in statuses or "collecting" in statuses:
            pair_status[pair_id] = "missing"
        else:
            pair_status[pair_id] = "accepted"
    blocking = [row for row in records if row["status"] == "blocking_repair"]
    accepted_pairs = {pair_id for pair_id, status in pair_status.items() if status == "accepted"}
    result = {
        "scheduled_episodes": len(rows),
        "scheduled_pairs": len(groups),
        "collected_episodes": sum(row["status"] not in {"missing", "collecting"} for row in records),
        "accepted_episodes": sum(row["status"].startswith("accepted") for row in records),
        "accepted_pairs": len(accepted_pairs),
        "blocking_repair_pairs": sorted({row["pair_id"] for row in blocking}),
        "blocking_repair_episodes": blocking,
        "episode_status": dict(sorted(Counter(row["status"] for row in records).items())),
        "accepted_operator_pair_counts": dict(sorted(Counter(
            pair[0]["target_operator"] for pair_id, pair in groups.items() if pair_id in accepted_pairs
        ).items())),
        "policy": "Only blocking_repair requires new simulation; appearance warnings and independently force-certified O4 validator false negatives are retained.",
    }
    print(json.dumps(result, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
