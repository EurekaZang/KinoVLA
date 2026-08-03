#!/usr/bin/env python3
"""Freeze the final early-region A1 construct correction before confirmatory scale."""

from __future__ import annotations

import hashlib
import json
from datetime import UTC, datetime
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]


def _sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main() -> None:
    prior = ROOT / "configs/data/kinofail_realistic_a1_matched_formal_v4.json"
    schedule = ROOT / "outputs/kinofail_realistic/design_a1_matched_v2/schedule.jsonl"
    audit = ROOT / "outputs/kinofail_realistic/design_a1_matched_v2/audit.json"
    collector = ROOT / "scripts/isaac_collect_kinofail_realistic_a1_matched_v4.py"
    output = ROOT / "configs/data/kinofail_realistic_a1_matched_formal_v5.json"
    if output.exists():
        raise FileExistsError(f"refusing to overwrite {output}")
    protocol = json.loads(prior.read_text(encoding="utf-8"))
    protocol.update({
        "protocol_id": "kinofail_realistic_a1_matched_o2_o4_36_v5",
        "frozen_utc": datetime.now(UTC).isoformat(),
        "supersedes_runtime_qa_failed_protocol": str(prior.relative_to(ROOT)),
        "superseded_protocol_sha256": _sha(prior),
        "amendment_reason": (
            "The one-pair v4 engineering preflight showed both episodes fell before the downstream "
            "route material was visible in the bounded window. Move the same O2/O4 region to a "
            "shared early route center 0.35 m with 0.10 m half-length, capture the first 16 RTX "
            "frames, and stop at 120 steps. Physics parameters, paired seeds/cameras, PBR causes, "
            "analysis, and acceptance thresholds are unchanged. The failed v4 pair is excluded."
        ),
        "collector_path": str(collector.relative_to(ROOT)),
        "collector_sha256": _sha(collector),
    })
    protocol["collection_contract"].update({
        "construct_horizon_control_steps": 120,
        "construct_horizon_s": 2.4,
        "rgb_capture_start_route_progress_m": 0.0,
        "rgb_capture_max_frames_per_view": 16,
        "operator_region_center_route_progress_m": 0.35,
        "operator_region_half_length_m": 0.10,
        "operator_region_start_route_progress_m": 0.25,
        "runtime_qa_preflight_pairs": 1,
        "runtime_qa_preflight_result": "v4 excluded: route material not visible before downstream fall",
    })
    output.write_text(json.dumps(protocol, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps({"output": str(output), "sha256": _sha(output), "pairs": 18}, indent=2))


if __name__ == "__main__":
    main()
