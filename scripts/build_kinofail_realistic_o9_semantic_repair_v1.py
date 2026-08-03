#!/usr/bin/env python3
"""Build the 90-pair O9 semantic-repair schedule from frozen scale-v8."""

from __future__ import annotations

import argparse
import hashlib
import json
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_SOURCE = (
    ROOT
    / "outputs/kinofail_realistic/design_scale_v8_replication_v1/full_schedule.jsonl"
)
DEFAULT_OUT = (
    ROOT / "outputs/kinofail_realistic/design_o9_semantic_repair_v1/schedule.jsonl"
)
DEFAULT_AUDIT = (
    ROOT
    / "outputs/kinofail_realistic/design_o9_semantic_repair_v1/design_audit.json"
)
DEFAULT_BENCHMARK_ID = "kinofail_realistic_o9_semantic_repair_v1"
DIMENSIONS = {
    "moderate": {
        "residual_support": 0.46,
        "ridge_height_m": 0.35,
        "ridge_width_m": 0.12,
    },
    "severe": {
        "residual_support": 0.22,
        "ridge_height_m": 0.36,
        "ridge_width_m": 0.14,
    },
}


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    return [
        json.loads(line)
        for line in path.read_text(encoding="utf-8").splitlines()
        if line
    ]


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--source", type=Path, default=DEFAULT_SOURCE)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUT)
    parser.add_argument("--audit", type=Path, default=DEFAULT_AUDIT)
    parser.add_argument("--benchmark-id", default=DEFAULT_BENCHMARK_ID)
    parser.add_argument("--moderate-height-m", type=float, default=0.35)
    args = parser.parse_args()

    source_rows = read_jsonl(args.source)
    source_o9 = [
        row for row in source_rows if row.get("target_operator") == "O9_high_centering"
    ]
    if len(source_o9) != 180:
        raise RuntimeError(f"expected 180 O9 schedule rows, got {len(source_o9)}")

    repaired = []
    for source_row in source_o9:
        row = dict(source_row)
        severity = str(row["severity_id"])
        if severity not in DIMENSIONS:
            raise RuntimeError(f"unexpected O9 severity {severity}")
        row["benchmark_id"] = str(args.benchmark_id)
        row["design_mode"] = "targeted_o9_semantic_repair"
        row["physical_realization"] = "pallet_edge"
        row["geometry_profile"] = "cross_path_pallet_edge_belly_crossbar"
        row["geometry_id"] = (
            f"{row['scene_family']}::cross_path_pallet_edge_belly_crossbar"
        )
        row["artifact_state"] = "planned"
        row["evaluation_eligible"] = False
        if row["condition"] == "anomaly":
            row["physics_parameters"] = dict(DIMENSIONS[severity])
            if severity == "moderate":
                row["physics_parameters"]["ridge_height_m"] = float(
                    args.moderate_height_m
                )
        elif row["condition"] == "nominal_counterfactual":
            row["physics_parameters"] = {
                "residual_support": 1.0,
                "ridge_height_m": 0.0,
                "ridge_width_m": 0.0,
            }
        else:
            raise RuntimeError(f"unexpected condition {row['condition']}")
        repaired.append(row)

    by_pair = Counter(row["counterfactual_group_id"] for row in repaired)
    pair_conditions: dict[str, set[str]] = {}
    for row in repaired:
        pair_conditions.setdefault(row["counterfactual_group_id"], set()).add(
            row["condition"]
        )
    checks = {
        "rows_180": len(repaired) == 180,
        "pairs_90": len(by_pair) == 90,
        "exactly_two_rows_per_pair": set(by_pair.values()) == {2},
        "complete_counterfactual_conditions": all(
            values == {"anomaly", "nominal_counterfactual"}
            for values in pair_conditions.values()
        ),
        "operator_o9_only": {row["target_operator"] for row in repaired}
        == {"O9_high_centering"},
        "pallet_edge_only": {row["physical_realization"] for row in repaired}
        == {"pallet_edge"},
        "all_evaluation_ineligible_before_collection": not any(
            row["evaluation_eligible"] for row in repaired
        ),
        "scene_split_domain_seed_balance_preserved": Counter(
            (
                row["scene_family"],
                row["split"],
                row["domain"],
                row["scene_seed"],
                row["severity_id"],
                row["condition"],
            )
            for row in repaired
        )
        == Counter(
            (
                row["scene_family"],
                row["split"],
                row["domain"],
                row["scene_seed"],
                row["severity_id"],
                row["condition"],
            )
            for row in source_o9
        ),
    }
    if not all(checks.values()):
        raise RuntimeError(f"O9 semantic repair schedule audit failed: {checks}")

    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("w", encoding="utf-8") as handle:
        for row in repaired:
            handle.write(json.dumps(row, sort_keys=True) + "\n")

    audit = {
        "schema_version": "kinofail.o9-semantic-repair-design-audit.v1",
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "status": "passed",
        "benchmark_id": str(args.benchmark_id),
        "source_schedule": str(args.source),
        "source_schedule_sha256": sha256(args.source),
        "repair_schedule": str(args.output),
        "repair_schedule_sha256": sha256(args.output),
        "checks": checks,
        "counts": {
            "rows": len(repaired),
            "counterfactual_pairs": len(by_pair),
            "scenes": len({row["scene_family"] for row in repaired}),
            "domains": len({row["domain"] for row in repaired}),
            "splits": dict(Counter(row["split"] for row in repaired)),
            "severities": dict(Counter(row["severity_id"] for row in repaired)),
        },
        "semantic_change": {
            "old": (
                "0.18/0.23 m ridge and region-exposure admission; head/limb "
                "impact could be mislabeled as high-centering"
            ),
            "new": (
                "0.35/0.36 m route-transverse pallet crossbar, matched start at "
                "ridge center, and direct sustained base-contact admission"
            ),
            "moderate": {
                **DIMENSIONS["moderate"],
                "ridge_height_m": float(args.moderate_height_m),
            },
            "severe": DIMENSIONS["severe"],
        },
    }
    args.audit.parent.mkdir(parents=True, exist_ok=True)
    args.audit.write_text(
        json.dumps(audit, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    print(json.dumps(audit, indent=2))


if __name__ == "__main__":
    main()
