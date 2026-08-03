#!/usr/bin/env python3
"""Analyze a frozen direct realistic C4 selective-versus-always-safe corpus."""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
from datetime import UTC, datetime
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from kino_vla.eval.realistic_c4 import evaluate_direct_c4


def _sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _json(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise TypeError(path)
    return value


def _jsonl(path: Path) -> list[dict[str, Any]]:
    return [
        json.loads(line)
        for line in path.read_text(encoding="utf-8").splitlines()
        if line
    ]


def main() -> int:
    parser = argparse.ArgumentParser()
    source = parser.add_mutually_exclusive_group(required=True)
    source.add_argument("--results", type=Path)
    source.add_argument("--results-root", type=Path)
    parser.add_argument("--protocol", type=Path, required=True)
    parser.add_argument(
        "--a0",
        type=Path,
        default=ROOT / "outputs/eval/realistic_a0_a7_v6/a0_certificate.json",
    )
    parser.add_argument(
        "--out",
        type=Path,
        default=ROOT / "outputs/eval/realistic_a0_a7_v6/a5_c4_direct.json",
    )
    args = parser.parse_args()

    protocol_path = args.protocol.resolve()
    a0_path = args.a0.resolve()
    result_paths = (
        [args.results.resolve()]
        if args.results is not None
        else sorted(args.results_root.resolve().glob("*/results.jsonl"))
    )
    if not result_paths:
        raise RuntimeError("no direct C4 result files found")
    result_rows = [
        row for path in result_paths for row in _jsonl(path)
    ]
    report = evaluate_direct_c4(result_rows, _json(protocol_path))
    a0 = _json(a0_path)
    result_hashes = {
        str(path.relative_to(ROOT)): _sha(path) for path in result_paths
    }
    first_branch_counts: dict[str, int] = {}
    isolation_passed = True
    for case in report["paired_cases"]:
        selective = next(
            row
            for row in result_rows
            if row["case_id"] == case["case_id"]
            and row["branch"] == "selective"
        )
        order = selective.get("branch_order", [])
        first = str(order[0]) if len(order) == 2 else ""
        first_branch_counts[first] = first_branch_counts.get(first, 0) + 1
        isolation_passed = isolation_passed and bool(
            selective.get("fresh_isaac_app_per_case_pair")
        )
    order_balance_passed = (
        set(first_branch_counts) == {"selective", "always_safe"}
        and abs(
            first_branch_counts["selective"]
            - first_branch_counts["always_safe"]
        )
        <= 1
    )
    report["execution_design_audit"] = {
        "first_branch_counts": first_branch_counts,
        "global_order_imbalance": (
            abs(
                first_branch_counts.get("selective", 0)
                - first_branch_counts.get("always_safe", 0)
            )
        ),
        "counterbalanced_branch_order_passed": order_balance_passed,
        "fresh_isaac_app_per_case_pair_passed": isolation_passed,
    }
    report["acceptance"]["counterbalanced_branch_order_passed"] = (
        order_balance_passed
    )
    report["acceptance"]["fresh_isaac_app_per_case_pair_passed"] = (
        isolation_passed
    )
    report["passed"] = bool(
        report["passed"] and order_balance_passed and isolation_passed
    )
    report.update(
        {
            "created_utc": datetime.now(UTC).isoformat(),
            "dataset_scope": "kinofail_realistic",
            "protocol": str(protocol_path.relative_to(ROOT)),
            "protocol_sha256": _sha(protocol_path),
            "results": sorted(result_hashes),
            "result_sha256": result_hashes,
            "a0_certificate": str(a0_path.relative_to(ROOT)),
            "a0_certificate_sha256": _sha(a0_path),
            "a0_evidence_bundle_sha256": a0["a0_evidence_bundle_sha256"],
        }
    )
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(
        json.dumps(report, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    print(
        json.dumps(
            {
                "out": str(args.out),
                "status": report["status"],
                "passed": report["passed"],
                "paired_case_count": report["paired_case_count"],
                "release_coverage": report["release_coverage"],
                "released_attribution_precision": report[
                    "released_attribution_precision"
                ],
                "primary_endpoint": report["primary_endpoint"],
                "acceptance": report["acceptance"],
            },
            indent=2,
            sort_keys=True,
        )
    )
    return 0 if report["passed"] else 2


if __name__ == "__main__":
    raise SystemExit(main())
