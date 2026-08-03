#!/usr/bin/env python3
"""Reconstruct and audit the frozen C2-v5 hard-router family decision.

The v5 structured model and the formal proprioception-only baseline fit the
same balanced four-class logistic proposal (C=1) on the same development
features and weights.  The structured router selects the body route iff that
proposal is one of the two T3 classes; otherwise it selects the T2 visual
specialist.  The frozen prediction ledger therefore contains every value needed
to reconstruct the route without refitting a model.
"""

from __future__ import annotations

import argparse
import hashlib
import json
from collections import Counter, defaultdict
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
T2 = "T2_vision_decisive"
T3 = "T3_proprio_decisive"
T3_CLASSES = {"invisible_obstacle", "low_friction"}


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def read_jsonl(path: Path) -> list[dict]:
    return [
        json.loads(line)
        for line in path.read_text(encoding="utf-8").splitlines()
        if line
    ]


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--predictions",
        type=Path,
        default=ROOT
        / "outputs/eval/c2_bidirectional_v5/formal/predictions.jsonl",
    )
    parser.add_argument(
        "--formal-report",
        type=Path,
        default=ROOT / "outputs/eval/c2_bidirectional_v5/formal/report.json",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=ROOT
        / "outputs/eval/c2_route_fidelity_audit_v1/report.json",
    )
    args = parser.parse_args()

    predictions = args.predictions.resolve()
    formal_report = args.formal_report.resolve()
    rows = read_jsonl(predictions)
    if not rows:
        raise RuntimeError("empty formal prediction ledger")

    confusion: Counter[tuple[str, str]] = Counter()
    by_case: dict[str, list[bool]] = defaultdict(list)
    by_scene: dict[str, list[bool]] = defaultdict(list)
    for row in rows:
        expected = str(row["cell"])
        proposal = str(row["proprio_only_prediction"])
        selected = T3 if proposal in T3_CLASSES else T2
        correct = selected == expected
        confusion[(expected, selected)] += 1
        by_case[str(row["case_id"])].append(correct)
        by_scene[str(row["scene_cluster"])].append(correct)

    sample_success = sum(sum(values) for values in by_case.values())
    case_success = sum(all(values) for values in by_case.values())
    scene_accuracy = {
        scene: sum(values) / len(values)
        for scene, values in sorted(by_scene.items())
    }
    report = {
        "schema_version": "kinofail.c2-route-fidelity-audit.v1",
        "status": "registered_reanalysis_of_frozen_formal_predictions",
        "reconstruction_contract": (
            "The structured v5 proposal and proprio-only baseline are the "
            "same balanced four-class logistic(C=1) trained on the same "
            "development rows; select T3/body for low_friction or "
            "invisible_obstacle, otherwise T2/vision."
        ),
        "sample_route_fidelity": {
            "success": sample_success,
            "total": len(rows),
            "accuracy": sample_success / len(rows),
        },
        "case_route_fidelity": {
            "all_views_correct": case_success,
            "total": len(by_case),
            "accuracy": case_success / len(by_case),
        },
        "scene_route_accuracy": scene_accuracy,
        "confusion": {
            f"{expected} -> {selected}": count
            for (expected, selected), count in sorted(confusion.items())
        },
        "passed": (
            sample_success == len(rows)
            and case_success == len(by_case)
            and set(scene_accuracy.values()) == {1.0}
        ),
        "source_sha256": {
            "predictions": sha256(predictions),
            "formal_report": sha256(formal_report),
        },
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        json.dumps(report, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    print(json.dumps(report, indent=2, sort_keys=True))
    return 0 if report["passed"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
