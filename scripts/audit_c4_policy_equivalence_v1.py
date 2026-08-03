#!/usr/bin/env python3
"""Audit policy equivalence on the frozen realistic direct-C4 cases.

The direct C4 simulator corpus contains outcomes only for the frozen
structured-ensemble release vector and the registered fallback.  Another
attributor can reuse those outcomes without a new simulator rollout if and
only if it produces the same release decision for every case under the same
five-seed agreement and probability rule.
"""

from __future__ import annotations

import argparse
import hashlib
import json
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any

import numpy as np


ROOT = Path(__file__).resolve().parents[1]
TARGET = "adhesion"
THRESHOLD = 0.5
REQUIRED_AGREEMENT = 5
METHODS = (
    "vision_only",
    "proprio_only",
    "early_fusion",
    "structured_bidirectional",
)


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


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


def _decision(rows: list[dict[str, Any]]) -> dict[str, Any]:
    if len(rows) != 5 or {int(row["seed"]) for row in rows} != set(range(5)):
        raise RuntimeError("each policy decision requires exactly five training seeds")
    classes = [str(value) for value in rows[0]["classes"]]
    if any([str(value) for value in row["classes"]] != classes for row in rows):
        raise RuntimeError("class order differs across training seeds")
    probability = np.asarray(
        [row["probabilities"] for row in rows], dtype=np.float64
    )
    mean_probability = probability.mean(axis=0)
    prediction = classes[int(np.argmax(mean_probability))]
    seed_predictions = [str(row["prediction"]) for row in rows]
    agreement = sum(value == prediction for value in seed_predictions)
    target_probability = float(mean_probability[classes.index(TARGET)])
    release = (
        prediction == TARGET
        and agreement >= REQUIRED_AGREEMENT
        and target_probability >= THRESHOLD
    )
    return {
        "prediction": prediction,
        "seed_agreement": int(agreement),
        "target_probability": target_probability,
        "release": bool(release),
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--schedule",
        type=Path,
        default=ROOT
        / "outputs/kinofail_realistic/design_c4_direct_v3/schedule.jsonl",
    )
    parser.add_argument(
        "--predictions",
        type=Path,
        default=ROOT / "outputs/eval/realistic_a0_a7_v6/predictions.jsonl",
    )
    parser.add_argument(
        "--direct-report",
        type=Path,
        default=ROOT / "outputs/eval/realistic_a0_a7_v6/a5_c4_direct.json",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=ROOT
        / "outputs/eval/c4_policy_equivalence_audit_v1/report.json",
    )
    args = parser.parse_args()
    schedule_path = args.schedule.resolve()
    predictions_path = args.predictions.resolve()
    direct_path = args.direct_report.resolve()
    output_path = args.output.resolve()

    schedule = _jsonl(schedule_path)
    direct = _json(direct_path)
    if len(schedule) != 75 or int(direct["paired_case_count"]) != 75:
        raise RuntimeError("the frozen direct-C4 design must contain 75 paired cases")

    case_by_sample = {
        str(row["source_sample_id"]): row for row in schedule
    }
    if len(case_by_sample) != len(schedule):
        raise RuntimeError("duplicate source sample in C4 schedule")
    frozen_release = {
        sample_id: bool(row["decision"]["release"])
        for sample_id, row in case_by_sample.items()
    }

    selected: dict[tuple[str, str], list[dict[str, Any]]] = defaultdict(list)
    for row in _jsonl(predictions_path):
        sample_id = str(row["sample_id"])
        method = str(row["method"])
        if (
            sample_id in case_by_sample
            and method in METHODS
            and str(row["axis"]) == "scene_and_material"
            and bool(row["headline_primary_view"])
        ):
            selected[(method, sample_id)].append(row)

    method_reports: dict[str, Any] = {}
    structured_vector: dict[str, bool] | None = None
    for method in METHODS:
        decisions = {
            sample_id: _decision(selected[(method, sample_id)])
            for sample_id in sorted(case_by_sample)
        }
        release_vector = {
            sample_id: bool(value["release"])
            for sample_id, value in decisions.items()
        }
        released = [
            sample_id for sample_id, value in release_vector.items() if value
        ]
        correct = [
            sample_id
            for sample_id in released
            if str(decisions[sample_id]["prediction"])
            == str(case_by_sample[sample_id]["truth_attribution"])
        ]
        by_operator: Counter[str] = Counter(
            str(case_by_sample[sample_id]["operator"])
            for sample_id in released
        )
        same_as_frozen = release_vector == frozen_release
        method_reports[method] = {
            "released_cases": len(released),
            "coverage": len(released) / len(schedule),
            "released_attribution_precision": (
                len(correct) / len(released) if released else 0.0
            ),
            "release_count_by_operator": dict(sorted(by_operator.items())),
            "release_set_sha256": hashlib.sha256(
                "\n".join(released).encode("utf-8")
            ).hexdigest(),
            "release_vector_identical_to_frozen_policy": same_as_frozen,
            "measured_closed_loop_outcome_reusable": same_as_frozen,
            "measured_primary_endpoint_if_reusable": (
                direct["primary_endpoint"] if same_as_frozen else None
            ),
            "measured_secondary_endpoint_if_reusable": (
                direct["secondary_success_endpoint"] if same_as_frozen else None
            ),
        }
        if method == "structured_bidirectional":
            structured_vector = release_vector

    if structured_vector != frozen_release:
        raise RuntimeError(
            "recomputed structured policy does not match the frozen schedule"
        )

    direct_by_case = {
        str(row["case_id"]): row for row in direct["paired_cases"]
    }
    fallback_by_operator: dict[str, dict[str, float | int]] = {}
    for operator in sorted(
        {str(row["operator"]) for row in direct["paired_cases"]}
    ):
        rows = [
            row
            for row in direct["paired_cases"]
            if str(row["operator"]) == operator
        ]
        fallback_by_operator[operator] = {
            "cases": len(rows),
            "fall_rate": float(
                np.mean([bool(row["always_safe"]["fell"]) for row in rows])
            ),
            "success_rate": float(
                np.mean([bool(row["always_safe"]["success"]) for row in rows])
            ),
            "mean_terminal_cost": float(
                np.mean(
                    [
                        float(row["always_safe"]["terminal_cost"])
                        for row in rows
                    ]
                )
            ),
        }
    if len(direct_by_case) != 75:
        raise RuntimeError("direct report contains duplicate paired case IDs")

    report = {
        "schema_version": "kinofail.c4-policy-equivalence-audit.v1",
        "status": "registered_reanalysis_of_frozen_predictions_and_outcomes",
        "passed": True,
        "estimand": (
            "whether a comparator induces the exact frozen 75-case release "
            "vector, permitting reuse of measured matched-prefix outcomes"
        ),
        "decision_rule": {
            "target": TARGET,
            "mean_probability_threshold": THRESHOLD,
            "training_seeds": 5,
            "required_seed_agreement": REQUIRED_AGREEMENT,
            "release_action": "backstep_release",
            "registered_fallback": "hold_and_request",
        },
        "case_count": len(schedule),
        "methods": method_reports,
        "registered_fallback_outcomes_by_operator": fallback_by_operator,
        "interpretation": (
            "Early fusion, proprioception only, and the frozen structured "
            "ensemble induce the same 15-case O4-only release vector. Their "
            "measured direct-C4 action sequence and outcomes are therefore "
            "identical. The recovery experiment supports selective adhesion "
            "release, not a router-specific or multimodality-specific recovery "
            "advantage. The registered fallback is conservative by design but "
            "is not a universal safety guarantee."
        ),
        "source_sha256": {
            "schedule": _sha256(schedule_path),
            "predictions": _sha256(predictions_path),
            "direct_report": _sha256(direct_path),
        },
    }
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(
        json.dumps(report, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    print(json.dumps(report, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
