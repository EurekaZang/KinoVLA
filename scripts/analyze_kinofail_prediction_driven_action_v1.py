#!/usr/bin/env python3
"""Link frozen cause predictions to complete same-checkpoint action matrices."""

from __future__ import annotations

import argparse
import csv
import hashlib
import itertools
import json
import math
import sys
from collections import defaultdict
from datetime import UTC, datetime
from pathlib import Path
from statistics import mean
from typing import Any, Callable

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from scripts.build_kinofail_action_consequence_evidence_v1 import read_complete_cases


METHODS = (
    "vision",
    "proprioception",
    "joint_early_fusion",
    "late_fusion",
    "concat_mlp",
    "tfn",
    "lmf",
    "gmu",
    "embracenet",
    "moddrop",
)
PRIMARY_METHOD = "gmu"
ACTION_FAMILIES = (
    "continue",
    "safe_halt",
    "hold_request",
    "low_friction",
    "effort_decay",
    "compliance",
    "collapse",
    "adhesion_release",
    "push_reflex",
    "invisible_obstacle",
    "high_centering",
)
CLASS_TO_ACTION = {
    "nominal": "continue",
    "low_friction": "low_friction",
    "compliant_terrain": "compliance",
    "region_collapse": "collapse",
    "adhesion": "adhesion_release",
    "overload": "hold_request",
    "external_push": "push_reflex",
    "invisible_obstacle": "invisible_obstacle",
    "high_centering": "high_centering",
    "effort_decay": "effort_decay",
    "obs_bias": "hold_request",
}
OPERATOR_TO_ACTION = {
    "O1_mu_field": "low_friction",
    "O2_compliance": "compliance",
    "O3_collapse": "collapse",
    "O4_tether": "adhesion_release",
    "O5_payload": "hold_request",
    "O6_push": "push_reflex",
    "O7_visual_remap": "low_friction",
    "O8_invisible_collider": "invisible_obstacle",
    "O9_high_centering": "high_centering",
    "O10_effort_decay": "effort_decay",
    "O11_obs_bias": "hold_request",
}
ACTION_TO_LOCAL = {
    "continue": "continue",
    "safe_halt": "always_safe_halt",
    "hold_request": "recover_as_hold_request",
    "low_friction": "recover_as_low_friction",
    "effort_decay": "recover_as_O10_effort_decay",
    "compliance": "recover_as_O2_compliance",
    "collapse": "recover_as_O3_collapse",
    "adhesion_release": "recover_as_O4_tether",
    "invisible_obstacle": "recover_as_O8_invisible_collider",
    "high_centering": "recover_as_O9_high_centering",
}


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def jsonl(path: Path) -> list[dict[str, Any]]:
    return [json.loads(line) for line in path.read_text().splitlines() if line]


def local_action(case: dict[str, Any], family: str) -> str:
    if family == "push_reflex":
        candidates = sorted(
            action for action in case["arms"] if action.startswith("recover_as_O6_")
        )
        if len(candidates) != 1:
            raise RuntimeError(f"case has no unique O6 response: {case['case_id']}")
        return candidates[0]
    return ACTION_TO_LOCAL[family]


def arm(case: dict[str, Any], family: str) -> dict[str, Any]:
    return case["arms"][local_action(case, family)]


def cluster_bootstrap_ci(
    rows: list[dict[str, Any]], value: Callable[[dict[str, Any]], float], *, seed: int
) -> list[float]:
    by_scene: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        by_scene[str(row["scene_id"])].append(row)
    scenes = sorted(by_scene)
    rng = np.random.default_rng(seed)
    estimates = np.empty(20_000, dtype=np.float64)
    for draw in range(len(estimates)):
        selected = rng.integers(0, len(scenes), size=len(scenes))
        values = [value(row) for index in selected for row in by_scene[scenes[index]]]
        estimates[draw] = float(np.mean(values))
    return [float(np.quantile(estimates, 0.025)), float(np.quantile(estimates, 0.975))]


def exact_scene_signflip_less(rows: list[dict[str, Any]], key: str) -> float:
    by_scene: dict[str, list[float]] = defaultdict(list)
    for row in rows:
        by_scene[str(row["scene_id"])].append(float(row[key]))
    scene_deltas = np.asarray([mean(by_scene[value]) for value in sorted(by_scene)])
    observed = float(scene_deltas.mean())
    if len(scene_deltas) > 20:
        raise RuntimeError("exact scene sign-flip enumeration is unexpectedly large")
    extreme = 0
    total = 0
    for signs in itertools.product((-1.0, 1.0), repeat=len(scene_deltas)):
        total += 1
        if float(np.mean(scene_deltas * np.asarray(signs))) <= observed + 1.0e-12:
            extreme += 1
    return extreme / total


def holm(raw: dict[str, float]) -> dict[str, float]:
    ordered = sorted(raw, key=lambda key: raw[key])
    adjusted: dict[str, float] = {}
    running = 0.0
    for index, key in enumerate(ordered):
        running = max(running, (len(ordered) - index) * raw[key])
        adjusted[key] = min(1.0, running)
    return adjusted


def mcnemar_exact_two_sided(left: list[bool], right: list[bool]) -> float:
    discordant_left = sum(a and not b for a, b in zip(left, right, strict=True))
    discordant_right = sum(b and not a for a, b in zip(left, right, strict=True))
    n = discordant_left + discordant_right
    if n == 0:
        return 1.0
    k = min(discordant_left, discordant_right)
    tail = sum(math.comb(n, value) for value in range(k + 1)) / (2**n)
    return min(1.0, 2.0 * tail)


def fixed_action_loso(cases: list[dict[str, Any]]) -> dict[str, str]:
    scenes = sorted({str(case["scene_id"]) for case in cases})
    selected: dict[str, str] = {}
    for held_out in scenes:
        development = [case for case in cases if str(case["scene_id"]) != held_out]
        score = {
            family: mean(float(arm(case, family)["terminal_cost"]) for case in development)
            for family in ACTION_FAMILIES
        }
        selected[held_out] = min(ACTION_FAMILIES, key=lambda family: (score[family], family))
    return selected


def evaluate_method(
    method: str,
    cases: list[dict[str, Any]],
    predictions: dict[str, dict[str, Any]],
    fixed_by_scene: dict[str, str],
) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    rows: list[dict[str, Any]] = []
    for case in cases:
        group = str(case["case_id"]).rsplit("__", 1)[-1]
        prediction = predictions[group]
        predicted_class = str(prediction["predictions"][method])
        selected_family = CLASS_TO_ACTION[predicted_class]
        oracle_family = OPERATOR_TO_ACTION[str(case["operator"])]
        fixed_family = fixed_by_scene[str(case["scene_id"])]
        selected = arm(case, selected_family)
        oracle = arm(case, oracle_family)
        fixed = arm(case, fixed_family)
        continuing = arm(case, "continue")
        halt = arm(case, "safe_halt")
        row = {
            "case_id": case["case_id"],
            "counterfactual_group_id": group,
            "scene_id": case["scene_id"],
            "operator": case["operator"],
            "truth": prediction["truth"],
            "method": method,
            "predicted_class": predicted_class,
            "selected_action_family": selected_family,
            "oracle_action_family": oracle_family,
            "fixed_action_family": fixed_family,
            "exact_cause_correct": predicted_class == str(prediction["truth"]),
            "action_equivalent_correct": selected_family == oracle_family,
            "selected_terminal_cost": float(selected["terminal_cost"]),
            "oracle_terminal_cost": float(oracle["terminal_cost"]),
            "fixed_terminal_cost": float(fixed["terminal_cost"]),
            "continue_terminal_cost": float(continuing["terminal_cost"]),
            "halt_terminal_cost": float(halt["terminal_cost"]),
            "selected_success": bool(selected["operator_recovery_success"]),
            "oracle_success": bool(oracle["operator_recovery_success"]),
            "fixed_success": bool(fixed["operator_recovery_success"]),
            "continue_success": bool(continuing["operator_recovery_success"]),
            "halt_success": bool(halt["operator_recovery_success"]),
            "selected_fell": bool(selected["fell"]),
            "oracle_fell": bool(oracle["fell"]),
            "fixed_fell": bool(fixed["fell"]),
            "continue_fell": bool(continuing["fell"]),
            "halt_fell": bool(halt["fell"]),
        }
        row["selected_minus_oracle_cost"] = row["selected_terminal_cost"] - row["oracle_terminal_cost"]
        row["selected_minus_fixed_cost"] = row["selected_terminal_cost"] - row["fixed_terminal_cost"]
        row["selected_minus_continue_cost"] = row["selected_terminal_cost"] - row["continue_terminal_cost"]
        row["selected_minus_halt_cost"] = row["selected_terminal_cost"] - row["halt_terminal_cost"]
        rows.append(row)

    result: dict[str, Any] = {
        "method": method,
        "cases": len(rows),
        "exact_cause_accuracy": mean(float(row["exact_cause_correct"]) for row in rows),
        "action_equivalent_accuracy": mean(float(row["action_equivalent_correct"]) for row in rows),
        "selected_mean_terminal_cost": mean(row["selected_terminal_cost"] for row in rows),
        "selected_success_rate": mean(float(row["selected_success"]) for row in rows),
        "selected_fall_rate": mean(float(row["selected_fell"]) for row in rows),
    }
    for comparator in ("oracle", "fixed", "continue", "halt"):
        key = f"selected_minus_{comparator}_cost"
        result[key] = mean(row[key] for row in rows)
        result[f"{key}_scene_cluster_ci95"] = cluster_bootstrap_ci(
            rows, lambda row, name=key: float(row[name]), seed=2026081300 + METHODS.index(method) * 10 + len(comparator)
        )
        result[f"{key}_scene_signflip_one_sided_p"] = exact_scene_signflip_less(rows, key)
    result["success_vs_fixed_mcnemar_two_sided_p"] = mcnemar_exact_two_sided(
        [row["selected_success"] for row in rows],
        [row["fixed_success"] for row in rows],
    )
    result["fall_vs_fixed_mcnemar_two_sided_p"] = mcnemar_exact_two_sided(
        [not row["selected_fell"] for row in rows],
        [not row["fixed_fell"] for row in rows],
    )
    return result, rows


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--protocol", type=Path, required=True)
    parser.add_argument("--predictions", type=Path, required=True)
    parser.add_argument("--prediction-provenance", type=Path, required=True)
    parser.add_argument("--formal-analysis", type=Path, required=True)
    parser.add_argument("--formal-corpus", type=Path, required=True)
    parser.add_argument("--o6-analysis", type=Path, required=True)
    parser.add_argument("--o6-corpus", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()

    protocol_path = args.protocol.resolve()
    protocol = json.loads(protocol_path.read_text())
    if protocol.get("status") != "frozen_before_action_outcome_linkage":
        raise RuntimeError("analysis protocol was not frozen before outcome linkage")
    expected = protocol["source_sha256"]
    actual_paths = {
        "analysis_script": Path(__file__).resolve(),
        "prediction_script": ROOT / "scripts/score_kinofail_prediction_driven_action_v1.py",
        "formal_analysis": args.formal_analysis.resolve(),
        "o6_analysis": args.o6_analysis.resolve(),
    }
    for key, path in actual_paths.items():
        if sha256(path) != expected[key]:
            raise RuntimeError(f"protocol seal failed for {key}")

    provenance_path = args.prediction_provenance.resolve()
    provenance = json.loads(provenance_path.read_text())
    predictions_path = args.predictions.resolve()
    if (
        provenance.get("status") != "frozen_models_scored_once_on_primary_views"
        or sha256(predictions_path) != provenance["artifact"]["sha256"]
    ):
        raise RuntimeError("prediction provenance seal failed")
    prediction_rows = jsonl(predictions_path)
    predictions = {str(row["counterfactual_group_id"]): row for row in prediction_rows}
    if len(predictions) != 277:
        raise RuntimeError("prediction file does not contain 277 unique cases")

    cases = read_complete_cases(
        args.formal_analysis.resolve(), args.formal_corpus.resolve(), "O6_push"
    ) + read_complete_cases(args.o6_analysis.resolve(), args.o6_corpus.resolve())
    cases = sorted(cases, key=lambda row: str(row["case_id"]))
    if len(cases) != 277:
        raise RuntimeError("action corpus does not contain 277 complete cases")
    case_groups = {str(case["case_id"]).rsplit("__", 1)[-1] for case in cases}
    if case_groups != set(predictions):
        raise RuntimeError("prediction/action join is not one-to-one")
    for case in cases:
        group = str(case["case_id"]).rsplit("__", 1)[-1]
        if CLASS_TO_ACTION[str(predictions[group]["truth"])] != OPERATOR_TO_ACTION[str(case["operator"])]:
            raise RuntimeError(f"cause/action ontology mismatch: {case['case_id']}")

    fixed_by_scene = fixed_action_loso(cases)
    results: list[dict[str, Any]] = []
    case_rows: list[dict[str, Any]] = []
    for method in METHODS:
        result, rows = evaluate_method(method, cases, predictions, fixed_by_scene)
        results.append(result)
        case_rows.extend(rows)
    adjusted = holm(
        {row["method"]: float(row["selected_minus_fixed_cost_scene_signflip_one_sided_p"]) for row in results}
    )
    for row in results:
        row["selected_minus_fixed_cost_holm_10"] = adjusted[row["method"]]

    primary = next(row for row in results if row["method"] == PRIMARY_METHOD)
    primary_rows = [row for row in case_rows if row["method"] == PRIMARY_METHOD]
    operator_primary: list[dict[str, Any]] = []
    for operator in sorted(OPERATOR_TO_ACTION):
        selected = [row for row in primary_rows if row["operator"] == operator]
        operator_primary.append(
            {
                "operator": operator,
                "cases": len(selected),
                "exact_cause_accuracy": mean(float(row["exact_cause_correct"]) for row in selected),
                "action_equivalent_accuracy": mean(float(row["action_equivalent_correct"]) for row in selected),
                "selected_success_rate": mean(float(row["selected_success"]) for row in selected),
                "selected_fall_rate": mean(float(row["selected_fell"]) for row in selected),
                "selected_minus_fixed_cost": mean(row["selected_minus_fixed_cost"] for row in selected),
                "selected_minus_oracle_cost": mean(row["selected_minus_oracle_cost"] for row in selected),
            }
        )

    fixed_summary = {
        scene: family for scene, family in sorted(fixed_by_scene.items())
    }
    report = {
        "schema_version": "kinofail.prediction-driven-action-analysis.v1",
        "created_utc": datetime.now(UTC).isoformat(),
        "status": "complete",
        "publication_evidence_eligible": True,
        "analysis_unit": "one physical case; all eleven action branches begin at the identical checkpoint",
        "policy_input": "one primary RGB appearance view plus its registered proprioceptive decision window",
        "prediction_to_action_uses_true_label": False,
        "counts": {
            "physical_cases": len(cases),
            "action_episodes": len(cases) * 11,
            "scenes": len({str(case["scene_id"]) for case in cases}),
            "operators": len({str(case["operator"]) for case in cases}),
            "methods": len(METHODS),
        },
        "primary_method": {
            "name": PRIMARY_METHOD,
            "selection_rule": "pre-specified as the highest worst-battery baseline in the already reported attribution table",
            "result": primary,
        },
        "comparators": {
            "oracle_cause": (
                "registered action selected from the true cause; true-cause registry "
                "reference, not a minimum-cost action oracle"
            ),
            "continue": "unchanged locomotion command",
            "safe_halt": "single conservative halt action",
            "fixed_action_loso": "one action family chosen on the other eleven scenes and applied without cause information to the held-out scene",
            "fixed_action_by_held_out_scene": fixed_summary,
        },
        "method_results": results,
        "primary_method_by_operator": operator_primary,
        "statistics": {
            "confidence_intervals": "20,000-draw scene-cluster bootstrap",
            "cost_tests": "exact one-sided sign-flip over twelve scene-mean paired differences",
            "multiplicity": "Holm adjustment across ten attribution methods for the prediction-selected versus LOSO fixed-action contrast",
            "binary_sensitivity": "two-sided exact McNemar test over paired physical cases",
        },
        "source_sha256": {
            "protocol": sha256(protocol_path),
            "predictions": sha256(predictions_path),
            "prediction_provenance": sha256(provenance_path),
            "formal_analysis": sha256(args.formal_analysis.resolve()),
            "o6_analysis": sha256(args.o6_analysis.resolve()),
            "analysis_script": sha256(Path(__file__).resolve()),
        },
    }

    output = args.output_dir.resolve()
    report_path = output / "prediction_driven_action_analysis.json"
    method_csv = output / "prediction_driven_method_results.csv"
    case_csv = output / "prediction_driven_case_results.csv"
    if report_path.exists() or method_csv.exists() or case_csv.exists():
        raise FileExistsError("prediction-driven action analysis is score-once")
    output.mkdir(parents=True, exist_ok=True)
    report_path.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    with method_csv.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(results[0]))
        writer.writeheader()
        writer.writerows(results)
    with case_csv.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(case_rows[0]))
        writer.writeheader()
        writer.writerows(case_rows)
    print(json.dumps({"primary_method": PRIMARY_METHOD, **primary}, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
