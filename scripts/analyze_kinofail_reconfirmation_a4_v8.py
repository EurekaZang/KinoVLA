#!/usr/bin/env python3
"""Score the sealed A4-v8 supported-recovery matrix once using F36 outputs."""

from __future__ import annotations

import argparse
import hashlib
import json
from collections import Counter, defaultdict
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import numpy as np


ROOT = Path(__file__).resolve().parents[1]
SEAL = ROOT / "outputs/freeze/kinofail_reconfirmation_a4_v8/seal_manifest.json"
COLLECTION = ROOT / "outputs/kinofail_reconfirmation_a4_v8/run/final_audit.json"
CORPUS = Path("/data/eureka/KinoVLA/outputs/kinofail_reconfirmation_a4_v8/corpus")
F36 = ROOT / "outputs/eval/unified_moe_v3_reconfirmation_v2_direct_o9_f36"
BOOTSTRAP_DRAWS = 20_000
BOOTSTRAP_SEED = 2_027_014_801


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def load(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text())
    if not isinstance(value, dict):
        raise TypeError(path)
    return value


def jsonl(path: Path) -> list[dict[str, Any]]:
    return [json.loads(line) for line in path.read_text().splitlines() if line]


def clustered_interval(rows: list[dict[str, Any]], field: str, *, seed: int) -> dict[str, Any]:
    by_scene: dict[str, list[float]] = defaultdict(list)
    for row in rows:
        by_scene[str(row["scene_id"])].append(float(row[field]))
    scenes = sorted(by_scene)
    means = np.asarray([np.mean(by_scene[scene]) for scene in scenes], dtype=np.float64)
    rng = np.random.default_rng(seed)
    sampled = rng.integers(0, len(scenes), size=(BOOTSTRAP_DRAWS, len(scenes)))
    distribution = means[sampled].mean(axis=1)
    return {
        "estimate": float(means.mean()),
        "ci95": [float(np.quantile(distribution, 0.025)), float(np.quantile(distribution, 0.975))],
        "one_sided_97_5_percent_lower_bound": float(np.quantile(distribution, 0.025)),
        "one_sided_97_5_percent_upper_bound": float(np.quantile(distribution, 0.975)),
        "scene_cluster_count": len(scenes),
        "physical_case_count": len(rows),
        "bootstrap_draws": BOOTSTRAP_DRAWS,
        "bootstrap_seed": seed,
    }


def stable_majority(values: list[str], group_id: str) -> str:
    counts = Counter(values)
    best = max(counts.values())
    tied = [label for label, count in counts.items() if count == best]
    return min(tied, key=lambda label: hashlib.sha256(f"{group_id}|{label}".encode()).hexdigest())


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--predictions", type=Path, default=F36 / "blind_predictions/blind_predictions.jsonl")
    parser.add_argument("--truth", type=Path, default=F36 / "blind_bundle/truth_key.jsonl")
    parser.add_argument("--out", type=Path, default=F36 / "a4_actual_action_policy_report_v8.json")
    args = parser.parse_args()
    predictions_path = args.predictions.resolve()
    truth_path = args.truth.resolve()
    out = args.out.resolve()
    for path in (SEAL, COLLECTION, predictions_path, truth_path):
        if not path.is_file():
            raise FileNotFoundError(path)
    if out.exists():
        raise FileExistsError(out)
    seal = load(SEAL)
    collection = load(COLLECTION)
    if (
        SEAL.with_name("seal_manifest.sha256").read_text().split()[0] != sha256(SEAL)
        or seal.get("analyzer_sha256") != sha256(Path(__file__).resolve())
        or collection.get("passed") is not True
        or collection.get("result_dependent_retry_or_selection") is not False
    ):
        raise RuntimeError("invalid A4-v8 seal or collection")

    results = [row for path in sorted(CORPUS.glob("*/*/results.jsonl")) for row in jsonl(path)]
    audits = [row for path in sorted(CORPUS.glob("*/*/case_audits.jsonl")) for row in jsonl(path)]
    if len(results) != 350 or len(audits) != 50:
        raise RuntimeError(f"A4-v8 incomplete: results={len(results)} audits={len(audits)}")
    if len({(row["case_id"], row["action"]) for row in results}) != 350:
        raise RuntimeError("A4-v8 duplicate result key")
    for row in results:
        trace = Path(str(row["trace"]))
        if not trace.is_file() or sha256(trace) != row["trace_sha256"]:
            raise RuntimeError(f"A4-v8 trace hash mismatch: {trace}")

    truth = {
        str(row["sample_id"]): row
        for row in jsonl(truth_path)
        if row.get("dataset") == "scale" and bool(row.get("valid", True))
    }
    primary = {}
    for row in truth.values():
        if row.get("condition") == "anomaly" and row.get("appearance_view_id") == "primary":
            group_id = str(row["group_id"])
            if group_id in primary:
                raise RuntimeError(f"duplicate primary anomaly: {group_id}")
            primary[group_id] = row
    predictions: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in jsonl(predictions_path):
        if row.get("method") == "learned_router":
            predictions[str(row["sample_id"])].append(row)

    by_case: dict[str, dict[str, dict[str, Any]]] = defaultdict(dict)
    for row in results:
        by_case[str(row["case_id"])][str(row["action"])] = row
    action_map = dict(seal["prediction_to_action"])
    fallback = str(action_map.pop("all_other_classes"))
    paired = []
    action_counts: Counter[str] = Counter()
    for case_id, arms in sorted(by_case.items()):
        if set(arms) != set(seal["action_arms"]):
            raise RuntimeError(f"arm mismatch: {case_id}")
        exemplar = next(iter(arms.values()))
        group_id = str(exemplar["source_counterfactual_group_id"])
        truth_row = primary[group_id]
        votes = predictions[str(truth_row["sample_id"])]
        if len(votes) != 5 or {str(row["checkpoint_id"]) for row in votes} != {f"seed{i}" for i in range(5)}:
            raise RuntimeError(f"prediction coverage mismatch: {group_id}")
        predicted = stable_majority([str(row["prediction"]) for row in votes], group_id)
        selected_action = str(action_map.get(predicted, fallback))
        action_counts[selected_action] += 1
        true_operator = str(exemplar["true_operator"])
        correct_action = f"recover_as_{true_operator}"
        selected = arms[selected_action]
        correct = arms[correct_action]
        continued = arms["continue"]
        safe = arms["always_safe_halt"]
        wrong = [
            row for action, row in arms.items()
            if action.startswith("recover_as_") and action != correct_action
        ]
        exact_binding = (
            str(exemplar["source_f35_decision"]["sample_id"]) == str(truth_row["sample_id"])
            and all(
                row["source_physical_nuisance"] == exemplar["source_physical_nuisance"]
                and row["source_f35_decision"] == exemplar["source_f35_decision"]
                and row.get("source_proprio_replay_certificate", {}).get("passed") is True
                for row in arms.values()
            )
        )
        paired.append(
            {
                "case_id": case_id,
                "scene_id": str(exemplar["scene_id"]),
                "domain": str(exemplar["domain"]),
                "true_operator": true_operator,
                "source_counterfactual_group_id": group_id,
                "predicted_class": predicted,
                "selected_action": selected_action,
                "attribution_correct": predicted == str(truth_row["truth"]),
                "exact_f36_source_binding": exact_binding,
                "correct_success": float(correct["operator_recovery_success"]),
                "correct_minus_mean_wrong_cost": float(correct["terminal_cost"]) - float(np.mean([row["terminal_cost"] for row in wrong])),
                "correct_minus_mean_wrong_success": float(correct["operator_recovery_success"]) - float(np.mean([row["operator_recovery_success"] for row in wrong])),
                "selective_minus_safe_cost": float(selected["terminal_cost"]) - float(safe["terminal_cost"]),
                "selective_minus_safe_fall": float(selected["fell"]) - float(safe["fell"]),
                "selective_minus_safe_success": float(selected["operator_recovery_success"]) - float(safe["operator_recovery_success"]),
                "selective_minus_continue_cost": float(selected["terminal_cost"]) - float(continued["terminal_cost"]),
                "selective_minus_continue_fall": float(selected["fell"]) - float(continued["fell"]),
                "selective_minus_continue_success": float(selected["operator_recovery_success"]) - float(continued["operator_recovery_success"]),
            }
        )

    metrics = {
        "correct_recovery_success": clustered_interval(paired, "correct_success", seed=BOOTSTRAP_SEED),
        "correct_recovery_minus_mean_wrong_cost": clustered_interval(paired, "correct_minus_mean_wrong_cost", seed=BOOTSTRAP_SEED + 1),
        "correct_recovery_minus_mean_wrong_success": clustered_interval(paired, "correct_minus_mean_wrong_success", seed=BOOTSTRAP_SEED + 2),
        "selective_policy_minus_always_safe_cost": clustered_interval(paired, "selective_minus_safe_cost", seed=BOOTSTRAP_SEED + 3),
        "selective_policy_minus_always_safe_fall": clustered_interval(paired, "selective_minus_safe_fall", seed=BOOTSTRAP_SEED + 4),
        "selective_policy_minus_always_safe_success": clustered_interval(paired, "selective_minus_safe_success", seed=BOOTSTRAP_SEED + 5),
        "selective_policy_minus_continue_cost": clustered_interval(paired, "selective_minus_continue_cost", seed=BOOTSTRAP_SEED + 6),
        "selective_policy_minus_continue_fall": clustered_interval(paired, "selective_minus_continue_fall", seed=BOOTSTRAP_SEED + 7),
        "selective_policy_minus_continue_success": clustered_interval(paired, "selective_minus_continue_success", seed=BOOTSTRAP_SEED + 8),
    }
    operator_cells = {}
    positive = 0
    harmful = 0
    for index, operator in enumerate(sorted({row["true_operator"] for row in paired})):
        rows = [row for row in paired if row["true_operator"] == operator]
        endpoint = clustered_interval(rows, "correct_minus_mean_wrong_cost", seed=BOOTSTRAP_SEED + 100 + index)
        identified_benefit = endpoint["one_sided_97_5_percent_upper_bound"] < 0.0
        identified_harm = endpoint["one_sided_97_5_percent_lower_bound"] > 0.0
        positive += int(identified_benefit)
        harmful += int(identified_harm)
        operator_cells[operator] = {
            "correct_recovery_minus_mean_wrong_cost": endpoint,
            "identified_benefit_bonferroni_alpha_0_025": identified_benefit,
            "identified_harm_bonferroni_alpha_0_025": identified_harm,
        }

    acceptance = {
        "all_25_unexposed_scenes_present": len({row["scene_id"] for row in paired}) == 25,
        "all_50_cases_and_350_episodes_present": len(paired) == 50 and len(results) == 350,
        "all_matched_prefix_and_replay_audits_pass": all(row.get("passed") is True and row.get("source_proprio_replay_passed") is True for row in audits),
        "all_five_recovery_trajectories_distinct": all(row.get("five_recovery_numeric_trajectories_distinct") is True for row in audits),
        "all_cases_bind_exact_f36_source": all(row["exact_f36_source_binding"] for row in paired),
        "all_unfavorable_outcomes_retained": all(row.get("unfavorable_outcome_retained") is True for row in results),
        "attribution_accuracy_at_least_0_90": float(np.mean([row["attribution_correct"] for row in paired])) >= 0.90,
        "correct_recovery_success_lower_bound_above_0_65": metrics["correct_recovery_success"]["one_sided_97_5_percent_lower_bound"] > 0.65,
        "correct_recovery_beats_mean_wrong_cost": metrics["correct_recovery_minus_mean_wrong_cost"]["ci95"][1] < 0.0,
        "at_least_one_of_two_operator_benefits_identified": positive >= 1,
        "no_supported_operator_identified_harmful": harmful == 0,
        "selective_beats_always_safe_cost": metrics["selective_policy_minus_always_safe_cost"]["ci95"][1] < 0.0,
        "selective_fall_noninferior_to_always_safe": metrics["selective_policy_minus_always_safe_fall"]["one_sided_97_5_percent_upper_bound"] <= 0.05,
        "selective_beats_continue_cost": metrics["selective_policy_minus_continue_cost"]["ci95"][1] < 0.0,
        "selective_fall_noninferior_to_continue": metrics["selective_policy_minus_continue_fall"]["one_sided_97_5_percent_upper_bound"] <= 0.05,
    }
    accuracy = float(np.mean([row["attribution_correct"] for row in paired]))
    report = {
        "schema_version": "kinofail.reconfirmation-a4-v8-report.v1",
        "created_utc": datetime.now(UTC).isoformat(),
        "status": "confirmatory_passed" if all(acceptance.values()) else "confirmatory_gate_failed",
        "passed": all(acceptance.values()),
        "physical_episode_count": len(results),
        "paired_case_count": len(paired),
        "scene_cluster_count": len({row["scene_id"] for row in paired}),
        "domains": sorted({row["domain"] for row in paired}),
        "supported_operators": sorted({row["true_operator"] for row in paired}),
        "attribution_accuracy_on_action_cases": accuracy,
        "policy_action_counts": dict(sorted(action_counts.items())),
        "primary_endpoints": metrics,
        "operator_cells": operator_cells,
        "identified_benefit_operator_count": positive,
        "identified_harm_operator_count": harmful,
        "acceptance": acceptance,
        "paired_cases": paired,
        "statistical_contract": seal["statistics"],
        "source_sha256": {
            "a4_seal": sha256(SEAL),
            "a4_collection_audit": sha256(COLLECTION),
            "f36_blind_predictions": sha256(predictions_path),
            "f36_truth_key": sha256(truth_path),
            "analysis_script": sha256(Path(__file__).resolve()),
        },
        "claim_boundary": (
            "This confirmatory action study supports only O4 adhesion and O5 overload, "
            "the two recovery cells executable by the frozen shared actor. O2, O8, and "
            "O9 remain part of the independent attribution battery but are explicitly "
            "excluded from autonomous-recovery capability claims."
        ),
    }
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n")
    print(json.dumps(report, indent=2, sort_keys=True))
    return 0 if report["passed"] else 2


if __name__ == "__main__":
    raise SystemExit(main())
