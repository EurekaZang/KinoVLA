#!/usr/bin/env python3
"""Score the frozen expanded A4 label-swap matrix once against F36 policy outputs."""

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
SEAL = ROOT / "outputs/freeze/kinofail_reconfirmation_a4_v7/seal_manifest.json"
COLLECTION = ROOT / "outputs/kinofail_reconfirmation_a4_v7/run/final_audit.json"
CORPUS = Path("/data/eureka/KinoVLA/outputs/kinofail_reconfirmation_a4_v7/corpus")
F36 = ROOT / "outputs/eval/unified_moe_v3_reconfirmation_v2_direct_o9_f36"
BOOTSTRAP_DRAWS = 20_000
BOOTSTRAP_SEED = 2_027_014_401
FALL_NI_MARGIN = 0.02
RECOVERY_SUCCESS_LOWER_BOUND = 0.75


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


def clustered_interval(
    rows: list[dict[str, Any]], field: str, *, seed: int
) -> dict[str, Any]:
    by_scene: dict[str, list[float]] = defaultdict(list)
    for row in rows:
        by_scene[str(row["scene_id"])].append(float(row[field]))
    scenes = sorted(by_scene)
    scene_means = np.asarray(
        [np.mean(by_scene[scene]) for scene in scenes], dtype=np.float64
    )
    rng = np.random.default_rng(seed)
    sampled = rng.integers(0, len(scenes), size=(BOOTSTRAP_DRAWS, len(scenes)))
    values = scene_means[sampled].mean(axis=1)
    return {
        "estimate": float(scene_means.mean()),
        "ci95": [float(np.quantile(values, 0.025)), float(np.quantile(values, 0.975))],
        "one_sided_97_5_percent_lower_bound": float(np.quantile(values, 0.025)),
        "one_sided_97_5_percent_upper_bound": float(np.quantile(values, 0.975)),
        "one_sided_99_percent_lower_bound": float(np.quantile(values, 0.01)),
        "one_sided_99_percent_upper_bound": float(np.quantile(values, 0.99)),
        "scene_cluster_count": len(scenes),
        "physical_case_count": len(rows),
        "bootstrap_draws": BOOTSTRAP_DRAWS,
        "bootstrap_seed": seed,
    }


def stable_majority(values: list[str], group_id: str) -> str:
    counts = Counter(values)
    best = max(counts.values())
    tied = [label for label, count in counts.items() if count == best]
    return min(
        tied,
        key=lambda label: hashlib.sha256(f"{group_id}|{label}".encode()).hexdigest(),
    )


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--predictions",
        type=Path,
        default=F36 / "blind_predictions/blind_predictions.jsonl",
    )
    parser.add_argument(
        "--truth", type=Path, default=F36 / "blind_bundle/truth_key.jsonl"
    )
    parser.add_argument(
        "--out", type=Path, default=F36 / "a4_actual_action_policy_report.json"
    )
    args = parser.parse_args()
    predictions_path, truth_path, out = (
        args.predictions.resolve(),
        args.truth.resolve(),
        args.out.resolve(),
    )
    for path in (SEAL, COLLECTION, predictions_path, truth_path):
        if not path.is_file():
            raise FileNotFoundError(path)
    if out.exists():
        raise FileExistsError(out)
    seal = load(SEAL)
    sidecar = SEAL.with_name("seal_manifest.sha256")
    collection = load(COLLECTION)
    if (
        sidecar.read_text().split()[0] != sha256(SEAL)
        or seal.get("status")
        != "sealed_before_f36_prediction_and_confirmatory_a4_outcomes"
        or seal.get("analyzer_sha256") != sha256(Path(__file__).resolve())
        or collection.get("passed") is not True
        or collection.get("result_dependent_retry_or_selection") is not False
    ):
        raise RuntimeError("A4-v7 seal or collection audit is invalid")

    results: list[dict[str, Any]] = []
    audits: list[dict[str, Any]] = []
    artifact_hashes: dict[str, str] = {}
    for path in sorted(CORPUS.glob("*/*/results.jsonl")):
        artifact_hashes[str(path)] = sha256(path)
        results.extend(jsonl(path))
    for path in sorted(CORPUS.glob("*/*/case_audits.jsonl")):
        artifact_hashes[str(path)] = sha256(path)
        audits.extend(jsonl(path))
    if len(results) != 2100 or len(audits) != 300:
        raise RuntimeError(f"A4-v7 is incomplete: results={len(results)}, audits={len(audits)}")
    if len({(row["case_id"], row["action"]) for row in results}) != 2100:
        raise RuntimeError("A4-v7 result keys are not unique")
    if len({row["case_id"] for row in audits}) != 300:
        raise RuntimeError("A4-v7 case audits are not unique")
    for row in results:
        trace = ROOT / str(row["trace"])
        if not trace.is_file() or sha256(trace) != row["trace_sha256"]:
            raise RuntimeError(f"A4 trace hash mismatch: {trace}")

    truth = {
        str(row["sample_id"]): row
        for row in jsonl(truth_path)
        if row.get("dataset") == "scale" and bool(row.get("valid", True))
    }
    primary_anomaly_by_group: dict[str, dict[str, Any]] = {}
    for row in truth.values():
        if row.get("condition") == "anomaly" and row.get("appearance_view_id") == "primary":
            group = str(row["group_id"])
            if group in primary_anomaly_by_group:
                raise RuntimeError(f"duplicate A4 primary anomaly sample: {group}")
            primary_anomaly_by_group[group] = row
    predictions_by_sample: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in jsonl(predictions_path):
        if row.get("method") == "learned_router":
            predictions_by_sample[str(row["sample_id"])].append(row)

    action_map = dict(seal["prediction_to_action"])
    fallback = str(action_map.pop("all_other_classes"))
    by_case: dict[str, dict[str, dict[str, Any]]] = defaultdict(dict)
    for row in results:
        by_case[str(row["case_id"])][str(row["action"])] = row
    paired: list[dict[str, Any]] = []
    policy_counts: Counter[str] = Counter()
    for case_id, arms in sorted(by_case.items()):
        if set(arms) != set(seal["action_arms"]):
            raise RuntimeError(f"A4 arm coverage mismatch: {case_id}")
        exemplar = next(iter(arms.values()))
        group_id = str(exemplar["source_counterfactual_group_id"])
        truth_row = primary_anomaly_by_group[group_id]
        exact_source_binding = (
            str(exemplar["source_f35_decision"]["sample_id"])
            == str(truth_row["sample_id"])
            and all(
                row["source_physical_nuisance"]
                == exemplar["source_physical_nuisance"]
                and row["source_f35_decision"] == exemplar["source_f35_decision"]
                and abs(float(row["decision_time_skew_s"])) <= 0.021
                and row.get("source_proprio_replay_certificate", {}).get(
                    "passed"
                )
                is True
                and float(
                    row["source_proprio_replay_certificate"][
                        "maximum_tolerance_normalized_difference"
                    ]
                )
                <= 1.0
                for row in arms.values()
            )
        )
        votes = predictions_by_sample[str(truth_row["sample_id"])]
        if len(votes) != 5 or {str(row["checkpoint_id"]) for row in votes} != {
            f"seed{index}" for index in range(5)
        }:
            raise RuntimeError(f"A4 prediction coverage mismatch: {group_id}")
        predicted_class = stable_majority(
            [str(row["prediction"]) for row in votes], group_id
        )
        selected_action = str(action_map.get(predicted_class, fallback))
        policy_counts[selected_action] += 1
        true_operator = str(exemplar["true_operator"])
        correct_action = f"recover_as_{true_operator}"
        selected, safe, continued, correct = (
            arms[selected_action],
            arms["always_safe_halt"],
            arms["continue"],
            arms[correct_action],
        )
        wrong_recoveries = [
            row
            for action, row in arms.items()
            if action.startswith("recover_as_") and action != correct_action
        ]
        if len(wrong_recoveries) != 4:
            raise RuntimeError(f"A4 wrong-recovery coverage mismatch: {case_id}")
        paired.append(
            {
                "case_id": case_id,
                "scene_id": str(exemplar["scene_id"]),
                "domain": str(exemplar["domain"]),
                "true_operator": true_operator,
                "severity_id": str(exemplar["severity_id"]),
                "source_counterfactual_group_id": group_id,
                "predicted_class": predicted_class,
                "selected_action": selected_action,
                "exact_f36_source_prefix_binding": exact_source_binding,
                "attribution_correct": predicted_class == str(truth_row["truth"]),
                "correct_recovery_minus_continue_cost": float(correct["terminal_cost"])
                - float(continued["terminal_cost"]),
                "correct_recovery_minus_continue_fall": float(correct["fell"])
                - float(continued["fell"]),
                "correct_recovery_success": float(
                    correct["operator_recovery_success"]
                ),
                "correct_recovery_minus_continue_success": float(
                    correct["operator_recovery_success"]
                )
                - float(continued["operator_recovery_success"]),
                "correct_recovery_minus_mean_wrong_cost": float(
                    correct["terminal_cost"]
                )
                - float(np.mean([row["terminal_cost"] for row in wrong_recoveries])),
                "correct_recovery_minus_mean_wrong_success": float(
                    correct["operator_recovery_success"]
                )
                - float(
                    np.mean(
                        [row["operator_recovery_success"] for row in wrong_recoveries]
                    )
                ),
                "selective_minus_always_safe_cost": float(selected["terminal_cost"])
                - float(safe["terminal_cost"]),
                "selective_minus_always_safe_fall": float(selected["fell"])
                - float(safe["fell"]),
                "selective_minus_always_safe_success": float(
                    selected["operator_recovery_success"]
                )
                - float(safe["operator_recovery_success"]),
                "selective_minus_continue_cost": float(selected["terminal_cost"])
                - float(continued["terminal_cost"]),
                "selective_minus_continue_fall": float(selected["fell"])
                - float(continued["fell"]),
                "selective_minus_continue_success": float(
                    selected["operator_recovery_success"]
                )
                - float(continued["operator_recovery_success"]),
                "selected": {
                    key: selected[key]
                    for key in (
                        "action",
                        "fell",
                        "operator_recovery_success",
                        "operator_recovery_endpoint_type",
                        "terminal_cost",
                    )
                },
                "always_safe": {
                    key: safe[key]
                    for key in (
                        "action",
                        "fell",
                        "operator_recovery_success",
                        "operator_recovery_endpoint_type",
                        "terminal_cost",
                    )
                },
            }
        )

    recovery_cost = clustered_interval(
        paired, "correct_recovery_minus_continue_cost", seed=BOOTSTRAP_SEED
    )
    selective_cost = clustered_interval(
        paired, "selective_minus_always_safe_cost", seed=BOOTSTRAP_SEED + 1
    )
    selective_fall = clustered_interval(
        paired, "selective_minus_always_safe_fall", seed=BOOTSTRAP_SEED + 2
    )
    selective_success = clustered_interval(
        paired, "selective_minus_always_safe_success", seed=BOOTSTRAP_SEED + 3
    )
    recovery_success = clustered_interval(
        paired, "correct_recovery_success", seed=BOOTSTRAP_SEED + 4
    )
    recovery_success_gain = clustered_interval(
        paired,
        "correct_recovery_minus_continue_success",
        seed=BOOTSTRAP_SEED + 5,
    )
    attribution_specific_cost = clustered_interval(
        paired,
        "correct_recovery_minus_mean_wrong_cost",
        seed=BOOTSTRAP_SEED + 9,
    )
    attribution_specific_success = clustered_interval(
        paired,
        "correct_recovery_minus_mean_wrong_success",
        seed=BOOTSTRAP_SEED + 10,
    )
    selective_continue_cost = clustered_interval(
        paired, "selective_minus_continue_cost", seed=BOOTSTRAP_SEED + 6
    )
    selective_continue_fall = clustered_interval(
        paired, "selective_minus_continue_fall", seed=BOOTSTRAP_SEED + 7
    )
    selective_continue_success = clustered_interval(
        paired, "selective_minus_continue_success", seed=BOOTSTRAP_SEED + 8
    )
    operator_cells: dict[str, Any] = {}
    positive_operators = 0
    harmed_operators = 0
    for index, operator in enumerate(sorted({row["true_operator"] for row in paired})):
        rows = [row for row in paired if row["true_operator"] == operator]
        endpoint = clustered_interval(
            rows,
            "correct_recovery_minus_mean_wrong_cost",
            seed=BOOTSTRAP_SEED + 100 + index,
        )
        continue_endpoint = clustered_interval(
            rows,
            "correct_recovery_minus_continue_cost",
            seed=BOOTSTRAP_SEED + 200 + index,
        )
        # Five operator-level benefit checks use Bonferroni alpha=.05/5=.01.
        positive = endpoint["one_sided_99_percent_upper_bound"] < 0.0
        harmed = endpoint["one_sided_99_percent_lower_bound"] > 0.0
        positive_operators += int(positive)
        harmed_operators += int(harmed)
        operator_cells[operator] = {
            "correct_recovery_minus_mean_wrong_cost": endpoint,
            "correct_recovery_minus_continue_cost": continue_endpoint,
            "identified_benefit": positive,
            "identified_harm": harmed,
        }

    acceptance = {
        "all_30_scenes_present": len({row["scene_id"] for row in paired}) == 30,
        "all_300_cases_and_2100_episodes_present": len(paired) == 300
        and len(results) == 2100,
        "all_matched_prefixes_passed": all(row.get("passed") is True for row in audits),
        "all_frozen_f35_proprio_observations_replayed": all(
            row.get("source_proprio_replay_passed") is True for row in audits
        ),
        "all_five_recovery_numeric_trajectories_distinct_in_every_case": all(
            row.get("five_recovery_numeric_trajectories_distinct") is True
            and int(row.get("distinct_recovery_numeric_trace_count", 0)) == 5
            for row in audits
        ),
        "all_cases_bind_exact_f36_source_seed_nuisance_and_decision": all(
            row["exact_f36_source_prefix_binding"] for row in paired
        ),
        "all_unfavorable_outcomes_retained": all(
            row.get("unfavorable_outcome_retained") is True for row in results
        ),
        "recovery_vs_continue_cost_ci_upper_below_zero": recovery_cost["ci95"][1]
        < 0.0,
        "correct_recovery_beats_mean_wrong_recovery_cost_ci_upper_below_zero": attribution_specific_cost[
            "ci95"
        ][1]
        < 0.0,
        "correct_recovery_success_lower_bound_above_0_75": recovery_success[
            "one_sided_97_5_percent_lower_bound"
        ]
        > RECOVERY_SUCCESS_LOWER_BOUND,
        "at_least_three_of_five_operator_recoveries_identified": positive_operators >= 3,
        "no_operator_recovery_identified_as_harmful": harmed_operators == 0,
        "selective_vs_always_safe_cost_ci_upper_below_zero": selective_cost["ci95"][1]
        < 0.0,
        "selective_fall_rate_noninferior_to_always_safe": selective_fall[
            "one_sided_97_5_percent_upper_bound"
        ]
        <= FALL_NI_MARGIN,
        "selective_vs_continue_cost_ci_upper_below_zero": selective_continue_cost[
            "ci95"
        ][1]
        < 0.0,
        "selective_fall_rate_noninferior_to_continue": selective_continue_fall[
            "one_sided_97_5_percent_upper_bound"
        ]
        <= FALL_NI_MARGIN,
    }
    report = {
        "schema_version": "kinofail.reconfirmation-a4-v7-report.v1",
        "created_utc": datetime.now(UTC).isoformat(),
        "status": "confirmatory_passed" if all(acceptance.values()) else "confirmatory_gate_failed",
        "passed": all(acceptance.values()),
        "physical_episode_count": len(results),
        "paired_case_count": len(paired),
        "scene_cluster_count": len({row["scene_id"] for row in paired}),
        "domains": sorted({row["domain"] for row in paired}),
        "policy_action_counts": dict(sorted(policy_counts.items())),
        "attribution_accuracy_on_action_cases": float(
            np.mean([row["attribution_correct"] for row in paired])
        ),
        "primary_endpoints": {
            "correct_recovery_minus_continue_cost": recovery_cost,
            "correct_recovery_success": recovery_success,
            "correct_recovery_minus_continue_success": recovery_success_gain,
            "correct_recovery_minus_mean_wrong_cost": attribution_specific_cost,
            "correct_recovery_minus_mean_wrong_success": attribution_specific_success,
            "selective_policy_minus_always_safe_cost": selective_cost,
            "selective_policy_minus_always_safe_fall": selective_fall,
            "selective_policy_minus_always_safe_success": selective_success,
            "selective_policy_minus_continue_cost": selective_continue_cost,
            "selective_policy_minus_continue_fall": selective_continue_fall,
            "selective_policy_minus_continue_success": selective_continue_success,
        },
        "operator_cells": operator_cells,
        "identified_benefit_operator_count": positive_operators,
        "identified_harm_operator_count": harmed_operators,
        "acceptance": acceptance,
        "paired_cases": paired,
        "statistical_contract": seal["statistics"],
        "source_sha256": {
            "a4_seal": sha256(SEAL),
            "a4_collection_audit": sha256(COLLECTION),
            "f36_blind_predictions": sha256(predictions_path),
            "f36_truth_key": sha256(truth_path),
            "result_ledgers": artifact_hashes,
            "analysis_script": sha256(Path(__file__).resolve()),
        },
        "claim_boundary": (
            "The selective policy uses only the frozen F36 learned-router vote; the true "
            "operator chooses neither its action nor its fallback. All five recovery labels "
            "were physically executed for every case, so wrong attributions retain their "
            "measured consequence rather than receiving an oracle action."
        ),
    }
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n")
    print(json.dumps(report, indent=2, sort_keys=True))
    return 0 if report["passed"] else 2


if __name__ == "__main__":
    raise SystemExit(main())
