#!/usr/bin/env python3
"""Analyze the frozen realistic A4-v5 actual-action matrix."""

from __future__ import annotations

import argparse
import hashlib
import json
from collections import defaultdict
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import numpy as np


ROOT = Path(__file__).resolve().parents[1]
OPERATORS = (
    "O2_compliance",
    "O4_tether",
    "O5_payload",
    "O8_invisible_collider",
    "O9_high_centering",
)
RECOVERY_ACTION = {
    "O2_compliance": "slow_high_step",
    "O4_tether": "backstep_release",
    "O5_payload": "hold_and_request",
    "O8_invisible_collider": "backstep_detour_replan",
    "O9_high_centering": "raise_body_slow_cross",
}
PREFIX_SCALARS = (
    "time_s", "progress_m", "lateral_m", "heading_rad", "base_height_m",
    "tilt_rad", "slip_ratio", "effort_ratio", "support_ratio",
)
PREFIX_ARRAYS = ("position_xy_m", "velocity_body_mps", "command_body")


def _sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _load_rows(root: Path) -> tuple[list[dict[str, Any]], dict[str, str]]:
    rows: list[dict[str, Any]] = []
    hashes: dict[str, str] = {}
    for path in sorted(root.glob("*/results.jsonl")):
        hashes[str(path.relative_to(ROOT))] = _sha(path)
        rows.extend(json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line)
    return rows, hashes


def _prefix_audit(a: dict[str, Any], b: dict[str, Any]) -> dict[str, Any]:
    same_step = int(a["decision_step"]) == int(b["decision_step"])
    n = min(int(a["decision_step"]), int(b["decision_step"])) + 1
    maximum = 0.0
    by_field: dict[str, float] = {}
    for key in PREFIX_SCALARS:
        value = max(abs(float(a["telemetry_rows"][i][key]) - float(b["telemetry_rows"][i][key])) for i in range(n))
        by_field[key] = value
        maximum = max(maximum, value)
    for key in PREFIX_ARRAYS:
        value = max(
            float(np.max(np.abs(
                np.asarray(a["telemetry_rows"][i][key], dtype=np.float64)
                - np.asarray(b["telemetry_rows"][i][key], dtype=np.float64)
            )))
            for i in range(n)
        )
        by_field[key] = value
        maximum = max(maximum, value)
    both_shared = all(
        a["telemetry_rows"][i]["action_phase"] == "shared_prefix"
        and b["telemetry_rows"][i]["action_phase"] == "shared_prefix"
        for i in range(n)
    )
    return {
        "same_decision_step": same_step,
        "prefix_rows": n,
        "maximum_absolute_difference": maximum,
        "maximum_by_field": by_field,
        "both_actions_start_strictly_after_decision": both_shared,
        "passed": same_step and both_shared and maximum <= 0.001,
    }


def _cluster_bootstrap(
    pairs: list[dict[str, Any]], field: str, *, seed: int, draws: int = 20_000
) -> dict[str, Any]:
    by_scene: dict[str, list[float]] = defaultdict(list)
    for row in pairs:
        by_scene[row["scene_cluster"]].append(float(row[field]))
    scenes = sorted(by_scene)
    scene_means = np.asarray([np.mean(by_scene[scene]) for scene in scenes], dtype=np.float64)
    rng = np.random.default_rng(seed)
    sampled = rng.integers(0, len(scenes), size=(draws, len(scenes)))
    values = scene_means[sampled].mean(axis=1)
    estimate = float(scene_means.mean())
    lower, upper = np.quantile(values, [0.025, 0.975])
    return {
        "estimate": estimate,
        "scene_cluster_count": len(scenes),
        "case_count": len(pairs),
        "scene_cluster_means": {scene: float(np.mean(by_scene[scene])) for scene in scenes},
        "bootstrap_draws": draws,
        "bootstrap_seed": seed,
        "ci95": [float(lower), float(upper)],
        "two_sided_sign_p": float(min(1.0, 2.0 * min(np.mean(values <= 0.0), np.mean(values >= 0.0)))),
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--corpus-root", type=Path,
        default=ROOT / "outputs/kinofail_realistic/corpus_a4_actual_action_v5",
    )
    parser.add_argument(
        "--protocol", type=Path,
        default=ROOT / "configs/data/kinofail_realistic_a4_actual_action_formal_v5.json",
    )
    parser.add_argument(
        "--out", type=Path,
        default=ROOT / "outputs/eval/realistic_a0_a7_v4/a4_consequence.json",
    )
    args = parser.parse_args()
    corpus, protocol_path = args.corpus_root.resolve(), args.protocol.resolve()
    protocol = json.loads(protocol_path.read_text(encoding="utf-8"))
    a0_path = ROOT / "outputs/eval/realistic_a0_a7_v4/a0_certificate.json"
    a0 = json.loads(a0_path.read_text(encoding="utf-8"))
    rows, artifacts = _load_rows(corpus)
    by_case: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        by_case[row["case_id"]].append(row)
    pair_rows: list[dict[str, Any]] = []
    prefix_audits: dict[str, dict[str, Any]] = {}
    for case_id, group in sorted(by_case.items()):
        if len(group) != 2:
            raise RuntimeError(f"incomplete pair {case_id}: {len(group)}")
        by_action = {row["action"]: row for row in group}
        operator = group[0]["operator"]
        if set(by_action) != {"continue", RECOVERY_ACTION[operator]}:
            raise RuntimeError(f"wrong action arms: {case_id}")
        audit = _prefix_audit(by_action["continue"], by_action[RECOVERY_ACTION[operator]])
        prefix_audits[case_id] = audit
        pair_rows.append({
            "case_id": case_id,
            "scene_cluster": group[0]["scene_cluster"],
            "domain": group[0]["domain"],
            "operator": operator,
            "reset_seed": group[0]["reset_seed"],
            "recovery_action": RECOVERY_ACTION[operator],
            "prefix_passed": audit["passed"],
            "recovery_minus_continue_cost": (
                float(by_action[RECOVERY_ACTION[operator]]["terminal_cost"])
                - float(by_action["continue"]["terminal_cost"])
            ),
            "recovery_minus_continue_success": (
                float(by_action[RECOVERY_ACTION[operator]]["success"])
                - float(by_action["continue"]["success"])
            ),
            "continue": {
                key: by_action["continue"][key]
                for key in ("success", "fell", "terminal_cost", "final_progress_m")
            },
            "recovery": {
                key: by_action[RECOVERY_ACTION[operator]][key]
                for key in ("success", "fell", "terminal_cost", "final_progress_m")
            },
        })
    eligible = [row for row in pair_rows if row["prefix_passed"]]
    cells: dict[str, Any] = {}
    positive, harmful, weak = [], [], []
    for op_index, operator in enumerate(OPERATORS):
        op_pairs = [row for row in eligible if row["operator"] == operator]
        actions = [row for row in rows if row["operator"] == operator]
        continue_rows = [row for row in actions if row["action"] == "continue"]
        recovery_rows = [row for row in actions if row["action"] == RECOVERY_ACTION[operator]]
        cost = _cluster_bootstrap(op_pairs, "recovery_minus_continue_cost", seed=4100 + op_index)
        success = _cluster_bootstrap(op_pairs, "recovery_minus_continue_success", seed=5100 + op_index)
        if cost["ci95"][1] < 0.0:
            interpretation = "identified_recovery_cost_reduction"
            positive.append(operator)
        elif cost["ci95"][0] > 0.0:
            interpretation = "identified_recovery_harm"
            harmful.append(operator)
        else:
            interpretation = "weak_or_non_identifiable"
            weak.append(operator)
        cells[operator] = {
            "recovery_action": RECOVERY_ACTION[operator],
            "episode_seeds_per_action": len(continue_rows),
            "scene_cluster_count": len({row["scene_cluster"] for row in op_pairs}),
            "continue": {
                "success_rate": float(np.mean([row["success"] for row in continue_rows])),
                "fall_rate": float(np.mean([row["fell"] for row in continue_rows])),
                "mean_terminal_cost": float(np.mean([row["terminal_cost"] for row in continue_rows])),
            },
            "recovery": {
                "success_rate": float(np.mean([row["success"] for row in recovery_rows])),
                "fall_rate": float(np.mean([row["fell"] for row in recovery_rows])),
                "mean_terminal_cost": float(np.mean([row["terminal_cost"] for row in recovery_rows])),
            },
            "paired_recovery_minus_continue_cost": cost,
            "paired_recovery_minus_continue_success": success,
            "interpretation": interpretation,
            "all_unfavorable_outcomes_retained": True,
        }
    overall_cost = _cluster_bootstrap(eligible, "recovery_minus_continue_cost", seed=6100)
    overall_success = _cluster_bootstrap(eligible, "recovery_minus_continue_success", seed=7100)
    counts = {
        f"{operator}|{action}": sum(
            row["operator"] == operator and row["action"] == action for row in rows
        )
        for operator in OPERATORS
        for action in ("continue", RECOVERY_ACTION[operator])
    }
    acceptance = {
        "actual_action_required": all(
            any(row["action_phase"] not in {"shared_prefix", "continue"} for row in item["telemetry_rows"][int(item["decision_step"]) + 1 :])
            for item in rows if item["action"] != "continue"
        ),
        "minimum_episode_seeds_per_action_cell_10": min(counts.values()) >= 10,
        "required_headline_operators_present": set(OPERATORS) == {row["operator"] for row in rows},
        "scene_clustered_paired_bootstrap_present": all(
            cells[operator]["paired_recovery_minus_continue_cost"]["scene_cluster_count"] >= 2
            for operator in OPERATORS
        ),
        "weak_or_non_identifiable_cells_retained": set(weak) <= set(cells),
        "all_pairs_complete": len(rows) == 100 and len(pair_rows) == 50,
        "all_prefixes_within_frozen_tolerance": len(eligible) == len(pair_rows),
    }
    report = {
        "schema_version": "kinofail.realistic-a4-actual-action-report.v1",
        "created_utc": datetime.now(UTC).isoformat(),
        "status": "confirmatory_complete",
        "dataset_scope": "kinofail_realistic",
        "a0_evidence_bundle_sha256": a0["a0_evidence_bundle_sha256"],
        "a0_certificate_sha256": _sha(a0_path),
        "passed": all(acceptance.values()),
        "protocol": str(protocol_path.relative_to(ROOT)),
        "protocol_sha256": _sha(protocol_path),
        "protocol_id": protocol["protocol_id"],
        "corpus_root": str(corpus.relative_to(ROOT)),
        "input_artifacts_sha256": artifacts,
        "physical_episode_count": len(rows),
        "paired_case_count": len(pair_rows),
        "scene_cluster_count": len({row["scene_cluster"] for row in rows}),
        "domains": sorted({row["domain"] for row in rows}),
        "counts_by_operator_action": counts,
        "prefix_audit": {
            "tolerance": 0.001,
            "passed_pairs": len(eligible),
            "failed_pairs": len(pair_rows) - len(eligible),
            "maximum_absolute_difference": max(
                audit["maximum_absolute_difference"] for audit in prefix_audits.values()
            ),
            "per_pair": prefix_audits,
        },
        "cells": cells,
        "overall": {
            "paired_recovery_minus_continue_cost": overall_cost,
            "paired_recovery_minus_continue_success": overall_success,
        },
        "identified_cost_reduction_operators": positive,
        "identified_harm_operators": harmful,
        "weak_or_non_identifiable_operators": weak,
        "acceptance": acceptance,
        "claim_boundary": (
            "A4 measures the frozen actions actually executed in corrected realistic Isaac physics. "
            "Completeness is distinct from efficacy: negative, harmful, and non-identifiable cells "
            "are retained and must not be described as successful recovery."
        ),
        "paired_cases": pair_rows,
    }
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps({
        "passed": report["passed"],
        "physical_episode_count": len(rows),
        "prefix_audit": report["prefix_audit"] | {"per_pair": "omitted"},
        "identified_cost_reduction_operators": positive,
        "identified_harm_operators": harmful,
        "weak_or_non_identifiable_operators": weak,
        "overall_cost": overall_cost,
        "cells": {op: {
            "effect": cells[op]["paired_recovery_minus_continue_cost"]["estimate"],
            "ci95": cells[op]["paired_recovery_minus_continue_cost"]["ci95"],
            "interpretation": cells[op]["interpretation"],
        } for op in OPERATORS},
    }, indent=2, sort_keys=True))
    return 0 if report["passed"] else 2


if __name__ == "__main__":
    raise SystemExit(main())
