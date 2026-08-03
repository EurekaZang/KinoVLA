#!/usr/bin/env python3
"""Compare legacy and realistic-route policies on the same A4 development cell."""

from __future__ import annotations

import argparse
import hashlib
import json
from collections import defaultdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parents[1]


def _sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _rows(path: Path) -> list[dict[str, Any]]:
    with path.open() as handle:
        return [json.loads(line) for line in handle if line.strip()]


def _arm(action: str) -> str:
    return "continue" if action == "continue" else "recovery"


def _summarize(rows: list[dict[str, Any]]) -> dict[str, Any]:
    cells: dict[tuple[str, str], list[dict[str, Any]]] = defaultdict(list)
    by_case: dict[str, dict[str, dict[str, Any]]] = defaultdict(dict)
    for row in rows:
        arm = _arm(str(row["action"]))
        cells[(str(row["operator"]), arm)].append(row)
        by_case[str(row["case_id"])][arm] = row
    cell_summary = {}
    for (operator, arm), group in sorted(cells.items()):
        cell_summary[f"{operator}|{arm}"] = {
            "n": len(group),
            "mean_terminal_cost": sum(float(row["terminal_cost"]) for row in group)
            / len(group),
            "success_rate": sum(bool(row["success"]) for row in group) / len(group),
            "fall_rate": sum(bool(row["fell"]) for row in group) / len(group),
            "mean_final_progress_m": sum(
                float(row["final_progress_m"]) for row in group
            )
            / len(group),
        }
    complete = [pair for pair in by_case.values() if set(pair) == {"continue", "recovery"}]
    return {
        "episode_count": len(rows),
        "case_count": len(by_case),
        "all_pairs_complete": len(complete) == len(by_case),
        "all_prefixes_exactly_matched": all(
            pair["continue"]["predecision_rows_sha256"]
            == pair["recovery"]["predecision_rows_sha256"]
            for pair in complete
        ),
        "mean_recovery_minus_continue_cost": sum(
            float(pair["recovery"]["terminal_cost"])
            - float(pair["continue"]["terminal_cost"])
            for pair in complete
        )
        / len(complete),
        "cells": cell_summary,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--legacy",
        default=(
            "outputs/kinofail_realistic/corpus_a4_actual_action_v5/"
            "indoor_office_186/results.jsonl"
        ),
    )
    parser.add_argument(
        "--candidate",
        default=(
            "outputs/kinofail_realistic/development_c4_policy_swap_v1/"
            "indoor_office_186/results.jsonl"
        ),
    )
    parser.add_argument(
        "--candidate-policy-manifest",
        default="outputs/locomotion/realistic_route_v1/training_manifest.json",
    )
    parser.add_argument(
        "--out",
        default=(
            "outputs/eval/realistic_a0_a7_v6/"
            "a4_policy_swap_development_analysis.json"
        ),
    )
    args = parser.parse_args()

    legacy_path = (ROOT / args.legacy).resolve()
    candidate_path = (ROOT / args.candidate).resolve()
    manifest_path = (ROOT / args.candidate_policy_manifest).resolve()
    out_path = (ROOT / args.out).resolve()
    legacy = _summarize(_rows(legacy_path))
    candidate = _summarize(_rows(candidate_path))
    manifest = json.loads(manifest_path.read_text())
    command = manifest["command_contract"]
    mass = manifest["domain_randomization"]["add_base_mass_range"]

    action_support = {
        "slow_high_step": {
            "velocity_command_supported": True,
            "posture_residual_trained": False,
        },
        "backstep_release": {
            "velocity_command_supported": float(command["lin_vel_x"][0]) <= -0.32,
            "requested_vx_mps": -0.32,
        },
        "hold_and_request": {
            "velocity_command_supported": float(command["rel_standing_envs"]) > 0.0,
            "requested_vx_mps": 0.0,
        },
        "backstep_detour_replan": {
            "reverse_command_supported": float(command["lin_vel_x"][0]) <= -0.24,
            "lateral_command_supported": float(command["lin_vel_y"][1]) >= 0.45,
            "requested_reverse_vx_mps": -0.24,
            "maximum_controller_lateral_command_mps": 0.45,
        },
        "raise_body_slow_cross": {
            "velocity_command_supported": True,
            "posture_residual_trained": False,
        },
        "O5_payload_mass": {
            "training_maximum_added_mass_kg": float(mass[1]),
            "development_operator_added_mass_kg": 14.0,
            "severity_within_training_support": float(mass[1]) >= 14.0,
        },
    }
    unsupported_actions = [
        key
        for key, value in action_support.items()
        if any(
            flag is False
            for flag_name, flag in value.items()
            if (
                flag_name.endswith("_supported")
                or flag_name.endswith("_trained")
                or flag_name == "severity_within_training_support"
            )
        )
    ]
    legacy_o4_o9_falls = sum(
        legacy["cells"][f"{operator}|continue"]["fall_rate"]
        for operator in ("O4_tether", "O9_high_centering")
    )
    candidate_o4_o9_falls = sum(
        candidate["cells"][f"{operator}|continue"]["fall_rate"]
        for operator in ("O4_tether", "O9_high_centering")
    )
    report = {
        "schema_version": "kinofail.realistic-policy-swap-development-analysis.v1",
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "status": "development_complete",
        "passed": False,
        "counts_as_a0_a7_evidence": False,
        "artifacts": {
            "legacy_results": {
                "path": str(legacy_path.relative_to(ROOT)),
                "sha256": _sha(legacy_path),
            },
            "candidate_results": {
                "path": str(candidate_path.relative_to(ROOT)),
                "sha256": _sha(candidate_path),
            },
            "candidate_policy_manifest": {
                "path": str(manifest_path.relative_to(ROOT)),
                "sha256": _sha(manifest_path),
            },
        },
        "legacy_policy": legacy,
        "realistic_route_policy": candidate,
        "controlled_findings": {
            "continue_stability_improved_for_O4_and_O9": (
                candidate_o4_o9_falls < legacy_o4_o9_falls
            ),
            "legacy_O4_plus_O9_continue_fall_rate_sum": legacy_o4_o9_falls,
            "realistic_route_O4_plus_O9_continue_fall_rate_sum": candidate_o4_o9_falls,
            "recovery_cost_delta_improved": (
                candidate["mean_recovery_minus_continue_cost"]
                < legacy["mean_recovery_minus_continue_cost"]
            ),
            "legacy_mean_recovery_minus_continue_cost": legacy[
                "mean_recovery_minus_continue_cost"
            ],
            "realistic_route_mean_recovery_minus_continue_cost": candidate[
                "mean_recovery_minus_continue_cost"
            ],
        },
        "action_support_audit": action_support,
        "unsupported_action_or_severity_contracts": unsupported_actions,
        "root_cause": (
            "The realistic-route actor improves ordinary forward-route stability, "
            "but the historical recovery controller requests reverse, stand, large "
            "lateral, posture-residual, and payload conditions outside that actor's "
            "training support. Swapping only the actor therefore cannot rescue C4."
        ),
        "required_fix": (
            "Train one attribution-blind recovery-capable low-level actor shared by "
            "ours and all baselines; validate its action primitives on development "
            "scenes; then freeze and run a new direct selective-versus-always-safe "
            "comparison on an untouched extension."
        ),
        "claim_boundary": (
            "This is a controlled development diagnosis, not positive C4 evidence. "
            "All unfavorable outcomes are retained."
        ),
    }
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n")
    print(json.dumps(report["controlled_findings"], indent=2))
    print(f"wrote {out_path}")


if __name__ == "__main__":
    main()
