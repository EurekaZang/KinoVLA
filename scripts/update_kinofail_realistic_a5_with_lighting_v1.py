#!/usr/bin/env python3
"""Attach the frozen held-out-lighting result to A5 without changing other outcomes."""

from __future__ import annotations

import hashlib
import json
from datetime import UTC, datetime
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
A5 = ROOT / "outputs/eval/realistic_a0_a7_v4/a5_selective.json"
LIGHTING = ROOT / "outputs/eval/realistic_a0_a7_v4/a5_lighting_heldout.json"
REALIZATION = ROOT / "outputs/eval/realistic_a0_a7_v4/a5_physical_realization_heldout.json"
A4 = ROOT / "outputs/eval/realistic_a0_a7_v4/a4_consequence.json"


def _sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _json(path: Path) -> dict:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise TypeError(path)
    return value


def main() -> int:
    a5, lighting, realization, a4 = (
        _json(A5), _json(LIGHTING), _json(REALIZATION), _json(A4)
    )
    a5["updated_utc"] = datetime.now(UTC).isoformat()
    a5["heldout_lighting"] = {
        "artifact": str(LIGHTING.relative_to(ROOT)),
        "sha256": _sha(LIGHTING),
        "status": lighting.get("status"),
        "passed_integrity_gates": lighting.get("passed") is True,
        "valid_counterfactual_pairs": lighting.get("valid_counterfactual_pairs"),
        "excluded_scheduled_pairs": lighting.get("excluded_scheduled_pairs"),
        "summaries": lighting.get("summaries", {}),
        "paired_scene_cluster_statistics": lighting.get(
            "paired_scene_cluster_statistics", {}
        ),
    }
    a5["actual_action_consequence"] = {
        "artifact": str(A4.relative_to(ROOT)),
        "sha256": _sha(A4),
        "confirmatory_complete": a4.get("status") == "confirmatory_complete",
        "efficacy_acceptance": a4.get("passed") is True,
        "overall_recovery_minus_continue_cost": a4.get("overall", {}).get(
            "paired_recovery_minus_continue_cost"
        ),
        "interpretation": (
            "Actual-action evidence exists, but frozen recoveries did not demonstrate selective "
            "benefit; consequence availability must not be relabelled as efficacy."
        ),
    }
    a5["heldout_physical_realization"] = {
        "artifact": str(REALIZATION.relative_to(ROOT)),
        "sha256": _sha(REALIZATION),
        "status": realization.get("status"),
        "passed_integrity_gates": realization.get("passed") is True,
        "valid_counterfactual_pairs": realization.get("valid_counterfactual_pairs"),
        "excluded_scheduled_pairs": realization.get("excluded_scheduled_pairs"),
        "operators": realization.get("operators", []),
        "summaries": realization.get("summaries", {}),
        "scene_cluster_statistics": realization.get("scene_cluster_statistics", {}),
        "claim_boundary": realization.get("claim_boundary"),
    }
    acceptance = dict(a5["acceptance"])
    acceptance.pop("selective_consequence_available_after_A4", None)
    acceptance["lighting_heldout_axis_present"] = lighting.get("passed") is True
    acceptance["physical_realization_heldout_axis_present"] = realization.get("passed") is True
    acceptance["actual_action_consequence_available"] = (
        a4.get("status") == "confirmatory_complete"
    )
    acceptance["direct_selective_vs_always_safe_evaluation_present"] = False
    acceptance["selective_minus_always_safe_cluster_ci_upper_le_0"] = False
    acceptance["released_attribution_precision_ge_0_95"] = False
    a5["acceptance"] = acceptance
    measurements_complete = (
        lighting.get("status") == "confirmatory_complete"
        and realization.get("status") == "confirmatory_complete"
        and a4.get("status") == "confirmatory_complete"
    )
    a5["status"] = "confirmatory_complete" if measurements_complete else "partial_complete"
    a5["passed"] = all(acceptance.values())
    a5["claim_boundary"] = (
        "Scene/material/camera/lighting and texture robustness may be reported from their frozen "
        "cells. A direct selective-policy versus always-safe-policy comparison has not yet been "
        "run. A4 recovery-minus-continue is side evidence and is not the C4 estimand."
    )
    A5.write_text(json.dumps(a5, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps({
        "out": str(A5), "status": a5["status"], "passed": a5["passed"],
        "acceptance": acceptance,
    }, indent=2))
    return 0 if a5["passed"] else 2


if __name__ == "__main__":
    raise SystemExit(main())
