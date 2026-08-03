#!/usr/bin/env python3
"""Correct two v27/v32 compatibility checks in the sealed v35 offline audit.

This adapter never executes Isaac Sim and never changes any recorded artifact.  It accepts the
v35 nominal/emergency Backstep phase names and the collector's frozen one-step posture-urgency
trigger, while retaining every physical, visual, route, provenance, and hash check from v1.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from scripts import audit_o4_action_consequence_v35_development_postrun as v1  # noqa: E402
from scripts.audit_o4_action_consequence_v34_confirmation_postrun import (  # noqa: E402
    _telemetry_rows,
)


_ORIGINAL_AUDIT_CASE = v1._audit_v27_case


def _corrected_early_branch_checks(
    manifest: dict[str, Any], runtime: dict[str, Any]
) -> dict[str, bool]:
    summaries = [manifest.get("takes", {}).get(name, {}) for name in ("continue", "backstep")]
    predecision_steps = [summary.get("predecision_step") for summary in summaries]
    first_attachment_steps = [summary.get("first_attachment_step") for summary in summaries]
    reasons = [summary.get("decision_trigger_reason") for summary in summaries]
    decision = manifest.get("decision_contract", {})
    return {
        "both_takes_branch_one_step_after_attachment": all(
            isinstance(pre, int) and isinstance(first, int) and pre == first + 1
            for pre, first in zip(predecision_steps, first_attachment_steps)
        ),
        "both_takes_use_same_frozen_one_step_trigger": len(set(reasons)) == 1
        and reasons[0] in {"maximum_dwell", "posture_urgency"},
        "decision_steps_match_between_takes": len(set(predecision_steps)) == 1
        and len(set(first_attachment_steps)) == 1,
        "manifest_contract_uses_one_step_dwell": decision.get(
            "minimum_decision_dwell_steps"
        )
        == runtime["minimum_decision_dwell_steps"]
        == 1
        and decision.get("maximum_decision_dwell_steps")
        == runtime["maximum_decision_dwell_steps"]
        == 1,
    }


def _corrected_audit_case(
    case: dict[str, Any], runtime: dict[str, Any]
) -> dict[str, Any]:
    result = _ORIGINAL_AUDIT_CASE(case, runtime)
    manifest = v1._json(Path(result["manifest"]))
    summaries = manifest.get("takes", {})
    decision_step = summaries.get("continue", {}).get("predecision_step")
    branch_step = decision_step + 1 if isinstance(decision_step, int) else None
    rows = _telemetry_rows(manifest)
    continue_rows = rows.get("continue", [])
    backstep_rows = rows.get("backstep", [])
    result["outcome_checks"]["actions_diverge_only_after_decision"] = (
        isinstance(branch_step, int)
        and len(continue_rows) > branch_step
        and len(backstep_rows) > branch_step
        and continue_rows[branch_step].get("action_phase") == "matched_forward"
        and backstep_rows[branch_step].get("action_phase")
        in {"backstep_nominal_clearance", "backstep_emergency_clearance"}
    )
    return result


def _correction_freeze_valid(path: Path, expected_original_out: Path) -> bool:
    payload = v1._json(path)
    records = payload.get("locked_files", [])
    return (
        payload.get("schema_version")
        == "kinofail.o4-action-consequence-v35-development-audit-correction-v2-freeze.v1"
        and payload.get("freeze_status") == "frozen_before_v35_v2_offline_reaudit"
        and payload.get("simulation_reexecution_allowed") is False
        and payload.get("original_postrun", {}).get("path") == str(
            expected_original_out.relative_to(ROOT)
        )
        and bool(records)
        and all(v1._locked(record) for record in records)
    )


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--preflight", type=Path, required=True)
    parser.add_argument("--correction-freeze", type=Path, required=True)
    parser.add_argument("--original-postrun", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    original_postrun = args.original_postrun.resolve()
    original = v1._json(original_postrun)
    if original.get("sealed") is not True or original.get("integrity_passed") is not True:
        raise RuntimeError("v35 v1 audit is not sealed with intact artifacts")
    if not _correction_freeze_valid(args.correction_freeze.resolve(), original_postrun):
        raise RuntimeError("v35 v2 correction freeze is invalid")
    if args.out.resolve().exists():
        raise FileExistsError(args.out.resolve())

    original_argv = sys.argv
    original_audit = v1._audit_v27_case
    original_early = v1._early_branch_checks
    try:
        v1._audit_v27_case = _corrected_audit_case
        v1._early_branch_checks = _corrected_early_branch_checks
        sys.argv = [
            str(Path(v1.__file__)),
            "--config", str(args.config.resolve()),
            "--preflight", str(args.preflight.resolve()),
            "--out", str(args.out.resolve()),
        ]
        code = v1.main()
    finally:
        v1._audit_v27_case = original_audit
        v1._early_branch_checks = original_early
        sys.argv = original_argv

    payload = v1._json(args.out.resolve())
    payload["schema_version"] = "kinofail.o4-action-consequence-v35-development-postrun.v2"
    payload["audit_correction"] = {
        "simulation_reexecuted": False,
        "recorded_artifacts_changed": False,
        "original_postrun": str(original_postrun),
        "original_postrun_sha256": v1._sha256(original_postrun),
        "corrected_checks": [
            "v35_backstep_phase_names",
            "frozen_one_step_posture_urgency_trigger",
        ],
    }
    payload["interpretation"] = (
        "This v2 audit corrects two offline compatibility checks over the unchanged sealed v35 "
        "artifacts. It remains development-only and cannot confirm O4 or A0-A7."
    )
    args.out.resolve().write_text(
        json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    return code


if __name__ == "__main__":
    raise SystemExit(main())
