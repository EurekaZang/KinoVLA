#!/usr/bin/env python3
"""Evaluate C2 v5 with the frozen class-proposal router."""

from __future__ import annotations

import json
from pathlib import Path

import eval_kinofail_realistic_c2_bidirectional_v4 as implementation


implementation.OURS = "structured_post_interaction_router_v5"
implementation.METHODS = (
    *implementation.BASELINES,
    implementation.OURS,
)
implementation.__file__ = __file__

_development_v4 = implementation._development
_formal_v4 = implementation._formal


def _development_v5(directory: Path, output: Path) -> int:
    return_code = _development_v4(directory, output)
    path = output / "development_report.json"
    report = json.loads(path.read_text(encoding="utf-8"))
    report.update(
        {
            "schema_version": (
                "kinofail.realistic-c2-v5-development-screen.v1"
            ),
            "status": (
                "development_complete_v3_v4_outcomes_consumed"
            ),
            "v3_failure_used_as_development": True,
            "v4_confirmation_outcomes_used": True,
        }
    )
    report["selection"]["proprio_proposal"] = (
        "balanced four-class logistic(C=1) on an 80-dimensional "
        "rotation-invariant post-interaction proprio summary"
    )
    report["selection"]["temporal_contract"] = (
        "first 21 native-rate samples ending 0.30 s after the first "
        "0.35 m footprint-to-region encounter"
    )
    report["selection"]["contract_rationale"] = (
        "measure operator response after physical encounter without "
        "aligning to failure, outcome, or truth metadata"
    )
    path.write_text(
        json.dumps(report, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    return return_code


def _formal_v5(
    development_directory: Path,
    test_directory: Path,
    geometry_directory: Path,
    protocol_path: Path,
    output: Path,
) -> int:
    return_code = _formal_v4(
        development_directory,
        test_directory,
        geometry_directory,
        protocol_path,
        output,
    )
    path = output / "report.json"
    report = json.loads(path.read_text(encoding="utf-8"))
    report["schema_version"] = (
        "kinofail.realistic-c2-formal-confirmation.v5"
    )
    report["selected_method"] = implementation.OURS
    report["architecture"] = {
        "proprio_proposal": (
            "balanced four-class logistic(C=1) on an 80-dimensional "
            "rotation-invariant post-interaction proprio summary"
        ),
        "T2_specialist": (
            "balanced logistic(C=0.1) on CLIP plus generic HOG"
        ),
        "composition": (
            "accept a proprio proposal only for a T3 class; otherwise "
            "invoke the T2 visual specialist"
        ),
        "temporal_contract": (
            "first 21 native-rate samples ending 0.30 s after the first "
            "0.35 m footprint-to-region encounter"
        ),
    }
    path.write_text(
        json.dumps(report, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    return return_code


implementation._development = _development_v5
implementation._formal = _formal_v5


if __name__ == "__main__":
    raise SystemExit(implementation.main())
