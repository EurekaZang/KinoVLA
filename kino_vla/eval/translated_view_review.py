"""Audit a manual translated-view review against immutable render artifacts."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Mapping

from kino_vla.eval.visual_shell_preflight import sha256_file


def audit_translated_view_review(review: Mapping[str, Any], *, root: Path) -> dict[str, Any]:
    contact_sheet = root / str(review["contact_sheet"])
    machine_audit_path = root / str(review["machine_audit"])
    machine_audit = json.loads(machine_audit_path.read_text(encoding="utf-8"))
    rubric = review["rubric"]
    critical = tuple(rubric["critical_criteria"])
    refinement = tuple(rubric["refinement_criteria"])
    criteria = review["criteria"]
    checks = {
        "contact_sheet_hash_matches": contact_sheet.is_file()
        and sha256_file(contact_sheet) == review["contact_sheet_sha256"],
        "machine_audit_hash_matches": machine_audit_path.is_file()
        and sha256_file(machine_audit_path) == review["machine_audit_sha256"],
        "machine_preflight_passed": machine_audit.get("passed") is True,
        "machine_audit_not_admitted": machine_audit.get(
            "counts_as_scene_registry_admission"
        )
        is False,
        "all_criteria_recorded": set(criteria) == set(critical) | set(refinement),
        "all_critical_criteria_passed": all(criteria.get(name) is True for name in critical),
        "publication_refinement_passed": sum(
            criteria.get(name) is True for name in refinement
        )
        >= int(rubric["minimum_refinement_passes_for_publication_candidate"]),
    }
    may_proceed = all(
        checks[name]
        for name in (
            "contact_sheet_hash_matches",
            "machine_audit_hash_matches",
            "machine_preflight_passed",
            "machine_audit_not_admitted",
            "all_criteria_recorded",
            "all_critical_criteria_passed",
        )
    )
    publication_candidate = may_proceed and checks["publication_refinement_passed"]
    return {
        "schema_version": "kinofail.embodiedgen-translated-view-review-audit.v1",
        "review_id": review["review_id"],
        "review_status": review["status"],
        "reviewer": review["reviewer"],
        "scene_id": review["scene_id"],
        "domain": review["domain"],
        "checks": checks,
        "critical_passes": sum(criteria.get(name) is True for name in critical),
        "critical_total": len(critical),
        "refinement_passes": sum(criteria.get(name) is True for name in refinement),
        "refinement_total": len(refinement),
        "may_proceed_to_isaac_visual_shell_composition": may_proceed,
        "translated_view_publication_candidate": publication_candidate,
        "scene_registry_eligible": False,
        "counts_as_a0_a7_evidence": False,
        "independent_human_review_required_before_registry": review[
            "independent_human_review_required_before_registry"
        ],
        "notes": review["notes"],
        "interpretation": (
            "A development review can prioritize Isaac visual-shell composition. It cannot "
            "admit a scene or replace body-fixed Go2 RTX, Kino collision/heightfield, operator "
            "capability, independent human QA, or realistic A0-A7 evidence."
        ),
    }
