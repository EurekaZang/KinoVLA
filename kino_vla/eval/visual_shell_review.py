"""Reproducible join of manual visual review with immutable panorama artifacts."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any, Mapping


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def audit_visual_shell_reviews(review: Mapping[str, Any], *, root: Path) -> dict[str, Any]:
    rubric = review["rubric"]
    critical = tuple(rubric["critical_criteria"])
    refinement = tuple(rubric["refinement_criteria"])
    required_refinement = int(rubric["minimum_refinement_passes_for_publication_candidate"])
    rows = []
    for raw in review["reviews"]:
        image_path = root / str(raw["image"])
        preflight_path = root / str(raw["automated_preflight"])
        preflight = json.loads(preflight_path.read_text(encoding="utf-8"))
        criteria = raw["criteria"]
        checks = {
            "image_exists_and_hash_matches": image_path.is_file()
            and _sha256(image_path) == raw["image_sha256"],
            "automated_preflight_passed": preflight.get("passed") is True,
            "all_rubric_criteria_recorded": set(criteria) == set(critical) | set(refinement),
            "all_critical_criteria_passed": all(criteria.get(name) is True for name in critical),
            "publication_refinement_passed": sum(criteria.get(name) is True for name in refinement)
            >= required_refinement,
        }
        may_proceed = all(
            checks[name]
            for name in (
                "image_exists_and_hash_matches",
                "automated_preflight_passed",
                "all_rubric_criteria_recorded",
                "all_critical_criteria_passed",
            )
        )
        publication_candidate = may_proceed and checks["publication_refinement_passed"]
        rows.append(
            {
                "scene_id": raw["scene_id"],
                "domain": raw["domain"],
                "checks": checks,
                "critical_passes": sum(criteria.get(name) is True for name in critical),
                "critical_total": len(critical),
                "refinement_passes": sum(criteria.get(name) is True for name in refinement),
                "refinement_total": len(refinement),
                "may_proceed_to_3d_visual_shell_development": may_proceed,
                "panorama_publication_candidate": publication_candidate,
                "scene_registry_eligible": False,
                "notes": raw["notes"],
            }
        )
    return {
        "schema_version": "kinofail.embodiedgen-visual-shell-review-audit.v1",
        "review_id": review["review_id"],
        "review_status": review["status"],
        "reviewer": review["reviewer"],
        "review_input_sha256": hashlib.sha256(
            json.dumps(review, sort_keys=True, separators=(",", ":")).encode("utf-8")
        ).hexdigest(),
        "scene_reviews": rows,
        "counts": {
            "reviewed": len(rows),
            "may_proceed_to_3d": sum(
                row["may_proceed_to_3d_visual_shell_development"] for row in rows
            ),
            "panorama_publication_candidates": sum(
                row["panorama_publication_candidate"] for row in rows
            ),
            "scene_registry_eligible": 0,
        },
        "independent_human_review_required_before_registry": rubric[
            "independent_human_review_required_before_registry"
        ],
        "interpretation": (
            "Manual panorama review can prioritize 3D development but cannot admit a scene. "
            "Geometry, collision/heightfield, body-fixed Go2 RTX, operator capability, and an "
            "independent human review remain mandatory."
        ),
    }
