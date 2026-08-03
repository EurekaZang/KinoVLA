from __future__ import annotations

import json

from kino_vla.eval.translated_view_review import audit_translated_view_review
from kino_vla.eval.visual_shell_preflight import sha256_file


def test_review_can_proceed_without_becoming_publication_candidate(tmp_path) -> None:
    image = tmp_path / "sheet.png"
    image.write_bytes(b"image")
    machine = tmp_path / "audit.json"
    machine.write_text(
        json.dumps({"passed": True, "counts_as_scene_registry_admission": False}),
        encoding="utf-8",
    )
    review = {
        "review_id": "test",
        "status": "development_review_not_independent_formal_qa",
        "reviewer": "test",
        "scene_id": "forest",
        "domain": "wild",
        "contact_sheet": "sheet.png",
        "contact_sheet_sha256": sha256_file(image),
        "machine_audit": "audit.json",
        "machine_audit_sha256": sha256_file(machine),
        "rubric": {
            "critical_criteria": ["coherent"],
            "refinement_criteria": ["hole_free"],
            "minimum_refinement_passes_for_publication_candidate": 1,
        },
        "criteria": {"coherent": True, "hole_free": False},
        "notes": [],
        "independent_human_review_required_before_registry": True,
    }

    audit = audit_translated_view_review(review, root=tmp_path)

    assert audit["may_proceed_to_isaac_visual_shell_composition"] is True
    assert audit["translated_view_publication_candidate"] is False
    assert audit["scene_registry_eligible"] is False
