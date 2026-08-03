from __future__ import annotations

import hashlib
import json
from pathlib import Path

from kino_vla.eval.visual_shell_review import audit_visual_shell_reviews


def test_review_can_advance_without_admitting_scene(tmp_path: Path) -> None:
    image = tmp_path / "pano.png"
    image.write_bytes(b"image")
    preflight = tmp_path / "preflight.json"
    preflight.write_text(json.dumps({"passed": True}), encoding="utf-8")
    review = {
        "review_id": "test",
        "status": "development",
        "reviewer": "reviewer",
        "rubric": {
            "critical_criteria": ["domain", "route"],
            "refinement_criteria": ["detail"],
            "minimum_refinement_passes_for_publication_candidate": 1,
            "independent_human_review_required_before_registry": True,
        },
        "reviews": [
            {
                "scene_id": "scene",
                "domain": "wild",
                "image": "pano.png",
                "image_sha256": hashlib.sha256(b"image").hexdigest(),
                "automated_preflight": "preflight.json",
                "criteria": {"domain": True, "route": True, "detail": True},
                "notes": [],
            }
        ],
    }
    result = audit_visual_shell_reviews(review, root=tmp_path)
    row = result["scene_reviews"][0]
    assert row["may_proceed_to_3d_visual_shell_development"] is True
    assert row["panorama_publication_candidate"] is True
    assert row["scene_registry_eligible"] is False
