#!/usr/bin/env python3

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from kino_vla.eval.translated_view_review import audit_translated_view_review  # noqa: E402


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--review",
        type=Path,
        default=ROOT
        / "configs/data/kinofail_embodiedgen_forest_translated_view_review_v1.json",
    )
    parser.add_argument(
        "--out",
        type=Path,
        default=ROOT
        / "outputs/kinofail_realistic/scene_sources/embodiedgen_visual_shell_v1/forest_trail_shell_dev01/mesh_development/translated_view_review_audit.json",
    )
    args = parser.parse_args()
    review = json.loads(args.review.read_text(encoding="utf-8"))
    audit = audit_translated_view_review(review, root=ROOT)
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(audit, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(
        json.dumps(
            {
                "out": str(args.out),
                "may_proceed": audit["may_proceed_to_isaac_visual_shell_composition"],
                "publication_candidate": audit["translated_view_publication_candidate"],
                "scene_registry_eligible": audit["scene_registry_eligible"],
            },
            sort_keys=True,
        )
    )


if __name__ == "__main__":
    main()
