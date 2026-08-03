#!/usr/bin/env python3

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from kino_vla.eval.visual_shell_review import audit_visual_shell_reviews  # noqa: E402


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--review",
        type=Path,
        default=ROOT / "configs/data/kinofail_embodiedgen_visual_shell_review_v1.json",
    )
    parser.add_argument(
        "--out",
        type=Path,
        default=ROOT
        / "outputs/kinofail_realistic/scene_sources/embodiedgen_visual_shell_v1/review_audit.json",
    )
    args = parser.parse_args()
    review = json.loads(args.review.read_text(encoding="utf-8"))
    audit = audit_visual_shell_reviews(review, root=ROOT)
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(audit, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps({"out": str(args.out), **audit["counts"]}, sort_keys=True))


if __name__ == "__main__":
    main()
