#!/usr/bin/env python3
"""Audit the three-profile body-fixed Go2 matrix for the forest development shell."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from kino_vla.eval.visual_shell_go2_matrix import audit_go2_profile_matrix


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--config",
        type=Path,
        default=ROOT / "configs/data/kinofail_forest_go2_profile_matrix_dev_v1.json",
    )
    parser.add_argument(
        "--out",
        type=Path,
        default=ROOT
        / "outputs/kinofail_realistic/scene_sources/embodiedgen_visual_shell_v1/forest_trail_shell_dev01/isaac_composition_dev_v1/go2_profile_matrix_audit.json",
    )
    args = parser.parse_args()
    config = json.loads(args.config.read_text(encoding="utf-8"))
    audit = audit_go2_profile_matrix(config, root=ROOT)
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(audit, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(
        json.dumps(
            {
                "passed": audit["passed"],
                "profiles": [row["profile"] for row in audit["profiles"]],
                "scene_registry_eligible": audit["scene_registry_eligible"],
                "counts_as_a0_a7_evidence": audit["counts_as_a0_a7_evidence"],
                "out": str(args.out),
            },
            indent=2,
        )
    )
    return 0 if audit["passed"] else 2


if __name__ == "__main__":
    raise SystemExit(main())
