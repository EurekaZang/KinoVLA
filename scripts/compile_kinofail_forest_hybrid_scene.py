#!/usr/bin/env python3
"""Compile the route-complete EmbodiedGen-background forest hybrid."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from kino_vla.sim.forest_hybrid_composition import compile_forest_hybrid_scene


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--config",
        type=Path,
        default=ROOT / "configs/data/kinofail_forest_hybrid_composition_dev_v3.json",
    )
    args = parser.parse_args()
    result = compile_forest_hybrid_scene(args.config, root=ROOT)
    print(
        json.dumps(
            {
                "passed": result["passed"],
                "scene_id": result["scene_id"],
                "episode_usd": result["episode_usd"],
                "audit_path": result["audit_path"],
                "scene_registry_eligible": result["scene_registry_eligible"],
                "counts_as_a0_a7_evidence": result["counts_as_a0_a7_evidence"],
            },
            indent=2,
        )
    )
    return 0 if result["passed"] else 2


if __name__ == "__main__":
    raise SystemExit(main())
