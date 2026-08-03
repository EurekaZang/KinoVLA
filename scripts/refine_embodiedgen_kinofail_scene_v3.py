#!/usr/bin/env python3
"""Author a development-only pitch-aware lighting candidate over corridor-v2."""

from __future__ import annotations

import argparse
import json

from kino_vla.sim.embodiedgen_scene_v3 import refine_embodiedgen_kinofail_scene_v3


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--corridor-v2-audit", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--candidate-id", required=True)
    parser.add_argument("--dome-intensity", type=float, required=True)
    parser.add_argument("--ceiling-sphere-intensity", type=float, required=True)
    args = parser.parse_args()
    result = refine_embodiedgen_kinofail_scene_v3(
        args.corridor_v2_audit,
        args.output_dir,
        candidate_id=args.candidate_id,
        dome_intensity=args.dome_intensity,
        ceiling_sphere_intensity=args.ceiling_sphere_intensity,
    )
    print(
        json.dumps(
            {
                "passed": result["passed"],
                "audit_path": result["audit_path"],
                "episode_usd": result["episode_usd"],
                "lighting_contract": result["lighting_contract"],
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
