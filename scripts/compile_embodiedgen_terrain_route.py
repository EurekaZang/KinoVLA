#!/usr/bin/env python3
"""Compile an explicit nominal-floor material layer for EmbodiedGen terrain operators."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from kino_vla.sim.embodiedgen_terrain_route import compile_embodiedgen_terrain_route


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--base-audit", type=Path, required=True)
    parser.add_argument("--expected-base-audit-sha256")
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--static-friction", type=float, default=0.8)
    parser.add_argument("--dynamic-friction", type=float, default=0.6)
    args = parser.parse_args()
    result = compile_embodiedgen_terrain_route(
        base_audit_path=args.base_audit,
        output_dir=args.out,
        expected_base_audit_sha256=args.expected_base_audit_sha256,
        static_friction=args.static_friction,
        dynamic_friction=args.dynamic_friction,
    )
    print(
        json.dumps(
            {
                "out": str(args.out.resolve()),
                "scene_id": result["scene_id"],
                "passed": result["passed"],
                "counts_as_a0_a7_evidence": result["counts_as_a0_a7_evidence"],
            },
            indent=2,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
