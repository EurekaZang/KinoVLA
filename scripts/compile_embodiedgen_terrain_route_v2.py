#!/usr/bin/env python3
"""Compile the dense, operator-ready EmbodiedGen terrain-route v2 layer."""

from __future__ import annotations

import argparse
import hashlib
import json
from datetime import UTC, datetime
from pathlib import Path

from kino_vla.sim.embodiedgen_terrain_route_v2 import (
    compile_embodiedgen_terrain_route_v2,
)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--base-audit", type=Path, required=True)
    parser.add_argument("--expected-base-audit-sha256")
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--static-friction", type=float, default=0.8)
    parser.add_argument("--dynamic-friction", type=float, default=0.6)
    parser.add_argument("--max-spacing-m", type=float, default=0.04)
    args = parser.parse_args()
    try:
        result = compile_embodiedgen_terrain_route_v2(
            base_audit_path=args.base_audit,
            output_dir=args.out,
            expected_base_audit_sha256=args.expected_base_audit_sha256,
            static_friction=args.static_friction,
            dynamic_friction=args.dynamic_friction,
            max_spacing_m=args.max_spacing_m,
        )
    except Exception as exc:
        output = args.out.resolve()
        output.mkdir(parents=True, exist_ok=True)
        input_path = args.base_audit.resolve()
        rejection = {
            "schema_version": "kinofail.embodiedgen-terrain-route-v2-compile-rejection.v1",
            "created_utc": datetime.now(UTC).isoformat(),
            "passed": False,
            "admission_state": "terrain_route_v2_compile_rejected",
            "base_compiled_audit": {
                "path": str(input_path),
                "sha256": (
                    hashlib.sha256(input_path.read_bytes()).hexdigest()
                    if input_path.is_file()
                    else None
                ),
            },
            "exception_type": type(exc).__name__,
            "reason": str(exc),
            "formal_operator_run_authorized": False,
        }
        (output / "compiled_scene_audit.json").write_text(
            json.dumps(rejection, indent=2, ensure_ascii=False) + "\n",
            encoding="utf-8",
        )
        print(json.dumps(rejection, indent=2, ensure_ascii=False))
        return 2
    print(
        json.dumps(
            {
                "out": str(args.out.resolve()),
                "scene_id": result["scene_id"],
                "passed": result["passed"],
                "dense_route_surface": result["physics_contract"]["dense_route_surface"],
                "counts_as_a0_a7_evidence": result["counts_as_a0_a7_evidence"],
            },
            indent=2,
        )
    )
    return 0 if result["passed"] else 2


if __name__ == "__main__":
    raise SystemExit(main())
