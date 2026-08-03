#!/usr/bin/env python3
"""Author a development-only render surface and bidirectional lighting layer."""

from __future__ import annotations

import argparse
import hashlib
import json
from datetime import UTC, datetime
from pathlib import Path

from kino_vla.sim.embodiedgen_scene_v4 import refine_embodiedgen_kinofail_scene_v4


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--corridor-v2-audit", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--material-id", required=True)
    parser.add_argument(
        "--material-lock",
        default="outputs/assets/terrain_pbr_v1/terrain_assets.lock.json",
    )
    parser.add_argument("--dome-intensity", type=float, default=350.0)
    parser.add_argument("--panel-intensity", type=float, default=450.0)
    parser.add_argument("--panel-exposure", type=float, default=7.0)
    parser.add_argument("--route-surface-width-m", type=float, default=1.20)
    args = parser.parse_args()
    try:
        result = refine_embodiedgen_kinofail_scene_v4(
            args.corridor_v2_audit,
            args.output_dir,
            material_id=args.material_id,
            material_lock=args.material_lock,
            dome_intensity=args.dome_intensity,
            panel_intensity=args.panel_intensity,
            panel_exposure=args.panel_exposure,
            route_surface_width_m=args.route_surface_width_m,
        )
    except Exception as exc:
        output = Path(args.output_dir).resolve()
        output.mkdir(parents=True, exist_ok=True)
        input_path = Path(args.corridor_v2_audit).resolve()
        rejection = {
            "schema_version": "kinofail.embodiedgen-route-surface-v4-rejection.v1",
            "created_utc": datetime.now(UTC).isoformat(),
            "passed": False,
            "admission_state": "route_surface_v4_rejected",
            "corridor_v2_audit": {
                "path": str(input_path),
                "sha256": (
                    hashlib.sha256(input_path.read_bytes()).hexdigest()
                    if input_path.is_file()
                    else None
                ),
            },
            "material_id": args.material_id,
            "exception_type": type(exc).__name__,
            "reason": str(exc),
            "formal_operator_run_authorized": False,
        }
        (output / "compiled_scene_audit.json").write_text(
            json.dumps(rejection, indent=2, ensure_ascii=False) + "\n",
            encoding="utf-8",
        )
        print(json.dumps(rejection, indent=2, ensure_ascii=False))
        raise SystemExit(2) from None
    print(
        json.dumps(
            {
                "passed": result["passed"],
                "audit_path": result["audit_path"],
                "episode_usd": result["episode_usd"],
                "appearance_contract": result["appearance_contract"],
                "lighting_contract": result["lighting_contract"],
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
