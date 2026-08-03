#!/usr/bin/env python3
"""Locate invalid rendered UV primvars in the fully composed Isaac runtime stage."""

from __future__ import annotations

import argparse
import json
from datetime import UTC, datetime
from pathlib import Path

import numpy as np


ROOT = Path(__file__).resolve().parents[1]


def main() -> int:
    preliminary = argparse.ArgumentParser(add_help=False)
    preliminary.add_argument("--episode-usd", type=Path, required=True)
    preliminary.add_argument("--out", type=Path, required=True)
    pre, _ = preliminary.parse_known_args()

    from isaaclab.app import AppLauncher

    parser = argparse.ArgumentParser(parents=[preliminary])
    AppLauncher.add_app_launcher_args(parser)
    args = parser.parse_args()
    app = AppLauncher(args).app
    try:
        import omni.usd
        from pxr import Usd, UsdGeom

        from kino_vla.sim.isaac_policy_backend import IsaacPolicyBackend
        from kino_vla.utils.config import load_config

        config = load_config("sim/go2_skeleton.yaml")
        backend = IsaacPolicyBackend(
            config,
            np.asarray([0.0, 0.0], dtype=np.float64),
            0.0,
        )
        scene_prim = backend.load_realistic_scene(str(args.episode_usd.resolve()))
        stage = omni.usd.get_context().get_stage()
        invalid = []
        checked = 0
        predicate = Usd.TraverseInstanceProxies()
        for prim in Usd.PrimRange.Stage(stage, predicate):
            if not prim.IsA(UsdGeom.Mesh):
                continue
            mesh = UsdGeom.Mesh(prim)
            point_count = len(mesh.GetPointsAttr().Get() or [])
            face_counts = mesh.GetFaceVertexCountsAttr().Get() or []
            face_vertex_count = len(mesh.GetFaceVertexIndicesAttr().Get() or [])
            expected_by_interpolation = {
                UsdGeom.Tokens.constant: 1,
                UsdGeom.Tokens.uniform: len(face_counts),
                UsdGeom.Tokens.vertex: point_count,
                UsdGeom.Tokens.varying: point_count,
                UsdGeom.Tokens.faceVarying: face_vertex_count,
            }
            for primvar in UsdGeom.PrimvarsAPI(mesh).GetPrimvars():
                name = primvar.GetPrimvarName()
                if not str(name).startswith("st"):
                    continue
                checked += 1
                values = primvar.Get()
                value_count = len(values) if values is not None else 0
                interpolation = primvar.GetInterpolation()
                expected = expected_by_interpolation.get(interpolation)
                index_count = len(primvar.GetIndices() or []) if primvar.IsIndexed() else 0
                actual = index_count if primvar.IsIndexed() else value_count
                if expected is None or actual == expected:
                    continue
                source_layers = []
                try:
                    source_layers = sorted(
                        {
                            str(spec.layer.realPath or spec.layer.identifier)
                            for spec in prim.GetPrimStack()
                        }
                    )
                except Exception:
                    source_layers = []
                invalid.append(
                    {
                        "prim_path": str(prim.GetPath()),
                        "primvar": str(name),
                        "interpolation": str(interpolation),
                        "expected_count": expected,
                        "value_count": value_count,
                        "indexed": primvar.IsIndexed(),
                        "index_count": index_count,
                        "instance_proxy": prim.IsInstanceProxy(),
                        "source_layers": source_layers,
                    }
                )
        by_scope = {
            "go2": sum("/Robot/" in row["prim_path"] for row in invalid),
            "kinofail_scene": sum(scene_prim in row["prim_path"] for row in invalid),
            "other": sum(
                "/Robot/" not in row["prim_path"] and scene_prim not in row["prim_path"]
                for row in invalid
            ),
        }
        result = {
            "schema_version": "kinofail.isaac-runtime-uv-primvar-audit.v1",
            "created_utc": datetime.now(UTC).isoformat(),
            "episode_usd": str(args.episode_usd.resolve()),
            "scene_prim": scene_prim,
            "checked_uv_primvars": checked,
            "invalid_uv_primvars": invalid,
            "invalid_count_by_scope": by_scope,
            "kinofail_scene_uv_primvars_valid": by_scope["kinofail_scene"] == 0,
            "passed": by_scope["kinofail_scene"] == 0,
        }
        output = args.out.resolve()
        output.parent.mkdir(parents=True, exist_ok=True)
        output.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n", encoding="utf-8")
        print(json.dumps(result, indent=2, sort_keys=True))
        return 0 if result["passed"] else 1
    finally:
        app.close()


if __name__ == "__main__":
    raise SystemExit(main())
