#!/usr/bin/env python3
"""Collect every frozen C2 v3 O7/O8 pair for one scene per Isaac app."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import sys
import threading
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import yaml


ROOT = Path(__file__).resolve().parents[1]


def _sha(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _json(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise TypeError(path)
    return value


def _jsonl(path: Path) -> list[dict[str, Any]]:
    return [
        json.loads(line)
        for line in path.read_text(encoding="utf-8").splitlines()
        if line
    ]


def main() -> int:
    preliminary = argparse.ArgumentParser(add_help=False)
    preliminary.add_argument("--schedule", required=True)
    preliminary.add_argument("--scene-registry", required=True)
    preliminary.add_argument("--protocol", required=True)
    preliminary.add_argument("--corpus-root", required=True)
    preliminary.add_argument("--scene", required=True)
    pre, _ = preliminary.parse_known_args()
    parser = argparse.ArgumentParser()
    parser.add_argument("--schedule", default=pre.schedule)
    parser.add_argument("--scene-registry", default=pre.scene_registry)
    parser.add_argument("--protocol", default=pre.protocol)
    parser.add_argument("--corpus-root", default=pre.corpus_root)
    parser.add_argument("--scene", default=pre.scene)
    parser.add_argument("--resume", action="store_true")
    from isaaclab.app import AppLauncher

    AppLauncher.add_app_launcher_args(parser)
    args = parser.parse_args()
    args.enable_cameras = True
    app = AppLauncher(args).app
    passed = False
    try:
        from isaac_collect_kinofail_realistic_pair_v2 import (
            _install_runtime_validation_adapter,
            _load_v1,
        )
        from kino_vla.sim.isaac_o4_v3_backend import (
            IsaacPolicyBackendO4V3,
        )
        from kino_vla.sim.realistic_route_protocol_v2 import (
            scene_route_binding,
        )
        from kino_vla.sim.terrain_materials import (
            appearance_binding_from_record,
            load_terrain_asset_lock,
        )
        from kino_vla.utils.config import CONFIGS_DIR, load_config

        _install_runtime_validation_adapter()
        implementation = _load_v1()
        schedule_path = (ROOT / args.schedule).resolve()
        registry_path = (ROOT / args.scene_registry).resolve()
        protocol_path = (ROOT / args.protocol).resolve()
        corpus_root = (ROOT / args.corpus_root).resolve()
        schedule = [
            row
            for row in _jsonl(schedule_path)
            if row["scene_family"] == args.scene
        ]
        if not schedule:
            raise RuntimeError(
                f"no C2 v3 T3 records for scene {args.scene}"
            )
        by_group: dict[str, list[dict[str, Any]]] = {}
        for row in schedule:
            by_group.setdefault(
                str(row["counterfactual_group_id"]), []
            ).append(row)
        if not all(
            len(rows) == 2
            and {str(row["condition"]) for row in rows}
            == {"nominal_counterfactual", "anomaly"}
            for rows in by_group.values()
        ):
            raise RuntimeError("C2 v3 T3 schedule has incomplete pairs")

        protocol = _json(protocol_path)
        for key, actual in (
            ("schedule_sha256", _sha(schedule_path)),
            ("scene_registry_sha256", _sha(registry_path)),
            ("collector_sha256", _sha(Path(__file__).resolve())),
            (
                "runtime_manifest_sha256",
                _sha(ROOT / "kino_vla/data/runtime_manifest.py"),
            ),
        ):
            if protocol.get(key) != actual:
                raise RuntimeError(
                    f"C2 v3 T3 protocol {key} mismatch"
                )
        registry = _json(registry_path)
        scene = implementation._registry_row(
            registry, str(args.scene)
        )
        episode = ROOT / scene["episode_usd"]
        compiled_path = ROOT / scene["compiled_audit"]
        if (
            _sha(episode) != scene["episode_sha256"]
            or _sha(compiled_path)
            != scene["compiled_audit_sha256"]
        ):
            raise RuntimeError("C2 v3 T3 scene hash mismatch")
        compiled = _json(compiled_path)
        binding = scene_route_binding(compiled)
        benchmark = yaml.safe_load(
            (CONFIGS_DIR / "data/kinofail_realistic.yaml").read_text(
                encoding="utf-8"
            )
        )
        camera_profiles = {
            str(row["camera_profile"]) for row in schedule
        }
        if len(camera_profiles) != 1:
            raise RuntimeError(
                "C2 v3 T3 scene requires one camera profile"
            )
        camera = dict(
            benchmark["camera_profile_specs"][
                next(iter(camera_profiles))
            ]
        )
        mount = camera["mount_xyz_m"]
        config = load_config(
            "sim/go2_skeleton.yaml",
            {
                "perception_cam_width": camera["width"],
                "perception_cam_height": camera["height"],
                "perception_focal_mm": camera["focal_length_mm"],
                "perception_cam_x_m": mount[0],
                "perception_cam_y_m": mount[1],
                "perception_cam_z_m": mount[2],
                "perception_cam_pitch_down_rad": camera[
                    "pitch_down_rad"
                ],
            },
        )
        backend = IsaacPolicyBackendO4V3(
            config,
            binding.frame.point(0.0, 0.0),
            binding.frame.heading_rad,
            perception_cam=True,
        )
        scene_prim = backend.load_realistic_scene(str(episode))
        route_surface_path = binding.route_surface_prim_path(
            scene_prim
        )
        implementation._tag_scene(
            scene_prim, route_surface_path
        )
        asset_lock_path = (
            ROOT
            / "outputs/assets/terrain_pbr_v1/terrain_assets.lock.json"
        )
        asset_lock = load_terrain_asset_lock(asset_lock_path)
        asset_root = Path(asset_lock["asset_root"])
        if not asset_root.is_absolute():
            asset_root = ROOT / asset_root
        schedule_sha = _sha(schedule_path)
        collector_sha = _sha(Path(__file__).resolve())
        results = []
        pair_dir = corpus_root / "pair_summaries"
        pair_dir.mkdir(parents=True, exist_ok=True)
        for group_id in sorted(by_group):
            summary_path = pair_dir / f"{group_id}.json"
            if args.resume and summary_path.exists():
                summary = _json(summary_path)
                if summary.get("passed") is True:
                    results.append(summary)
                    print(
                        json.dumps(
                            {
                                "counterfactual_group_id": group_id,
                                "passed": True,
                                "resumed": True,
                            },
                            sort_keys=True,
                        ),
                        flush=True,
                    )
                    continue
            pair_results = []
            pair = sorted(
                by_group[group_id],
                key=lambda row: row["condition"] == "anomaly",
            )
            for record in pair:
                bindings = {
                    str(view["appearance_view_id"]): (
                        appearance_binding_from_record(
                            view,
                            lock=asset_lock,
                            asset_root=asset_root,
                        )
                    )
                    for view in record["appearance_views"]
                }
                episode_dir = corpus_root / Path(
                    record["required_outputs"]["episode_manifest"]
                ).parent
                result = implementation._collect_one(
                    backend,
                    record,
                    scene,
                    compiled,
                    binding.frame,
                    route_surface_path,
                    bindings,
                    asset_lock_path,
                    episode_dir,
                    protocol_path,
                    schedule_sha,
                    collector_sha,
                    camera,
                    False,
                )
                pair_results.append(result)
                print(
                    json.dumps(result, sort_keys=True), flush=True
                )
            summary = {
                "schema_version": (
                    "kinofail.realistic-c2-v3-t3-pair.v1"
                ),
                "created_utc": datetime.now(UTC).isoformat(),
                "counterfactual_group_id": group_id,
                "scene_cluster": str(args.scene),
                "passed": all(
                    row["passed"] for row in pair_results
                ),
                "results": pair_results,
            }
            summary_path.write_text(
                json.dumps(summary, indent=2, sort_keys=True)
                + "\n",
                encoding="utf-8",
            )
            results.append(summary)
        passed = (
            len(results) == len(by_group)
            and all(row["passed"] for row in results)
        )
        scene_summary = {
            "schema_version": (
                "kinofail.realistic-c2-v3-t3-scene.v1"
            ),
            "created_utc": datetime.now(UTC).isoformat(),
            "scene_cluster": str(args.scene),
            "pair_count": len(by_group),
            "passed": passed,
        }
        scene_summary_path = (
            corpus_root
            / "scene_summaries"
            / f"{args.scene}.json"
        )
        scene_summary_path.parent.mkdir(parents=True, exist_ok=True)
        scene_summary_path.write_text(
            json.dumps(scene_summary, indent=2, sort_keys=True)
            + "\n",
            encoding="utf-8",
        )
    finally:
        sys.stdout.flush()
        closer = threading.Thread(target=app.close, daemon=True)
        closer.start()
        closer.join(timeout=15.0)
    os._exit(0 if passed else 2)


if __name__ == "__main__":
    main()
