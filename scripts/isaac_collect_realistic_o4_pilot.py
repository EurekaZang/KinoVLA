#!/usr/bin/env python3
"""Collect the frozen formal O4 indoor-workstudio pilot subset.

This is the first evaluation-eligible slice of the 396-episode realistic pilot.  Scope is
intentionally narrow and preregistered: two severity-matched counterfactual pairs in the one
scene family whose authored USD/compiler/runtime path has already passed review.  The protocol
cannot authorize the other eight scene families or any other operator.
"""

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

import numpy as np
import yaml

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from scripts.isaac_collect_realistic_smoke_pair import (  # noqa: E402
    _collect_episode,
    _read_jsonl,
    _sha256,
)


def _collector_bundle_sha256(paths: list[Path]) -> str:
    digest = hashlib.sha256()
    for path in paths:
        relative = str(path.resolve().relative_to(REPO_ROOT)).encode("utf-8")
        digest.update(len(relative).to_bytes(4, "big"))
        digest.update(relative)
        payload = path.read_bytes()
        digest.update(len(payload).to_bytes(8, "big"))
        digest.update(payload)
    return digest.hexdigest()


def _load_protocol(path: Path, *, schedule_path: Path) -> tuple[dict[str, Any], str]:
    protocol = json.loads(path.read_text(encoding="utf-8"))
    if protocol.get("schema_version") != "kinofail.formal-collection-protocol.v1":
        raise ValueError("unsupported formal protocol schema")
    if protocol.get("status") != "frozen":
        raise ValueError("formal protocol is not frozen")
    schedule_sha256 = _sha256(schedule_path)
    if protocol.get("schedule_sha256") != schedule_sha256:
        raise ValueError("pilot schedule changed after protocol freeze")
    shared = REPO_ROOT / "scripts/isaac_collect_realistic_smoke_pair.py"
    collector_sha256 = _collector_bundle_sha256([Path(__file__).resolve(), shared])
    if protocol.get("collector_sha256") != collector_sha256:
        raise ValueError("collector bundle changed after protocol freeze")
    runtime_sha256 = _sha256(REPO_ROOT / "kino_vla/data/runtime_manifest.py")
    if protocol.get("runtime_manifest_sha256") != runtime_sha256:
        raise ValueError("runtime validator changed after protocol freeze")
    return protocol, collector_sha256


def _select_protocol_records(
    schedule: list[dict[str, Any]], protocol: dict[str, Any]
) -> list[dict[str, Any]]:
    allowed = protocol["allowed"]
    group_ids = [str(value) for value in allowed["counterfactual_group_ids"]]
    selected = [row for row in schedule if row["counterfactual_group_id"] in group_ids]
    if len(selected) != 2 * len(group_ids):
        raise RuntimeError("frozen formal groups are not complete in the live schedule")
    for group_id in group_ids:
        pair = [row for row in selected if row["counterfactual_group_id"] == group_id]
        if len(pair) != 2 or {row["condition"] for row in pair} != {
            "anomaly",
            "nominal_counterfactual",
        }:
            raise RuntimeError(f"formal group {group_id} is not a complete counterfactual pair")
    expected_fields = {
        "target_operator": set(allowed["target_operators"]),
        "scene_family": set(allowed["scene_families"]),
        "physical_realization": set(allowed["physical_realizations"]),
        "geometry_profile": set(allowed["geometry_profiles"]),
    }
    for key, expected in expected_fields.items():
        actual = {str(row[key]) for row in selected}
        if not actual <= expected:
            raise ValueError(f"formal protocol does not authorize {key}={sorted(actual)}")
    if {row["target_operator"] for row in selected} != {"O4_tether"}:
        raise ValueError("this collector implements only O4_tether")
    if {row["scene_family"] for row in selected} != {"indoor_workstudio_01"}:
        raise ValueError("this frozen collector implements only the reviewed workstudio scene")
    if len({row["camera_profile"] for row in selected}) != 1:
        raise ValueError("one Isaac process requires one frozen camera profile")
    return sorted(
        selected,
        key=lambda row: (
            group_ids.index(str(row["counterfactual_group_id"])),
            0 if row["condition"] == "nominal_counterfactual" else 1,
        ),
    )


def main() -> int:
    preliminary = argparse.ArgumentParser(add_help=False)
    preliminary.add_argument(
        "--schedule", default="outputs/kinofail_realistic/design_v1/pilot_schedule.jsonl"
    )
    preliminary.add_argument(
        "--protocol", default="configs/data/kinofail_o4_formal_pilot_v1.json"
    )
    preliminary.add_argument(
        "--corpus-root", default="outputs/kinofail_realistic/corpus_v1_formal"
    )
    pre_args, _ = preliminary.parse_known_args()

    parser = argparse.ArgumentParser(description="Collect frozen formal O4 pilot subset")
    parser.add_argument("--schedule", default=pre_args.schedule)
    parser.add_argument("--protocol", default=pre_args.protocol)
    parser.add_argument("--corpus-root", default=pre_args.corpus_root)
    parser.add_argument("--overwrite", action="store_true")
    from isaaclab.app import AppLauncher

    AppLauncher.add_app_launcher_args(parser)
    args = parser.parse_args()
    args.enable_cameras = True
    app = AppLauncher(args).app

    from kino_vla.data.runtime_manifest import audit_runtime_corpus
    from kino_vla.sim.isaac_policy_backend import IsaacPolicyBackend
    from kino_vla.sim.realistic_scene import (
        build_indoor_adhesion_scene_spec,
        compile_indoor_scene_layers,
    )
    from kino_vla.sim.terrain_materials import (
        appearance_binding_from_record,
        load_terrain_asset_lock,
    )
    from kino_vla.utils.config import CONFIGS_DIR, load_config

    schedule_path = (REPO_ROOT / args.schedule).resolve()
    protocol_path = (REPO_ROOT / args.protocol).resolve()
    schedule = _read_jsonl(schedule_path)
    protocol, collector_sha256 = _load_protocol(protocol_path, schedule_path=schedule_path)
    selected = _select_protocol_records(schedule, protocol)
    representative = selected[0]
    corpus_root = (REPO_ROOT / args.corpus_root).resolve()
    corpus_root.mkdir(parents=True, exist_ok=True)

    data_config = yaml.safe_load(
        (CONFIGS_DIR / "data/kinofail_realistic.yaml").read_text(encoding="utf-8")
    )
    camera_profile = dict(data_config["camera_profile_specs"][representative["camera_profile"]])
    mount = camera_profile["mount_xyz_m"]
    sim_cfg = load_config(
        "sim/go2_skeleton.yaml",
        {
            "perception_cam_width": int(camera_profile["width"]),
            "perception_cam_height": int(camera_profile["height"]),
            "perception_focal_mm": float(camera_profile["focal_length_mm"]),
            "perception_cam_x_m": float(mount[0]),
            "perception_cam_y_m": float(mount[1]),
            "perception_cam_z_m": float(mount[2]),
            "perception_cam_pitch_down_rad": float(camera_profile["pitch_down_rad"]),
        },
    )
    scene_spec = build_indoor_adhesion_scene_spec(
        int(representative["scene_seed"]),
        scene_id=str(representative["scene_family"]),
        source_kind=str(representative["scene_source"]),
        film_color=(0.62, 0.66, 0.64),
        film_emissive=(0.02, 0.025, 0.022),
    )
    scene_cache = corpus_root / "_scene_cache" / (
        f"{scene_spec.scene_id}_{int(representative['scene_seed'])}"
    )
    compiled = compile_indoor_scene_layers(scene_spec, scene_cache)
    if not compiled["audit"]["passed"]:
        raise RuntimeError(f"scene compile audit failed: {compiled['audit']}")

    asset_lock_path = REPO_ROOT / "outputs/assets/terrain_pbr_v1/terrain_assets.lock.json"
    asset_lock = load_terrain_asset_lock(asset_lock_path)
    demo_cfg = load_config("demo/indoor_adhesion_icra.yaml")
    backend = IsaacPolicyBackend(
        sim_cfg,
        np.asarray(demo_cfg.start.pos, dtype=np.float64),
        float(demo_cfg.start.heading),
        perception_cam=True,
    )
    backend.load_realistic_scene(str(compiled["episode_usd"]))

    schedule_sha256 = _sha256(schedule_path)
    results: list[dict[str, Any]] = []
    for record in selected:
        episode_dir = corpus_root / Path(record["required_outputs"]["episode_manifest"]).parent
        bindings = {
            str(view["appearance_view_id"]): appearance_binding_from_record(
                view,
                lock=asset_lock,
                asset_root=REPO_ROOT / str(asset_lock["asset_root"]),
            )
            for view in record["appearance_views"]
        }
        result = _collect_episode(
            backend,
            record,
            episode_dir=episode_dir,
            scene_spec=scene_spec,
            compiled=compiled,
            appearance_bindings=bindings,
            asset_lock_path=asset_lock_path,
            demo_cfg=demo_cfg,
            camera_profile=camera_profile,
            overwrite=bool(args.overwrite),
            collection_status="formal_pilot",
            schedule_sha256=schedule_sha256,
            collector_sha256=collector_sha256,
            formal_protocol_path=protocol_path,
        )
        results.append(result)
        print(
            f"[formal-o4] {result['episode_id']}: passed={result['passed']} "
            f"rgb={result['rgb_frames']} views={result['appearance_views']}"
        )

    audit = audit_runtime_corpus(
        schedule,
        corpus_root=corpus_root,
        gate_overrides=data_config["runtime_quality_gates"],
        write_validated=True,
        require_complete=False,
    )
    summary = {
        "schema_version": "kinofail.formal-o4-pilot-subset.v1",
        "created_utc": datetime.now(UTC).isoformat(),
        "passed": all(bool(row["passed"]) for row in results) and audit.passed,
        "publication_status": "formal_pilot_partial_not_publication_freeze",
        "protocol_id": protocol["protocol_id"],
        "protocol_path": str(protocol_path),
        "protocol_sha256": _sha256(protocol_path),
        "schedule_sha256": schedule_sha256,
        "collector_bundle_sha256": collector_sha256,
        "results": results,
        "runtime_audit": audit.summary,
    }
    summary_path = corpus_root / "o4_formal_pilot_summary.json"
    summary_path.write_text(
        json.dumps(summary, indent=2, sort_keys=True, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )
    ok = bool(summary["passed"])
    print("PASS: formal O4 pilot subset" if ok else "FAIL: formal O4 pilot subset")
    sys.stdout.flush()
    closer = threading.Thread(target=app.close, daemon=True)
    closer.start()
    closer.join(timeout=15.0)
    os._exit(0 if ok else 1)


if __name__ == "__main__":
    main()
