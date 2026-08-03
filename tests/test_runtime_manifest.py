from __future__ import annotations

import json
import struct
import zlib
from pathlib import Path

import numpy as np

from kino_vla.data import runtime_manifest as runtime_manifest_module
from kino_vla.data.realistic_benchmark import build_realistic_corpus
from kino_vla.data.runtime_manifest import (
    FORMAL_PROTOCOL_SCHEMA_VERSION,
    audit_runtime_corpus,
    build_collected_manifest,
    schedule_record_sha256,
    sha256_file,
    validate_runtime_episode,
    write_collected_manifest,
)


def _png_bytes(width: int, height: int, value: int) -> bytes:
    signature = b"\x89PNG\r\n\x1a\n"

    def chunk(name: bytes, payload: bytes) -> bytes:
        body = name + payload
        return struct.pack(">I", len(payload)) + body + struct.pack(">I", zlib.crc32(body))

    ihdr = struct.pack(">IIBBBBB", width, height, 8, 2, 0, 0, 0)
    rows = []
    for y in range(height):
        row = bytearray(b"\x00")
        for x in range(width):
            pixel = min(
                220,
                value + (100 * x) // max(width, 1) + (50 * y) // max(height, 1),
            )
            row.extend((pixel, min(230, pixel + x % 31), min(225, pixel + y % 23)))
        rows.append(bytes(row))
    image = zlib.compress(b"".join(rows))
    return signature + chunk(b"IHDR", ihdr) + chunk(b"IDAT", image) + chunk(b"IEND", b"")


def _collected_episode(tmp_path: Path):
    schedule = build_realistic_corpus(mode="pilot").records[0]
    rgb_views = {}
    for view in schedule["appearance_views"]:
        view_id = view["appearance_view_id"]
        relative_dir = "rgb" if view["is_primary"] else f"rgb_views/{view_id}"
        (tmp_path / relative_dir).mkdir(parents=True)
        view_frames = []
        for index, timestamp_s in enumerate(np.arange(5, dtype=np.float64) * 0.05):
            relative = f"{relative_dir}/{index:06d}.png"
            value = 30 + index + 35 * int(view["appearance_view_index"])
            (tmp_path / relative).write_bytes(_png_bytes(320, 180, value))
            view_frames.append({"path": relative, "timestamp_s": float(timestamp_s)})
        rgb_views[view_id] = view_frames
    frames = rgb_views["primary"]
    timestamps = np.arange(25, dtype=np.float64) * 0.01
    np.savez_compressed(
        tmp_path / "proprio.npz",
        features=np.ones((25, 11), dtype=np.float32),
        timestamp_s=timestamps,
    )
    with (tmp_path / "privileged.jsonl").open("w", encoding="utf-8") as stream:
        for timestamp_s in timestamps:
            stream.write(f'{{"timestamp_s": {float(timestamp_s):.6f}, "base_height_m": 0.31}}\n')
    provenance = tmp_path / "provenance"
    provenance.mkdir()
    (provenance / "scene.usda").write_text("#usda 1.0\n", encoding="utf-8")
    semantic_payload = {
        "n_frames": 5,
        "mean_scheduled_appearance_pixel_fraction": 0.55,
        "mean_scene_context_pixel_fraction": 0.25,
    }
    for view in schedule["appearance_views"]:
        view_id = view["appearance_view_id"]
        (provenance / f"appearance_{view_id}.lock.json").write_text("{}\n", encoding="utf-8")
        (provenance / f"semantic_summary_{view_id}.json").write_text(
            json.dumps(semantic_payload) + "\n", encoding="utf-8"
        )

    schedule_sha256 = "b" * 64
    collector_sha256 = "a" * 64
    runtime_manifest_sha256 = sha256_file(Path(runtime_manifest_module.__file__))
    formal_protocol = {
        "schema_version": FORMAL_PROTOCOL_SCHEMA_VERSION,
        "protocol_id": "unit_test_formal_pilot_v1",
        "status": "frozen",
        "benchmark_id": schedule["benchmark_id"],
        "schedule_sha256": schedule_sha256,
        "collector_sha256": collector_sha256,
        "runtime_manifest_sha256": runtime_manifest_sha256,
        "allowed": {
            "counterfactual_group_ids": [schedule["counterfactual_group_id"]],
            "target_operators": [schedule["target_operator"]],
            "scene_families": [schedule["scene_family"]],
            "physical_realizations": [schedule["physical_realization"]],
            "geometry_profiles": [schedule["geometry_profile"]],
        },
    }
    formal_protocol_path = provenance / "formal_protocol.json"
    formal_protocol_path.write_text(
        json.dumps(formal_protocol, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )

    def artifact(relative: str) -> dict[str, str]:
        import hashlib

        return {
            "path": relative,
            "sha256": hashlib.sha256((tmp_path / relative).read_bytes()).hexdigest(),
        }

    manifest = build_collected_manifest(
        schedule,
        episode_dir=tmp_path,
        rgb_frames=frames,
        rgb_views=rgb_views,
        camera={
            "profile": schedule["camera_profile"],
            "body_fixed": True,
            "pose_sync_method": "rigid_base_transform_each_step",
            "base_to_camera_xyz_quat_xyzw": [0.335, 0.0, 0.065, 0.0, 0.0, 0.0, 1.0],
        },
        operator_readback={
            "operator_id": schedule["target_operator"],
            "active": True,
            "qa_passed": True,
            "applied_parameters": schedule["physics_parameters"],
        },
        scene_readback={
            "qa_passed": True,
            "scene_family": schedule["scene_family"],
            "scene_source": schedule["scene_source"],
            "scene_seed": schedule["scene_seed"],
            "semantic_summary_path": "provenance/semantic_summary_primary.json",
            "artifacts": [
                artifact("provenance/scene.usda"),
                artifact("provenance/semantic_summary_primary.json"),
            ],
        },
        appearance_readback={
            "qa_passed": True,
            "appearance_id": schedule["appearance_id"],
            "material_family": schedule["material_family"],
            "surface_state": schedule["surface_state"],
            "uv_scale": schedule["uv_scale"],
            "uv_rotation_deg": schedule["uv_rotation_deg"],
            "uv_offset": schedule["uv_offset"],
            "albedo_brightness_multiplier": schedule["albedo_brightness_multiplier"],
            "normal_strength": schedule["normal_strength"],
            "roughness_multiplier": schedule["roughness_multiplier"],
            "semantic_summary_path": "provenance/semantic_summary_primary.json",
            "artifacts": [
                artifact("provenance/appearance_primary.lock.json"),
                artifact("provenance/semantic_summary_primary.json"),
            ],
            "views": {
                view["appearance_view_id"]: {
                    "qa_passed": True,
                    "appearance_id": view["appearance_id"],
                    "material_family": view["material_family"],
                    "surface_state": view["surface_state"],
                    "uv_scale": view["uv_scale"],
                    "uv_rotation_deg": view["uv_rotation_deg"],
                    "uv_offset": view["uv_offset"],
                    "albedo_brightness_multiplier": view["albedo_brightness_multiplier"],
                    "normal_strength": view["normal_strength"],
                    "roughness_multiplier": view["roughness_multiplier"],
                    "semantic_summary_path": (
                        f"provenance/semantic_summary_{view['appearance_view_id']}.json"
                    ),
                    "artifacts": [
                        artifact(f"provenance/appearance_{view['appearance_view_id']}.lock.json"),
                        artifact(f"provenance/semantic_summary_{view['appearance_view_id']}.json"),
                    ],
                }
                for view in schedule["appearance_views"]
            },
        },
        geometry_readback={
            "qa_passed": True,
            "geometry_id": schedule["geometry_id"],
            "geometry_profile": schedule["geometry_profile"],
            "physical_realization": schedule["physical_realization"],
        },
        collection={
            "status": "formal_pilot",
            "seed": schedule["scene_seed"],
            "simulator": "unit-test",
            "schedule_sha256": schedule_sha256,
            "collector_sha256": collector_sha256,
            "runtime_manifest_sha256": runtime_manifest_sha256,
            "formal_protocol": {
                "path": "provenance/formal_protocol.json",
                "sha256": sha256_file(formal_protocol_path),
                "protocol_id": formal_protocol["protocol_id"],
                "schedule_sha256": schedule_sha256,
                "collector_sha256": collector_sha256,
                "runtime_manifest_sha256": runtime_manifest_sha256,
            },
        },
    )
    return schedule, manifest


def test_validated_is_the_only_evaluation_eligible_runtime_state(tmp_path: Path) -> None:
    schedule, manifest = _collected_episode(tmp_path)
    path = write_collected_manifest(manifest, tmp_path)
    assert path.exists()
    result = validate_runtime_episode(schedule, episode_dir=tmp_path, write_validated=True)
    assert result.passed, result.issues
    assert result.validated_record["artifact_state"] == "validated"
    assert result.validated_record["evaluation_eligible"] is True
    assert result.manifest["runtime_validation"]["measured"]["max_rgb_proprio_skew_s"] < 1e-9


def test_hash_tampering_cannot_be_hidden_by_a_validated_claim(tmp_path: Path) -> None:
    schedule, manifest = _collected_episode(tmp_path)
    manifest["artifact_state"] = "validated"
    manifest["evaluation_eligible"] = True
    (tmp_path / "rgb/000000.png").write_bytes(_png_bytes(320, 180, 255))
    result = validate_runtime_episode(schedule, episode_dir=tmp_path, manifest=manifest)
    assert not result.passed
    assert "rgb_entry_0_hash_mismatch" in result.issues
    assert result.validated_record["artifact_state"] == "collected"
    assert result.validated_record["evaluation_eligible"] is False


def test_formal_status_without_a_valid_frozen_protocol_is_not_eligible(tmp_path: Path) -> None:
    schedule, manifest = _collected_episode(tmp_path)
    del manifest["collection"]["formal_protocol"]
    result = validate_runtime_episode(schedule, episode_dir=tmp_path, manifest=manifest)
    assert not result.passed
    assert "formal_protocol_reference_missing" in result.issues
    assert result.validated_record["evaluation_eligible"] is False


def test_tampered_formal_protocol_or_scope_is_rejected(tmp_path: Path) -> None:
    schedule, manifest = _collected_episode(tmp_path)
    protocol_path = tmp_path / manifest["collection"]["formal_protocol"]["path"]
    protocol = json.loads(protocol_path.read_text(encoding="utf-8"))
    protocol["allowed"]["scene_families"] = ["different_scene"]
    protocol_path.write_text(json.dumps(protocol) + "\n", encoding="utf-8")
    manifest["collection"]["formal_protocol"]["sha256"] = sha256_file(protocol_path)
    result = validate_runtime_episode(schedule, episode_dir=tmp_path, manifest=manifest)
    assert not result.passed
    assert "formal_protocol_scene_family_not_authorized" in result.issues
    assert result.validated_record["evaluation_eligible"] is False


def test_smoke_artifacts_can_pass_qa_without_becoming_evaluation_eligible(
    tmp_path: Path,
) -> None:
    schedule, manifest = _collected_episode(tmp_path)
    manifest["collection"]["status"] = "smoke_pair_not_formal_pilot"
    result = validate_runtime_episode(schedule, episode_dir=tmp_path, manifest=manifest)
    assert result.passed, result.issues
    assert result.validated_record["artifact_state"] == "validated_smoke"
    assert result.validated_record["evaluation_eligible"] is False
    assert result.gates["promotion_blockers"] == ["collection_protocol_not_formal"]

    schedule = dict(schedule)
    schedule["required_outputs"] = dict(schedule["required_outputs"])
    schedule["required_outputs"]["episode_manifest"] = "manifest.json"
    manifest["schedule_record_sha256"] = schedule_record_sha256(schedule)
    (tmp_path / "manifest.json").write_text(json.dumps(manifest), encoding="utf-8")
    audit = audit_runtime_corpus([schedule], corpus_root=tmp_path)
    assert audit.passed
    assert audit.summary["state_counts"] == {"validated_smoke": 1}
    assert audit.summary["artifact_qa_passed_records"] == 1
    assert audit.summary["evaluation_eligible_records"] == 0
    assert audit.records == ()


def test_camera_and_operator_are_causal_runtime_gates(tmp_path: Path) -> None:
    schedule, manifest = _collected_episode(tmp_path)
    manifest["camera"]["body_fixed"] = False
    manifest["operator_readback"]["applied_parameters"]["mu_d"] = 0.6
    result = validate_runtime_episode(schedule, episode_dir=tmp_path, manifest=manifest)
    assert not result.passed
    assert "camera_not_body_fixed" in result.issues
    assert "operator_parameter_mu_d_mismatch" in result.issues


def test_scene_appearance_and_geometry_are_causal_runtime_gates(tmp_path: Path) -> None:
    schedule, manifest = _collected_episode(tmp_path)
    manifest["scene_readback"]["scene_seed"] += 1
    manifest["appearance_readback"]["appearance_id"] = "wrong_appearance"
    manifest["geometry_readback"]["geometry_profile"] = "rectangle_proxy"
    result = validate_runtime_episode(schedule, episode_dir=tmp_path, manifest=manifest)
    assert not result.passed
    assert "scene_scene_seed_mismatch" in result.issues
    assert "appearance_appearance_id_mismatch" in result.issues
    assert "geometry_geometry_profile_mismatch" in result.issues


def test_artifact_paths_cannot_escape_episode_directory(tmp_path: Path) -> None:
    schedule, manifest = _collected_episode(tmp_path)
    manifest["artifacts"]["rgb_frames"][0]["path"] = "../escape.png"
    manifest["artifacts"]["rgb_views"]["primary"][0]["path"] = "../escape.png"
    result = validate_runtime_episode(schedule, episode_dir=tmp_path, manifest=manifest)
    assert not result.passed
    assert any(issue.startswith("rgb_entry_0_unreadable") for issue in result.issues)


def test_all_scheduled_appearance_views_are_required(tmp_path: Path) -> None:
    schedule, manifest = _collected_episode(tmp_path)
    del manifest["artifacts"]["rgb_views"]["swap_02"]
    result = validate_runtime_episode(schedule, episode_dir=tmp_path, manifest=manifest)
    assert not result.passed
    assert "rgb_view_ids_mismatch" in result.issues


def test_appearance_views_must_share_exact_timestamps(tmp_path: Path) -> None:
    schedule, manifest = _collected_episode(tmp_path)
    manifest["artifacts"]["rgb_views"]["swap_01"][0]["timestamp_s"] = 0.01
    result = validate_runtime_episode(schedule, episode_dir=tmp_path, manifest=manifest)
    assert not result.passed
    assert "rgb_view_swap_01_timestamps_not_synchronized" in result.issues


def test_metadata_only_texture_swap_is_rejected(tmp_path: Path) -> None:
    schedule, manifest = _collected_episode(tmp_path)
    primary = manifest["artifacts"]["rgb_views"]["primary"]
    manifest["artifacts"]["rgb_views"]["swap_01"] = [dict(entry) for entry in primary]
    result = validate_runtime_episode(schedule, episode_dir=tmp_path, manifest=manifest)
    assert not result.passed
    assert "rgb_view_swap_01_appearance_effect_too_small" in result.issues


def test_corpus_audit_separates_missing_from_invalid_artifacts(tmp_path: Path) -> None:
    schedule, manifest = _collected_episode(tmp_path)
    schedule = dict(schedule)
    schedule["required_outputs"] = dict(schedule["required_outputs"])
    schedule["required_outputs"]["episode_manifest"] = "episode/manifest.json"
    episode = tmp_path / "episode"
    episode.mkdir()
    # Move the already-created fixtures below the schedule's episode directory.
    (tmp_path / "rgb").rename(episode / "rgb")
    (tmp_path / "rgb_views").rename(episode / "rgb_views")
    (tmp_path / "proprio.npz").rename(episode / "proprio.npz")
    (tmp_path / "privileged.jsonl").rename(episode / "privileged.jsonl")
    (tmp_path / "provenance").rename(episode / "provenance")
    # The required-output edit changes the canonical schedule hash, so rebuild the manifest.
    rgb_frames = [
        {"path": f"rgb/{index:06d}.png", "timestamp_s": index * 0.05} for index in range(5)
    ]
    rgb_views = {
        view_id: [{"path": entry["path"], "timestamp_s": entry["timestamp_s"]} for entry in entries]
        for view_id, entries in manifest["artifacts"]["rgb_views"].items()
    }
    manifest = build_collected_manifest(
        schedule,
        episode_dir=episode,
        rgb_frames=rgb_frames,
        rgb_views=rgb_views,
        camera=manifest["camera"],
        operator_readback=manifest["operator_readback"],
        scene_readback=manifest["scene_readback"],
        appearance_readback=manifest["appearance_readback"],
        geometry_readback=manifest["geometry_readback"],
        collection=manifest["collection"],
    )
    write_collected_manifest(manifest, episode)
    audit = audit_runtime_corpus([schedule], corpus_root=tmp_path)
    assert audit.passed
    assert audit.summary["state_counts"] == {"validated": 1}
    assert audit.summary["evaluation_eligible_records"] == 1

    missing_root = tmp_path / "empty"
    missing = audit_runtime_corpus([schedule], corpus_root=missing_root)
    assert missing.passed
    assert missing.summary["state_counts"] == {"planned": 1}
    assert not missing.summary["publication_freeze_ready"]
    frozen = audit_runtime_corpus([schedule], corpus_root=missing_root, require_complete=True)
    assert not frozen.passed
    assert "runtime_corpus_incomplete" in frozen.summary["issues"]
