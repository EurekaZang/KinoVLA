#!/usr/bin/env python3
"""Run one model-blind, failure-closed indoor-scene admission candidate.

This entry point is frozen at F0 and is intentionally unaware of model
checkpoints, labels, predictions, or endpoint metrics.  It may only establish
that a newly generated RoomGen scene is structurally valid, renderable from the
Go2 front camera, and traversable under nominal physics.  Every requested
candidate leaves a terminal audit, including failures.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import subprocess
import sys
from datetime import UTC, datetime
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parents[1]
GENERATOR = ROOT / "scripts/run_embodiedgen_room_source_failure_closed_v1.py"
SOURCE_PREFLIGHT = ROOT / "scripts/audit_embodiedgen_source_preflight_v1.py"
BASE_COMPILER = ROOT / "scripts/compile_embodiedgen_kinofail_scene.py"
CORRIDOR_COMPILER = ROOT / "scripts/refine_embodiedgen_kinofail_scene_v2.py"
SURFACE_COMPILER = ROOT / "scripts/refine_embodiedgen_kinofail_scene_v4.py"
TERRAIN_COMPILER = ROOT / "scripts/compile_embodiedgen_terrain_route_v2.py"
TERRAIN_AUDITOR = ROOT / "scripts/audit_embodiedgen_terrain_route_v2.py"
RTX_QA = ROOT / "scripts/isaac_render_embodiedgen_scene_qa_v5.py"
GO2_QA = ROOT / "scripts/run_embodiedgen_go2_scene_qa_failure_closed_v1.py"
ISAAC_PYTHON = Path("/home/eureka/nvidia/isaacsim/python.sh")


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _load_json(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise TypeError(f"expected object: {path}")
    return value


def _run(stage: str, command: list[str], ledger: list[dict[str, Any]]) -> bool:
    started = datetime.now(UTC).isoformat()
    environment = os.environ.copy()
    prior_pythonpath = environment.get("PYTHONPATH", "")
    environment["PYTHONPATH"] = (
        str(ROOT)
        if not prior_pythonpath
        else str(ROOT) + os.pathsep + prior_pythonpath
    )
    completed = subprocess.run(
        command,
        cwd=ROOT,
        env=environment,
        check=False,
    )
    ledger.append(
        {
            "stage": stage,
            "started_utc": started,
            "completed_utc": datetime.now(UTC).isoformat(),
            "command": command,
            "returncode": int(completed.returncode),
            "passed": completed.returncode == 0,
        }
    )
    return completed.returncode == 0


def _candidate(config: dict[str, Any], scene_id: str) -> dict[str, Any]:
    matches = [
        dict(row)
        for row in config["indoor_candidates"]
        if str(row["scene_id"]) == scene_id
    ]
    if len(matches) != 1:
        raise KeyError(f"scene_id must resolve exactly once: {scene_id}")
    return matches[0]


def _expected_qa_material(
    config: dict[str, Any],
    candidate: dict[str, Any],
) -> str:
    domain = str(candidate["domain"])
    stream = [
        row
        for row in config["indoor_candidates"]
        if str(row["domain"]) == domain
    ]
    index = next(
        position
        for position, row in enumerate(stream)
        if str(row["scene_id"]) == str(candidate["scene_id"])
    )
    return f"confirm_v1_{domain}_pbr_{index % 10:02d}"


def _admission_decision(
    *,
    failure_stage: str | None,
    evidence: dict[str, dict[str, Any]],
    expected_names: set[str],
) -> bool:
    """Require every audit to pass while treating the source manifest as identity evidence."""

    audit_names = expected_names - {"source_manifest"}
    return (
        failure_stage is None
        and set(evidence) == expected_names
        and "source_manifest" in evidence
        and all(evidence[name].get("passed") is True for name in audit_names)
    )


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--config",
        type=Path,
        default=ROOT
        / "configs/data/kinofail_unified_confirmatory_scene_candidates_v1.json",
    )
    parser.add_argument("--scene-id", required=True)
    parser.add_argument("--material-lock", type=Path, required=True)
    parser.add_argument("--material-id", required=True)
    args = parser.parse_args()

    config_path = args.config.resolve()
    material_lock = args.material_lock.resolve()
    config = _load_json(config_path)
    row = _candidate(config, args.scene_id)
    expected_material_id = _expected_qa_material(config, row)
    if args.material_id != expected_material_id:
        raise RuntimeError(
            "candidate QA material differs from the F0 rule: "
            f"expected {expected_material_id}, received {args.material_id}"
        )
    indoor = dict(config["indoor_pipeline"])
    source_root = (ROOT / str(indoor["source_root"])).resolve()
    compiled_root = (ROOT / str(indoor["compiled_root"])).resolve()
    candidate_root = compiled_root / args.scene_id
    if candidate_root.exists():
        raise FileExistsError(
            f"refusing to overwrite confirmatory candidate: {candidate_root}"
        )
    if not material_lock.is_file():
        raise FileNotFoundError(material_lock)
    if not ISAAC_PYTHON.is_file():
        raise FileNotFoundError(ISAAC_PYTHON)
    lock = _load_json(material_lock)
    material_ids = {str(value["id"]) for value in lock.get("materials", [])}
    if args.material_id not in material_ids:
        raise KeyError(f"material absent from confirmatory lock: {args.material_id}")

    candidate_root.mkdir(parents=True, exist_ok=False)
    terminal_path = candidate_root / "terminal_scene_admission.json"
    ledger: list[dict[str, Any]] = []
    room_type = str(row["room_type"])
    source_seed = int(row["source_seed"])
    runtime_seed = int(row["runtime_seed"])
    asset_root = source_root / f"{room_type}_seed{source_seed}"
    source_manifest = asset_root / "source_manifest.json"
    source_preflight = (
        asset_root
        / "source_geometry_preflight_v1"
        / "source_geometry_preflight_audit.json"
    )
    base_root = candidate_root / "kinofail_base_v1"
    corridor_root = candidate_root / "kinofail_corridor_v2"
    surface_root = candidate_root / "route_surface_v4"
    terrain_root = surface_root / "terrain_route_v2"
    rtx_root = surface_root / "rtx_qa"
    go2_root = terrain_root / "go2_qa"

    source_python = sys.executable
    isaac_python = str(ISAAC_PYTHON)
    commands = [
        (
            "source_generation",
            [
                source_python,
                str(GENERATOR),
                "--embodiedgen-root",
                str(Path(indoor["embodiedgen_root"]).resolve()),
                "--output-root",
                str(source_root),
                "--scene-id",
                args.scene_id,
                "--room-type",
                room_type,
                "--seed",
                str(source_seed),
                "--complexity",
                str(row["complexity"]),
                "--exception-out",
                str(candidate_root / "source_generation_exception.json"),
            ],
        ),
        (
            "source_geometry_preflight",
            [
                isaac_python,
                str(SOURCE_PREFLIGHT),
                "--source-manifest",
                str(source_manifest),
                "--out",
                str(source_preflight),
            ],
        ),
        (
            "base_compile",
            [
                isaac_python,
                str(BASE_COMPILER),
                "--source-manifest",
                str(source_manifest),
                "--output-dir",
                str(base_root),
            ],
        ),
        (
            "corridor_compile",
            [
                isaac_python,
                str(CORRIDOR_COMPILER),
                "--base-compiled-audit",
                str(base_root / "compiled_scene_audit.json"),
                "--output-dir",
                str(corridor_root),
            ],
        ),
        (
            "route_surface_compile",
            [
                isaac_python,
                str(SURFACE_COMPILER),
                "--corridor-v2-audit",
                str(corridor_root / "compiled_scene_audit.json"),
                "--output-dir",
                str(surface_root),
                "--material-id",
                args.material_id,
                "--material-lock",
                str(material_lock),
                "--dome-intensity",
                str(indoor["dome_intensity"]),
                "--panel-intensity",
                str(indoor["panel_intensity"]),
                "--panel-exposure",
                str(indoor["panel_exposure"]),
                "--route-surface-width-m",
                str(indoor["route_surface_width_m"]),
            ],
        ),
        (
            "terrain_compile",
            [
                isaac_python,
                str(TERRAIN_COMPILER),
                "--base-audit",
                str(surface_root / "compiled_scene_audit.json"),
                "--out",
                str(terrain_root),
                "--static-friction",
                str(indoor["terrain_static_friction"]),
                "--dynamic-friction",
                str(indoor["terrain_dynamic_friction"]),
                "--max-spacing-m",
                str(indoor["terrain_max_spacing_m"]),
            ],
        ),
        (
            "terrain_offline_audit",
            [
                isaac_python,
                str(TERRAIN_AUDITOR),
                "--compiled-audit",
                str(terrain_root / "compiled_scene_audit.json"),
                "--out",
                str(terrain_root / "offline_usd_audit.json"),
            ],
        ),
        (
            "rtx_qa",
            [
                isaac_python,
                str(RTX_QA),
                "--episode-usd",
                str(surface_root / "episode_v4.usda"),
                "--compiled-audit",
                str(surface_root / "compiled_scene_audit.json"),
                "--out",
                str(rtx_root),
                "--width",
                str(indoor["rtx_width"]),
                "--height",
                str(indoor["rtx_height"]),
                "--warmup-frames",
                str(indoor["rtx_warmup_frames"]),
                "--headless",
            ],
        ),
        (
            "go2_nominal_qa",
            [
                source_python,
                str(GO2_QA),
                "--scene-id",
                args.scene_id,
                "--episode-usd",
                str(terrain_root / "episode_terrain_v2.usda"),
                "--compiled-audit",
                str(terrain_root / "compiled_scene_audit.json"),
                "--out",
                str(go2_root),
                "--seed",
                str(runtime_seed),
                "--steps",
                str(indoor["go2_qa_steps"]),
                "--target-progress-m",
                str(indoor["go2_qa_target_progress_m"]),
                "--command-speed-mps",
                str(indoor["go2_qa_command_speed_mps"]),
                "--cross-track-gain",
                "1.0",
                "--heading-gain",
                "2.0",
                "--camera-profile",
                str(indoor["go2_qa_camera_profile"]),
            ],
        ),
    ]

    failure_stage: str | None = None
    for stage, command in commands:
        if not _run(stage, command, ledger):
            failure_stage = stage
            break

    evidence_paths = {
        "source_manifest": source_manifest,
        "source_preflight": source_preflight,
        "base_audit": base_root / "compiled_scene_audit.json",
        "corridor_audit": corridor_root / "compiled_scene_audit.json",
        "surface_audit": surface_root / "compiled_scene_audit.json",
        "terrain_audit": terrain_root / "compiled_scene_audit.json",
        "terrain_offline_audit": terrain_root / "offline_usd_audit.json",
        "rtx_audit": rtx_root / "rtx_scene_audit.json",
        "go2_audit": go2_root / "go2_scene_audit.json",
    }
    evidence: dict[str, Any] = {}
    for name, path in evidence_paths.items():
        if path.is_file():
            value = _load_json(path)
            evidence[name] = {
                "path": str(path),
                "sha256": _sha256(path),
                "passed": value.get("passed"),
            }
    admitted = _admission_decision(
        failure_stage=failure_stage,
        evidence=evidence,
        expected_names=set(evidence_paths),
    )
    terminal = {
        "schema_version": "kinofail.confirmatory-scene-admission.v1",
        "created_utc": datetime.now(UTC).isoformat(),
        "scene_id": args.scene_id,
        "domain": row["domain"],
        "candidate": row,
        "model_blind": True,
        "model_or_endpoint_files_read": False,
        "single_attempt": True,
        "material_id": args.material_id,
        "material_lock": {
            "path": str(material_lock),
            "sha256": _sha256(material_lock),
        },
        "config": {"path": str(config_path), "sha256": _sha256(config_path)},
        "ledger": ledger,
        "evidence": evidence,
        "failure_stage": failure_stage,
        "admitted": admitted,
        "episode_usd": (
            str(terrain_root / "episode_terrain_v2.usda") if admitted else None
        ),
        "confirmatory_model_inference_authorized": False,
    }
    terminal_path.write_text(
        json.dumps(terminal, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    print(json.dumps({"scene_id": args.scene_id, "admitted": admitted}))
    return 0 if admitted else 2


if __name__ == "__main__":
    raise SystemExit(main())
