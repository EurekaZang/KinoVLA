#!/usr/bin/env python3
"""Execute one frozen v28 operator-blind realistic-scene candidate fail-closed."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import subprocess
from datetime import UTC, datetime
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parents[1]
SHELL = Path("/bin/bash")
ISAAC_SETUP = Path("/home/eureka/nvidia/isaacsim/setup_conda_env.sh")
ISAAC_PYTHON = Path("/home/eureka/miniconda3/envs/kinovla/bin/python")


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _json(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise TypeError(f"expected JSON object: {path}")
    return value


def _resolve(value: str | Path) -> Path:
    path = Path(value)
    return (path if path.is_absolute() else ROOT / path).resolve()


def _environment() -> dict[str, str]:
    environment = os.environ.copy()
    environment["PYTHONPATH"] = str(ROOT) + (
        ":" + environment["PYTHONPATH"] if environment.get("PYTHONPATH") else ""
    )
    environment["OMNI_KIT_ACCEPT_EULA"] = "YES"
    return environment


def _isaac_command(script: Path, arguments: list[str]) -> list[str]:
    return [
        str(SHELL),
        "-lc",
        'source "$1"; shift; exec "$@"',
        "kinofail-v28-scene-candidate",
        str(ISAAC_SETUP),
        str(ISAAC_PYTHON),
        str(script),
        *arguments,
    ]


def _run_stage(
    *,
    name: str,
    command: list[str],
    audit_path: Path,
    environment: dict[str, str],
) -> tuple[bool, dict[str, Any]]:
    if audit_path.exists():
        raise FileExistsError(f"stage audit already exists: {audit_path}")
    started = datetime.now(UTC).isoformat()
    completed = subprocess.run(
        command,
        cwd=ROOT,
        env=environment,
        check=False,
    )
    audit_exists = audit_path.is_file()
    audit: dict[str, Any] = {}
    parse_error: str | None = None
    if audit_exists:
        try:
            audit = _json(audit_path)
        except (OSError, json.JSONDecodeError, TypeError) as exc:
            parse_error = f"{type(exc).__name__}:{exc}"
    passed = (
        completed.returncode == 0
        and audit_exists
        and parse_error is None
        and audit.get("passed") is True
    )
    return passed, {
        "stage": name,
        "started_utc": started,
        "finished_utc": datetime.now(UTC).isoformat(),
        "command": command,
        "returncode": int(completed.returncode),
        "audit": str(audit_path),
        "audit_exists": audit_exists,
        "audit_sha256": _sha256(audit_path) if audit_exists else None,
        "audit_passed": audit.get("passed") if parse_error is None else None,
        "audit_parse_error": parse_error,
        "passed": passed,
    }


def _write_receipt(path: Path, payload: dict[str, Any]) -> None:
    if path.exists():
        raise FileExistsError(f"refusing to overwrite candidate receipt: {path}")
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )


def _script(name: str) -> Path:
    return ROOT / "scripts" / name


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--preflight", type=Path, required=True)
    parser.add_argument("--expected-config-sha256", required=True)
    parser.add_argument("--scene-id", required=True)
    args = parser.parse_args()

    for required in (SHELL, ISAAC_SETUP, ISAAC_PYTHON):
        if not required.is_file():
            raise FileNotFoundError(required)
    config_path = args.config.resolve()
    preflight_path = args.preflight.resolve()
    config = _json(config_path)
    preflight = _json(preflight_path)
    config_hash = _sha256(config_path)
    if config_hash != args.expected_config_sha256:
        raise RuntimeError("config no longer matches frozen command hash")
    if preflight.get("passed") is not True or preflight.get("config_sha256") != config_hash:
        raise RuntimeError("v28 preflight is absent, failed, or bound to another config")
    if config.get("schema_version") != "kinofail.o4-scene-stream-v28-freeze.v1":
        raise RuntimeError("unsupported v28 config schema")
    candidates = [
        item for item in config["candidate_stream"] if item["scene_id"] == args.scene_id
    ]
    if len(candidates) != 1:
        raise RuntimeError(f"scene must appear exactly once in frozen stream: {args.scene_id}")
    scene = candidates[0]
    pipeline = config["pipeline"]
    source_root = _resolve(pipeline["source_root"])
    source = source_root / scene["output_directory"]
    request = source_root / f"{scene['output_directory']}_request.json"
    generation_exception = (
        source_root / f"{scene['output_directory']}_generation_exception_audit.json"
    )
    receipt = source_root / f"{scene['output_directory']}_v28_candidate_receipt.json"
    if any(path.exists() for path in (source, request, generation_exception, receipt)):
        raise FileExistsError(f"candidate has already been opened: {scene['scene_id']}")

    route = source / pipeline["route_output_name"] / scene["material_id"]
    terrain = route / pipeline["terrain_output_name"]
    environment = _environment()
    stages: list[dict[str, Any]] = []
    common_receipt = {
        "schema_version": "kinofail.o4-scene-candidate-v28-receipt.v1",
        "created_utc": datetime.now(UTC).isoformat(),
        "config": str(config_path),
        "config_sha256": config_hash,
        "preflight": str(preflight_path),
        "preflight_sha256": _sha256(preflight_path),
        "scene_id": scene["scene_id"],
        "room_type": scene["room_type"],
        "source_seed": scene["source_seed"],
        "runtime_seed": scene["runtime_seed"],
        "material_id": scene["material_id"],
        "development_only": True,
        "counts_as_a0_a7_evidence": False,
        "o4_outcomes_observed": False,
    }

    def finish(*, passed: bool, terminal_stage: str, reason: str) -> int:
        _write_receipt(
            receipt,
            {
                **common_receipt,
                "finished_utc": datetime.now(UTC).isoformat(),
                "stages": stages,
                "terminal_stage": terminal_stage,
                "reason": reason,
                "passed": passed,
                "fully_admitted": passed,
                "sealed": True,
            },
        )
        print(
            json.dumps(
                {
                    "scene_id": scene["scene_id"],
                    "receipt": str(receipt),
                    "passed": passed,
                    "terminal_stage": terminal_stage,
                },
                indent=2,
            )
        )
        return 0 if passed else 2

    try:
        generation_command = [
            str(ISAAC_PYTHON),
            str(_resolve(pipeline["generation_runner"])),
            "--embodiedgen-root",
            str(_resolve(pipeline["embodiedgen_root"])),
            "--output-root",
            str(source_root),
            "--scene-id",
            scene["scene_id"],
            "--room-type",
            scene["room_type"],
            "--seed",
            str(scene["source_seed"]),
            "--complexity",
            scene["complexity"],
            "--exception-out",
            str(generation_exception),
        ]
        started = datetime.now(UTC).isoformat()
        generated = subprocess.run(
            generation_command, cwd=ROOT, env=environment, check=False
        )
        generation_audit = (
            source / "source_integrity_audit.json"
            if (source / "source_integrity_audit.json").is_file()
            else generation_exception
        )
        generation_payload = _json(generation_audit) if generation_audit.is_file() else {}
        generation_passed = (
            generated.returncode == 0
            and request.is_file()
            and (source / "source_manifest.json").is_file()
            and generation_audit.name == "source_integrity_audit.json"
            and generation_payload.get("passed") is True
        )
        stages.append(
            {
                "stage": "generation",
                "started_utc": started,
                "finished_utc": datetime.now(UTC).isoformat(),
                "command": generation_command,
                "returncode": int(generated.returncode),
                "request": str(request),
                "request_exists": request.is_file(),
                "request_sha256": _sha256(request) if request.is_file() else None,
                "audit": str(generation_audit),
                "audit_exists": generation_audit.is_file(),
                "audit_sha256": _sha256(generation_audit) if generation_audit.is_file() else None,
                "audit_passed": generation_payload.get("passed"),
                "passed": generation_passed,
            }
        )
        if not generation_passed:
            return finish(
                passed=False,
                terminal_stage="generation",
                reason="generation wrapper did not produce a passed source-integrity audit",
            )

        stage_specs: list[tuple[str, list[str], Path]] = [
            (
                "source_preflight",
                [
                    str(ISAAC_PYTHON),
                    str(_script("audit_embodiedgen_source_preflight_v1.py")),
                    "--source-manifest",
                    str(source / "source_manifest.json"),
                    "--out",
                    str(source / "source_geometry_preflight_v1/source_geometry_preflight_audit.json"),
                ],
                source / "source_geometry_preflight_v1/source_geometry_preflight_audit.json",
            ),
            (
                "base_compile",
                [
                    str(ISAAC_PYTHON),
                    str(_script("compile_embodiedgen_kinofail_scene.py")),
                    "--source-manifest",
                    str(source / "source_manifest.json"),
                    "--output-dir",
                    str(source / pipeline["base_output_name"]),
                ],
                source / pipeline["base_output_name"] / "compiled_scene_audit.json",
            ),
            (
                "corridor",
                [
                    str(ISAAC_PYTHON),
                    str(_script("refine_embodiedgen_kinofail_scene_v2.py")),
                    "--base-compiled-audit",
                    str(source / pipeline["base_output_name"] / "compiled_scene_audit.json"),
                    "--output-dir",
                    str(source / pipeline["corridor_output_name"]),
                ],
                source / pipeline["corridor_output_name"] / "compiled_scene_audit.json",
            ),
            (
                "route",
                [
                    str(ISAAC_PYTHON),
                    str(_script("refine_embodiedgen_kinofail_scene_v4.py")),
                    "--corridor-v2-audit",
                    str(source / pipeline["corridor_output_name"] / "compiled_scene_audit.json"),
                    "--output-dir",
                    str(route),
                    "--material-id",
                    scene["material_id"],
                    "--material-lock",
                    str(_resolve(config["material_lock"]["path"])),
                    "--dome-intensity",
                    str(pipeline["dome_intensity"]),
                    "--panel-intensity",
                    str(pipeline["panel_intensity"]),
                    "--panel-exposure",
                    str(pipeline["panel_exposure"]),
                    "--route-surface-width-m",
                    str(pipeline["route_surface_width_m"]),
                ],
                route / "compiled_scene_audit.json",
            ),
        ]
        for name, command, audit_path in stage_specs:
            passed, record = _run_stage(
                name=name,
                command=command,
                audit_path=audit_path,
                environment=environment,
            )
            stages.append(record)
            if not passed:
                return finish(
                    passed=False,
                    terminal_stage=name,
                    reason=f"{name} did not produce a passed immutable audit",
                )

        isaac_specs: list[tuple[str, Path, list[str], Path]] = [
            (
                "rtx",
                _script("isaac_render_embodiedgen_scene_qa_v5.py"),
                [
                    "--episode-usd",
                    str(route / "episode_v4.usda"),
                    "--compiled-audit",
                    str(route / "compiled_scene_audit.json"),
                    "--out",
                    str(route / "rtx_qa"),
                    "--width",
                    str(pipeline["rtx_width"]),
                    "--height",
                    str(pipeline["rtx_height"]),
                    "--warmup-frames",
                    str(pipeline["rtx_warmup_frames"]),
                    "--headless",
                ],
                route / "rtx_qa/rtx_scene_audit.json",
            ),
            (
                "motion_proxy",
                _script("isaac_render_embodiedgen_route_motion_qa_v1.py"),
                [
                    "--episode-usd",
                    str(route / "episode_v4.usda"),
                    "--compiled-audit",
                    str(route / "compiled_scene_audit.json"),
                    "--out",
                    str(route / pipeline["motion_proxy_output_name"]),
                    "--width",
                    str(pipeline["motion_proxy_width"]),
                    "--height",
                    str(pipeline["motion_proxy_height"]),
                    "--warmup-frames",
                    str(pipeline["motion_proxy_warmup_frames"]),
                    "--camera-profile",
                    pipeline["go2_camera_profile"],
                    "--headless",
                ],
                route / pipeline["motion_proxy_output_name"] / "route_motion_rtx_audit.json",
            ),
        ]
        for name, script, arguments, audit_path in isaac_specs:
            passed, record = _run_stage(
                name=name,
                command=_isaac_command(script, arguments),
                audit_path=audit_path,
                environment=environment,
            )
            stages.append(record)
            if not passed:
                return finish(
                    passed=False,
                    terminal_stage=name,
                    reason=f"{name} did not produce a passed immutable audit",
                )

        terrain_specs: list[tuple[str, list[str], Path]] = [
            (
                "terrain_compile",
                [
                    str(ISAAC_PYTHON),
                    str(_script("compile_embodiedgen_terrain_route_v2.py")),
                    "--base-audit",
                    str(route / "compiled_scene_audit.json"),
                    "--expected-base-audit-sha256",
                    _sha256(route / "compiled_scene_audit.json"),
                    "--out",
                    str(terrain),
                    "--static-friction",
                    str(pipeline["terrain_static_friction"]),
                    "--dynamic-friction",
                    str(pipeline["terrain_dynamic_friction"]),
                    "--max-spacing-m",
                    str(pipeline["terrain_max_spacing_m"]),
                ],
                terrain / "compiled_scene_audit.json",
            ),
            (
                "terrain_offline",
                [
                    str(ISAAC_PYTHON),
                    str(_script("audit_embodiedgen_terrain_route_v2.py")),
                    "--compiled-audit",
                    str(terrain / "compiled_scene_audit.json"),
                    "--out",
                    str(terrain / "offline_usd_audit.json"),
                ],
                terrain / "offline_usd_audit.json",
            ),
        ]
        for name, command, audit_path in terrain_specs:
            passed, record = _run_stage(
                name=name,
                command=command,
                audit_path=audit_path,
                environment=environment,
            )
            stages.append(record)
            if not passed:
                return finish(
                    passed=False,
                    terminal_stage=name,
                    reason=f"{name} did not produce a passed immutable audit",
                )

        go2_command = [
            str(ISAAC_PYTHON),
            str(_resolve(pipeline["go2_runner"])),
            "--scene-id",
            scene["scene_id"],
            "--episode-usd",
            str(terrain / "episode_terrain_v2.usda"),
            "--compiled-audit",
            str(terrain / "compiled_scene_audit.json"),
            "--out",
            str(terrain / "go2_qa"),
            "--seed",
            str(scene["runtime_seed"]),
            "--steps",
            str(pipeline["go2_steps"]),
            "--target-progress-m",
            str(pipeline["go2_target_progress_m"]),
            "--command-speed-mps",
            str(pipeline["go2_command_speed_mps"]),
            "--cross-track-gain",
            str(pipeline["go2_cross_track_gain"]),
            "--heading-gain",
            str(pipeline["go2_heading_gain"]),
            "--camera-profile",
            pipeline["go2_camera_profile"],
        ]
        passed, record = _run_stage(
            name="go2",
            command=go2_command,
            audit_path=terrain / "go2_qa/go2_launcher_audit.json",
            environment=environment,
        )
        stages.append(record)
        native_go2 = terrain / "go2_qa/go2_scene_audit.json"
        if passed:
            native_payload = _json(native_go2) if native_go2.is_file() else {}
            passed = native_go2.is_file() and native_payload.get("passed") is True
            record["native_audit"] = str(native_go2)
            record["native_audit_exists"] = native_go2.is_file()
            record["native_audit_sha256"] = _sha256(native_go2) if native_go2.is_file() else None
            record["native_audit_passed"] = native_payload.get("passed")
            record["passed"] = passed
        if not passed:
            return finish(
                passed=False,
                terminal_stage="go2",
                reason="Go2 launcher/native audit did not both pass",
            )

        stack_specs: list[tuple[str, list[str], Path]] = [
            (
                "stack_v16",
                [
                    str(ISAAC_PYTHON),
                    str(_script("audit_embodiedgen_realistic_stack_v16_admission.py")),
                    "--source-preflight",
                    str(source / "source_geometry_preflight_v1/source_geometry_preflight_audit.json"),
                    "--route-compiled-audit",
                    str(route / "compiled_scene_audit.json"),
                    "--rtx-audit",
                    str(route / "rtx_qa/rtx_scene_audit.json"),
                    "--terrain-compiled-audit",
                    str(terrain / "compiled_scene_audit.json"),
                    "--terrain-offline-audit",
                    str(terrain / "offline_usd_audit.json"),
                    "--go2-audit",
                    str(native_go2),
                    "--out",
                    str(terrain / "realistic_stack_v16_admission.json"),
                ],
                terrain / "realistic_stack_v16_admission.json",
            ),
            (
                "stack_v17",
                [
                    str(ISAAC_PYTHON),
                    str(_script("audit_embodiedgen_realistic_stack_v17_admission.py")),
                    "--stack-v16-audit",
                    str(terrain / "realistic_stack_v16_admission.json"),
                    "--motion-proxy-audit",
                    str(route / pipeline["motion_proxy_output_name"] / "route_motion_rtx_audit.json"),
                    "--route-compiled-audit",
                    str(route / "compiled_scene_audit.json"),
                    "--out",
                    str(terrain / "realistic_stack_v17_admission.json"),
                ],
                terrain / "realistic_stack_v17_admission.json",
            ),
        ]
        for name, command, audit_path in stack_specs:
            passed, record = _run_stage(
                name=name,
                command=command,
                audit_path=audit_path,
                environment=environment,
            )
            stages.append(record)
            if not passed:
                return finish(
                    passed=False,
                    terminal_stage=name,
                    reason=f"{name} did not produce a passed immutable audit",
                )
        return finish(
            passed=True,
            terminal_stage="stack_v17",
            reason="all frozen nominal-only realistic-stack gates passed",
        )
    except Exception as exc:  # noqa: BLE001 - process failure is immutable evidence
        return finish(
            passed=False,
            terminal_stage="runner_exception",
            reason=f"{type(exc).__name__}:{exc}",
        )


if __name__ == "__main__":
    raise SystemExit(main())
