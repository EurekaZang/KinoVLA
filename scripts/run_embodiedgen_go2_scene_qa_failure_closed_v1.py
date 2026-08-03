#!/usr/bin/env python3
"""Run the frozen Go2 scene QA through one audited Isaac Lab launcher.

This wrapper owns process selection only.  The native QA script still owns all
scene, locomotion, camera, and admission decisions.  A process-level failure is
preserved as an immutable exception artifact; a completed native run receives a
launcher receipt that binds the exact executable, setup script, QA script, and
native audit.
"""

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
ISAAC_SETUP = Path("/home/eureka/nvidia/isaacsim/setup_conda_env.sh")
ISAACLAB_PYTHON = Path("/home/eureka/miniconda3/envs/kinovla/bin/python")
GO2_SCRIPT = ROOT / "scripts/isaac_embodiedgen_go2_scene_qa.py"
SHELL = Path("/bin/bash")


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _json(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise TypeError(path)
    return value


def launcher_components() -> list[dict[str, str]]:
    paths = [SHELL, ISAAC_SETUP, ISAACLAB_PYTHON, GO2_SCRIPT]
    missing = [str(path) for path in paths if not path.is_file()]
    if missing:
        raise FileNotFoundError(", ".join(missing))
    return [
        {"path": str(path.resolve()), "sha256": _sha256(path.resolve())}
        for path in paths
    ]


def _write(path: Path, payload: dict[str, Any]) -> None:
    if path.exists():
        raise FileExistsError(f"refusing to overwrite frozen launcher artifact: {path}")
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--scene-id", required=True)
    parser.add_argument("--episode-usd", type=Path, required=True)
    parser.add_argument("--compiled-audit", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--seed", type=int, required=True)
    parser.add_argument("--steps", type=int, default=140)
    parser.add_argument("--target-progress-m", type=float, default=0.5)
    parser.add_argument("--command-speed-mps", type=float, default=0.32)
    parser.add_argument("--cross-track-gain", type=float, default=1.0)
    parser.add_argument("--heading-gain", type=float, default=2.0)
    parser.add_argument("--camera-profile", default="go2_front_calib_c")
    args = parser.parse_args()

    episode = args.episode_usd.resolve()
    compiled = args.compiled_audit.resolve()
    out = args.out.resolve()
    native_audit = out / "go2_scene_audit.json"
    exception_audit = out / "go2_scene_exception_audit.json"
    launcher_audit = out / "go2_launcher_audit.json"
    for path in (episode, compiled):
        if not path.is_file():
            raise FileNotFoundError(path)
    if any(path.exists() for path in (native_audit, exception_audit, launcher_audit)):
        raise FileExistsError(f"Go2 output already opened: {out}")

    components = launcher_components()
    native_args = [
        str(ISAACLAB_PYTHON.resolve()),
        str(GO2_SCRIPT.resolve()),
        "--episode-usd",
        str(episode),
        "--compiled-audit",
        str(compiled),
        "--out",
        str(out),
        "--seed",
        str(args.seed),
        "--steps",
        str(args.steps),
        "--target-progress-m",
        str(args.target_progress_m),
        "--command-speed-mps",
        str(args.command_speed_mps),
        "--route-tracking",
        "--cross-track-gain",
        str(args.cross_track_gain),
        "--heading-gain",
        str(args.heading_gain),
        "--camera-profile",
        args.camera_profile,
        "--headless",
    ]
    command = [
        str(SHELL.resolve()),
        "-lc",
        'source "$1"; shift; exec "$@"',
        "kinofail-go2-launcher",
        str(ISAAC_SETUP.resolve()),
        *native_args,
    ]
    environment = os.environ.copy()
    environment["PYTHONPATH"] = (
        str(ROOT)
        + (":" + environment["PYTHONPATH"] if environment.get("PYTHONPATH") else "")
    )
    environment["OMNI_KIT_ACCEPT_EULA"] = "YES"
    completed = subprocess.run(command, cwd=ROOT, env=environment, check=False)
    common = {
        "created_utc": datetime.now(UTC).isoformat(),
        "scene_id": args.scene_id,
        "episode_usd": {"path": str(episode), "sha256": _sha256(episode)},
        "compiled_audit": {"path": str(compiled), "sha256": _sha256(compiled)},
        "launcher_components": components,
        "native_arguments": native_args[2:],
        "returncode": int(completed.returncode),
    }
    # The native QA intentionally returns 2 for a scientifically valid scene
    # rejection.  That is not a launcher exception: the immutable native audit
    # is the terminal evidence and must remain distinguishable from a process
    # failure that produced no audit at all.
    if not native_audit.is_file():
        payload = {
            "schema_version": "kinofail.embodiedgen-go2-scene-qa-exception.v2",
            **common,
            "passed": False,
            "reason": "native_audit_missing",
            "replacement_authorized": False,
            "formal_operator_run_authorized": False,
        }
        _write(exception_audit, payload)
        print(json.dumps({"out": str(exception_audit), "passed": False}, indent=2))
        return completed.returncode or 2

    native = _json(native_audit)
    payload = {
        "schema_version": "kinofail.embodiedgen-go2-launcher-audit.v1",
        **common,
        "passed": True,
        "native_audit": {
            "path": str(native_audit),
            "sha256": _sha256(native_audit),
            "native_passed": native.get("passed"),
        },
        "admission_state": "launcher_completed_native_audit_preserved",
    }
    _write(launcher_audit, payload)
    print(
        json.dumps(
            {
                "out": str(launcher_audit),
                "passed": True,
                "native_passed": native.get("passed"),
            },
            indent=2,
        )
    )
    # A completed native rejection is represented by native_passed=false, not
    # by a wrapper failure.  Downstream adjudication reads the native audit.
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
