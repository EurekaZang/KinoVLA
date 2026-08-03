#!/usr/bin/env python3
"""Generate one frozen EmbodiedGen V2 RoomGen source and write Kino-Fail provenance."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import subprocess
from datetime import UTC, datetime
from pathlib import Path

from kino_vla.sim.embodiedgen_asset import (
    COMPLEXITIES,
    COMPATIBILITY_SHIM_ID,
    COMPATIBILITY_SHIM_PATH,
    OFFICIAL_COMMIT,
    OFFICIAL_INFINIGEN_COMMIT,
    OFFICIAL_RELEASE,
    ROOM_TYPES,
    ROOM_TYPE_CLI,
    EmbodiedGenRoomSource,
    audit_embodiedgen_room_manifest,
    write_embodiedgen_room_manifest,
)


def _git(repo: Path, *args: str) -> str:
    return subprocess.check_output(["git", "-C", str(repo), *args], text=True).strip()


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _verify_generation_environment(
    blender_python: Path,
    infinigen_root: Path,
    environment: dict[str, str],
    repo: Path,
) -> None:
    """Fail early unless the child process sees the audited Gin filesystem route."""
    probe = """
import json
import sys
import gin
from gin import config

resource_readers = [
    getattr(exists, "__module__", "")
    for _, exists in config._FILE_READERS
    if getattr(exists, "__module__", "") == "gin.resource_reader"
]
print(json.dumps({
    "sitecustomize_loaded": "sitecustomize" in sys.modules,
    "location_prefixes": config._LOCATION_PREFIXES,
    "resource_readers": resource_readers,
}))
"""
    completed = subprocess.run(
        [str(blender_python), "-c", probe],
        cwd=repo,
        env=environment,
        check=True,
        capture_output=True,
        text=True,
    )
    try:
        state = json.loads(completed.stdout.strip().splitlines()[-1])
    except (IndexError, json.JSONDecodeError) as exc:
        raise RuntimeError("unable to audit EmbodiedGen child Python environment") from exc
    if state.get("sitecustomize_loaded") is not True:
        raise RuntimeError("EmbodiedGen compatibility shim was not loaded")
    if str(infinigen_root) not in state.get("location_prefixes", []):
        raise RuntimeError("frozen Infinigen Gin search prefix was not registered")
    if state.get("resource_readers"):
        raise RuntimeError("incompatible Gin namespace resource reader remains active")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--embodiedgen-root", type=Path, required=True)
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--scene-id", required=True)
    parser.add_argument("--room-type", choices=sorted(ROOM_TYPES), default="Office")
    parser.add_argument("--seed", type=int, required=True)
    parser.add_argument("--complexity", choices=sorted(COMPLEXITIES), default="simple")
    parser.add_argument(
        "--skip-generation",
        action="store_true",
        help="Only freeze an already-complete RoomGen output.",
    )
    args = parser.parse_args()

    repo = args.embodiedgen_root.resolve()
    if _git(repo, "rev-parse", "HEAD") != OFFICIAL_COMMIT:
        raise SystemExit(f"EmbodiedGen HEAD is not frozen {OFFICIAL_RELEASE}/{OFFICIAL_COMMIT}")
    if _git(repo, "describe", "--tags", "--exact-match") != OFFICIAL_RELEASE:
        raise SystemExit(f"EmbodiedGen checkout is not exact tag {OFFICIAL_RELEASE}")
    infinigen_root = repo / "thirdparty/infinigen"
    if _git(infinigen_root, "rev-parse", "HEAD") != OFFICIAL_INFINIGEN_COMMIT:
        raise SystemExit(f"Infinigen checkout is not frozen {OFFICIAL_INFINIGEN_COMMIT}")
    blender_python = repo / "thirdparty/infinigen/blender/4.2/python/bin/python3.11"
    if not blender_python.is_file():
        raise SystemExit(f"RoomGen Blender Python missing: {blender_python}")

    output_root = args.output_root.resolve()
    asset_root = output_root / f"{args.room_type}_seed{args.seed}"
    command = [
        str(blender_python),
        "-m",
        "embodied_gen.scripts.room_gen.gen_room",
        "--output-root",
        str(output_root),
        "--room-type",
        ROOM_TYPE_CLI[args.room_type],
        "--seed",
        str(args.seed),
        "--complexity",
        args.complexity,
        "--no-urdf",
    ]
    request = {
        "schema_version": "kinofail.embodiedgen-generation-request.v1",
        "created_utc": datetime.now(UTC).isoformat(),
        "scene_id": args.scene_id,
        "embodiedgen_release": OFFICIAL_RELEASE,
        "embodiedgen_commit": OFFICIAL_COMMIT,
        "infinigen_commit": OFFICIAL_INFINIGEN_COMMIT,
        "room_type": args.room_type,
        "seed": args.seed,
        "complexity": args.complexity,
        "routing": "explicit_no_gpt",
        "command": command,
        "compatibility_shim": {
            "id": COMPATIBILITY_SHIM_ID,
            "path": str(COMPATIBILITY_SHIM_PATH),
            "sha256": _sha256(COMPATIBILITY_SHIM_PATH),
            "modifies_upstream_source": False,
        },
    }
    output_root.mkdir(parents=True, exist_ok=True)
    request_path = output_root / f"{args.room_type}_seed{args.seed}_request.json"
    request_path.write_text(json.dumps(request, indent=2) + "\n", encoding="utf-8")

    if not args.skip_generation:
        environment = os.environ.copy()
        environment["BLENDER_PYTHON"] = str(blender_python)
        environment["KINO_EMBODIEDGEN_INFINIGEN_ROOT"] = str(infinigen_root)
        existing_pythonpath = environment.get("PYTHONPATH", "")
        shim_dir = str(COMPATIBILITY_SHIM_PATH.parent)
        environment["PYTHONPATH"] = (
            f"{shim_dir}{os.pathsep}{existing_pythonpath}" if existing_pythonpath else shim_dir
        )
        _verify_generation_environment(blender_python, infinigen_root, environment, repo)
        subprocess.run(command, cwd=repo, env=environment, check=True)

    visual_usd = asset_root / "usd/export_scene/export_scene.usdc"
    source = EmbodiedGenRoomSource(
        scene_id=args.scene_id,
        asset_root=asset_root,
        visual_usd=visual_usd,
        room_type=args.room_type,
        seed=args.seed,
        complexity=args.complexity,
        generation_command=tuple(command),
    )
    manifest_path = asset_root / "source_manifest.json"
    write_embodiedgen_room_manifest(source, manifest_path)
    audit = audit_embodiedgen_room_manifest(manifest_path)
    audit_path = asset_root / "source_integrity_audit.json"
    audit_path.write_text(json.dumps(audit, indent=2) + "\n", encoding="utf-8")
    if not audit["passed"]:
        raise SystemExit(f"generated RoomGen source audit failed: {audit}")
    print(manifest_path)
    print(audit_path)


if __name__ == "__main__":
    main()
