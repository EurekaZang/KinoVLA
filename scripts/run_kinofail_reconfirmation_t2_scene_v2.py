#!/usr/bin/env python3
"""Launch one frozen T2 scene once, with a pre-execution audit record."""

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
ISAACLAB = Path("/home/eureka/IsaacLab-v2.3.0/isaaclab.sh")
EXPERIENCE = Path(
    "/home/eureka/IsaacLab-v2.3.0/apps/"
    "isaaclab.python.headless.rendering.kit"
)
CONDA_PREFIX = Path("/home/eureka/miniconda3/envs/kinovla")
DEFAULT_COLLECTOR = (
    ROOT / "scripts/isaac_collect_kinofail_confirmatory_t2_v1.py"
)
ISAACLAB_PYTHONPATH = (
    "/home/eureka/IsaacLab-v2.3.0/source/isaaclab:"
    "/home/eureka/IsaacLab-v2.3.0/source/isaaclab_tasks:"
    "/home/eureka/IsaacLab-v2.3.0/source/isaaclab_assets:"
    "/home/eureka/IsaacLab-v2.3.0/source/isaaclab_rl:"
    "/home/eureka/KinoVLA"
)


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


def _jsonl(path: Path) -> list[dict[str, Any]]:
    return [
        json.loads(line)
        for line in path.read_text(encoding="utf-8").splitlines()
        if line
    ]


def _write_json(path: Path, value: dict[str, Any]) -> None:
    temporary = path.with_suffix(".tmp")
    temporary.write_text(
        json.dumps(value, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    os.replace(temporary, path)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--schedule", type=Path, required=True)
    parser.add_argument("--scene-registry", type=Path, required=True)
    parser.add_argument("--asset-lock", type=Path, required=True)
    parser.add_argument("--protocol", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--scene", required=True)
    parser.add_argument("--collector", type=Path, default=DEFAULT_COLLECTOR)
    args = parser.parse_args()

    schedule = args.schedule.resolve()
    registry = args.scene_registry.resolve()
    asset_lock = args.asset_lock.resolve()
    protocol_path = args.protocol.resolve()
    collector = args.collector.resolve()
    out = args.out.resolve()
    for path in (
        schedule,
        registry,
        asset_lock,
        protocol_path,
        collector,
        ISAACLAB,
        EXPERIENCE,
        CONDA_PREFIX / "bin/python",
    ):
        if not path.is_file():
            raise FileNotFoundError(path)

    protocol = _json(protocol_path)
    if protocol.get("status") != "frozen":
        raise RuntimeError("T2 protocol is not frozen")
    for key, actual in (
        ("schedule_sha256", _sha256(schedule)),
        ("scene_registry_sha256", _sha256(registry)),
        ("asset_lock_sha256", _sha256(asset_lock)),
        ("collector_sha256", _sha256(collector)),
    ):
        if protocol.get(key) != actual:
            raise RuntimeError(f"frozen T2 protocol {key} mismatch")
    scene_rows = [
        row
        for row in _jsonl(schedule)
        if str(row["scene_cluster"]) == args.scene
    ]
    if not scene_rows:
        raise RuntimeError(f"no frozen T2 cases for {args.scene}")

    out.mkdir(parents=True, exist_ok=True)
    audit_dir = out / "launcher_audits"
    log_dir = out / "launcher_logs"
    audit_dir.mkdir(parents=True, exist_ok=True)
    log_dir.mkdir(parents=True, exist_ok=True)
    audit_path = audit_dir / f"{args.scene}.json"
    summary_path = out / "scene_summaries" / f"{args.scene}.json"
    if audit_path.is_file():
        audit = _json(audit_path)
        if (
            audit.get("schedule_sha256") != _sha256(schedule)
            or audit.get("scene_id") != args.scene
        ):
            raise RuntimeError("existing T2 launch audit belongs to another design")
        return (
            0
            if summary_path.is_file()
            and _json(summary_path).get("passed") is True
            else 2
        )

    log_path = log_dir / f"{args.scene}.log"
    audit = {
        "schema_version": "kinofail.reconfirmation-t2-launch.v2",
        "state": "started",
        "started_utc": datetime.now(UTC).isoformat(),
        "retry_authorized": False,
        "scientific_content_changed": False,
        "scene_id": args.scene,
        "case_count": len(scene_rows),
        "schedule": str(schedule),
        "schedule_sha256": _sha256(schedule),
        "protocol": str(protocol_path),
        "protocol_sha256": _sha256(protocol_path),
        "scene_registry_sha256": _sha256(registry),
        "asset_lock_sha256": _sha256(asset_lock),
        "collector_sha256": _sha256(collector),
        "log": str(log_path),
    }
    _write_json(audit_path, audit)

    environment = os.environ.copy()
    environment["CONDA_PREFIX"] = str(CONDA_PREFIX)
    environment["PATH"] = (
        f"{CONDA_PREFIX / 'bin'}:{environment.get('PATH', '')}"
    )
    environment["OMNI_KIT_ACCEPT_EULA"] = "YES"
    environment["PYTHONPATH"] = ISAACLAB_PYTHONPATH
    command = [
        str(ISAACLAB),
        "-p",
        str(collector),
        "--schedule",
        str(schedule),
        "--scene-registry",
        str(registry),
        "--asset-lock",
        str(asset_lock),
        "--protocol",
        str(protocol_path),
        "--out",
        str(out),
        "--scene",
        args.scene,
        "--headless",
        "--enable_cameras",
        "--experience",
        str(EXPERIENCE),
    ]
    with log_path.open("x", encoding="utf-8") as log:
        completed = subprocess.run(
            command,
            cwd=ROOT,
            env=environment,
            stdout=log,
            stderr=subprocess.STDOUT,
            check=False,
        )
    summary = _json(summary_path) if summary_path.is_file() else {}
    terminal = {
        **audit,
        "state": "terminal",
        "completed_utc": datetime.now(UTC).isoformat(),
        "returncode": int(completed.returncode),
        "summary_exists": summary_path.is_file(),
        "passed": summary.get("passed") is True,
    }
    _write_json(audit_path, terminal)
    print(json.dumps(terminal, indent=2, sort_keys=True))
    return 0 if terminal["passed"] else 2


if __name__ == "__main__":
    raise SystemExit(main())
