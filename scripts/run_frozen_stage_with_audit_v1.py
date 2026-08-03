#!/usr/bin/env python3
"""Execute one frozen stage and preserve pre-audit exceptions as evidence."""

from __future__ import annotations

import argparse
import hashlib
import json
import subprocess
from datetime import UTC, datetime
from pathlib import Path


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--stage", required=True)
    parser.add_argument("--scene-id", required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--required-input", type=Path, action="append", default=[])
    parser.add_argument("command", nargs=argparse.REMAINDER)
    args = parser.parse_args()
    command = list(args.command)
    if command and command[0] == "--":
        command = command[1:]
    if not command:
        parser.error("a command is required after --")
    out = args.out.resolve()
    if out.exists():
        raise FileExistsError(f"refusing to overwrite stage audit: {out}")
    inputs = [path.resolve() for path in args.required_input]
    for path in inputs:
        if not path.is_file():
            raise FileNotFoundError(path)

    completed = subprocess.run(command, capture_output=True, text=True, check=False)
    command_files = []
    for token in command:
        path = Path(token)
        if path.is_file():
            resolved = path.resolve()
            command_files.append({"path": str(resolved), "sha256": _sha256(resolved)})
    payload = {
        "schema_version": "kinofail.frozen-stage-execution-audit.v1",
        "created_utc": datetime.now(UTC).isoformat(),
        "stage": args.stage,
        "scene_id": args.scene_id,
        "command": command,
        "command_files": command_files,
        "required_inputs": [
            {"path": str(path), "sha256": _sha256(path)} for path in inputs
        ],
        "exit_code": int(completed.returncode),
        "stdout": completed.stdout,
        "stderr": completed.stderr,
        "passed": completed.returncode == 0,
        "admission_state": (
            "stage_execution_completed"
            if completed.returncode == 0
            else "stage_execution_failed_before_native_audit"
        ),
    }
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(payload, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    print(json.dumps(payload, indent=2, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
