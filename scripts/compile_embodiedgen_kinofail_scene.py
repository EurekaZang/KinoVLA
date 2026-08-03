#!/usr/bin/env python3
"""Compile Kino-owned collision, route, and O4 metadata around a frozen RoomGen scene."""

from __future__ import annotations

import argparse
import hashlib
import json
from datetime import UTC, datetime
from pathlib import Path

from kino_vla.sim.embodiedgen_scene import compile_embodiedgen_kinofail_scene


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--source-manifest", required=True)
    parser.add_argument("--output-dir", required=True)
    args = parser.parse_args()
    try:
        result = compile_embodiedgen_kinofail_scene(args.source_manifest, args.output_dir)
    except Exception as exc:
        output = Path(args.output_dir).resolve()
        output.mkdir(parents=True, exist_ok=True)
        source_path = Path(args.source_manifest).resolve()
        rejection = {
            "schema_version": "kinofail.embodiedgen-base-compile-rejection.v1",
            "created_utc": datetime.now(UTC).isoformat(),
            "passed": False,
            "admission_state": "base_compile_rejected",
            "source_manifest": {
                "path": str(source_path),
                "sha256": (
                    hashlib.sha256(source_path.read_bytes()).hexdigest()
                    if source_path.is_file()
                    else None
                ),
            },
            "exception_type": type(exc).__name__,
            "reason": str(exc),
            "formal_operator_run_authorized": False,
        }
        rejection_path = output / "compiled_scene_audit.json"
        rejection_path.write_text(
            json.dumps(rejection, indent=2, ensure_ascii=False) + "\n",
            encoding="utf-8",
        )
        print(json.dumps(rejection, indent=2, ensure_ascii=False))
        raise SystemExit(2) from None
    print(json.dumps({
        "passed": result["passed"],
        "audit_path": result["audit_path"],
        "episode_usd": result["episode_usd"],
        "route": result["route"],
    }, indent=2))
    if not result["passed"]:
        raise SystemExit(2)


if __name__ == "__main__":
    main()
