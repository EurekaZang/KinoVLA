#!/usr/bin/env python3
"""Refine a frozen v1 scene compilation into a robust-corridor v2 episode."""

from __future__ import annotations

import argparse
import hashlib
import json
from datetime import UTC, datetime
from pathlib import Path

from kino_vla.sim.embodiedgen_scene_v2 import refine_embodiedgen_kinofail_scene_v2


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--base-compiled-audit", required=True)
    parser.add_argument("--output-dir", required=True)
    args = parser.parse_args()
    try:
        result = refine_embodiedgen_kinofail_scene_v2(
            args.base_compiled_audit, args.output_dir
        )
    except Exception as exc:
        output = Path(args.output_dir).resolve()
        output.mkdir(parents=True, exist_ok=True)
        base_path = Path(args.base_compiled_audit).resolve()
        rejection = {
            "schema_version": "kinofail.embodiedgen-corridor-v2-refinement-rejection.v1",
            "created_utc": datetime.now(UTC).isoformat(),
            "passed": False,
            "admission_state": "corridor_v2_refinement_rejected",
            "base_compiled_audit": {
                "path": str(base_path),
                "sha256": _sha256(base_path) if base_path.is_file() else None,
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
    print(
        json.dumps(
            {
                "passed": result["passed"],
                "audit_path": result["audit_path"],
                "episode_usd": result["episode_usd"],
                "route": result["route"],
                "corridor_search": result["corridor_search"],
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
