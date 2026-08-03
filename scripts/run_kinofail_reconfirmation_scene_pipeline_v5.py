#!/usr/bin/env python3
"""F21 scene pipeline using fresh-process atomic T2 collection."""

from __future__ import annotations

import argparse
import sys
from datetime import UTC, datetime
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from scripts import run_kinofail_reconfirmation_scene_pipeline_v4 as predecessor
from scripts import run_kinofail_reconfirmation_t2_atomic_f21 as t2


AMENDMENT = t2.F21
ATOMIC_T2 = Path(t2.__file__).resolve()


def _validate_f21() -> dict[str, Any]:
    return t2._validate_f21()


def install_f21_t2_runner() -> None:
    predecessor.predecessor.SLOTTED_T2 = ATOMIC_T2


def _augment_state(scene: str, amendment: dict[str, Any]) -> None:
    state_path = (
        predecessor.predecessor.base.ORCHESTRATION_ROOT
        / scene
        / "pipeline_state.json"
    )
    if not state_path.is_file():
        return
    state = predecessor.predecessor.base._json(state_path)
    state["schema_version"] = "kinofail.reconfirmation-scene-pipeline.v5"
    state["f21_atomic_t2_amendment"] = {
        "path": str(AMENDMENT),
        "sha256": predecessor.predecessor.base._sha256(AMENDMENT),
        "status": amendment["status"],
        "fresh_isaac_process_per_t2_case": True,
        "atomic_case_commit": True,
        "case_hard_timeout_seconds": t2.CASE_HARD_TIMEOUT_SECONDS,
        "maximum_operational_attempts_per_case": (
            t2.MAX_OPERATIONAL_ATTEMPTS_PER_CASE
        ),
        "scientific_content_changed": False,
    }
    state["f21_state_wrapped_utc"] = datetime.now(UTC).isoformat()
    predecessor.predecessor.base._write_json(state_path, state)


def main() -> int:
    parser = argparse.ArgumentParser(add_help=False)
    parser.add_argument("--scene", required=True)
    known, _ = parser.parse_known_args()
    amendment = _validate_f21()
    install_f21_t2_runner()
    try:
        return int(predecessor.main())
    finally:
        _augment_state(known.scene, amendment)


if __name__ == "__main__":
    raise SystemExit(main())
