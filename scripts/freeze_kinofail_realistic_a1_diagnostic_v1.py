#!/usr/bin/env python3
"""Freeze the realistic A1 diagnostic before CLIP features exist."""

from __future__ import annotations

import hashlib
import json
from datetime import UTC, datetime
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]


def _sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main() -> None:
    sources = [
        ROOT / "configs/eval/kinofail_realistic_a1_diagnostic_v1.json",
        ROOT / "scripts/run_kinofail_realistic_a1_diagnostic_v1.py",
    ]
    absent = [
        ROOT / "outputs/eval/realistic_a0_a7_v4/features/features.npz",
        ROOT / "outputs/eval/realistic_a0_a7_v4/a1_construct_validity.json",
    ]
    value = {
        "schema_version": "kinofail.realistic-a1-diagnostic-freeze.v1",
        "created_utc": datetime.now(UTC).isoformat(),
        "source_sha256": {str(path.relative_to(ROOT)): _sha(path) for path in sources},
        "formal_artifacts_absent": {str(path.relative_to(ROOT)): not path.exists() for path in absent},
        "scope": "Diagnostic only; cannot promote A1 without a separately frozen matched O2/O4 extension.",
    }
    output = ROOT / "outputs/eval/realistic_a0_a7_v4/a1_diagnostic_protocol_freeze.json"
    output.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps({"output": str(output), "sha256": _sha(output)}, indent=2))


if __name__ == "__main__":
    main()
