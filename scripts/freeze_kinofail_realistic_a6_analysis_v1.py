#!/usr/bin/env python3
"""Freeze A6 analysis before any A6 simulation exists."""

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
        ROOT / "configs/data/kinofail_realistic_a6_boundary_formal_v2.json",
        ROOT / "configs/eval/kinofail_realistic_a6_analysis_v1.json",
        ROOT / "scripts/analyze_kinofail_realistic_a6_v1.py",
    ]
    corpus = ROOT / "outputs/kinofail_realistic/corpus_a6_boundary_v2"
    report = ROOT / "outputs/eval/realistic_a0_a7_v4/a6_boundary.json"
    value = {
        "schema_version": "kinofail.realistic-a6-analysis-freeze.v1",
        "created_utc": datetime.now(UTC).isoformat(),
        "source_sha256": {str(path.relative_to(ROOT)): _sha(path) for path in sources},
        "a6_corpus_absent": not corpus.exists(),
        "a6_report_absent": not report.exists(),
        "a8_in_scope": False,
    }
    output = ROOT / "outputs/eval/realistic_a0_a7_v4/a6_analysis_protocol_freeze.json"
    output.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps({"output": str(output), "sha256": _sha(output)}, indent=2))


if __name__ == "__main__":
    main()
