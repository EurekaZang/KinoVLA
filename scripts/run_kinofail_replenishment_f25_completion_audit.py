#!/usr/bin/env python3
"""Continuously audit F25 and run the strict final gate after acquisition.

This watcher never launches, retries, or modifies physical episodes.  It only
reads the sealed F25 cohort and its append-only attempt ledger.  Partial audit
reports are descriptive; the strict audit is invoked exactly once all frozen
replacement pairs have terminal attempt records.
"""

from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
import time
from datetime import UTC, datetime
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parents[1]
PYTHON = Path("/home/eureka/miniconda3/envs/kinovla/bin/python")
AUDITOR = ROOT / "scripts/audit_kinofail_replenishment_f25.py"


def _json(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise TypeError(path)
    return value


def _atomic_json(path: Path, value: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(
        json.dumps(value, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    os.replace(temporary, path)


def _attempt_counts(corpus_root: Path, planned: int) -> dict[str, int]:
    attempts = [
        _json(path) for path in sorted((corpus_root / "attempts").glob("*.json"))
    ]
    terminal = [row for row in attempts if row.get("state") == "terminal"]
    passed = [row for row in terminal if row.get("passed") is True]
    started = [row for row in attempts if row.get("state") == "started"]
    return {
        "planned_pairs": planned,
        "attempt_records": len(attempts),
        "terminal_pairs": len(terminal),
        "passed_pairs": len(passed),
        "failed_pairs": len(terminal) - len(passed),
        "started_pairs": len(started),
        "unattempted_pairs": planned - len(attempts),
    }


def _run_audit(
    *, freeze: Path, corpus_root: Path, output: Path, allow_incomplete: bool
) -> subprocess.CompletedProcess[str]:
    command = [
        str(PYTHON),
        str(AUDITOR),
        "--freeze",
        str(freeze),
        "--corpus-root",
        str(corpus_root),
        "--output",
        str(output),
    ]
    if allow_incomplete:
        command.append("--allow-incomplete")
    return subprocess.run(
        command,
        cwd=ROOT,
        check=False,
        capture_output=True,
        text=True,
    )


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--freeze", type=Path, required=True)
    parser.add_argument("--corpus-root", type=Path, required=True)
    parser.add_argument("--live-audit", type=Path, required=True)
    parser.add_argument("--final-audit", type=Path, required=True)
    parser.add_argument("--state", type=Path, required=True)
    parser.add_argument("--poll-s", type=float, default=60.0)
    args = parser.parse_args()

    freeze = args.freeze.resolve()
    corpus_root = args.corpus_root.resolve()
    live_audit = args.live_audit.resolve()
    final_audit = args.final_audit.resolve()
    state_path = args.state.resolve()
    planned = int(_json(freeze)["scheduled_pairs"])
    if planned != 1_417:
        raise RuntimeError(f"unexpected frozen cohort size: {planned}")

    while True:
        counts = _attempt_counts(corpus_root, planned)
        partial = _run_audit(
            freeze=freeze,
            corpus_root=corpus_root,
            output=live_audit,
            allow_incomplete=True,
        )
        state: dict[str, Any] = {
            "schema_version": "kinofail.f25-completion-audit-state.v1",
            "updated_utc": datetime.now(UTC).isoformat(),
            "pid": os.getpid(),
            "state": "watching",
            "freeze": str(freeze),
            "corpus_root": str(corpus_root),
            "counts": counts,
            "live_audit": str(live_audit),
            "live_audit_returncode": partial.returncode,
            "live_audit_stderr": partial.stderr[-4000:],
        }
        if partial.returncode != 0:
            state["state"] = "partial_audit_error"
        _atomic_json(state_path, state)
        print(json.dumps(state, sort_keys=True), flush=True)

        all_terminal = (
            counts["terminal_pairs"] == planned
            and counts["started_pairs"] == 0
            and counts["unattempted_pairs"] == 0
        )
        if all_terminal:
            final = _run_audit(
                freeze=freeze,
                corpus_root=corpus_root,
                output=final_audit,
                allow_incomplete=False,
            )
            final_report = _json(final_audit) if final_audit.is_file() else {}
            state.update(
                {
                    "updated_utc": datetime.now(UTC).isoformat(),
                    "state": "completed" if final.returncode == 0 else "gate_failed",
                    "final_audit": str(final_audit),
                    "final_audit_returncode": final.returncode,
                    "final_audit_passed": final_report.get("passed"),
                    "final_audit_stderr": final.stderr[-4000:],
                }
            )
            _atomic_json(state_path, state)
            print(final.stdout, end="", flush=True)
            print(json.dumps(state, sort_keys=True), flush=True)
            return final.returncode
        time.sleep(max(10.0, args.poll_s))


if __name__ == "__main__":
    sys.exit(main())
