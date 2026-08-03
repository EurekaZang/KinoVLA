#!/usr/bin/env python3
"""Seal an interrupted/failed formal O4 run without fabricating a passed pair manifest."""

from __future__ import annotations

import argparse
import hashlib
import json
from datetime import UTC, datetime
from pathlib import Path


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _evidence_file(run_dir: Path, relative: str) -> dict[str, object]:
    path = run_dir / relative
    if not path.is_file():
        raise FileNotFoundError(path)
    return {"path": relative, "bytes": path.stat().st_size, "sha256": _sha256(path)}


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--run-dir", type=Path, required=True)
    parser.add_argument("--protocol", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()

    run_dir = args.run_dir.resolve()
    protocol_path = args.protocol.resolve()
    output = args.out.resolve()
    if output.exists():
        raise FileExistsError(f"refusing to overwrite sealed failure audit: {output}")
    if (run_dir / "pair_manifest.json").exists():
        raise RuntimeError("run already has a pair manifest; incomplete-run audit is inapplicable")

    protocol = json.loads(protocol_path.read_text(encoding="utf-8"))
    takes: dict[str, dict[str, object]] = {}
    evidence: list[dict[str, object]] = []
    for take in ("nominal", "o4_adhesion"):
        summary_path = run_dir / take / "summary.json"
        summary = json.loads(summary_path.read_text(encoding="utf-8"))
        takes[take] = summary
        evidence.extend(
            [
                _evidence_file(run_dir, f"{take}/summary.json"),
                _evidence_file(run_dir, str(summary["telemetry"])),
                _evidence_file(run_dir, str(summary["frames_manifest"])),
            ]
        )

    anomaly = takes["o4_adhesion"]
    if int(anomaly["attachment_events"]) == 0:
        reason = "operator_not_triggered_no_attachment_event"
    else:
        reason = "collector_terminated_before_pair_manifest"
    audit = {
        "schema_version": "kinofail.embodiedgen-o4-run-failure.v1",
        "sealed_utc": datetime.now(UTC).isoformat(),
        "status": "failed",
        "formal_attempt_counted": True,
        "admitted_benchmark_pair": False,
        "must_not_be_rerun_or_retroactively_passed": True,
        "failure_reason": reason,
        "collector_failure": "paired consequence computation had no O4 attachment event",
        "run_dir": str(run_dir),
        "protocol": {
            "path": str(protocol_path),
            "sha256": _sha256(protocol_path),
            "protocol_id": protocol["protocol_id"],
            "formal_episode_seeds": protocol["formal_episode_seeds"],
            "run_contract": protocol["run_contract"],
            "frozen_files": protocol["frozen_files"],
        },
        "takes": takes,
        "preserved_evidence": evidence,
        "diagnosis": {
            "nominal_and_o4_physical_summaries_equal": all(
                takes["nominal"][key] == takes["o4_adhesion"][key]
                for key in (
                    "steps",
                    "max_progress_m",
                    "final_progress_m",
                    "min_base_height_m",
                    "max_tilt_rad",
                    "fell",
                )
            ),
            "o4_peak_force_n": anomaly["peak_adhesion_force_n"],
            "o4_attachment_events": anomaly["attachment_events"],
            "o4_peel_events": anomaly["peel_events"],
            "root_cause": (
                "operator placement was defined at 78% of total route extent while collection "
                "started at the midpoint; the longer held-out route put the patch beyond the "
                "frozen forward exposure window"
            ),
        },
    }
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(audit, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    print(output)
    print(_sha256(output))


if __name__ == "__main__":
    main()
