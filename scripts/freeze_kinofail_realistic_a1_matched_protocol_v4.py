#!/usr/bin/env python3
"""Freeze the valid-surface-state correction before any A1 manifest exists."""

from __future__ import annotations

import hashlib
import json
from datetime import UTC, datetime
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]


def _sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main() -> None:
    prior = ROOT / "configs/data/kinofail_realistic_a1_matched_formal_v3.json"
    schedule = ROOT / "outputs/kinofail_realistic/design_a1_matched_v2/schedule.jsonl"
    audit = ROOT / "outputs/kinofail_realistic/design_a1_matched_v2/audit.json"
    output = ROOT / "configs/data/kinofail_realistic_a1_matched_formal_v4.json"
    if output.exists():
        raise FileExistsError(f"refusing to overwrite {output}")
    protocol = json.loads(prior.read_text(encoding="utf-8"))
    rows = [json.loads(line) for line in schedule.read_text(encoding="utf-8").splitlines() if line]
    design = json.loads(audit.read_text(encoding="utf-8"))
    if design.get("passed") is not True or len(rows) != 36:
        raise RuntimeError("corrected A1 design audit did not pass")
    protocol.update({
        "protocol_id": "kinofail_realistic_a1_matched_o2_o4_36_v4",
        "frozen_utc": datetime.now(UTC).isoformat(),
        "supersedes_uncollected_protocol": str(prior.relative_to(ROOT)),
        "superseded_protocol_sha256": _sha(prior),
        "amendment_reason": (
            "Correct invalid development-only surface-state tokens primary/swap_a/swap_b to "
            "the renderer-supported clean/scuffed/wet states. The scanned PBR assets, physics, "
            "scenes, seeds, cameras, horizon, analysis, and pair IDs are unchanged; no prior "
            "A1 manifest exists."
        ),
        "schedule_path": str(schedule.relative_to(ROOT)),
        "schedule_sha256": _sha(schedule),
        "design_audit_path": str(audit.relative_to(ROOT)),
        "design_audit_sha256": _sha(audit),
    })
    protocol["allowed"] = {
        "counterfactual_group_ids": sorted({row["counterfactual_group_id"] for row in rows}),
        "target_operators": ["O2_compliance", "O4_tether"],
        "scene_families": sorted({row["scene_family"] for row in rows}),
        "physical_realizations": sorted({row["physical_realization"] for row in rows}),
        "geometry_profiles": sorted({row["geometry_profile"] for row in rows}),
        "severity_ids": sorted({row["severity_id"] for row in rows}),
        "conditions": ["anomaly", "nominal_counterfactual"],
    }
    output.write_text(json.dumps(protocol, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps({"output": str(output), "sha256": _sha(output), "pairs": 18}, indent=2))


if __name__ == "__main__":
    main()
