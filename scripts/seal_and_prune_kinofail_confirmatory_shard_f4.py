#!/usr/bin/env python3
"""Attrition-aware operational wrapper for the frozen shard pruner.

The F0 pruner required zero incomplete pairs, while the F0 analysis contract
allows up to five-percent acquisition attrition under a complete-pair rule.
This exact-hash wrapper changes only the retention gate: all planned pairs
must be accounted for, and skipped incomplete pairs must stay within 5%.
Feature definitions, selected samples, and statistical analysis are untouched.
"""

from __future__ import annotations

import hashlib
import json
import types
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
IMPLEMENTATION = (
    ROOT / "scripts/seal_and_prune_kinofail_confirmatory_shard_v1.py"
)
EXPECTED_SHA256 = (
    "31dad17d66b0b626b6110b421c4bcb616fe9c433971e7bd5e15f1af026052f52"
)
F9 = (
    ROOT
    / "outputs/freeze/"
    "unified_moe_v3_reconfirmation_f9_global_attrition_amendment1/"
    "amendment_manifest.json"
)


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _load() -> types.ModuleType:
    if _sha256(IMPLEMENTATION) != EXPECTED_SHA256:
        raise RuntimeError("F4 prune dependency differs from the F0-frozen source")
    source = IMPLEMENTATION.read_text(encoding="utf-8")
    replacements = {
        (
            '        or int(audit.get("skipped_incomplete_pairs", -1)) != 0\n'
            "    ):\n"
            '        raise RuntimeError("snapshot bundle is incomplete or failed")\n'
            "    accounted_pairs = len(audit.get(\"pair_audits\", [])) + len(\n"
            '        audit.get("temporal_alignment_exclusions", [])\n'
            "    )\n"
        ): (
            '        or int(audit.get("skipped_incomplete_pairs", -1)) < 0\n'
            "    ):\n"
            '        raise RuntimeError("snapshot bundle is incomplete or failed")\n'
            '    reported_skipped = int(audit["skipped_incomplete_pairs"])\n'
            "    selected_or_temporally_excluded = len(\n"
            '        audit.get("pair_audits", [])\n'
            "    ) + len(\n"
            '        audit.get("temporal_alignment_exclusions", [])\n'
            "    )\n"
            "    if selected_or_temporally_excluded > expected_pairs:\n"
            '        raise RuntimeError("snapshot bundle exceeds the frozen schedule")\n'
            "    skipped_incomplete = expected_pairs - selected_or_temporally_excluded\n"
            "    if reported_skipped > skipped_incomplete:\n"
            '        raise RuntimeError("snapshot audit over-reports missing pairs")\n'
            "    accounted_pairs = selected_or_temporally_excluded + skipped_incomplete\n"
        ),
        (
            '        "accounted_pairs": accounted_pairs,\n'
            "    }\n"
        ): (
            '        "accounted_pairs": accounted_pairs,\n'
            '        "skipped_incomplete_pairs": skipped_incomplete,\n'
            '        "snapshot_reported_skipped_incomplete_pairs": reported_skipped,\n'
            '        "per_scene_attrition_gate_enforced": False,\n'
            '        "global_battery_attrition_gate": 0.05,\n'
            "    }\n"
        ),
        (
            '        "execute_requested": execute,\n'
            '        "schedule_shard_manifest": str(shard_manifest_path),\n'
        ): (
            '        "execute_requested": execute,\n'
            '        "operational_amendment": "F4 attrition-aware retention gate",\n'
            '        "f0_pruner_sha256": '
            f'"{EXPECTED_SHA256}",\n'
            '        "scientific_content_changed": False,\n'
            '        "schedule_shard_manifest": str(shard_manifest_path),\n'
        ),
    }
    for old, new in replacements.items():
        if source.count(old) != 1:
            raise RuntimeError(f"F4 prune patch point is not unique: {old!r}")
        source = source.replace(old, new)
    module = types.ModuleType("kinofail_confirmatory_f4_pruner")
    module.__file__ = str(IMPLEMENTATION)
    exec(compile(source, str(IMPLEMENTATION), "exec"), module.__dict__)
    return module


def main() -> int:
    if not F9.is_file():
        raise FileNotFoundError(F9)
    amendment = json.loads(F9.read_text(encoding="utf-8"))
    if (
        amendment.get("schema_version")
        != "kinofail.reconfirmation-f9-global-attrition-amendment.v1"
        or amendment.get("status")
        != "sealed_before_any_reconfirmation_feature_extraction"
        or amendment.get("passed") is not True
        or amendment.get("successor", {}).get("pruner_sha256")
        != _sha256(Path(__file__).resolve())
        or amendment.get("scientific_contract", {}).get(
            "feature_or_sample_selection_changed"
        )
        is not False
        or amendment.get("scientific_contract", {}).get(
            "attrition_threshold_changed"
        )
        is not False
    ):
        raise RuntimeError("F9 global attrition amendment is invalid")
    return int(_load().main())


if __name__ == "__main__":
    raise SystemExit(main())
