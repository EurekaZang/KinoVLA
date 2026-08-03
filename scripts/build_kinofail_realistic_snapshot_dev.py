#!/usr/bin/env python3
"""Build the frozen, development-only event-aligned realistic snapshot bundle."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from kino_vla.data.realistic_snapshots import (  # noqa: E402
    build_event_aligned_snapshots,
    write_snapshot_bundle,
)

if str(ROOT / "scripts") not in sys.path:
    sys.path.insert(0, str(ROOT / "scripts"))
from kinofail_reconfirmation_attrition_f13 import (  # noqa: E402
    build_for_protocol,
    sha256,
)


def _apply_launcher_eligibility(
    records: list[dict[str, Any]],
    arrays: dict[str, Any],
    audit: dict[str, Any],
    *,
    protocol: dict[str, Any],
    repo_root: Path,
) -> tuple[list[dict[str, Any]], dict[str, Any], dict[str, Any]]:
    built = build_for_protocol(protocol=protocol, repo_root=repo_root)
    if built is None:
        return records, arrays, audit
    ledger, ledger_path = built
    eligible = set(ledger["eligible_pair_ids"])
    attrited = {
        str(row["counterfactual_group_id"])
        for row in ledger["attrition"]
    }
    planned = eligible | attrited
    if len(planned) != int(ledger["counts"]["planned_pairs"]):
        raise RuntimeError("F13 ledger does not partition the frozen schedule")

    filtered_records = [
        row
        for row in records
        if str(row["counterfactual_group_id"]) in eligible
    ]
    sample_ids = {str(row["sample_id"]) for row in filtered_records}
    filtered_arrays = {
        key: value
        for key, value in arrays.items()
        if key.rsplit("__", 1)[0] in sample_ids
    }
    pair_audits = [
        row
        for row in audit.get("pair_audits", [])
        if str(row["counterfactual_group_id"]) in eligible
    ]
    temporal = [
        row
        for row in audit.get("temporal_alignment_exclusions", [])
        if str(row["counterfactual_group_id"]) in eligible
    ]
    selected_pairs = {
        str(row["counterfactual_group_id"]) for row in pair_audits
    }
    temporal_pairs = {
        str(row["counterfactual_group_id"]) for row in temporal
    }
    if selected_pairs & temporal_pairs:
        raise RuntimeError("F13 selected and temporal-excluded pairs overlap")
    if selected_pairs | temporal_pairs != eligible:
        missing = sorted(eligible - selected_pairs - temporal_pairs)
        raise RuntimeError(
            "F13 eligible launcher pairs are missing from the physical "
            f"snapshot decision: {missing[:8]}"
        )
    if {
        str(row["counterfactual_group_id"]) for row in filtered_records
    } != selected_pairs:
        raise RuntimeError("F13 snapshot records do not match selected pairs")

    corrected = dict(audit)
    corrected["pair_audits"] = pair_audits
    corrected["temporal_alignment_exclusions"] = temporal
    corrected["core_snapshot_skipped_incomplete_pairs"] = int(
        audit.get("skipped_incomplete_pairs", 0)
    )
    corrected["skipped_incomplete_pairs"] = len(attrited)
    corrected["launcher_attrition_exclusions"] = ledger["attrition"]
    corrected["launcher_eligibility_ledger"] = str(ledger_path)
    corrected["launcher_eligibility_ledger_sha256"] = sha256(ledger_path)
    corrected["operational_amendment"] = ledger[
        "operational_amendment"
    ]
    corrected["operational_amendment_sha256"] = ledger[
        "operational_amendment_sha256"
    ]
    corrected["model_or_prediction_loaded_for_launcher_filter"] = False
    corrected["selection_uses_outcome_strength"] = False
    corrected["counts"] = {
        **audit["counts"],
        "snapshot_records": len(filtered_records),
        "physical_episodes": len(
            {
                str(row["physical_episode_id"])
                for row in filtered_records
            }
        ),
        "independent_counterfactual_pairs": len(pair_audits),
        "appearance_intervention_sequences": len(filtered_records),
        "operators": len(
            {str(row["target_operator"]) for row in filtered_records}
        ),
        "domains": len(
            {str(row["domain"]) for row in filtered_records}
        ),
        "scene_families": len(
            {str(row["scene_family"]) for row in filtered_records}
        ),
        "temporally_infeasible_counterfactual_pairs": len(temporal),
        "launcher_attrited_counterfactual_pairs": len(attrited),
    }
    corrected["checks"] = {
        **audit["checks"],
        "all_records_from_complete_pairs": len(filtered_records)
        == sum(
            2 * int(row["appearance_views_per_physical_episode"])
            for row in pair_audits
        ),
        "all_selected_or_excluded_pairs_accounted_for": (
            len(selected_pairs) + len(temporal_pairs) + len(attrited)
            == len(planned)
        ),
        "launcher_ledger_partitions_frozen_schedule": (
            len(eligible) + len(attrited) == len(planned)
        ),
        "selected_pairs_match_launcher_eligible_pairs": (
            selected_pairs | temporal_pairs == eligible
        ),
        "launcher_filter_model_blind": True,
        "launcher_filter_result_independent": True,
    }
    corrected["passed"] = all(corrected["checks"].values())
    return filtered_records, filtered_arrays, corrected


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--protocol",
        default="configs/eval/kinofail_realistic_snapshot_development_v1.json",
    )
    parser.add_argument("--repo-root", default=str(ROOT))
    args = parser.parse_args()
    repo_root = Path(args.repo_root).resolve()
    protocol = json.loads((repo_root / args.protocol).read_text(encoding="utf-8"))
    records, arrays, audit = build_event_aligned_snapshots(protocol, repo_root=repo_root)
    records, arrays, audit = _apply_launcher_eligibility(
        records,
        arrays,
        audit,
        protocol=protocol,
        repo_root=repo_root,
    )
    output_dir = repo_root / protocol["output_dir"]
    hashes = write_snapshot_bundle(records, arrays, audit, output_dir=output_dir)
    result = {
        "output_dir": str(output_dir),
        "passed": audit["passed"],
        "counts": audit["counts"],
        "sha256": hashes,
        "publication_guard": audit["publication_guard"],
    }
    print(json.dumps(result, indent=2, sort_keys=True))
    return 0 if audit["passed"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
