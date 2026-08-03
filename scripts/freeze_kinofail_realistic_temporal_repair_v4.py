#!/usr/bin/env python3
"""Freeze the final broad temporal-repair protocol before scale-v8 model fitting."""

from __future__ import annotations

import hashlib
import json
from collections import defaultdict
from datetime import UTC, datetime
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
SCHEDULE = ROOT / "outputs/kinofail_realistic/design_scale_v8_replication_v1/full_schedule.jsonl"
BASE_PROTOCOL = ROOT / "configs/data/kinofail_realistic_temporal_repair_formal_v3.json"
BASE_SNAPSHOT = ROOT / "configs/eval/kinofail_realistic_snapshot_scale_v8_v5.json"
BASE_MULTIMODAL = ROOT / "configs/eval/kinofail_realistic_multimodal_scale_v8_v5.json"
COLLECTOR = ROOT / "scripts/isaac_collect_kinofail_realistic_temporal_repair_v4.py"
PROTOCOL = ROOT / "configs/data/kinofail_realistic_temporal_repair_formal_v4.json"
SNAPSHOT = ROOT / "configs/eval/kinofail_realistic_snapshot_scale_v8_v6.json"
MULTIMODAL = ROOT / "configs/eval/kinofail_realistic_multimodal_scale_v8_v6.json"
BASE_CORPUS = ROOT / "outputs/kinofail_realistic/corpus_scale_v8_replication_v1"


def _sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _json(path: Path) -> dict:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise TypeError(path)
    return value


def _in_scope(row: dict) -> bool:
    operator = str(row["target_operator"])
    return (
        operator == "O8_invisible_collider"
        or (operator == "O3_collapse" and row["severity_id"] == "severe")
    )


def main() -> int:
    for destination in (PROTOCOL, SNAPSHOT, MULTIMODAL):
        if destination.exists():
            raise FileExistsError(f"refusing to overwrite frozen artifact: {destination}")
    for outcome in (
        ROOT / "outputs/eval/realistic_a0_a7_v6/training_manifest.json",
        ROOT / "outputs/eval/realistic_a0_a7_v6/predictions.jsonl",
    ):
        if outcome.exists():
            raise RuntimeError(f"model outcome already exists: {outcome}")

    rows = [
        json.loads(line)
        for line in SCHEDULE.read_text(encoding="utf-8").splitlines()
        if line
    ]
    pair_ids = sorted(
        {
            str(row["counterfactual_group_id"])
            for row in rows
            if row["condition"] == "anomaly" and _in_scope(row)
        }
    )
    if len(pair_ids) != 135:
        raise RuntimeError(f"expected 135 complete-block pairs, found {len(pair_ids)}")

    schedule_by_pair: dict[str, dict] = {}
    for row in rows:
        schedule_by_pair.setdefault(str(row["counterfactual_group_id"]), row)
    observed = defaultdict(lambda: {"finished_pairs": 0, "too_short_pairs": 0, "profiles": set()})
    for pair_id in pair_ids:
        summary_path = BASE_CORPUS / "pair_summaries" / f"{pair_id}.json"
        if not summary_path.is_file():
            continue
        summary = _json(summary_path)
        representative = schedule_by_pair[pair_id]
        block = (
            f"{representative['target_operator']}/"
            f"{representative['severity_id']}"
        )
        observed[block]["finished_pairs"] += 1
        manifests = [
            _json(Path(str(result["manifest"])))
            for result in summary.get("results", [])
            if Path(str(result.get("manifest", ""))).is_file()
        ]
        issues = {
            str(issue)
            for manifest in manifests
            for issue in manifest.get("runtime_validation", {}).get("issues", [])
        }
        if "too_few_proprio_samples" in issues or "too_few_rgb_frames" in issues:
            observed[block]["too_short_pairs"] += 1
            for manifest in manifests:
                profile = (
                    manifest.get("collection", {})
                    .get("physical_nuisance", {})
                    .get("profile_index")
                )
                if profile is not None:
                    observed[block]["profiles"].add(int(profile))
    observed_json = {
        key: {
            "finished_pairs": value["finished_pairs"],
            "too_short_pairs": value["too_short_pairs"],
            "observed_too_short_profiles": sorted(value["profiles"]),
        }
        for key, value in sorted(observed.items())
    }
    frozen_utc = datetime.now(UTC).isoformat()

    protocol = _json(BASE_PROTOCOL)
    protocol.update(
        {
            "protocol_id": "kinofail_realistic_scale_v8_temporal_repair_v4",
            "frozen_utc": frozen_utc,
            "collector_path": str(COLLECTOR.relative_to(ROOT)),
            "collector_sha256": _sha(COLLECTOR),
            "scope": (
                "Uniform all-scene, all-profile repair of O3-severe and the complete "
                "O8 operator block at both severities; no other pair is admitted."
            ),
            "supersedes": {
                "protocol": str(BASE_PROTOCOL.relative_to(ROOT)),
                "protocol_sha256": _sha(BASE_PROTOCOL),
                "reason": (
                    "O8 immediate collision was observed at moderate as well as severe "
                    "strength before any scale-v8 model training or prediction."
                ),
            },
        }
    )
    protocol["allowed"] = {
        **dict(protocol["allowed"]),
        "counterfactual_group_ids": pair_ids,
        "design_blocks": [
            {
                "target_operator": "O3_collapse",
                "severity_ids": ["severe"],
                "physical_nuisance_profile_indices": [0, 1, 2, 3, 4],
            },
            {
                "target_operator": "O8_invisible_collider",
                "severity_ids": ["moderate", "severe"],
                "physical_nuisance_profile_indices": [0, 1, 2, 3, 4],
            },
        ],
        "physical_nuisance_profile_indices": [0, 1, 2, 3, 4],
        "severity_ids": ["moderate", "severe"],
        "target_operators": ["O3_collapse", "O8_invisible_collider"],
    }
    protocol["amendment"] = {
        "changed": (
            "Install O3-severe and every O8 instance after the same recorded "
            "30-control-step (0.6 s) pre-roll."
        ),
        "model_predictions_available_at_freeze": False,
        "pair_selection_is_complete_design_blocks_not_outcome_cherry_picking": True,
        "selection_basis": {
            "complete_design_blocks": [
                "O3_collapse / severe / all five profiles / all nine scenes",
                "O8_invisible_collider / moderate+severe / all five profiles / all nine scenes",
            ],
            "observed_before_freeze": observed_json,
            "all_unobserved_block_members_included_prospectively": True,
        },
        "unchanged": [
            "all scheduled O3 and O8 physical parameters",
            "five-profile route-entry nuisance definition and balance",
            "scene geometry and operator region",
            "controller and commanded speed",
            "body-fixed Go2 front camera",
            "three synchronized PBR appearance views",
            "counterfactual pairing",
            "10 Hz RGB capture",
            "runtime quality gates",
        ],
    }
    protocol["collection_contract"] = {
        **dict(protocol["collection_contract"]),
        "counterfactual_pairs": 135,
        "physical_episodes": 270,
    }
    PROTOCOL.write_text(
        json.dumps(protocol, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )

    snapshot = _json(BASE_SNAPSHOT)
    snapshot.update(
        {
            "protocol_id": "realistic_snapshot_scale_v8_temporal_overlay_v6",
            "status": "frozen_after_runtime_qa_before_model_training",
        }
    )
    snapshot["amendment"] = {
        "base_snapshot_protocol": str(BASE_SNAPSHOT.relative_to(ROOT)),
        "base_snapshot_protocol_sha256": _sha(BASE_SNAPSHOT),
        "created_utc": frozen_utc,
        "model_architecture_or_split_changed": False,
        "model_outcomes_available_at_freeze": False,
        "repair_protocol": str(PROTOCOL.relative_to(ROOT)),
        "repair_protocol_sha256": _sha(PROTOCOL),
        "scope": (
            "Use the uniform repair overlay for all O3-severe and all O8 pairs; "
            "use the immutable base corpus for the remaining 855 pairs."
        ),
        "selection_uses_model_outcomes": False,
    }
    snapshot["publication_guard"]["reason"] = (
        "All five profiles remain balanced; complete mechanism-defined blocks receive "
        "one pre-outcome temporal repair before model training."
    )
    snapshot["source_corpus_overlays"] = [
        {
            "corpus_root": "outputs/kinofail_realistic/corpus_scale_v8_temporal_repair_v4",
            "counterfactual_group_ids": pair_ids,
            "repair_protocol": str(PROTOCOL.relative_to(ROOT)),
            "repair_protocol_sha256": _sha(PROTOCOL),
            "required_collection_protocol_id": protocol["protocol_id"],
        }
    ]
    SNAPSHOT.write_text(
        json.dumps(snapshot, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )

    multimodal = _json(BASE_MULTIMODAL)
    multimodal.update(
        {
            "protocol_id": "kinofail_realistic_multimodal_scale_v8_temporal_overlay_v6",
            "source_snapshot_protocol": str(SNAPSHOT.relative_to(ROOT)),
            "status": "frozen_after_runtime_qa_before_model_training",
        }
    )
    multimodal["freeze_context"] = (
        "Architecture, thresholds, splits and statistics are unchanged; the final "
        "mechanism-complete temporal overlay is bound before model outcomes."
    )
    multimodal["freeze_provenance"] = {
        "base_multimodal_protocol": str(BASE_MULTIMODAL.relative_to(ROOT)),
        "base_multimodal_protocol_sha256": _sha(BASE_MULTIMODAL),
        "model_outcomes_available_at_freeze": False,
        "snapshot_protocol": str(SNAPSHOT.relative_to(ROOT)),
        "snapshot_protocol_sha256": _sha(SNAPSHOT),
    }
    multimodal["leakage_guards"][-1] = (
        "all O3-severe and all O8 pairs across all five profiles use the same "
        "pre-outcome temporal amendment"
    )
    MULTIMODAL.write_text(
        json.dumps(multimodal, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    print(
        json.dumps(
            {
                "protocol": str(PROTOCOL),
                "protocol_sha256": _sha(PROTOCOL),
                "snapshot": str(SNAPSHOT),
                "snapshot_sha256": _sha(SNAPSHOT),
                "multimodal": str(MULTIMODAL),
                "multimodal_sha256": _sha(MULTIMODAL),
                "pairs": len(pair_ids),
                "observed_before_freeze": observed_json,
            },
            indent=2,
            sort_keys=True,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
