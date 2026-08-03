#!/usr/bin/env python3
"""Assemble one machine-readable ICRA readiness verdict for realistic Kino-Fail."""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
from collections import Counter, defaultdict
from datetime import UTC, datetime
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from kino_vla.data.realistic_snapshots import _runtime_is_accepted  # noqa: E402


def _sha(path: Path) -> str | None:
    return hashlib.sha256(path.read_bytes()).hexdigest() if path.is_file() else None


def _json(path: Path) -> dict[str, Any]:
    if not path.is_file():
        return {}
    value = json.loads(path.read_text(encoding="utf-8"))
    return value if isinstance(value, dict) else {}


def _jsonl(path: Path) -> list[dict[str, Any]]:
    if not path.is_file():
        return []
    return [
        json.loads(line)
        for line in path.read_text(encoding="utf-8").splitlines()
        if line
    ]


def _profile_index(schedule: list[dict[str, Any]]) -> dict[tuple[str, int], int]:
    by_scene: dict[str, set[int]] = defaultdict(set)
    for row in schedule:
        by_scene[str(row["scene_family"])].add(int(row["scene_seed"]))
    return {
        (scene, seed): index
        for scene, seeds in sorted(by_scene.items())
        for index, seed in enumerate(sorted(seeds))
    }


def _runtime_progress(
    schedule: list[dict[str, Any]],
    corpus: Path,
    allowed_suffixes: list[str],
    *,
    repair_corpus: Path | None = None,
    repair_protocol: dict[str, Any] | None = None,
) -> dict[str, Any]:
    schedule_by_episode = {str(row["episode_id"]): row for row in schedule}
    profile_by_seed = _profile_index(schedule)
    summaries = []
    repair_ids = set((repair_protocol or {}).get("allowed", {}).get(
        "counterfactual_group_ids", []
    ))
    repair_overlay_pairs = []
    for path in sorted((corpus / "pair_summaries").glob("*.json")):
        base = _json(path)
        pair_id = str(base.get("counterfactual_group_id", path.stem))
        repair_path = (
            repair_corpus / "pair_summaries" / f"{pair_id}.json"
            if repair_corpus is not None and pair_id in repair_ids
            else None
        )
        repair = _json(repair_path) if repair_path is not None else {}
        if isinstance(repair.get("results"), list) and len(repair["results"]) == 2:
            summaries.append(repair)
            repair_overlay_pairs.append(pair_id)
        else:
            summaries.append(base)
    eligible_pairs = []
    rejected_pairs = []
    operator_finished = Counter()
    operator_eligible = Counter()
    profile_finished = Counter()
    profile_eligible = Counter()
    issue_counts = Counter()
    for summary in summaries:
        pair_ok = True
        representative: dict[str, Any] | None = None
        for episode in summary.get("results", []):
            record = schedule_by_episode.get(str(episode.get("episode_id")))
            if record is None:
                pair_ok = False
                issue_counts["episode_not_in_frozen_schedule"] += 1
                continue
            representative = record
            manifest = _json(Path(str(episode.get("manifest", ""))))
            accepted = bool(manifest) and _runtime_is_accepted(
                record, manifest, allowed_suffixes=allowed_suffixes
            )
            pair_ok &= accepted
            if not accepted:
                issue_counts.update(
                    str(value)
                    for value in manifest.get("runtime_validation", {}).get("issues", [])
                )
        if representative is None:
            continue
        operator = str(representative["target_operator"])
        profile = profile_by_seed[
            (str(representative["scene_family"]), int(representative["scene_seed"]))
        ]
        operator_finished[operator] += 1
        profile_finished[str(profile)] += 1
        if pair_ok:
            eligible_pairs.append(str(summary.get("counterfactual_group_id")))
            operator_eligible[operator] += 1
            profile_eligible[str(profile)] += 1
        else:
            rejected_pairs.append(str(summary.get("counterfactual_group_id")))
    total_pairs = len({str(row["counterfactual_group_id"]) for row in schedule})
    return {
        "scheduled_pairs": total_pairs,
        "finished_pairs": len(summaries),
        "scientifically_eligible_finished_pairs": len(eligible_pairs),
        "scientifically_rejected_finished_pairs": len(rejected_pairs),
        "progress_fraction": len(summaries) / max(total_pairs, 1),
        "eligible_fraction_among_finished": len(eligible_pairs) / max(len(summaries), 1),
        "pairs_by_operator": {
            "finished": dict(sorted(operator_finished.items())),
            "eligible": dict(sorted(operator_eligible.items())),
        },
        "pairs_by_physical_nuisance_profile": {
            "finished": dict(sorted(profile_finished.items())),
            "eligible": dict(sorted(profile_eligible.items())),
        },
        "rejection_issue_counts": dict(sorted(issue_counts.items())),
        "eligible_pair_ids": eligible_pairs,
        "rejected_pair_ids": rejected_pairs,
        "repair_overlay_pairs_used": sorted(repair_overlay_pairs),
        "repair_overlay_pair_count": len(repair_overlay_pairs),
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--contract",
        type=Path,
        default=ROOT / "configs/eval/kinofail_realistic_a0_a7_v6.json",
    )
    parser.add_argument(
        "--protocol",
        type=Path,
        default=ROOT
        / "configs/data/kinofail_realistic_scale_v8_replication_formal_v3.json",
    )
    parser.add_argument(
        "--snapshot-protocol",
        type=Path,
        default=ROOT / "configs/eval/kinofail_realistic_snapshot_scale_v8_v6.json",
    )
    parser.add_argument(
        "--design-audit",
        type=Path,
        default=ROOT
        / "outputs/kinofail_realistic/design_scale_v8_replication_v1/"
        "scale_v8_replication_audit.json",
    )
    parser.add_argument(
        "--nuisance-audit",
        type=Path,
        default=ROOT
        / "outputs/kinofail_realistic/design_scale_v8_replication_v1/"
        "scale_v8_physical_nuisance_audit.json",
    )
    parser.add_argument(
        "--corpus-root",
        type=Path,
        default=ROOT / "outputs/kinofail_realistic/corpus_scale_v8_replication_v1",
    )
    parser.add_argument(
        "--readiness",
        type=Path,
        default=ROOT / "outputs/eval/realistic_a0_a7_v6/readiness_audit.json",
    )
    parser.add_argument(
        "--repair-corpus-root",
        type=Path,
        default=ROOT
        / "outputs/kinofail_realistic/corpus_scale_v8_temporal_repair_v4",
    )
    parser.add_argument(
        "--repair-protocol",
        type=Path,
        default=ROOT
        / "configs/data/kinofail_realistic_temporal_repair_formal_v4.json",
    )
    parser.add_argument(
        "--out",
        type=Path,
        default=ROOT
        / "outputs/eval/realistic_a0_a7_v6/icra_benchmark_readiness.json",
    )
    args = parser.parse_args()

    contract_path = args.contract.resolve()
    protocol_path = args.protocol.resolve()
    snapshot_path = args.snapshot_protocol.resolve()
    design_path = args.design_audit.resolve()
    nuisance_path = args.nuisance_audit.resolve()
    corpus = args.corpus_root.resolve()
    readiness_path = args.readiness.resolve()
    repair_corpus = args.repair_corpus_root.resolve()
    repair_protocol_path = args.repair_protocol.resolve()
    contract = _json(contract_path)
    protocol = _json(protocol_path)
    snapshot = _json(snapshot_path)
    design = _json(design_path)
    nuisance = _json(nuisance_path)
    readiness = _json(readiness_path)
    repair_protocol = _json(repair_protocol_path)
    schedule_path = ROOT / str(protocol["schedule_path"])
    collector_path = ROOT / str(protocol["collector_path"])
    schedule = _jsonl(schedule_path)
    allowed_suffixes = [
        str(value)
        for value in snapshot.get("selection", {}).get(
            "allow_nonblocking_runtime_issue_suffixes", []
        )
    ]
    runtime = _runtime_progress(
        schedule,
        corpus,
        allowed_suffixes,
        repair_corpus=repair_corpus,
        repair_protocol=repair_protocol,
    )
    snapshot_repair_overlay = next(
        iter(snapshot.get("source_corpus_overlays", [])),
        {},
    )
    design_stats = design.get("design_audit", {})
    planned = {
        "physical_episodes": len(schedule),
        "counterfactual_pairs": len(
            {str(row["counterfactual_group_id"]) for row in schedule}
        ),
        "appearance_view_sequences": sum(
            int(row.get("appearance_view_count", 0)) for row in schedule
        ),
        "domains": len({str(row["domain"]) for row in schedule}),
        "scene_families": len({str(row["scene_family"]) for row in schedule}),
        "operators": len({str(row["target_operator"]) for row in schedule}),
        "severity_levels": len({str(row["severity_id"]) for row in schedule}),
        "camera_profiles": len({str(row["camera_profile"]) for row in schedule}),
        "material_families_all_views": len(
            {
                str(view["material_family"])
                for row in schedule
                for view in row["appearance_views"]
            }
        ),
        "appearance_ids": len(
            {
                str(view["appearance_id"])
                for row in schedule
                for view in row["appearance_views"]
            }
        ),
    }
    schedule_gate = {
        "design_audit_passed": design.get("passed") is True,
        "exact_planned_episode_count": planned["physical_episodes"]
        == int(contract["benchmark"]["physical_episodes"]),
        "exact_planned_pair_count": planned["counterfactual_pairs"]
        == int(contract["benchmark"]["counterfactual_groups"]),
        "all_11_operators": planned["operators"] == 11,
        "nine_scene_families": planned["scene_families"] == 9,
        "three_domains": planned["domains"] == 3,
        "three_camera_profiles": planned["camera_profiles"] == 3,
        "two_severity_levels": planned["severity_levels"] == 2,
        "operator_material_mi_negligible": float(
            design_stats.get("operator_material_normalized_mi", 1.0)
        )
        < 1.0e-6,
        "condition_material_mi_zero": float(
            design_stats.get("condition_material_normalized_mi", 1.0)
        )
        == 0.0,
        "minimum_18_material_families_per_operator": int(
            design_stats.get("min_material_families_per_operator", 0)
        )
        >= 18,
    }
    provenance_gate = {
        "protocol_frozen": protocol.get("status") == "frozen",
        "schedule_hash_matches": protocol.get("schedule_sha256") == _sha(schedule_path),
        "collector_hash_matches": protocol.get("collector_sha256")
        == _sha(collector_path),
        "contract_binds_protocol": contract.get("claim_policy", {}).get(
            "corpus_protocol_id"
        )
        == protocol.get("protocol_id"),
        "snapshot_binds_protocol": snapshot.get("selection", {}).get(
            "required_collection_protocol_id"
        )
        == protocol.get("protocol_id"),
        "snapshot_binds_repair_protocol": snapshot_repair_overlay.get(
            "repair_protocol_sha256"
        )
        == _sha(repair_protocol_path),
        "a8_excluded": contract.get("claim_policy", {}).get("a8_in_scope") is False
        and protocol.get("collection_contract", {}).get("a8_in_scope") is False,
        "repair_protocol_frozen": repair_protocol.get("status") == "frozen",
        "repair_schedule_hash_matches": repair_protocol.get("schedule_sha256")
        == _sha(schedule_path),
        "repair_collector_hash_matches": repair_protocol.get("collector_sha256")
        == _sha(ROOT / str(repair_protocol.get("collector_path", ""))),
        "repair_scope_is_complete_o3_severe_and_o8_blocks": len(
            repair_protocol.get("allowed", {}).get("counterfactual_group_ids", [])
        )
        == 135
        and repair_protocol.get("allowed", {}).get("target_operators")
        == ["O3_collapse", "O8_invisible_collider"]
        and repair_protocol.get("allowed", {}).get("severity_ids")
        == ["moderate", "severe"]
        and repair_protocol.get("allowed", {}).get("physical_nuisance_profile_indices")
        == [0, 1, 2, 3, 4]
        and repair_protocol.get("allowed", {}).get("design_blocks")
        == [
            {
                "physical_nuisance_profile_indices": [0, 1, 2, 3, 4],
                "severity_ids": ["severe"],
                "target_operator": "O3_collapse",
            },
            {
                "physical_nuisance_profile_indices": [0, 1, 2, 3, 4],
                "severity_ids": ["moderate", "severe"],
                "target_operator": "O8_invisible_collider",
            },
        ],
    }
    randomization_gate = {
        "physical_nuisance_audit_passed": nuisance.get("passed") is True,
        "five_balanced_physical_profiles": nuisance.get(
            "profile_counts_by_counterfactual_pair"
        )
        == {str(index): 198 for index in range(5)},
        "pair_shared_physical_nuisance": nuisance.get("checks", {}).get(
            "pair_shared_nuisance"
        )
        is True,
        "all_cells_cover_five_profiles": nuisance.get("checks", {}).get(
            "every_scene_operator_severity_cell_has_all_five_profiles"
        )
        is True,
        "three_synchronized_appearance_views": planned["appearance_view_sequences"]
        == 3 * planned["physical_episodes"],
        "appearance_not_counted_as_independent_physics": True,
    }
    completion_gate = {
        "all_scheduled_pairs_finished": runtime["finished_pairs"]
        == runtime["scheduled_pairs"],
        "formal_runtime_a0_ready": readiness.get("experiments", {})
        .get("A0", {})
        .get("ready")
        is True,
        "all_A0_A7_ready": readiness.get("realistic_a0_a7_complete") is True,
        "eight_experiments_ready": int(readiness.get("n_ready", 0)) == 8,
    }
    synthetic_ready = all(
        value is True
        for group in (schedule_gate, provenance_gate, randomization_gate, completion_gate)
        for value in group.values()
    )
    bottlenecks = []
    if not completion_gate["all_scheduled_pairs_finished"]:
        bottlenecks.append(
            {
                "gate": "physical_collection",
                "remaining_pairs": runtime["scheduled_pairs"] - runtime["finished_pairs"],
            }
        )
    missing = list(readiness.get("missing_experiments", []))
    if missing:
        bottlenecks.append({"gate": "A0-A7", "missing_experiments": missing})
    # No real-robot artifact is part of the current contract; keep the stronger term closed.
    bottlenecks.append(
        {
            "gate": "real_go2_anchor",
            "required_for": "sim2real_validated",
            "present": False,
        }
    )
    result = {
        "schema_version": "kinofail.realistic-icra-benchmark-readiness.v1",
        "created_utc": datetime.now(UTC).isoformat(),
        "status": "publication_ready" if synthetic_ready else "in_progress",
        "dataset_scope": "kinofail_realistic",
        "contract": {"path": str(contract_path.relative_to(ROOT)), "sha256": _sha(contract_path)},
        "protocol": {"path": str(protocol_path.relative_to(ROOT)), "sha256": _sha(protocol_path)},
        "repair_overlay": {
            "protocol": str(repair_protocol_path.relative_to(ROOT)),
            "protocol_sha256": _sha(repair_protocol_path),
            "corpus_root": str(repair_corpus.relative_to(ROOT)),
            "pairs_used": runtime["repair_overlay_pair_count"],
            "pairs_planned": len(
                repair_protocol.get("allowed", {}).get("counterfactual_group_ids", [])
            ),
        },
        "planned": planned,
        "runtime_progress": runtime,
        "gates": {
            "schedule_and_leakage": schedule_gate,
            "provenance": provenance_gate,
            "randomization": randomization_gate,
            "completion": completion_gate,
        },
        "verdict": {
            "icra_synthetic_benchmark_ready": synthetic_ready,
            "allowed_term": "sim2real-oriented synthetic causal benchmark",
            "sim2real_validated": False,
            "real_go2_anchor_present": False,
        },
        "bottlenecks": bottlenecks,
        "a8_in_scope": False,
    }
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(
        json.dumps(result, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    print(
        json.dumps(
            {
                "out": str(args.out),
                "status": result["status"],
                "finished_pairs": runtime["finished_pairs"],
                "scheduled_pairs": runtime["scheduled_pairs"],
                "n_ready": readiness.get("n_ready", 0),
                "icra_synthetic_benchmark_ready": synthetic_ready,
                "bottlenecks": bottlenecks,
            },
            indent=2,
        )
    )
    return 0 if synthetic_ready else 2


if __name__ == "__main__":
    raise SystemExit(main())
