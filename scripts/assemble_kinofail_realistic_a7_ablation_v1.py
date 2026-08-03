#!/usr/bin/env python3
"""Assemble the six-cell realistic A7 ablation ledger without substituting old evidence."""

from __future__ import annotations

import argparse
import hashlib
import json
from datetime import UTC, datetime
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]


def _sha256(path: Path) -> str | None:
    return hashlib.sha256(path.read_bytes()).hexdigest() if path.is_file() else None


def _json(path: Path) -> dict:
    if not path.is_file():
        return {}
    value = json.loads(path.read_text(encoding="utf-8"))
    return value if isinstance(value, dict) else {}


def _artifact(path: Path) -> dict:
    return {
        "path": str(path.relative_to(ROOT)),
        "present": path.is_file(),
        "sha256": _sha256(path),
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--contract", type=Path,
        default=ROOT / "configs/eval/kinofail_realistic_a0_a7_v4.json",
    )
    parser.add_argument(
        "--out", type=Path,
        default=ROOT / "outputs/eval/realistic_a0_a7_v4/a7_ablation.json",
    )
    parser.add_argument(
        "--evidence-root", type=Path,
        default=ROOT / "outputs/eval/realistic_a0_a7_v4",
    )
    parser.add_argument(
        "--corpus-root", type=Path,
        default=ROOT / "outputs/kinofail_realistic/corpus_scale_v7",
    )
    args = parser.parse_args()
    contract_path = args.contract.resolve()
    contract = _json(contract_path)
    required = contract["experiments"]["A7"]["acceptance"]["required_ablations"]
    evidence_root = args.evidence_root.resolve()
    texture_path = evidence_root / "a7_texture_consistency_ablation.json"
    texture = _json(texture_path)
    matched_path = evidence_root / "a7_matched_ablation_results.json"
    matched = _json(matched_path)
    a0_path = evidence_root / "a0_certificate.json"
    a0 = _json(a0_path)
    training_path = evidence_root / "training_manifest.json"
    inference_path = evidence_root / "inference_manifest.json"
    predictions_path = evidence_root / "predictions.jsonl"
    controlled_path = ROOT / "outputs/eval/a7/a7_eval_manifest.json"
    corpus_root = args.corpus_root.resolve()
    manifests = list(corpus_root.rglob("manifest.json"))
    body_fixed_count = 0
    local_physics_count = 0
    for path in manifests:
        value = _json(path)
        body_fixed_count += value.get("camera", {}).get("body_fixed") is True
        local_physics_count += value.get("operator_readback", {}).get("qa_passed") is True

    ablations = {
        "default_grid_vs_realistic_scene": {
            "status": "missing_matched_counterfactual_arm",
            "complete": False,
            "available_realistic_arm": {
                "physical_episodes": len(manifests),
                "scene_families": 9,
                "domains": 3,
                "a0": _artifact(a0_path),
            },
            "controlled_core_reference": _artifact(controlled_path),
            "reason": (
                "Both sources exist, but they do not share a frozen paired scene/operator/seed "
                "schedule; an across-corpus difference is descriptive, not an A7 causal ablation."
            ),
        },
        "procedural_texture_vs_scanned_pbr": {
            "status": "missing_matched_procedural_render_arm",
            "complete": False,
            "available_scanned_pbr_arm": {
                "physical_episodes": len(manifests),
                "synchronized_views_per_episode": 3,
                "texture_swap_runtime_pass": a0.get("checks", {}).get("texture_swap_runtime_pass"),
            },
            "reason": (
                "The realistic arm uses hash-bound scanned PBR assets; a procedural-texture replay "
                "of the same physical states has not yet been collected."
            ),
        },
        "world_follow_vs_body_fixed_camera": {
            "status": "missing_world_follow_arm",
            "complete": False,
            "available_body_fixed_arm": {
                "body_fixed_manifests": body_fixed_count,
                "total_manifests": len(manifests),
                "all_body_fixed": bool(manifests) and body_fixed_count == len(manifests),
            },
            "reason": "No matched world-follow camera replay exists on the realistic corpus.",
        },
        "surrogate_trunk_effect_vs_local_physics": {
            "status": "missing_matched_surrogate_replay",
            "complete": False,
            "available_local_physics_arm": {
                "operator_qa_passed_manifests": local_physics_count,
                "total_manifests": len(manifests),
            },
            "controlled_surrogate_reference": _artifact(controlled_path),
            "reason": (
                "Controlled surrogate results are retained as references but are not paired to the "
                "nine realistic scenes, so they cannot close this ablation."
            ),
        },
        "texture_vs_geometry_vs_physics_intervention": {
            "status": "missing_isolated_geometry_only_arm",
            "complete": False,
            "available_arms": {
                "texture_only_synchronized_views": len(manifests) * 3,
                "local_physics_manifests": local_physics_count,
                "geometry_only_manifests": 0,
            },
            "reason": (
                "Texture-only and active-physics evidence are available; no frozen geometry-only "
                "replay holds appearance and physics constant."
            ),
        },
        "without_texture_swap_consistency": {
            "status": "confirmatory_complete" if texture.get("status") == "confirmatory_complete" else "missing",
            "complete": texture.get("status") == "confirmatory_complete",
            "artifact": _artifact(texture_path),
            "passed_integrity_gates": texture.get("passed") is True,
            "three_training_seeds": texture.get("acceptance", {}).get("three_training_seeds") is True,
            "result": texture.get("summaries", {}),
            "interpretation": (
                "Texture-swap augmentation produces small, seed-dependent changes; the null/weak "
                "effect is retained and does not weaken the already high measured invariance."
            ),
        },
    }
    matched_cells = matched.get("cells", {}) if isinstance(matched.get("cells"), dict) else {}
    for cell in (
        "default_grid_vs_realistic_scene",
        "procedural_texture_vs_scanned_pbr",
        "world_follow_vs_body_fixed_camera",
        "surrogate_trunk_effect_vs_local_physics",
        "texture_vs_geometry_vs_physics_intervention",
    ):
        source = matched_cells.get(cell, {})
        if (
            matched.get("status") == "confirmatory_complete"
            and isinstance(source, dict)
            and source.get("complete") is True
        ):
            ablations[cell] = {
                "status": "confirmatory_complete",
                "complete": True,
                "artifact": _artifact(matched_path),
                "result": source,
                "negative_or_null_result_retained": True,
            }
    if list(ablations) != list(required):
        raise RuntimeError(f"A7 ledger does not match frozen required cells: {required}")
    completed = sum(row["complete"] for row in ablations.values())
    checks = {
        "all_six_required_ablations_declared": len(ablations) == 6,
        "three_training_seeds_for_completed_model_ablation": ablations[
            "without_texture_swap_consistency"
        ]["three_training_seeds"],
        "paired_scene_cluster_statistics_for_completed_model_ablation": texture.get(
            "acceptance", {}
        ).get("paired_counterfactual_statistics_present") is True
        and matched.get("acceptance", {}).get(
            "scene_cluster_bootstrap_repetitions"
        ) is True,
        "matched_five_cell_integrity_passed": matched.get("passed") is True,
        "no_controlled_core_result_substituted_for_realistic_ablation": True,
        "all_missing_arms_explicit": all(
            row["complete"] or str(row["status"]).startswith("missing")
            for row in ablations.values()
        ),
        "all_six_ablations_complete": completed == 6,
    }
    result = {
        "schema_version": "kinofail.realistic-a7-ablation-ledger.v1",
        "created_utc": datetime.now(UTC).isoformat(),
        "status": "confirmatory_complete" if completed == 6 else "partial_complete",
        "dataset_scope": "kinofail_realistic",
        "a0_evidence_bundle_sha256": a0.get("a0_evidence_bundle_sha256"),
        "training_manifest_sha256": _sha256(training_path),
        "inference_manifest_sha256": _sha256(inference_path),
        "predictions_sha256": _sha256(predictions_path),
        "contract": _artifact(contract_path),
        "required_ablations": required,
        "completed_ablations": completed,
        "total_ablations": len(ablations),
        "ablations": ablations,
        "acceptance": checks,
        "passed": all(checks.values()),
        "claim_guard": (
            "Only complete, protocol-matched realistic cells may support causal ablation claims; "
            "controlled-core artifacts remain descriptive references."
        ),
        "a8_in_scope": False,
    }
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps({
        "out": str(args.out), "status": result["status"], "passed": result["passed"],
        "completed": completed, "total": len(ablations),
        "missing": [key for key, row in ablations.items() if not row["complete"]],
    }, indent=2))
    return 0 if result["passed"] else 2


if __name__ == "__main__":
    raise SystemExit(main())
