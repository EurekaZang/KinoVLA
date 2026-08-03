#!/usr/bin/env python3
"""Hash-bind the predeclared texture ablation to completed scale-v8 features."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]


def _sha(path: Path) -> str:
    if not path.is_file():
        raise FileNotFoundError(path)
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--evidence-root",
        type=Path,
        default=ROOT / "outputs/eval/realistic_a0_a7_v5",
    )
    parser.add_argument(
        "--out",
        type=Path,
        default=ROOT
        / "configs/eval/kinofail_realistic_a7_texture_ablation_scale_v8_v1.json",
    )
    parser.add_argument(
        "--protocol-id",
        default="kinofail_realistic_a7_texture_ablation_scale_v8_v1",
    )
    args = parser.parse_args()
    evidence_root = args.evidence_root.resolve()
    output_path = args.out.resolve()
    paths = {
        "records": evidence_root / "snapshots/snapshot_records.jsonl",
        "features": evidence_root / "features/features.npz",
        "feature_manifest": evidence_root / "features/feature_manifest.json",
        "scene_registry": ROOT / "configs/data/kinofail_realistic_scale_scene_registry_v1.json",
        "model_implementation": ROOT / "kino_vla/eval/realistic_multimodal.py",
    }
    value = {
        "schema_version": "kinofail.realistic-a7-texture-ablation-protocol.v1",
        "protocol_id": str(args.protocol_id),
        "status": "frozen_before_scale_v8_a7_outcomes",
        "dataset_scope": "kinofail_realistic",
        "a8_in_scope": False,
        "freeze_policy": (
            "Architecture, split, seeds and statistics are copied unchanged from the scale-v7 "
            "ablation; this step only binds hashes after deterministic feature extraction and "
            "before fitting either ablation arm."
        ),
        **{
            key: str(path.relative_to(ROOT))
            for key, path in paths.items()
        },
        **{
            f"{key}_sha256": _sha(path)
            for key, path in paths.items()
        },
        "training_seeds": [0, 1, 2],
        "fit_split": "registry scene split in {train,val}",
        "test_split": "registry scene split test",
        "conditions": {
            "with_texture_swap_augmentation": "all three synchronized appearance views in fit",
            "without_texture_swap_augmentation": "primary appearance view only in fit",
        },
        "headline_view": "primary",
        "texture_consistency_views": ["primary", "swap_01", "swap_02"],
        "statistical_unit": "paired counterfactual physics group clustered over physical episodes",
        "bootstrap_repetitions": 10000,
        "bootstrap_seed": 2026072307,
        "negative_results_retained": True,
        "output": str(
            (evidence_root / "a7_texture_consistency_ablation.json").relative_to(ROOT)
        ),
    }
    if output_path.is_file():
        previous = json.loads(output_path.read_text(encoding="utf-8"))
        if previous != value:
            raise RuntimeError(f"existing frozen protocol differs: {output_path}")
        print(
            json.dumps(
                {"unchanged": str(output_path), "sha256": _sha(output_path)}, indent=2
            )
        )
        return 0
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(
        json.dumps(value, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    print(
        json.dumps(
            {"frozen": str(output_path), "sha256": _sha(output_path)}, indent=2
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
