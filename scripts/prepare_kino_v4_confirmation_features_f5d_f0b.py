#!/usr/bin/env python3
"""Prepare the F5 feature seal for the decision-window F0b checkpoint."""

from __future__ import annotations

import json
from pathlib import Path
import sys

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from scripts import prepare_kino_v4_confirmation_features_f5d as predecessor


F0B = ROOT / "outputs/freeze/kino_v4_decision_window_f0b/freeze_manifest.json"
F0B_BUILDER = ROOT / "scripts/build_kino_v4_confirmation_conflict_base_features_f0b.py"
LEGACY_BUILDER = "scripts/build_kino_v4_confirmation_conflict_base_features_v1.py"


def _sha256(path: Path) -> str:
    import hashlib

    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def main() -> int:
    f0b = json.loads(F0B.read_text(encoding="utf-8"))
    if (
        f0b.get("confirmation_prediction_truth_key_or_score_read") is not False
        or f0b.get("method", {}).get("deployment_inputs_only")
        != [
            "five-frame robot-front RGB sequence",
            "80-D decision-window proprioceptive summary",
        ]
    ):
        raise RuntimeError("F0b input contract is not frozen and model blind")

    module = predecessor.predecessor.module()
    module.F4 = predecessor.OBSERVATION
    module.DEFAULT_OUTPUT = predecessor.OUTPUT
    module.T3_CORPUS = predecessor.COMBINED_CORPUS
    original_prepare_scale = module._prepare_scale
    original_run = module._run

    def run(arguments: list[str]) -> None:
        values = list(arguments)
        if values and values[0] == LEGACY_BUILDER:
            values[0] = str(F0B_BUILDER.relative_to(ROOT))
        original_run(values)

    def prepare_scale(output: Path, seal: dict, scenes: list[str]) -> dict:
        report = original_prepare_scale(output, seal, scenes)
        features_path = output / "scale/features.npz"
        with np.load(features_path, allow_pickle=False) as archive:
            sample_ids = archive["sample_ids"].astype(str)
            visual = np.asarray(archive["visual"], dtype=np.float32)
            invariant = np.asarray(archive["invariant_proprio"], dtype=np.float32)
            full = np.asarray(archive["full_proprio"], dtype=np.float32)
        np.savez_compressed(
            features_path,
            sample_ids=sample_ids,
            visual=visual,
            full_proprio=np.zeros_like(full, dtype=np.float32),
            invariant_proprio=invariant,
        )
        report["output_sha256"]["features"] = module._sha256(features_path)
        report["decision_window_f0b"] = {
            "legacy_190d_compatibility_block_uniformly_disabled": True,
            "invariant_dimension": int(invariant.shape[1]),
            "applies_to_every_scale_sample": True,
        }
        scale_manifest = output / "scale/feature_manifest.json"
        module._write_json(scale_manifest, {"passed": True, **report})
        return report

    module._run = run
    module._prepare_scale = prepare_scale
    result = int(module.main())

    seal_path = predecessor.OUTPUT / "feature_seal.json"
    seal = json.loads(seal_path.read_text(encoding="utf-8"))
    with np.load(
        predecessor.OUTPUT / "scale/features.npz", allow_pickle=False
    ) as archive:
        scale_full = np.asarray(archive["full_proprio"], dtype=np.float32)
        scale_invariant = np.asarray(
            archive["invariant_proprio"], dtype=np.float32
        )
    with np.load(
        predecessor.OUTPUT / "conflict/base_features/features.npz",
        allow_pickle=False,
    ) as archive:
        conflict_full = np.asarray(archive["proprio"], dtype=np.float32)
    with np.load(
        predecessor.OUTPUT / "conflict/invariant_features/features.npz",
        allow_pickle=False,
    ) as archive:
        conflict_invariant = np.asarray(archive["proprio"], dtype=np.float32)
    exact_zero = bool(
        np.count_nonzero(scale_full) == 0
        and np.count_nonzero(conflict_full) == 0
    )
    invariant_contract = bool(
        scale_invariant.ndim == 2
        and conflict_invariant.ndim == 2
        and scale_invariant.shape[1] == 80
        and conflict_invariant.shape[1] == 80
        and np.isfinite(scale_invariant).all()
        and np.isfinite(conflict_invariant).all()
    )
    seal["schema_version"] = "kinofail.kino-v4-confirmation-feature-seal-f0b.v1"
    seal["status"] = "sealed_before_f0_checkpoint_inference"
    seal["checks"].update(
        {
            "legacy_190d_compatibility_block_uniformly_disabled": True,
            "only_registered_80d_decision_window_proprioception_deployed": True,
            "all_scale_and_conflict_compatibility_values_exactly_zero": exact_zero,
            "all_deployed_proprioception_is_finite_and_80d": invariant_contract,
            "confirmation_prediction_truth_key_or_score_read": False,
        }
    )
    seal["passed"] = all(seal["checks"].values())
    seal["method_f0b"] = {
        "manifest": str(F0B.relative_to(ROOT)),
        "sha256": _sha256(F0B),
    }
    seal["code_sha256"].update(
        {
            "f0b_preparer": _sha256(Path(__file__).resolve()),
            "conflict_base_f0b": _sha256(F0B_BUILDER),
        }
    )
    module._write_json(seal_path, seal)
    return result


if __name__ == "__main__":
    raise SystemExit(main())
