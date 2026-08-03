#!/usr/bin/env python3
"""Run write-once, checkpoint-only blind inference for confirmation data.

No training API is used.  The input archive is deliberately restricted to
``sample_ids``, ``visual``, and ``proprio`` so labels and experimental
metadata cannot enter the inference process.  Oracle routing belongs only in
the later one-shot unblinding scorer and is intentionally absent here.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import pickle
import sys
from pathlib import Path
from typing import Any

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from kino_vla.eval.realistic_multimodal import (  # noqa: E402
    ObservableClassifier,
)
from kino_vla.eval.unified_moe import (  # noqa: E402
    EXPERT_NAMES,
    UnifiedEvidenceMoE,
)
from scripts.audit_kinofail_confirmatory_freshness_v1 import (  # noqa: E402
    validate_f0,
)
from scripts.freeze_kinofail_unified_confirmatory_f0 import (  # noqa: E402
    EXPECTED_CHECKPOINT_SHA256,
    audit_checkpoint,
    checkpoint_paths,
    sha256_file,
)

ALLOWED_ARCHIVE_KEYS = {"sample_ids", "visual", "proprio"}
FORBIDDEN_KEY_FRAGMENTS = (
    "truth",
    "ground_truth",
    "label",
    "scene",
    "material",
    "operator",
    "severity",
    "domain",
    "split",
    "outcome",
    "cost",
    "target",
    "privileged",
)
LEGACY_BODY_CLASSES = {
    "effort_decay",
    "external_push",
    "high_centering",
    "invisible_obstacle",
    "low_friction",
    "obs_bias",
    "overload",
    "region_collapse",
}


def _sha256_text(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _load_json(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise TypeError(path)
    return value


def _ensure_new_output(path: Path) -> None:
    if path.exists():
        raise FileExistsError(f"refusing to overwrite blind inference output: {path}")


def load_blind_features(
    path: Path,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, list[str]]:
    with np.load(path, allow_pickle=False) as archive:
        keys = list(archive.files)
        lowered = [key.lower() for key in keys]
        forbidden = sorted(
            key
            for key, lower in zip(keys, lowered, strict=True)
            if any(fragment in lower for fragment in FORBIDDEN_KEY_FRAGMENTS)
        )
        if forbidden:
            raise ValueError(
                "blind feature bundle contains forbidden metadata: " + ", ".join(forbidden)
            )
        if set(keys) != ALLOWED_ARCHIVE_KEYS:
            raise ValueError(
                "blind feature bundle keys must be exactly "
                f"{sorted(ALLOWED_ARCHIVE_KEYS)}, received {sorted(keys)}"
            )
        sample_ids = archive["sample_ids"].astype(str)
        visual = np.asarray(archive["visual"], dtype=np.float32)
        proprio = np.asarray(archive["proprio"], dtype=np.float32)
    if (
        sample_ids.ndim != 1
        or visual.shape != (len(sample_ids), 1536)
        or proprio.shape != (len(sample_ids), 80)
    ):
        raise ValueError(
            "blind feature dimensions must be sample_ids=(N,), "
            f"visual=(N,1536), proprio=(N,80); received "
            f"{sample_ids.shape}, {visual.shape}, {proprio.shape}"
        )
    if len(set(sample_ids.tolist())) != len(sample_ids):
        raise ValueError("blind sample_ids must be unique")
    if not np.isfinite(visual).all() or not np.isfinite(proprio).all():
        raise ValueError("blind features contain non-finite values")
    return sample_ids, visual, proprio, keys


def _install_training_guards() -> dict[str, int]:
    counter = {"fit_attempts": 0}

    def blocked_fit(*_args: Any, **_kwargs: Any) -> Any:
        counter["fit_attempts"] += 1
        raise RuntimeError("fit/refit is forbidden in blind inference")

    UnifiedEvidenceMoE.fit = classmethod(blocked_fit)
    ObservableClassifier.fit = classmethod(blocked_fit)
    return counter


def _load_checkpoint(path: Path, seed: int) -> UnifiedEvidenceMoE:
    audit_checkpoint(path, seed)
    with path.open("rb") as stream:
        model = pickle.load(stream)  # noqa: S301 - hash-pinned local artifact
    if not isinstance(model, UnifiedEvidenceMoE):
        raise TypeError(path)
    return model


def _method_predictions(
    model: UnifiedEvidenceMoE,
    visual: np.ndarray,
    proprio: np.ndarray,
    *,
    random_seed: int,
) -> dict[str, dict[str, Any]]:
    routed = model.predict_with_routes(visual, proprio)
    expert = routed.expert_probabilities
    vision = EXPERT_NAMES.index("vision")
    prop = EXPERT_NAMES.index("proprio")
    joint = EXPERT_NAMES.index("joint")
    methods: dict[str, dict[str, Any]] = {
        "learned_router": {
            "probabilities": routed.probabilities,
            "routes": routed.routes,
            "evidence_routes": routed.evidence_routes,
            "route_scores": routed.route_scores,
            "route_probabilities": routed.route_probabilities,
            "disagreement": routed.disagreement,
        },
        "fixed_vision": {
            "probabilities": expert[:, vision, :],
            "routes": np.full(len(visual), "vision"),
            "evidence_routes": np.full(len(visual), "vision"),
        },
        "fixed_proprio": {
            "probabilities": expert[:, prop, :],
            "routes": np.full(len(visual), "proprio"),
            "evidence_routes": np.full(len(visual), "proprio"),
        },
        "fixed_joint": {
            "probabilities": expert[:, joint, :],
            "routes": np.full(len(visual), "joint"),
            "evidence_routes": np.full(len(visual), "cross_modal_interaction"),
        },
        "late_average": {
            "probabilities": expert.mean(axis=1),
            "routes": np.full(len(visual), "late_average"),
            "evidence_routes": np.full(len(visual), "all_modalities"),
        },
    }
    prop_prediction = np.asarray(model.classes)[expert[:, prop, :].argmax(axis=1)]
    use_prop = np.isin(prop_prediction, sorted(LEGACY_BODY_CLASSES))
    legacy = expert[:, vision, :].copy()
    legacy[use_prop] = expert[use_prop, prop, :]
    methods["legacy_class_rule"] = {
        "probabilities": legacy,
        "routes": np.where(use_prop, "proprio", "vision"),
        "evidence_routes": np.where(use_prop, "proprio", "vision"),
    }
    generator = np.random.default_rng(int(random_seed))
    random_route = generator.integers(0, len(EXPERT_NAMES), size=len(visual))
    methods["random_route"] = {
        "probabilities": expert[np.arange(len(visual)), random_route],
        "routes": np.asarray(EXPERT_NAMES)[random_route],
        "evidence_routes": np.asarray(EXPERT_NAMES)[random_route],
    }
    return methods


def _rows_for_model(
    *,
    sample_ids: np.ndarray,
    seed: int,
    checkpoint_hash: str,
    classes: list[str],
    methods: dict[str, dict[str, Any]],
) -> list[dict[str, Any]]:
    rows = []
    for method, result in methods.items():
        probabilities = np.asarray(result["probabilities"], dtype=np.float64)
        prediction = np.asarray(classes)[probabilities.argmax(axis=1)]
        confidence = probabilities.max(axis=1)
        for index, sample_id in enumerate(sample_ids):
            row: dict[str, Any] = {
                "sample_id": str(sample_id),
                "checkpoint_id": f"seed{seed}",
                "checkpoint_sha256": checkpoint_hash,
                "method": method,
                "probabilities": {
                    label: float(probabilities[index, class_index])
                    for class_index, label in enumerate(classes)
                },
                "prediction": str(prediction[index]),
                "confidence": float(confidence[index]),
                "route": str(result["routes"][index]),
                "evidence_route": str(result["evidence_routes"][index]),
            }
            if method == "learned_router":
                row.update(
                    {
                        "route_scores": np.asarray(result["route_scores"][index]).tolist(),
                        "route_probabilities": np.asarray(
                            result["route_probabilities"][index]
                        ).tolist(),
                        "expert_disagreement": bool(result["disagreement"][index]),
                    }
                )
            rows.append(row)
    return rows


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--f0-manifest", type=Path, required=True)
    parser.add_argument("--f1-manifest", type=Path, required=True)
    parser.add_argument("--freshness-audit", type=Path, required=True)
    parser.add_argument("--features", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()
    output = args.output_dir.resolve()
    _ensure_new_output(output)

    f0, f0_checks = validate_f0(args.f0_manifest)
    if not all(f0_checks.values()):
        raise RuntimeError(f"F0 freeze validation failed: {f0_checks}")
    freshness = _load_json(args.freshness_audit)
    if freshness.get("passed") is not True:
        raise RuntimeError("freshness audit did not pass")
    if freshness.get("source_sha256", {}).get("f0_manifest") != sha256_file(args.f0_manifest):
        raise RuntimeError("freshness audit is bound to another F0")
    f1 = _load_json(args.f1_manifest)
    f1_sidecar = args.f1_manifest.with_name("seal_manifest.sha256")
    if (
        f1.get("status")
        != "sealed_before_anomaly_collection_and_model_inference"
        or f1.get("confirmatory") is not True
        or not f1_sidecar.is_file()
        or f1_sidecar.read_text(encoding="utf-8").split()[0]
        != sha256_file(args.f1_manifest)
        or f1.get("input_sha256", {}).get("f0_manifest")
        != sha256_file(args.f0_manifest)
        or f1.get("input_sha256", {}).get("freshness_audit")
        != sha256_file(args.freshness_audit)
    ):
        raise RuntimeError("F1 seal validation failed or is bound to another F0/freshness audit")
    frozen_checkpoint_hashes = {
        int(item["seed"]): str(item["sha256"]) for item in f0["architecture"]["checkpoint_audits"]
    }
    if frozen_checkpoint_hashes != EXPECTED_CHECKPOINT_SHA256:
        raise RuntimeError("F0 checkpoint set differs from frozen v3")

    sample_ids, visual, proprio, input_keys = load_blind_features(args.features.resolve())
    models = {seed: _load_checkpoint(path, seed) for seed, path in checkpoint_paths().items()}
    fit_counter = _install_training_guards()

    rows: list[dict[str, Any]] = []
    random_seed = int(f0["analysis_contract"]["random_route_seed"])
    for seed, model in models.items():
        methods = _method_predictions(
            model,
            visual,
            proprio,
            random_seed=random_seed + seed,
        )
        rows.extend(
            _rows_for_model(
                sample_ids=sample_ids,
                seed=seed,
                checkpoint_hash=EXPECTED_CHECKPOINT_SHA256[seed],
                classes=model.classes,
                methods=methods,
            )
        )
    if fit_counter["fit_attempts"] != 0:
        raise RuntimeError("a training call was attempted")

    output.mkdir(parents=True, exist_ok=False)
    predictions_path = output / "blind_predictions.jsonl"
    predictions_path.write_text(
        "".join(json.dumps(row, sort_keys=True) + "\n" for row in rows),
        encoding="utf-8",
    )
    methods = sorted({str(row["method"]) for row in rows})
    manifest = {
        "schema_version": "kinofail.unified-moe-blind-predictions.v1",
        "status": "blind_predictions_sealed",
        "labels_or_outcomes_read": False,
        "oracle_route_computed": False,
        "fit_or_refit_called": False,
        "input_archive_keys": input_keys,
        "sample_count": len(sample_ids),
        "checkpoint_count": len(models),
        "methods": methods,
        "route_contract": {
            "threshold": 0.8,
            "comparison": "strictly_greater_than",
            "equal_vision_joint_excess_tie_break": "vision",
            "default_route": "proprio",
        },
        "source_sha256": {
            "f0_manifest": sha256_file(args.f0_manifest),
            "f1_manifest": sha256_file(args.f1_manifest),
            "freshness_audit": sha256_file(args.freshness_audit),
            "blind_features": sha256_file(args.features),
            "predictor": _sha256_text(Path(__file__).resolve()),
            "checkpoints": {
                str(seed): EXPECTED_CHECKPOINT_SHA256[seed]
                for seed in sorted(EXPECTED_CHECKPOINT_SHA256)
            },
        },
        "artifact": {
            "predictions": "blind_predictions.jsonl",
            "predictions_sha256": sha256_file(predictions_path),
            "rows": len(rows),
        },
    }
    manifest_path = output / "prediction_manifest.json"
    manifest_path.write_text(
        json.dumps(manifest, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    (output / "prediction_manifest.sha256").write_text(
        f"{sha256_file(manifest_path)}  prediction_manifest.json\n",
        encoding="utf-8",
    )
    print(
        json.dumps(
            {
                "passed": True,
                "status": manifest["status"],
                "samples": len(sample_ids),
                "rows": len(rows),
                "predictions_sha256": manifest["artifact"]["predictions_sha256"],
                "fit_or_refit_called": False,
            },
            indent=2,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
