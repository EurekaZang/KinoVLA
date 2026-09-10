#!/usr/bin/env python3
"""Evaluate the same-stage control for KiNO's evidence route.

The control pairs the original anticipatory T2 route decisions with persistent
robot-internal O5/O10 failures.  Both arms use 21x19 history-only body packets.
O5/O10 pairs are assigned the same recorded T2 RGB sequence; this certifies a
body-decisive input pair without changing KiNO's proprioceptive route score.
"""

from __future__ import annotations

import argparse
import hashlib
import importlib.util
import json
import sys
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import numpy as np
from sklearn.ensemble import ExtraTreesClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import balanced_accuracy_score, recall_score
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from kino_vla.eval.conflict_invariant_kino import (  # noqa: E402
    invariant_relative_proprio_features,
)

DEVELOPMENT_SCRIPT = ROOT / "scripts/develop_kinofail_conflict_invariant_kino_v4.py"
PREDICTIONS = (
    ROOT
    / "outputs/eval/kino_conflict_invariant_v4_development_dinov2large_terrain448_eta011"
    / "scene_disjoint_predictions.jsonl"
)
DEFAULT_OUTPUT = ROOT / "outputs/eval/kino_phase_matched_routing_control_v1"
SEED = 20260803
BOOTSTRAP_DRAWS = 20_000


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _jsonl(path: Path) -> list[dict[str, Any]]:
    return [json.loads(line) for line in path.read_text().splitlines() if line]


def _load_development_module() -> Any:
    spec = importlib.util.spec_from_file_location("kino_v4_development", DEVELOPMENT_SCRIPT)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"cannot import {DEVELOPMENT_SCRIPT}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _stable_order(sample_ids: np.ndarray, indices: np.ndarray) -> list[int]:
    return sorted(
        indices.tolist(),
        key=lambda index: hashlib.sha256(str(sample_ids[index]).encode("utf-8")).hexdigest(),
    )


def _acceleration_range(summary: np.ndarray) -> np.ndarray:
    values = np.asarray(summary, dtype=np.float32)
    if values.ndim != 2 or values.shape[1] != 190:
        raise ValueError(f"expected 190-D proprio summaries, received {values.shape}")
    blocks = values.reshape(len(values), 10, 19)
    minimum = blocks[:, 2, 6:9]
    maximum = blocks[:, 3, 6:9]
    return np.linalg.norm(maximum - minimum, axis=1).reshape(-1, 1)


def _balanced_accuracy(labels: np.ndarray, predictions: np.ndarray) -> float:
    return float(balanced_accuracy_score(labels, predictions))


def _scene_bootstrap(
    scenes: np.ndarray,
    labels: np.ndarray,
    kino: np.ndarray,
    proxy: np.ndarray,
) -> dict[str, Any]:
    unique = np.asarray(sorted(set(scenes.tolist())))
    lookup = {scene: np.flatnonzero(scenes == scene) for scene in unique}
    rng = np.random.default_rng(SEED)
    values = np.empty((BOOTSTRAP_DRAWS, 3), dtype=np.float64)
    for draw in range(BOOTSTRAP_DRAWS):
        sampled = rng.choice(unique, size=len(unique), replace=True)
        indices = np.concatenate([lookup[str(scene)] for scene in sampled])
        kino_ba = _balanced_accuracy(labels[indices], kino[indices])
        proxy_ba = _balanced_accuracy(labels[indices], proxy[indices])
        values[draw] = (kino_ba, proxy_ba, kino_ba - proxy_ba)
    names = ("kino", "phase_proxy", "kino_minus_phase_proxy")
    return {
        name: {
            "estimate": float(value),
            "ci95": [
                float(np.quantile(values[:, index], 0.025)),
                float(np.quantile(values[:, index], 0.975)),
            ],
        }
        for index, (name, value) in enumerate(
            zip(
                names,
                (
                    _balanced_accuracy(labels, kino),
                    _balanced_accuracy(labels, proxy),
                    _balanced_accuracy(labels, kino) - _balanced_accuracy(labels, proxy),
                ),
                strict=True,
            )
        )
    }


def _cross_validated_logistic(
    features: np.ndarray,
    labels: np.ndarray,
    folds: np.ndarray,
) -> np.ndarray:
    predictions = np.zeros(len(labels), dtype=np.int8)
    for fold in range(5):
        train = folds != fold
        test = ~train
        model = Pipeline(
            [
                ("scale", StandardScaler()),
                (
                    "classifier",
                    LogisticRegression(
                        C=1.0,
                        class_weight="balanced",
                        max_iter=2_000,
                        random_state=SEED + fold,
                    ),
                ),
            ]
        )
        model.fit(features[train], labels[train])
        predictions[test] = model.predict(features[test]).astype(np.int8)
    return predictions


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    args = parser.parse_args()
    output = args.output.resolve()
    if output.exists():
        raise FileExistsError(output)

    module = _load_development_module()
    scale = module._load_scale()
    conflict = module._load_conflict()
    prediction_rows = _jsonl(PREDICTIONS)
    prediction_lookup = {
        (str(row["dataset"]), str(row["sample_id"])): row for row in prediction_rows
    }

    conflict_primary = np.asarray(
        [row.get("appearance_view_id") == "primary" for row in conflict["rows"]]
    )
    vision_indices = np.flatnonzero(
        conflict_primary & conflict["vision_decisive"] & (conflict["labels"] == "compliant_terrain")
    )
    if len(vision_indices) != 1_500:
        raise RuntimeError(f"expected 1,500 T2 cases, found {len(vision_indices)}")

    scale_primary = np.asarray(
        [row.get("appearance_view_id") == "primary" for row in scale["rows"]]
    )
    body_pairs: list[tuple[int, int, int]] = []
    for scene in sorted(set(scale["scenes"].tolist())):
        per_class: dict[str, list[int]] = {}
        for label in ("overload", "effort_decay"):
            candidates = np.flatnonzero(
                scale_primary & (scale["scenes"] == scene) & (scale["labels"] == label)
            )
            ordered = _stable_order(scale["sample_ids"], candidates)
            if len(ordered) < 25:
                raise RuntimeError(f"{scene} has only {len(ordered)} {label} samples")
            per_class[label] = ordered[:25]

        scene_t2 = [
            index for index in vision_indices.tolist() if str(conflict["scenes"][index]) == scene
        ]
        scene_t2 = _stable_order(conflict["sample_ids"], np.asarray(scene_t2))
        if len(scene_t2) < 25:
            raise RuntimeError(f"{scene} has only {len(scene_t2)} T2 visual sources")
        for pair_index, (overload, decay) in enumerate(
            zip(per_class["overload"], per_class["effort_decay"], strict=True)
        ):
            body_pairs.append((overload, decay, scene_t2[pair_index]))

    body_indices = np.asarray(
        [index for overload, decay, _ in body_pairs for index in (overload, decay)],
        dtype=np.int64,
    )
    if len(body_pairs) != 750 or len(body_indices) != 1_500:
        raise RuntimeError("P0 construction must contain 750 pairs and 1,500 samples")

    rows: list[dict[str, Any]] = []
    full: list[np.ndarray] = []
    route_labels: list[int] = []
    route_predictions: list[int] = []
    folds: list[int] = []
    scenes: list[str] = []

    for index in vision_indices:
        sample_id = str(conflict["sample_ids"][index])
        prediction = prediction_lookup[("conflict", sample_id)]
        rows.append(
            {
                "cell": "V0_vision_history_only",
                "sample_id": sample_id,
                "scene": str(conflict["scenes"][index]),
                "fold": int(conflict["folds"][index]),
                "expected_route": "vision",
                "kino_route": ("vision" if str(prediction["route"]) == "vision" else "body"),
                "cause": str(conflict["labels"][index]),
                "physical_case_id": str(conflict["rows"][index]["case_id"]),
                "phase": "history_only_pre_external_contact",
                "support_certificate": "body packet byte-identical within T2 pair",
            }
        )
        full.append(conflict["full"][index])
        route_labels.append(1)
        route_predictions.append(int(str(prediction["route"]) == "vision"))
        folds.append(int(conflict["folds"][index]))
        scenes.append(str(conflict["scenes"][index]))

    pair_lookup: dict[int, tuple[int, str]] = {}
    for pair_number, (overload, decay, visual_source) in enumerate(body_pairs):
        source_id = str(conflict["sample_ids"][visual_source])
        pair_lookup[overload] = (pair_number, source_id)
        pair_lookup[decay] = (pair_number, source_id)
    for index in body_indices:
        sample_id = str(scale["sample_ids"][index])
        prediction = prediction_lookup[("scale", sample_id)]
        pair_number, source_id = pair_lookup[int(index)]
        rows.append(
            {
                "cell": "P0_body_history_only",
                "sample_id": sample_id,
                "scene": str(scale["scenes"][index]),
                "fold": int(scale["folds"][index]),
                "expected_route": "body",
                "kino_route": ("vision" if str(prediction["route"]) == "vision" else "body"),
                "cause": str(scale["labels"][index]),
                "physical_case_id": str(scale["rows"][index]["group_id"]),
                "matched_pair_id": f"p0_{pair_number:04d}",
                "shared_visual_source_sample_id": source_id,
                "phase": "history_only_pre_external_contact",
                "support_certificate": "same recorded RGB packet assigned within O5/O10 pair",
            }
        )
        full.append(scale["full"][index])
        route_labels.append(0)
        route_predictions.append(int(str(prediction["route"]) == "vision"))
        folds.append(int(scale["folds"][index]))
        scenes.append(str(scale["scenes"][index]))

    full_array = np.stack(full).astype(np.float32)
    route_truth = np.asarray(route_labels, dtype=np.int8)
    kino_route = np.asarray(route_predictions, dtype=np.int8)
    fold_array = np.asarray(folds, dtype=np.int8)
    scene_array = np.asarray(scenes)
    if not (
        len(rows) == 3_000
        and np.sum(route_truth == 1) == 1_500
        and np.sum(route_truth == 0) == 1_500
        and len(set(scene_array.tolist())) == 30
    ):
        raise RuntimeError("phase-matched route construction is unbalanced")

    phase_proxy = _acceleration_range(full_array)
    phase_proxy_route = _cross_validated_logistic(phase_proxy, route_truth, fold_array)
    for row, value, prediction in zip(rows, phase_proxy[:, 0], phase_proxy_route, strict=True):
        row["acceleration_range_proxy"] = float(value)
        row["phase_proxy_route"] = "vision" if int(prediction) else "body"

    # Verify that the chosen scalar really detects the original protocol stage.
    t3_candidates = np.flatnonzero(conflict_primary & ~conflict["vision_decisive"])
    t3_stage: list[int] = []
    for scene in sorted(set(conflict["scenes"].tolist())):
        candidates = t3_candidates[conflict["scenes"][t3_candidates] == scene]
        ordered = _stable_order(conflict["sample_ids"], candidates)
        if len(ordered) < 50:
            raise RuntimeError(f"{scene} has only {len(ordered)} T3 stage samples")
        t3_stage.extend(ordered[:50])
    stage_indices = np.concatenate([vision_indices, np.asarray(t3_stage, dtype=np.int64)])
    stage_truth = np.concatenate([np.zeros(1_500, dtype=np.int8), np.ones(1_500, dtype=np.int8)])
    stage_proxy = _acceleration_range(conflict["full"][stage_indices])
    stage_predictions = _cross_validated_logistic(
        stage_proxy, stage_truth, conflict["folds"][stage_indices]
    )

    # P0 body-decisive support check, evaluated without render-only replicas.
    p0_full = scale["full"][body_indices]
    p0_invariant = scale["invariant"][body_indices]
    p0_features = invariant_relative_proprio_features(p0_invariant, p0_full)
    p0_labels = scale["labels"][body_indices]
    p0_folds = scale["folds"][body_indices]
    p0_predictions = np.empty(len(p0_labels), dtype=object)
    for fold in range(5):
        train = p0_folds != fold
        test = ~train
        model = ExtraTreesClassifier(
            n_estimators=512,
            max_features="sqrt",
            min_samples_leaf=4,
            class_weight="balanced",
            random_state=SEED + 100 + fold,
            n_jobs=-1,
        )
        model.fit(p0_features[train], p0_labels[train])
        p0_predictions[test] = model.predict(p0_features[test])

    bootstrap = _scene_bootstrap(
        scene_array,
        route_truth,
        kino_route,
        phase_proxy_route,
    )
    body_cell = route_truth == 0
    p0_route_recall = {
        label: float(np.mean(kino_route[body_cell][p0_labels == label] == 0))
        for label in sorted(set(p0_labels.tolist()))
    }
    acceptance = {
        "kino_ba_at_least_0_97": bootstrap["kino"]["estimate"] >= 0.97,
        "kino_ci_lower_above_0_95": bootstrap["kino"]["ci95"][0] > 0.95,
        "gain_ci_lower_above_0_10": (bootstrap["kino_minus_phase_proxy"]["ci95"][0] > 0.10),
        "each_p0_route_recall_at_least_0_95": all(
            value >= 0.95 for value in p0_route_recall.values()
        ),
    }

    output.mkdir(parents=True, exist_ok=False)
    predictions_path = output / "physical_predictions.jsonl"
    predictions_path.write_text("".join(json.dumps(row, sort_keys=True) + "\n" for row in rows))
    report = {
        "schema_version": "kinofail.phase-matched-routing-control.v1",
        "created_utc": datetime.now(UTC).isoformat(),
        "status": "complete" if all(acceptance.values()) else "acceptance_failed",
        "analysis_role": "controlled diagnostic; not independent confirmation",
        "seed": SEED,
        "route_threshold": 0.11,
        "unit": "one primary-view physical route decision",
        "design": {
            "decision_stage": "history-only, before external terrain contact",
            "tensor_schema": "21x19 body packet; five RGB frames",
            "V0": (
                "1,500 T2 cases; body packet byte-identical within each "
                "compliant-terrain/adhesion pair"
            ),
            "P0": (
                "750 persistent internal-fault pairs; the same recorded T2 RGB "
                "packet is assigned to O5 payload and O10 effort-decay bodies"
            ),
            "views": "primary only; no render-only pseudoreplication",
            "split": "five scene-disjoint folds",
        },
        "counts": {
            "route_decisions": len(rows),
            "vision_route_decisions": int(np.sum(route_truth == 1)),
            "body_route_decisions": int(np.sum(route_truth == 0)),
            "V0_cases": len(vision_indices),
            "P0_pairs": len(body_pairs),
            "P0_samples": len(body_indices),
            "scenes": len(set(scene_array.tolist())),
            "appearance_views_per_physical_unit": 1,
        },
        "original_protocol_stage_probe": {
            "feature": "L2 range of xyz linear acceleration",
            "balanced_accuracy": _balanced_accuracy(stage_truth, stage_predictions),
            "precontact_recall": float(recall_score(stage_truth, stage_predictions, pos_label=0)),
            "postcontact_recall": float(recall_score(stage_truth, stage_predictions, pos_label=1)),
        },
        "same_stage_route_control": {
            "stage_only_balanced_accuracy": 0.5,
            "phase_proxy_feature": "L2 range of xyz linear acceleration",
            "phase_proxy": bootstrap["phase_proxy"],
            "kino": bootstrap["kino"],
            "kino_minus_phase_proxy": bootstrap["kino_minus_phase_proxy"],
            "V0_vision_route_recall": float(np.mean(kino_route[route_truth == 1] == 1)),
            "P0_body_route_recall": float(np.mean(kino_route[route_truth == 0] == 0)),
            "P0_body_route_recall_by_cause": p0_route_recall,
        },
        "support_checks": {
            "V0_body_only_ceiling": 0.5,
            "V0_reason": "body packet is byte-identical within each T2 pair",
            "P0_visual_only_ceiling": 0.5,
            "P0_reason": "the same recorded RGB packet is assigned within each pair",
            "P0_body_probe_balanced_accuracy": _balanced_accuracy(p0_labels, p0_predictions),
            "P0_body_probe_per_class_recall": {
                label: float(np.mean(p0_predictions[p0_labels == label] == label))
                for label in sorted(set(p0_labels.tolist()))
            },
        },
        "bootstrap": {
            "draws": BOOTSTRAP_DRAWS,
            "cluster": "scene",
            "interval": "percentile 95%",
        },
        "acceptance": acceptance,
        "passed": all(acceptance.values()),
        "source_sha256": {
            str(PREDICTIONS.relative_to(ROOT)): _sha256(PREDICTIONS),
            str(DEVELOPMENT_SCRIPT.relative_to(ROOT)): _sha256(DEVELOPMENT_SCRIPT),
        },
        "artifacts": {
            "predictions": str(predictions_path.relative_to(ROOT)),
            "predictions_sha256": _sha256(predictions_path),
        },
    }
    report_path = output / "report.json"
    report_path.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n")
    print(json.dumps(report, indent=2, sort_keys=True))
    return 0 if report["passed"] else 2


if __name__ == "__main__":
    raise SystemExit(main())
