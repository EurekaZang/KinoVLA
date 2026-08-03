#!/usr/bin/env python3
"""Evaluate the C2 v4 class-proposal modality router."""

from __future__ import annotations

import argparse
import json
import pickle
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import numpy as np

from eval_kinofail_realistic_c2_bidirectional_v2 import (
    BASELINES,
    T2,
    T3,
    _aligned,
    _bootstrap_delta,
    _json,
    _linear,
    _load_development,
    _load_test,
    _metric,
    _sha,
    _weights,
)
from eval_kinofail_realistic_c2_bidirectional_v3 import (
    _fit_baseline,
    _one_hot,
)


OURS = "structured_class_proposal_router_v4"
METHODS = (*BASELINES, OURS)
T2_CLASSES = ("adhesion", "compliant_terrain")
T3_CLASSES = ("invisible_obstacle", "low_friction")


class StructuredClassProposalRouterV4:
    """Use proprio class proposals for T3 and visual evidence inside T2."""

    def __init__(
        self,
        proprio_proposal: Any,
        t2_visual: Any,
        classes: list[str],
    ) -> None:
        self.proprio_proposal = proprio_proposal
        self.t2_visual = t2_visual
        self.classes = classes

    def family_prediction(
        self,
        visual: np.ndarray,
        geometry: np.ndarray,
        proprio: np.ndarray,
    ) -> np.ndarray:
        del visual, geometry
        probability = _aligned(
            self.proprio_proposal, proprio, self.classes
        )
        proposal = np.asarray(self.classes)[
            probability.argmax(axis=1)
        ]
        return np.where(np.isin(proposal, T3_CLASSES), T3, T2)

    def predict_proba(
        self,
        visual: np.ndarray,
        geometry: np.ndarray,
        proprio: np.ndarray,
    ) -> np.ndarray:
        proposal_probability = _aligned(
            self.proprio_proposal, proprio, self.classes
        )
        proposal = np.asarray(self.classes)[
            proposal_probability.argmax(axis=1)
        ]
        t2_probability = _aligned(
            self.t2_visual,
            np.concatenate([visual, geometry], axis=1),
            list(T2_CLASSES),
        )
        output = np.zeros(
            (len(visual), len(self.classes)), dtype=np.float64
        )
        target = {
            label: index for index, label in enumerate(self.classes)
        }
        for index, label in enumerate(proposal):
            if label in T3_CLASSES:
                values = np.asarray(
                    [
                        proposal_probability[index, target[value]]
                        for value in T3_CLASSES
                    ]
                )
                values = values / max(float(values.sum()), 1.0e-12)
                for value, probability in zip(
                    T3_CLASSES, values, strict=True
                ):
                    output[index, target[value]] = probability
            else:
                for position, value in enumerate(T2_CLASSES):
                    output[index, target[value]] = t2_probability[
                        index, position
                    ]
        return output


def _fit_ours(
    visual: np.ndarray,
    geometry: np.ndarray,
    proprio: np.ndarray,
    labels: np.ndarray,
    cells: np.ndarray,
    case_ids: np.ndarray,
) -> StructuredClassProposalRouterV4:
    classes = sorted(set(labels.astype(str)))
    proposal = _linear(c_value=1.0)
    proposal.fit(
        proprio,
        labels,
        classifier__sample_weight=_weights(case_ids),
    )
    t2 = cells == T2
    t2_visual = _linear(c_value=0.1)
    t2_visual.fit(
        np.concatenate([visual[t2], geometry[t2]], axis=1),
        labels[t2],
        classifier__sample_weight=_weights(case_ids[t2]),
    )
    return StructuredClassProposalRouterV4(
        proposal, t2_visual, classes
    )


def _fit(
    method: str,
    visual: np.ndarray,
    geometry: np.ndarray,
    proprio: np.ndarray,
    labels: np.ndarray,
    cells: np.ndarray,
    case_ids: np.ndarray,
) -> Any:
    if method == OURS:
        return _fit_ours(
            visual,
            geometry,
            proprio,
            labels,
            cells,
            case_ids,
        )
    return _fit_baseline(
        method,
        visual,
        geometry,
        proprio,
        labels,
        case_ids,
    )


def _development(directory: Path, output: Path) -> int:
    rows, visual, geometry, proprio, manifest_path = (
        _load_development(directory)
    )
    labels = np.asarray(
        [str(row["attribution_category"]) for row in rows]
    )
    cells = np.asarray([str(row["cell"]) for row in rows])
    cases = np.asarray([str(row["case_id"]) for row in rows])
    scenes = np.asarray(
        [str(row["scene_cluster"]) for row in rows]
    )
    classes = sorted(set(labels.astype(str)))
    predictions = {
        method: np.empty(len(rows), dtype=object)
        for method in METHODS
    }
    family = np.empty(len(rows), dtype=object)
    for scene in sorted(set(scenes)):
        train = scenes != scene
        evaluate = ~train
        for method in METHODS:
            model = _fit(
                method,
                visual[train],
                geometry[train],
                proprio[train],
                labels[train],
                cells[train],
                cases[train],
            )
            probability = model.predict_proba(
                visual[evaluate],
                geometry[evaluate],
                proprio[evaluate],
            )
            predictions[method][evaluate] = np.asarray(
                model.classes
            )[probability.argmax(axis=1)]
            if method == OURS:
                family[evaluate] = model.family_prediction(
                    visual[evaluate],
                    geometry[evaluate],
                    proprio[evaluate],
                )
    results = {
        method: _metric(
            rows,
            labels,
            _one_hot(predictions[method], classes),
            classes,
        )[0]
        for method in METHODS
    }
    best_baseline = max(
        BASELINES,
        key=lambda method: results[method]["balanced_accuracy"],
    )
    report = {
        "schema_version": (
            "kinofail.realistic-c2-v4-development-screen.v1"
        ),
        "created_utc": datetime.now(UTC).isoformat(),
        "status": "development_complete_v4_outcomes_unavailable",
        "v3_failure_used_as_development": True,
        "v4_confirmation_outcomes_used": False,
        "scene_disjoint_loso": True,
        "scene_clusters": sorted(set(scenes)),
        "results": results,
        "family_gate_accuracy": float(np.mean(family == cells)),
        "best_baseline_method": best_baseline,
        "selected_method": OURS,
        "selection": {
            "proprio_proposal": (
                "balanced four-class logistic(C=1) on the frozen "
                "130-dimensional scene-invariant proprio summary"
            ),
            "T2_specialist": (
                "balanced logistic(C=0.1) on CLIP+generic HOG"
            ),
            "composition": (
                "accept a proprio proposal only for a T3 class; otherwise "
                "invoke the T2 visual specialist"
            ),
            "deployment_forbidden": [
                "operator ID",
                "scene/domain ID",
                "material/severity metadata",
                "truth label",
                "test outcome",
            ],
        },
        "development_advantage": {
            "balanced_accuracy_delta_vs_best_baseline": (
                results[OURS]["balanced_accuracy"]
                - results[best_baseline]["balanced_accuracy"]
            ),
            "worst_scene_delta_vs_best_baseline": (
                results[OURS]["worst_scene_accuracy"]
                - results[best_baseline]["worst_scene_accuracy"]
            ),
        },
        "feature_manifest_sha256": _sha(manifest_path),
        "evaluator_sha256": _sha(Path(__file__).resolve()),
    }
    output.mkdir(parents=True, exist_ok=False)
    path = output / "development_report.json"
    path.write_text(
        json.dumps(report, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    print(json.dumps(report, indent=2, sort_keys=True))
    return 0


def _formal(
    development_directory: Path,
    test_directory: Path,
    geometry_directory: Path,
    protocol_path: Path,
    output: Path,
) -> int:
    protocol = _json(protocol_path)
    (
        development_rows,
        development_visual,
        development_geometry,
        development_proprio,
        development_manifest_path,
    ) = _load_development(development_directory)
    (
        test_rows,
        test_visual,
        test_geometry,
        test_proprio,
        test_manifest_path,
        geometry_manifest_path,
    ) = _load_test(test_directory, geometry_directory)
    for key, actual in (
        ("evaluator_sha256", _sha(Path(__file__).resolve())),
        (
            "development_feature_manifest_sha256",
            _sha(development_manifest_path),
        ),
    ):
        if protocol.get(key) != actual:
            raise RuntimeError(f"C2 v4 protocol {key} mismatch")
    development_labels = np.asarray(
        [
            str(row["attribution_category"])
            for row in development_rows
        ]
    )
    development_cells = np.asarray(
        [str(row["cell"]) for row in development_rows]
    )
    development_cases = np.asarray(
        [str(row["case_id"]) for row in development_rows]
    )
    test_labels = np.asarray(
        [str(row["attribution_category"]) for row in test_rows]
    )
    results = {}
    predictions = {}
    models = {}
    for method in METHODS:
        model = _fit(
            method,
            development_visual,
            development_geometry,
            development_proprio,
            development_labels,
            development_cells,
            development_cases,
        )
        probability = model.predict_proba(
            test_visual, test_geometry, test_proprio
        )
        metric, prediction = _metric(
            test_rows, test_labels, probability, model.classes
        )
        results[method] = metric
        predictions[method] = prediction
        models[method] = model
    best_baseline = max(
        BASELINES,
        key=lambda method: results[method]["balanced_accuracy"],
    )
    delta = _bootstrap_delta(
        test_rows,
        test_labels,
        predictions[OURS],
        predictions[best_baseline],
        draws=int(protocol["bootstrap"]["draws"]),
        seed=int(protocol["bootstrap"]["seed"]),
    )
    ours = results[OURS]
    acceptance = protocol["acceptance"]
    development_scenes = {
        str(row["scene_cluster"]) for row in development_rows
    }
    test_scenes = {
        str(row["scene_cluster"]) for row in test_rows
    }
    checks = {
        "minimum_balanced_accuracy": (
            ours["balanced_accuracy"]
            >= float(acceptance["minimum_balanced_accuracy"])
        ),
        "minimum_worst_direction_accuracy": (
            min(ours["per_cell_accuracy"].values())
            >= float(acceptance["minimum_worst_direction_accuracy"])
        ),
        "minimum_worst_scene_accuracy": (
            ours["worst_scene_accuracy"]
            >= float(acceptance["minimum_worst_scene_accuracy"])
        ),
        "minimum_texture_swap_hard_consistency": (
            ours["texture_swap_hard_consistency"]
            >= float(
                acceptance[
                    "minimum_texture_swap_hard_consistency"
                ]
            )
        ),
        "beats_every_baseline_point": all(
            ours["balanced_accuracy"]
            > results[method]["balanced_accuracy"]
            for method in BASELINES
        ),
        "matched_delta_noninferiority_ci": (
            delta["ci95"][0]
            >= float(acceptance["minimum_delta_ci95_lower"])
        ),
        "proprio_insufficient_on_t2": (
            results["proprio_only"]["per_cell_accuracy"][T2]
            <= float(
                acceptance[
                    "maximum_wrong_modality_direction_accuracy"
                ]
            )
        ),
        "vision_insufficient_on_t3": (
            results["vision_only"]["per_cell_accuracy"][T3]
            <= float(
                acceptance[
                    "maximum_wrong_modality_direction_accuracy"
                ]
            )
        ),
        "fresh_three_scene_three_domain_confirmation": (
            len(test_scenes) == 3
            and len({str(row["domain"]) for row in test_rows}) == 3
            and test_scenes.isdisjoint(development_scenes)
        ),
    }
    output.mkdir(parents=True, exist_ok=False)
    prediction_path = output / "predictions.jsonl"
    prediction_path.write_text(
        "".join(
            json.dumps(
                {
                    "sample_id": row["sample_id"],
                    "case_id": row["case_id"],
                    "scene_cluster": row["scene_cluster"],
                    "domain": row["domain"],
                    "cell": row["cell"],
                    "truth": test_labels[index],
                    **{
                        f"{method}_prediction": str(
                            predictions[method][index]
                        )
                        for method in METHODS
                    },
                },
                sort_keys=True,
            )
            + "\n"
            for index, row in enumerate(test_rows)
        ),
        encoding="utf-8",
    )
    checkpoint_hashes = {}
    for method, model in models.items():
        path = output / f"{method}.pkl"
        with path.open("wb") as handle:
            pickle.dump(
                model, handle, protocol=pickle.HIGHEST_PROTOCOL
            )
        checkpoint_hashes[method] = _sha(path)
    report = {
        "schema_version": (
            "kinofail.realistic-c2-formal-confirmation.v4"
        ),
        "created_utc": datetime.now(UTC).isoformat(),
        "protocol_id": protocol["protocol_id"],
        "protocol_sha256": _sha(protocol_path),
        "passed": all(checks.values()),
        "checks": checks,
        "test_counts": {
            "samples": len(test_rows),
            "matched_cases": len(
                {str(row["case_id"]) for row in test_rows}
            ),
            "cases_per_direction": {
                cell: len(
                    {
                        str(row["case_id"])
                        for row in test_rows
                        if row["cell"] == cell
                    }
                )
                for cell in (T2, T3)
            },
            "scene_clusters": len(test_scenes),
            "domains": len(
                {str(row["domain"]) for row in test_rows}
            ),
            "classes": len(set(test_labels.astype(str))),
        },
        "results": results,
        "best_baseline_on_formal_battery": best_baseline,
        "ours_minus_best_baseline": delta,
        "artifacts": {
            "predictions_sha256": _sha(prediction_path),
            "checkpoint_sha256": checkpoint_hashes,
            "test_feature_manifest_sha256": _sha(
                test_manifest_path
            ),
            "test_geometry_manifest_sha256": _sha(
                geometry_manifest_path
            ),
        },
        "claim_boundary": protocol["claim_boundary"],
    }
    report_path = output / "report.json"
    report_path.write_text(
        json.dumps(report, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    print(
        json.dumps(
            {
                "passed": report["passed"],
                "checks": checks,
                "results": results,
                "best_baseline": best_baseline,
                "delta": delta,
                "report": str(report_path),
            },
            indent=2,
            sort_keys=True,
        )
    )
    return 0 if report["passed"] else 2


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--mode", choices=("development", "formal"), required=True
    )
    parser.add_argument(
        "--development-features", type=Path, required=True
    )
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--test-features", type=Path)
    parser.add_argument("--test-geometry", type=Path)
    parser.add_argument("--protocol", type=Path)
    args = parser.parse_args()
    if args.mode == "development":
        return _development(
            args.development_features.resolve(),
            args.output.resolve(),
        )
    if (
        args.test_features is None
        or args.test_geometry is None
        or args.protocol is None
    ):
        raise ValueError("formal mode requires test features/geometry/protocol")
    return _formal(
        args.development_features.resolve(),
        args.test_features.resolve(),
        args.test_geometry.resolve(),
        args.protocol.resolve(),
        args.output.resolve(),
    )


if __name__ == "__main__":
    raise SystemExit(main())
