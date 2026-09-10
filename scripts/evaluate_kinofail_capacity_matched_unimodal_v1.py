#!/usr/bin/env python3
"""Capacity-matched unimodal neural controls on the frozen all-191 benchmark.

Both controls use the exact Concat-MLP architecture and training protocol.  The
unavailable stream is replaced by zeros during both fitting and inference, so
parameter count, optimizer, duration selection, and ensemble size remain
matched while information access is strictly unimodal.
"""

from __future__ import annotations

import json
import sys
from collections import defaultdict
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from kino_vla.eval.known_multimodal_fusions import (  # noqa: E402
    FusionDimensions,
    build_fusion_model,
)
from scripts import evaluate_kinofail_known_fusion_baselines_v1 as base  # noqa: E402


FEATURE_ROOT = Path("/data/eureka/kinovla_outputs/kino_v4_all191_v1")
REFERENCE = ROOT / "outputs/eval/kinofail_known_fusion_baselines_all191_v1_score_once"
OUTPUT = ROOT / "outputs/eval/kinofail_capacity_matched_unimodal_v1"
METHODS = ("visual_mlp", "proprio_mlp")


def _inputs(
    method: str, visual: torch.Tensor, proprio: torch.Tensor
) -> tuple[torch.Tensor, torch.Tensor]:
    if method == "visual_mlp":
        return visual, torch.zeros_like(proprio)
    if method == "proprio_mlp":
        return torch.zeros_like(visual), proprio
    raise KeyError(method)


def _train_bundle(device: torch.device) -> tuple[dict[str, Any], dict[str, Any]]:
    development = base._load_development()
    bundle: dict[str, Any] = {"methods": list(METHODS), "datasets": {}}
    training_report: dict[str, Any] = {}
    for battery, source in development.items():
        labels = np.asarray(source["labels"]).astype(str)
        groups = np.asarray(source["groups"]).astype(str)
        classes = sorted(set(labels.tolist()))
        encoded = base._encode_labels(labels, classes)
        train_mask = np.asarray(source["folds"]) != base.VALIDATION_FOLD
        validation_mask = ~train_mask
        train_visual_mean, train_visual_scale = base._fit_normalizer(
            source["visual_fusion"], train_mask
        )
        train_proprio_mean, train_proprio_scale = base._fit_normalizer(
            source["proprio_fusion"], train_mask
        )
        validation_visual = torch.from_numpy(
            base._normalize(
                source["visual_fusion"], train_visual_mean, train_visual_scale
            )
        )
        validation_proprio = torch.from_numpy(
            base._normalize(
                source["proprio_fusion"], train_proprio_mean, train_proprio_scale
            )
        )
        targets = torch.from_numpy(encoded)
        weights = torch.from_numpy(base._training_weights(labels, groups))
        dims = FusionDimensions(
            visual=validation_visual.shape[1],
            proprio=validation_proprio.shape[1],
            classes=len(classes),
        )
        selected_epochs: dict[str, int] = {}
        duration_reports: dict[str, Any] = {}
        for method_index, method in enumerate(METHODS):
            method_visual, method_proprio = _inputs(
                method, validation_visual, validation_proprio
            )
            _, report = base._train_model(
                "concat_mlp",
                dims,
                method_visual,
                method_proprio,
                targets,
                weights,
                train_mask,
                seed=base.TRAINING_SEEDS[0] + 700 + 100 * method_index,
                epochs=base.MAX_EPOCHS,
                device=device,
                validation=validation_mask,
                validation_labels=labels[validation_mask],
                validation_groups=groups[validation_mask],
                classes=classes,
            )
            selected_epochs[method] = int(report["selected_epoch"])
            duration_reports[method] = report

        visual_mean, visual_scale = base._fit_normalizer(
            source["visual_fusion"], np.ones(len(labels), dtype=bool)
        )
        proprio_mean, proprio_scale = base._fit_normalizer(
            source["proprio_fusion"], np.ones(len(labels), dtype=bool)
        )
        visual = torch.from_numpy(
            base._normalize(source["visual_fusion"], visual_mean, visual_scale)
        )
        proprio = torch.from_numpy(
            base._normalize(source["proprio_fusion"], proprio_mean, proprio_scale)
        )
        all_rows = np.ones(len(labels), dtype=bool)
        states: dict[str, list[dict[str, torch.Tensor]]] = {}
        refits: dict[str, Any] = {}
        for method_index, method in enumerate(METHODS):
            states[method] = []
            refits[method] = []
            method_visual, method_proprio = _inputs(method, visual, proprio)
            for seed_index, seed in enumerate(base.TRAINING_SEEDS):
                effective_seed = seed + 700 + 100 * method_index
                model, report = base._train_model(
                    "concat_mlp",
                    dims,
                    method_visual,
                    method_proprio,
                    targets,
                    weights,
                    all_rows,
                    seed=effective_seed,
                    epochs=selected_epochs[method],
                    device=device,
                )
                states[method].append(
                    {key: value.detach().cpu() for key, value in model.state_dict().items()}
                )
                refits[method].append(
                    {"seed_index": seed_index, "seed": effective_seed, **report}
                )
        bundle["datasets"][battery] = {
            "classes": classes,
            "dimensions": {
                "visual": dims.visual,
                "proprio": dims.proprio,
                "classes": dims.classes,
            },
            "normalizer": {
                "visual_mean": visual_mean,
                "visual_scale": visual_scale,
                "proprio_mean": proprio_mean,
                "proprio_scale": proprio_scale,
            },
            "states": states,
        }
        training_report[battery] = {
            "samples": int(len(labels)),
            "groups": int(len(set(groups.tolist()))),
            "selected_epochs": selected_epochs,
            "duration_selection": duration_reports,
            "full_refits": refits,
        }
    return bundle, training_report


def _score(
    bundle: dict[str, Any], device: torch.device
) -> tuple[dict[str, Any], dict[str, list[dict[str, Any]]]]:
    base.CONFIRMATION = FEATURE_ROOT
    base.REFERENCE = REFERENCE
    confirmation = base._load_confirmation()
    new_rows: dict[str, list[dict[str, Any]]] = {}
    view_metrics: dict[str, Any] = {}
    for battery, source in confirmation.items():
        saved = bundle["datasets"][battery]
        classes = [str(value) for value in saved["classes"]]
        dims = FusionDimensions(**saved["dimensions"])
        normalizer = saved["normalizer"]
        visual = torch.from_numpy(
            base._normalize(
                source["visual_fusion"],
                normalizer["visual_mean"],
                normalizer["visual_scale"],
            )
        )
        proprio = torch.from_numpy(
            base._normalize(
                source["proprio_fusion"],
                normalizer["proprio_mean"],
                normalizer["proprio_scale"],
            )
        )
        probabilities: dict[str, np.ndarray] = {}
        for method in METHODS:
            method_visual, method_proprio = _inputs(method, visual, proprio)
            seed_probabilities = []
            for state in saved["states"][method]:
                model = build_fusion_model("concat_mlp", dims)
                model.load_state_dict(state)
                model.to(device)
                seed_probabilities.append(
                    base._predict(
                        model, method_visual, method_proprio, device=device
                    )
                )
            probabilities[method] = np.mean(seed_probabilities, axis=0)
        view_metrics[battery] = {
            method: base._metrics(
                source["labels"], np.asarray(classes)[probability.argmax(axis=1)]
            )
            for method, probability in probabilities.items()
        }
        new_rows[battery] = base._aggregate_confirmation(
            source, probabilities, classes
        )

    reference_rows = base._jsonl(REFERENCE / "physical_units.jsonl")
    lookup = {
        (str(row["battery"]), str(row["unit_id"])): row for row in reference_rows
    }
    combined: dict[str, list[dict[str, Any]]] = {}
    for battery, rows in new_rows.items():
        combined[battery] = []
        for row in rows:
            reference = lookup[(battery, str(row["unit_id"]))]
            combined[battery].append(
                {
                    **row,
                    "predictions": {
                        **reference["predictions"],
                        **row["predictions"],
                    },
                }
            )
    return view_metrics, combined


def main() -> int:
    if OUTPUT.exists():
        raise FileExistsError(OUTPUT)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    bundle, training_report = _train_bundle(device)
    view_metrics, combined = _score(bundle, device)
    all_methods = sorted(combined["scale"][0]["predictions"])
    physical_metrics: dict[str, dict[str, Any]] = defaultdict(dict)
    per_cell: dict[str, dict[str, Any]] = {}
    for battery, rows in combined.items():
        labels = np.asarray([str(row["truth"]) for row in rows])
        for method in all_methods:
            predicted = np.asarray(
                [str(row["predictions"][method]) for row in rows]
            )
            physical_metrics[battery][method] = base._metrics(labels, predicted)
    for cell in ("T2_vision_decisive", "T3_proprio_decisive"):
        rows = [row for row in combined["conflict"] if row["cell"] == cell]
        labels = np.asarray([str(row["truth"]) for row in rows])
        per_cell[cell] = {
            method: base._metrics(
                labels,
                np.asarray([str(row["predictions"][method]) for row in rows]),
            )
            for method in all_methods
        }
    cross = {}
    for method in all_methods:
        scale = float(physical_metrics["scale"][method]["balanced_accuracy"])
        conflict = float(
            physical_metrics["conflict"][method]["balanced_accuracy"]
        )
        cross[method] = {
            "scale": scale,
            "t2": float(
                per_cell["T2_vision_decisive"][method]["balanced_accuracy"]
            ),
            "t3": float(
                per_cell["T3_proprio_decisive"][method]["balanced_accuracy"]
            ),
            "conflict": conflict,
            "macro": 0.5 * (scale + conflict),
            "worst": min(scale, conflict),
        }
    bootstrap = base._bootstrap(combined, all_methods)
    OUTPUT.mkdir(parents=True, exist_ok=False)
    torch.save(bundle, OUTPUT / "capacity_matched_unimodal.pt")
    (OUTPUT / "physical_units.jsonl").write_text(
        "".join(
            json.dumps({"battery": battery, **row}, sort_keys=True) + "\n"
            for battery, rows in combined.items()
            for row in rows
        ),
        encoding="utf-8",
    )
    report = {
        "schema_version": "kinofail.capacity-matched-unimodal.v1",
        "created_utc": datetime.now(UTC).isoformat(),
        "status": "complete",
        "evaluation_opened_only_after_training_completed": True,
        "information_control": {
            "visual_mlp": "Concat-MLP with proprioceptive tensor replaced by zeros",
            "proprio_mlp": "Concat-MLP with visual tensor replaced by zeros",
        },
        "shared_protocol": {
            "architecture": "exact Concat-MLP parameterization",
            "validation_fold": base.VALIDATION_FOLD,
            "max_epochs": base.MAX_EPOCHS,
            "patience": base.PATIENCE,
            "optimizer": "AdamW",
            "learning_rate": base.LEARNING_RATE,
            "weight_decay": base.WEIGHT_DECAY,
            "batch_size": base.BATCH_SIZE,
            "ensemble_seeds": list(base.TRAINING_SEEDS),
        },
        "training_report": training_report,
        "view_level_secondary_metrics": view_metrics,
        "physical_unit_primary_metrics": physical_metrics,
        "per_conflict_cell_physical_unit": per_cell,
        "cross_battery_physical_unit": cross,
        "physical_unit_crossed_cluster_bootstrap": bootstrap,
    }
    (OUTPUT / "report.json").write_text(
        json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    print(json.dumps({method: cross[method] for method in METHODS}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
