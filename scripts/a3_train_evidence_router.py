#!/usr/bin/env python
"""Train the five-seed A3.4 observable evidence router on appearance-train rows only."""

from __future__ import annotations

import argparse
import hashlib
import json
import time
from dataclasses import dataclass
from typing import Any

import numpy as np
import torch
from torch import nn

from kino_vla.eval.a7_ablation import artifact_meta, load_yaml, repo_path, write_json
from kino_vla.eval.appearance_library import load_appearance_library
from kino_vla.eval.evidence_routed import (
    EXPERT_NAMES,
    EvidenceRoutedNetwork,
    predict_category,
    regime_target,
)
from kino_vla.eval.material_support import (
    MATERIAL_CLASSES,
    dominant_chromatic_rgb,
    fit_material_prototypes,
    matched_material_pair_logits,
    material_evidence_features,
)
from kino_vla.map.clip_appearance import ClipAppearanceEncoder
from kino_vla.utils.config import load_config

CELLS = ("T1", "T2", "T3", "T4", "T5")


@dataclass(frozen=True)
class Row:
    sid: str
    cell: str
    t3_sub: str
    truth: str
    appearance_id: str
    appearance_split: str
    visual: np.ndarray
    proprio: np.ndarray
    material_logits: np.ndarray
    conflict_logits: np.ndarray
    theta: np.ndarray
    synthetic: bool = False


def _t3_sub(rec: dict[str, Any]) -> str:
    if rec.get("a3_direction") in {"looks_safe", "reverse"}:
        return str(rec["a3_direction"])
    if rec["snapshot"]["operator_name"] == "O8_invisible_collider":
        return "O8"
    return "other"


def _source_records(cfg: dict[str, Any]) -> tuple[list[dict[str, Any]], dict[str, np.ndarray]]:
    records: list[dict[str, Any]] = []
    arrays: dict[str, np.ndarray] = {}
    sources = [cfg["sources"]["a0_corpus"], cfg["sources"]["a3_t3_corpus"]]
    if cfg["sources"].get("successor_reverse_train"):
        sources.append(cfg["sources"]["successor_reverse_train"])
    for source in sources:
        root = repo_path(source)
        records.extend(
            json.loads(line)
            for line in (root / "samples.jsonl").read_text().splitlines()
            if line
        )
        npz = np.load(root / "frames.npz")
        arrays.update({key: np.asarray(npz[key]) for key in npz.files})
    return records, arrays


def _conflict_predictions(cfg: dict[str, Any]) -> dict[str, str]:
    sources = [cfg["sources"]["conflict_predictions"]]
    if cfg["sources"].get("successor_reverse_conflict_predictions"):
        sources.append(cfg["sources"]["successor_reverse_conflict_predictions"])
    predictions = {}
    for source in sources:
        rows = json.loads(repo_path(source).read_text())
        predictions.update(
            {str(row["sid"]): str(row.get("attribution") or "") for row in rows}
        )
    return predictions


def build_feature_cache(cfg: dict[str, Any], *, force: bool = False) -> dict[str, Any]:
    """Extract CLIP + train-prototype features without fitting on held-out appearances."""
    cache_path = repo_path(cfg["feature_cache"])
    meta_path = cache_path.with_suffix(".json")
    if cache_path.exists() and meta_path.exists() and not force:
        return json.loads(meta_path.read_text())
    records, arrays = _source_records(cfg)
    encoder = ClipAppearanceEncoder()
    rgbs = [np.asarray(arrays[f"{rec['sample_id']}__rgb"][-1], dtype=np.float32) for rec in records]
    chunks = []
    for start in range(0, len(rgbs), 64):
        chunks.append(encoder.embed_batch(rgbs[start : start + 64]).astype(np.float32))
    clip_visual = np.concatenate(chunks, axis=0)
    material_rgb = np.stack([dominant_chromatic_rgb(rgb) for rgb in rgbs]).astype(np.float32)
    appearance_library = load_appearance_library(
        str(cfg["protocol"].get("material_appearance_config", "eval/appearance_library.yaml")),
        register=False,
    )
    calibration = []
    calibration_sids = []
    for rec, feature in zip(records, material_rgb, strict=True):
        appearance_id = str(rec.get("appearance_id", ""))
        if str(rec.get("appearance_split", "")) != str(cfg["protocol"]["fit_appearance_split"]):
            continue
        if appearance_id not in appearance_library._appearances:
            continue
        semantic_class = appearance_library.semantic_class(appearance_id)
        if semantic_class not in MATERIAL_CLASSES:
            continue
        calibration.append(
            {
                "sample_id": str(rec["sample_id"]),
                "appearance_id": appearance_id,
                "semantic_class": semantic_class,
                "material_rgb": feature,
            }
        )
        calibration_sids.append(str(rec["sample_id"]))
    material_models = {
        name: fit_material_prototypes(calibration, target_class=name)
        for name in MATERIAL_CLASSES
    }
    material_visual = np.stack(
        [material_evidence_features(feature, material_models) for feature in material_rgb]
    )
    visual = np.concatenate([clip_visual, material_visual], axis=1).astype(np.float32)
    cache_path.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        cache_path,
        sid=np.asarray([rec["sample_id"] for rec in records]),
        visual=visual,
        material_rgb=material_rgb,
    )
    payload = {
        "n": len(records),
        "visual_dim": int(visual.shape[1]),
        "clip_visual_dim": int(clip_visual.shape[1]),
        "material_visual_dim": int(material_visual.shape[1]),
        "encoder": encoder.model_id,
        "source_record_ids_sha256": hashlib.sha256(
            "\n".join(str(rec["sample_id"]) for rec in records).encode()
        ).hexdigest(),
        "label_free_extraction": True,
        "material_prototype_fit_split": str(cfg["protocol"]["fit_appearance_split"]),
        "material_prototype_classes": list(MATERIAL_CLASSES),
        "material_models": material_models,
        "material_calibration_ids_sha256": hashlib.sha256(
            "\n".join(sorted(calibration_sids)).encode()
        ).hexdigest(),
        "material_calibration_n": len(calibration_sids),
        "heldout_appearance_labels_used_for_fit": False,
    }
    write_json(meta_path, payload)
    return payload


def load_rows(cfg: dict[str, Any]) -> list[Row]:
    records, arrays = _source_records(cfg)
    conflict = _conflict_predictions(cfg)
    cache = np.load(repo_path(cfg["feature_cache"]))
    feature_meta = json.loads(repo_path(cfg["feature_cache"]).with_suffix(".json").read_text())
    visual_by_sid = {
        str(sid): np.asarray(feature, dtype=np.float32)
        for sid, feature in zip(cache["sid"], cache["visual"], strict=True)
    }
    material_rgb_by_sid = {
        str(sid): np.asarray(feature, dtype=np.float32)
        for sid, feature in zip(cache["sid"], cache["material_rgb"], strict=True)
    }
    categories = sorted(
        {
            "nominal" if rec.get("taxonomy_cell") == "T5" else rec["ground_truth"]["category"]
            for rec in records
        }
    )
    scale = float(cfg["model"]["conflict_logit_scale"])
    material_scale = float(cfg["model"]["material_pair_logit_scale"])
    rows = []
    for rec in records:
        sid = str(rec["sample_id"])
        cell = str(rec["taxonomy_cell"])
        truth = "nominal" if cell == "T5" else str(rec["ground_truth"]["category"])
        logits = np.zeros(len(categories), dtype=np.float32)
        pred = conflict.get(sid, "")
        if pred in categories:
            logits[categories.index(pred)] = scale
        rows.append(
            Row(
                sid=sid,
                cell=cell,
                t3_sub=_t3_sub(rec),
                truth=truth,
                appearance_id=str(rec.get("appearance_id", "")),
                appearance_split=str(rec.get("appearance_split", "")),
                visual=visual_by_sid[sid],
                proprio=np.asarray(arrays[f"{sid}__proprio"], dtype=np.float32),
                material_logits=matched_material_pair_logits(
                    material_rgb_by_sid[sid],
                    feature_meta["material_models"],
                    categories,
                    scale=material_scale,
                ),
                conflict_logits=logits,
                theta=np.asarray(rec["target_theta"], dtype=np.float32),
            )
        )
    return rows


def add_counterfactual_nominal(
    rows: list[Row], *, categories: list[str], ratio: float, scale: float
) -> list[Row]:
    """Train-only hazard-looking RGB paired with train-only nominal proprioception."""
    fit = [row for row in rows if row.appearance_split == "train"]
    nominal = [row for row in fit if row.truth == "nominal"]
    alarms = [row for row in fit if row.truth == "adhesion"]
    if not nominal or not alarms or ratio <= 0.0:
        return list(rows)
    count = int(round(len(nominal) * float(ratio)))
    synthetic = []
    for i in range(count):
        body = nominal[i % len(nominal)]
        alarm = alarms[i % len(alarms)]
        conflict = np.zeros(len(categories), dtype=np.float32)
        conflict[categories.index("adhesion")] = float(scale)
        synthetic.append(
            Row(
                sid=f"cf_reverse_{i:04d}_{alarm.sid}_{body.sid}",
                cell="T3",
                t3_sub="reverse",
                truth="nominal",
                appearance_id=f"cf_{alarm.appearance_id}",
                appearance_split="train",
                visual=alarm.visual,
                proprio=body.proprio,
                material_logits=alarm.material_logits,
                conflict_logits=conflict,
                theta=body.theta,
                synthetic=True,
            )
        )
    return [*rows, *synthetic]


def _development_split(
    rows: list[Row], *, seed: int, fraction: float
) -> tuple[list[Row], list[Row]]:
    fit = [row for row in rows if row.appearance_split == "train"]
    train, dev = [], []
    for row in fit:
        digest = hashlib.sha256(f"{seed}|{row.sid}".encode()).digest()
        value = int.from_bytes(digest[:8], "big") / 2**64
        (dev if value < float(fraction) else train).append(row)
    for cell in CELLS:
        if not any(row.cell == cell for row in dev):
            candidate = next(row for row in train if row.cell == cell)
            train.remove(candidate)
            dev.append(candidate)
    return train, dev


def _arrays(
    rows: list[Row], categories: list[str], mean: np.ndarray, std: np.ndarray, device: torch.device
) -> dict[str, Any]:
    return {
        "visual": torch.tensor(np.stack([row.visual for row in rows]), device=device),
        "proprio": torch.tensor(
            (np.stack([row.proprio for row in rows]) - mean) / std, device=device
        ),
        "material": torch.tensor(np.stack([row.material_logits for row in rows]), device=device),
        "conflict": torch.tensor(np.stack([row.conflict_logits for row in rows]), device=device),
        "target": torch.tensor(
            [categories.index(row.truth) for row in rows], dtype=torch.long, device=device
        ),
        "intervene": torch.tensor(
            [row.truth != "nominal" for row in rows], dtype=torch.float32, device=device
        ),
        "regime": torch.tensor(
            [regime_target(row.cell, row.t3_sub, row.truth) for row in rows],
            dtype=torch.long,
            device=device,
        ),
        "theta": torch.tensor(np.stack([row.theta for row in rows]), device=device),
        "cells": [row.cell for row in rows],
    }


def _balanced_weights(values: list[str], device: torch.device) -> torch.Tensor:
    counts = {value: values.count(value) for value in set(values)}
    return torch.tensor([1.0 / counts[value] for value in values], device=device)


def _loss(
    out: dict[str, torch.Tensor], data: dict[str, Any], weights: dict[str, float]
) -> tuple[torch.Tensor, dict[str, float]]:
    sample_weight = _balanced_weights(data["cells"], data["target"].device)

    def weighted_ce(logits: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        per = nn.functional.cross_entropy(logits, target, reduction="none")
        return (per * sample_weight).sum() / sample_weight.sum()

    category = weighted_ce(out["category_logits"], data["target"])
    intervention_per = nn.functional.binary_cross_entropy_with_logits(
        out["intervention_logit"], data["intervene"], reduction="none"
    )
    intervention = (intervention_per * sample_weight).sum() / sample_weight.sum()
    router = weighted_ce(out["router_logits"], data["regime"])
    visual_mask = torch.tensor(
        [cell in {"T1", "T2"} for cell in data["cells"]],
        dtype=torch.bool,
        device=data["target"].device,
    )
    proprio_mask = torch.tensor(
        [cell in {"T3", "T4", "T5"} for cell in data["cells"]],
        dtype=torch.bool,
        device=data["target"].device,
    )
    visual = nn.functional.cross_entropy(
        out["visual_logits"][visual_mask], data["target"][visual_mask]
    )
    proprio = nn.functional.cross_entropy(
        out["proprio_logits"][proprio_mask], data["target"][proprio_mask]
    )
    joint = weighted_ce(out["joint_logits"], data["target"])
    theta = nn.functional.mse_loss(out["theta"], data["theta"])
    terms = {
        "category": category,
        "intervention": intervention,
        "router": router,
        "visual_expert": visual,
        "proprio_expert": proprio,
        "joint_expert": joint,
        "theta": theta,
    }
    total = sum(float(weights[name]) * value for name, value in terms.items())
    return total, {name: float(value.detach().cpu()) for name, value in terms.items()}


@torch.no_grad()
def evaluate(
    model: nn.Module,
    rows: list[Row],
    data: dict[str, Any],
    categories: list[str],
    thresholds: list[float],
) -> tuple[dict[str, Any], float, list[dict[str, Any]]]:
    model.eval()
    out = model(data["visual"], data["proprio"], data["material"], data["conflict"])
    best = None
    best_rows: list[dict[str, Any]] = []
    for threshold in thresholds:
        pred, p_intervene, router = predict_category(
            out, categories, continue_threshold=float(threshold)
        )
        scored = [
            {
                "sid": row.sid,
                "cell": row.cell,
                "t3_sub": row.t3_sub,
                "truth": row.truth,
                "appearance_id": row.appearance_id,
                "appearance_split": row.appearance_split,
                "attribution": guess,
                "attr_ok": guess == row.truth,
                "p_intervene": float(probability),
                "router_weights": {
                    name: float(value) for name, value in zip(EXPERT_NAMES, weights, strict=True)
                },
                "synthetic": row.synthetic,
            }
            for row, guess, probability, weights in zip(
                rows, pred, p_intervene, router, strict=True
            )
        ]
        per_cell = {
            cell: float(np.mean([item["attr_ok"] for item in scored if item["cell"] == cell]))
            for cell in CELLS
            if any(item["cell"] == cell for item in scored)
        }
        objective = (min(per_cell.values()), float(np.mean(list(per_cell.values()))))
        if best is None or objective > best[0]:
            best = (objective, float(threshold), per_cell)
            best_rows = scored
    assert best is not None
    metrics = {
        "per_cell": {cell: round(value, 6) for cell, value in best[2].items()},
        "worst_cell": round(best[0][0], 6),
        "macro": round(best[0][1], 6),
    }
    return metrics, best[1], best_rows


def train_seed(cfg: dict[str, Any], all_rows: list[Row], seed: int) -> dict[str, Any]:
    torch.manual_seed(seed)
    np.random.seed(seed)
    categories = sorted({row.truth for row in all_rows})
    feature_meta = json.loads(repo_path(cfg["feature_cache"]).with_suffix(".json").read_text())
    train_rows, dev_rows = _development_split(
        all_rows,
        seed=seed,
        fraction=float(cfg["protocol"]["development_fraction"]),
    )
    flat = np.concatenate([row.proprio for row in train_rows], axis=0)
    mean = flat.mean(axis=0).astype(np.float32)
    std = np.maximum(
        flat.std(axis=0).astype(np.float32), float(cfg["model"]["proprio_std_eps"])
    )
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    train = _arrays(train_rows, categories, mean, std, device)
    dev = _arrays(dev_rows, categories, mean, std, device)
    model = EvidenceRoutedNetwork.build(
        visual_dim=train_rows[0].visual.shape[0],
        proprio_dim=train_rows[0].proprio.shape[1],
        n_categories=len(categories),
        hidden=int(cfg["model"]["hidden"]),
        material_feature_dim=int(feature_meta["material_visual_dim"]),
    ).to(device)
    training = cfg["training"]
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=float(training["lr"]),
        weight_decay=float(training["weight_decay"]),
    )
    thresholds = [float(x) for x in cfg["selection"]["continue_thresholds"]]
    best_key = (-1.0, -1.0, float("-inf"))
    best_state = None
    best_threshold = 0.5
    stale = 0
    history = []
    started = time.perf_counter()
    for epoch in range(int(training["epochs"])):
        model.train()
        out = model(train["visual"], train["proprio"], train["material"], train["conflict"])
        loss, parts = _loss(out, train, training["loss"])
        optimizer.zero_grad()
        loss.backward()
        nn.utils.clip_grad_norm_(model.parameters(), float(training["grad_clip"]))
        optimizer.step()
        if epoch % 5 == 0 or epoch + 1 == int(training["epochs"]):
            # Accuracy saturates early on the small development split.  Use the preregistered
            # accuracy objective first, then structured dev loss to avoid saving the first
            # under-converged checkpoint that merely happens to reach 100%.
            model.train()
            dev_out = model(dev["visual"], dev["proprio"], dev["material"], dev["conflict"])
            dev_loss, _ = _loss(dev_out, dev, training["loss"])
            dev_metrics, threshold, _ = evaluate(model, dev_rows, dev, categories, thresholds)
            key = (
                float(dev_metrics["worst_cell"]),
                float(dev_metrics["macro"]),
                -float(dev_loss.detach().cpu()),
            )
            history.append(
                {
                    "epoch": epoch,
                    "loss": round(float(loss.detach().cpu()), 6),
                    "dev": dev_metrics,
                    "continue_threshold": threshold,
                    "dev_structured_loss": round(float(dev_loss.detach().cpu()), 6),
                    "parts": {name: round(value, 6) for name, value in parts.items()},
                }
            )
            if key > best_key:
                best_key = key
                best_state = {
                    name: value.detach().cpu().clone()
                    for name, value in model.state_dict().items()
                }
                best_threshold = threshold
                stale = 0
            else:
                stale += 5
            if stale >= int(training["patience"]):
                break
    assert best_state is not None
    model.load_state_dict(best_state)
    split_rows = {
        "development": dev_rows,
        "appearance_test": [row for row in all_rows if row.appearance_split == "test"],
        "legacy_full": [row for row in all_rows if not row.synthetic],
    }
    metrics = {}
    per_item = {}
    for name, rows in split_rows.items():
        data = _arrays(rows, categories, mean, std, device)
        result, _, scored = evaluate(model, rows, data, categories, [best_threshold])
        metrics[name] = result
        per_item[name] = scored
    seed_dir = repo_path(cfg["output_dir"]) / f"seed{seed}"
    seed_dir.mkdir(parents=True, exist_ok=True)
    prompt_cfg = load_config("data/hindsight.yaml")
    blob = {
        "state_dict": best_state,
        "categories": categories,
        "canonical": prompt_cfg.recovery.canonical.to_dict(),
        "visual_dim": int(train_rows[0].visual.shape[0]),
        "proprio_dim": int(train_rows[0].proprio.shape[1]),
        "hidden": int(cfg["model"]["hidden"]),
        "proprio_mean": mean,
        "proprio_std": std,
        "continue_threshold": best_threshold,
        "conflict_logit_scale": float(cfg["model"]["conflict_logit_scale"]),
        "material_models": feature_meta["material_models"],
        "material_classes": feature_meta["material_prototype_classes"],
        "clip_visual_dim": int(feature_meta["clip_visual_dim"]),
        "material_feature_dim": int(feature_meta["material_visual_dim"]),
        "material_pair_logit_scale": float(cfg["model"]["material_pair_logit_scale"]),
    }
    torch.save(blob, seed_dir / "router.pt")
    for name, rows in per_item.items():
        write_json(seed_dir / f"per_item_{name}.json", rows)
    result = {
        "seed": seed,
        "device": str(device),
        "n_train": len(train_rows),
        "n_dev": len(dev_rows),
        "continue_threshold": best_threshold,
        "best_dev_key": list(best_key),
        "metrics": metrics,
        "history": history,
        "wall_time_s": time.perf_counter() - started,
    }
    write_json(seed_dir / "metrics.json", result)
    return result


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/eval/a3_successor.yaml")
    parser.add_argument("--force-features", action="store_true")
    parser.add_argument("--seeds", default="", help="optional comma-separated override")
    args = parser.parse_args()
    cfg = load_yaml(args.config)
    build_feature_cache(cfg, force=args.force_features)
    rows = load_rows(cfg)
    categories = sorted({row.truth for row in rows})
    rows = add_counterfactual_nominal(
        rows,
        categories=categories,
        ratio=float(cfg["protocol"]["counterfactual_nominal_ratio"]),
        scale=float(cfg["model"]["conflict_logit_scale"]),
    )
    seeds = (
        [int(item) for item in args.seeds.split(",") if item]
        if args.seeds
        else [int(item) for item in cfg["training"]["seeds"]]
    )
    results = [train_seed(cfg, rows, seed) for seed in seeds]
    summary = {
        **artifact_meta(
            args.config,
            sources={
                "a0_corpus": cfg["sources"]["a0_corpus"],
                "a3_t3_corpus": cfg["sources"]["a3_t3_corpus"],
                "successor_reverse_train": cfg["sources"].get("successor_reverse_train"),
                "conflict_predictions": cfg["sources"]["conflict_predictions"],
                "successor_reverse_conflict_predictions": cfg["sources"].get(
                    "successor_reverse_conflict_predictions"
                ),
                "conflict_adapter": cfg["sources"]["conflict_adapter"],
            },
        ),
        "seeds": seeds,
        "n_rows_real": sum(not row.synthetic for row in rows),
        "n_rows_counterfactual": sum(row.synthetic for row in rows),
        "deployment_inputs": [
            "frozen_CLIP_RGB",
            "standardized_proprio_window",
            "frozen_conflict_VLA_logits",
        ],
        "forbidden_deployment_inputs": cfg["protocol"]["forbidden_deployment_inputs"],
        "results": results,
    }
    write_json(repo_path(cfg["output_dir"]) / "training_summary.json", summary)
    print(json.dumps({"seeds": seeds, "metrics": [r["metrics"] for r in results]}, indent=2))


if __name__ == "__main__":
    main()
