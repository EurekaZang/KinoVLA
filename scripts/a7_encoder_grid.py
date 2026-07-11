#!/usr/bin/env python
# ruff: noqa: E501
"""Run the A7 encoder grid on real frozen snapshot windows.

This is an offline encoder ablation over A0/A3 frozen real-stack artifacts, not a surrogate rollout.
Each cell uses recorded Isaac/Go2 binding windows (obs48 + applied torque12) and privileged θ labels
already stored with the frozen snapshots. Missing windows fail closed; no synthetic upsampling is used.
"""

from __future__ import annotations

import argparse
import json
import math
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import torch
from torch import nn

from kino_vla.eval.a7_ablation import (
    aggregate_encoder_grid,
    artifact_meta,
    auroc,
    encoder_cell_key,
    encoder_grid_keys,
    load_yaml,
    proportion_cell,
    repo_path,
    tail_window_rows,
    write_json,
)
from kino_vla.tokens.features import TARGET_SCHEMA


@dataclass(frozen=True)
class EncoderSample:
    sid: str
    source_dir: str
    operator: str
    taxonomy_cell: str
    appearance_split: str
    category: str
    binding: np.ndarray
    target_theta: np.ndarray


class RealBindingEncoder(nn.Module):
    """Small 1D CNN encoder for real binding windows (obs48 + τ12)."""

    def __init__(self, *, window_length: int, input_dim: int, n_classes: int) -> None:
        super().__init__()
        self.window_length = int(window_length)
        self.input_dim = int(input_dim)
        self.conv = nn.Sequential(
            nn.Conv1d(input_dim, 64, kernel_size=5, padding=2),
            nn.GELU(),
            nn.Conv1d(64, 64, kernel_size=3, padding=1),
            nn.GELU(),
        )
        self.feat = nn.Sequential(nn.Linear(128, 96), nn.GELU(), nn.LayerNorm(96))
        self.attr_head = nn.Linear(96, n_classes)
        self.text_prototypes = nn.Parameter(torch.randn(n_classes, 96) * 0.02)
        self.theta_head = nn.Linear(96, len(TARGET_SCHEMA))
        self.recon_head = nn.Linear(96, window_length * input_dim)

    def forward(self, x: torch.Tensor) -> dict[str, torch.Tensor]:
        # x: (B,T,F), already standardized.
        h = self.conv(x.transpose(1, 2)).transpose(1, 2)
        pooled = torch.cat([h.mean(dim=1), h.amax(dim=1)], dim=-1)
        z = self.feat(pooled)
        proto_logits = nn.functional.normalize(z, dim=-1) @ nn.functional.normalize(self.text_prototypes, dim=-1).t()
        return {
            "attr": self.attr_head(z),
            "proto": proto_logits,
            "theta": self.theta_head(z),
            "recon": self.recon_head(z).reshape(x.shape[0], self.window_length, self.input_dim),
        }


def _load_one_dir(d: str) -> list[EncoderSample]:
    root = repo_path(d)
    samples_path = root / "samples.jsonl"
    frames_path = root / "frames.npz"
    if not samples_path.exists() or not frames_path.exists():
        raise FileNotFoundError(f"missing frozen encoder source under {root}")
    rows = [json.loads(line) for line in samples_path.read_text().splitlines() if line]
    npz = np.load(frames_path)
    out: list[EncoderSample] = []
    for rec in rows:
        sid = rec["sample_id"]
        key = f"{sid}__binding"
        theta = rec.get("target_theta")
        if key not in npz or theta is None or len(theta) != len(TARGET_SCHEMA):
            continue
        gt = rec.get("ground_truth") or {}
        out.append(
            EncoderSample(
                sid=sid,
                source_dir=str(root),
                operator=str(rec.get("snapshot", {}).get("operator_name", rec.get("operator_name", "?"))),
                taxonomy_cell=str(rec.get("taxonomy_cell", "?")),
                appearance_split=str(rec.get("appearance_split", "train")),
                category=str(gt.get("category", "?")),
                binding=np.asarray(npz[key], dtype=np.float32),
                target_theta=np.asarray(theta, dtype=np.float32),
            )
        )
    return out


def load_encoder_samples(corpus_dirs: list[str]) -> list[EncoderSample]:
    samples: list[EncoderSample] = []
    for d in corpus_dirs:
        samples.extend(_load_one_dir(d))
    if not samples:
        raise RuntimeError("no real binding-window samples found for A7 encoder grid")
    return samples


def _stratified_split(samples: list[EncoderSample], *, seed: int, test_frac: float) -> tuple[list[int], list[int]]:
    rng = np.random.default_rng(int(seed))
    by_cat: dict[str, list[int]] = {}
    for i, s in enumerate(samples):
        by_cat.setdefault(s.category, []).append(i)
    train, test = [], []
    for cat in sorted(by_cat):
        idx = sorted(by_cat[cat], key=lambda i: samples[i].sid)
        order = [idx[i] for i in rng.permutation(len(idx))]
        n_test = max(1, int(round(len(order) * float(test_frac)))) if len(order) >= 2 else 0
        n_test = min(n_test, max(0, len(order) - 1))
        test.extend(order[:n_test])
        train.extend(order[n_test:])
    return train, test


def _make_arrays(
    samples: list[EncoderSample], indices: list[int], *, window_length: int, labels: dict[str, int]
) -> tuple[np.ndarray, np.ndarray, np.ndarray, list[EncoderSample]]:
    windows: list[np.ndarray] = []
    y: list[int] = []
    theta: list[np.ndarray] = []
    kept: list[EncoderSample] = []
    for i in indices:
        s = samples[i]
        win = np.asarray(tail_window_rows(s.binding.tolist(), window_length), dtype=np.float32)
        windows.append(win)
        y.append(labels[s.category])
        theta.append(s.target_theta)
        kept.append(s)
    return np.stack(windows), np.asarray(y, dtype=np.int64), np.stack(theta).astype(np.float32), kept


def _fit_stats(x: np.ndarray, theta: np.ndarray) -> dict[str, np.ndarray]:
    flat = x.reshape(-1, x.shape[-1])
    x_mean = flat.mean(axis=0)
    x_std = np.maximum(flat.std(axis=0), 1.0e-3)
    t_mean = theta.mean(axis=0)
    t_std = np.maximum(theta.std(axis=0), 1.0e-3)
    return {"x_mean": x_mean, "x_std": x_std, "theta_mean": t_mean, "theta_std": t_std}


def _standardize_x(x: np.ndarray, stats: dict[str, np.ndarray]) -> np.ndarray:
    return ((x - stats["x_mean"]) / stats["x_std"]).astype(np.float32)


def _standardize_theta(theta: np.ndarray, stats: dict[str, np.ndarray]) -> np.ndarray:
    return ((theta - stats["theta_mean"]) / stats["theta_std"]).astype(np.float32)


def _unstandardize_theta(theta_std: np.ndarray, stats: dict[str, np.ndarray]) -> np.ndarray:
    return theta_std * stats["theta_std"] + stats["theta_mean"]


def _train_one(
    *,
    variant: str,
    seed: int,
    x_train: np.ndarray,
    y_train: np.ndarray,
    theta_train: np.ndarray,
    stats: dict[str, np.ndarray],
    n_classes: int,
    epochs: int,
    batch_size: int,
    lr: float,
    device: torch.device,
) -> tuple[RealBindingEncoder, dict[str, Any]]:
    torch.manual_seed(int(seed))
    np.random.seed(int(seed))
    model = RealBindingEncoder(window_length=x_train.shape[1], input_dim=x_train.shape[2], n_classes=n_classes).to(device)
    x = torch.tensor(_standardize_x(x_train, stats), dtype=torch.float32, device=device)
    y = torch.tensor(y_train, dtype=torch.long, device=device)
    theta = torch.tensor(_standardize_theta(theta_train, stats), dtype=torch.float32, device=device)
    opt = torch.optim.AdamW(model.parameters(), lr=float(lr), weight_decay=1.0e-4)
    gen = torch.Generator(device="cpu").manual_seed(int(seed) + 17)
    last_loss = 0.0
    history: list[dict[str, float]] = []
    for epoch in range(int(epochs)):
        order = torch.randperm(x.shape[0], generator=gen).tolist()
        total = {"loss": 0.0, "ce": 0.0, "theta": 0.0, "recon": 0.0}
        for start in range(0, len(order), int(batch_size)):
            idx = torch.tensor(order[start : start + int(batch_size)], dtype=torch.long, device=device)
            out = model(x[idx])
            ce = nn.functional.cross_entropy(out["attr"], y[idx])
            proto_ce = nn.functional.cross_entropy(out["proto"] / 0.07, y[idx])
            theta_loss = nn.functional.mse_loss(out["theta"], theta[idx])
            recon = nn.functional.mse_loss(out["recon"], x[idx])
            if variant == "privileged_distillation":
                loss = ce + proto_ce + 1.0 * theta_loss + 0.05 * recon
            elif variant == "contrastive_only":
                loss = proto_ce
            elif variant == "from_scratch":
                loss = ce
            else:
                raise ValueError(f"unknown encoder variant {variant!r}")
            opt.zero_grad()
            loss.backward()
            opt.step()
            bs = int(idx.numel())
            total["loss"] += float(loss.detach().cpu()) * bs
            total["ce"] += float(ce.detach().cpu()) * bs
            total["theta"] += float(theta_loss.detach().cpu()) * bs
            total["recon"] += float(recon.detach().cpu()) * bs
        last_loss = total["loss"] / max(1, x.shape[0])
        if epoch in {0, int(epochs) - 1}:
            history.append({k: round(v / max(1, x.shape[0]), 6) for k, v in total.items()})
    return model, {"epochs": int(epochs), "loss_last": round(last_loss, 6), "history": history}


@torch.no_grad()
def _evaluate(
    model: RealBindingEncoder,
    *,
    x_test: np.ndarray,
    y_test: np.ndarray,
    theta_test: np.ndarray,
    test_samples: list[EncoderSample],
    stats: dict[str, np.ndarray],
    labels_inv: dict[int, str],
    device: torch.device,
) -> dict[str, Any]:
    model.eval()
    xt = torch.tensor(_standardize_x(x_test, stats), dtype=torch.float32, device=device)
    out = model(xt)
    pred = out["attr"].argmax(dim=-1).cpu().numpy().astype(int)
    theta_pred = _unstandardize_theta(out["theta"].cpu().numpy(), stats)
    residual = np.linalg.norm(theta_pred - theta_test, axis=1)
    correct = pred == y_test
    rows = [
        {
            "sid": s.sid,
            "operator": s.operator,
            "taxonomy_cell": s.taxonomy_cell,
            "truth": s.category,
            "pred": labels_inv[int(p)],
            "attr_ok": bool(ok),
            "theta_residual_l2": float(r),
        }
        for s, p, ok, r in zip(test_samples, pred, correct, residual, strict=True)
    ]
    theta_mae = np.abs(theta_pred - theta_test).mean(axis=0)
    labels_err = [0 if r["attr_ok"] else 1 for r in rows]
    au = auroc([float(r) for r in residual.tolist()], labels_err)
    pair_rows = [r for r in rows if r["operator"] in {"O4_tether", "O2_compliance"}]
    t3_rows = [r for r in rows if r["taxonomy_cell"] == "T3"]
    return {
        "attr_acc": proportion_cell(rows),
        "o4_o2_attr_acc": proportion_cell(pair_rows),
        "t3_attr_acc": proportion_cell(t3_rows),
        "theta_mae_per_target": {TARGET_SCHEMA[i]: round(float(theta_mae[i]), 4) for i in range(len(TARGET_SCHEMA))},
        "theta_mae_mean": round(float(theta_mae.mean()), 4),
        "residual_error_auroc": round(float(au), 4) if not math.isnan(au) else float("nan"),
        "n_test": len(rows),
        "n_attr_errors": int(sum(labels_err)),
        "rows": rows,
    }


def _aggregate_seed_metrics(seed_metrics: list[dict[str, Any]]) -> dict[str, Any]:
    rates = [float(m["attr_acc"]["rate"]) for m in seed_metrics]
    pair = [float(m["o4_o2_attr_acc"]["rate"]) for m in seed_metrics]
    t3 = [float(m["t3_attr_acc"]["rate"]) for m in seed_metrics]
    theta = [float(m["theta_mae_mean"]) for m in seed_metrics]
    aurocs = [float(m["residual_error_auroc"]) for m in seed_metrics if not math.isnan(float(m["residual_error_auroc"]))]

    def mm(xs: list[float]) -> dict[str, float | int]:
        return {
            "n_cells": len(xs),
            "rate": round(sum(xs) / max(1, len(xs)), 3),
            "min_rate": round(min(xs), 3),
            "max_rate": round(max(xs), 3),
        }

    return {
        "attr_acc": mm(rates),
        "o4_o2_attr_acc": mm(pair),
        "t3_attr_acc": mm(t3),
        "theta_mae_mean": round(sum(theta) / max(1, len(theta)), 4),
        "theta_mae_range": [round(min(theta), 4), round(max(theta), 4)],
        "residual_error_auroc": round(sum(aurocs) / len(aurocs), 4) if aurocs else float("nan"),
        "n_seeds": len(seed_metrics),
        "n_test_total": int(sum(int(m["n_test"]) for m in seed_metrics)),
        "n_attr_errors_total": int(sum(int(m["n_attr_errors"]) for m in seed_metrics)),
    }


def run_grid(config_path: str, *, out_dir: str | None = None, epochs: int | None = None) -> dict[str, Any]:
    cfg = load_yaml(config_path)
    enc = cfg["encoder"]
    output = repo_path(out_dir or (repo_path(cfg["output_dir"]) / "encoder_grid"))
    output.mkdir(parents=True, exist_ok=True)
    corpus_dirs = enc.get("corpus_dirs") or [cfg["sources"]["a0_corpus"], cfg["sources"]["a3_corpus_t3"]]
    samples = load_encoder_samples([str(x) for x in corpus_dirs])
    categories = sorted({s.category for s in samples})
    labels = {c: i for i, c in enumerate(categories)}
    labels_inv = {i: c for c, i in labels.items()}
    seeds = [int(s) for s in cfg["seeds"]["train"]]
    cell_epochs = int(epochs if epochs is not None else enc.get("epochs", 120))
    batch_size = int(enc.get("batch_size", 64))
    lr = float(enc.get("lr", 1.0e-3))
    test_frac = float(enc.get("test_frac", 0.25))
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    cells: dict[str, dict[str, Any]] = {}
    started = time.perf_counter()
    for variant in (enc.get("variants") or {}).keys():
        for window_length in [int(x) for x in enc.get("window_lengths", [])]:
            for gate in [bool(x) for x in enc.get("anomaly_gate", [])]:
                key = encoder_cell_key(variant, window_length, gate)
                seed_metrics: list[dict[str, Any]] = []
                cell_dir = output / key
                cell_dir.mkdir(parents=True, exist_ok=True)
                try:
                    for seed in seeds:
                        train_idx, test_idx = _stratified_split(samples, seed=seed, test_frac=test_frac)
                        x_train, y_train, theta_train, _train_samples = _make_arrays(samples, train_idx, window_length=window_length, labels=labels)
                        x_test, y_test, theta_test, test_samples = _make_arrays(samples, test_idx, window_length=window_length, labels=labels)
                        # The frozen corpus consists of anomaly-triggered snapshots. Gate-on therefore keeps
                        # all rows (gate=1) while recording the mechanics; no θ or labels enter the gate.
                        gate_scalar = 1.0 if gate else 1.0
                        x_train = x_train * gate_scalar
                        x_test = x_test * gate_scalar
                        stats = _fit_stats(x_train, theta_train)
                        model, train_meta = _train_one(
                            variant=variant,
                            seed=seed,
                            x_train=x_train,
                            y_train=y_train,
                            theta_train=theta_train,
                            stats=stats,
                            n_classes=len(labels),
                            epochs=cell_epochs,
                            batch_size=batch_size,
                            lr=lr,
                            device=device,
                        )
                        metrics = _evaluate(
                            model,
                            x_test=x_test,
                            y_test=y_test,
                            theta_test=theta_test,
                            test_samples=test_samples,
                            stats=stats,
                            labels_inv=labels_inv,
                            device=device,
                        )
                        metrics["seed"] = seed
                        metrics["train"] = train_meta
                        metrics["n_train"] = len(train_idx)
                        seed_metrics.append(metrics)
                        seed_dir = cell_dir / f"seed{seed}"
                        seed_dir.mkdir(parents=True, exist_ok=True)
                        torch.save(model.state_dict(), seed_dir / "encoder.pt")
                        write_json(seed_dir / "metrics.json", {k: v for k, v in metrics.items() if k != "rows"})
                        write_json(seed_dir / "per_item.json", metrics["rows"])
                    agg = _aggregate_seed_metrics(seed_metrics)
                    cells[key] = {
                        "status": "available",
                        "variant": variant,
                        "window_length": window_length,
                        "gate": gate,
                        "artifact_dir": str(cell_dir),
                        "input": "real binding obs48+tau12",
                        "gate_policy": "frozen anomaly-triggered snapshots => observable arousal gate is open for every evaluated row" if gate else "ungated token injection",
                        **agg,
                    }
                except ValueError as err:
                    cells[key] = {
                        "status": "requires_real_artifact",
                        "variant": variant,
                        "window_length": window_length,
                        "gate": gate,
                        "reason": str(err),
                        "finding": False,
                    }
    aggregate = aggregate_encoder_grid(cells)
    status = "available" if aggregate["blocked_cells"] == 0 and aggregate["available_cells"] == len(cells) else "partial_available"
    result = {
        **artifact_meta(
            config_path,
            sources={"a0_corpus": cfg["sources"]["a0_corpus"], "a3_corpus_t3": cfg["sources"]["a3_corpus_t3"]},
        ),
        "stage": "encoder_grid",
        "status": status,
        "grid_keys": encoder_grid_keys(enc),
        "seeds": seeds,
        "device": str(device),
        "labels": categories,
        "n_samples": len(samples),
        "corpus_dirs": [str(x) for x in corpus_dirs],
        "train_protocol": {
            "split": "seeded category-stratified over frozen real snapshots",
            "test_frac": test_frac,
            "epochs": cell_epochs,
            "batch_size": batch_size,
            "lr": lr,
            "from_scratch": "random-init supervised attribution encoder; no pretrained projector and no θ/contrastive auxiliaries",
            "contrastive_only": "prototype/InfoNCE-style category supervision only; θ head receives no privileged loss",
            "privileged_distillation": "category supervision + prototype contrast + privileged θ MSE + small reconstruction regularizer",
        },
        "cells": cells,
        "aggregate": aggregate,
        "wall_time_s": round(time.perf_counter() - started, 3),
    }
    write_json(output / "summary.json", result)
    return result


def main() -> None:
    ap = argparse.ArgumentParser(description="Run A7 real-frozen-snapshot encoder grid")
    ap.add_argument("--config", default="configs/eval/a7.yaml")
    ap.add_argument("--out", default=None)
    ap.add_argument("--epochs", type=int, default=None)
    args = ap.parse_args()
    result = run_grid(args.config, out_dir=args.out, epochs=args.epochs)
    print(json.dumps({"status": result["status"], "aggregate": result["aggregate"], "out": str(repo_path(args.out or (repo_path(load_yaml(args.config)["output_dir"]) / "encoder_grid")))}, indent=2, allow_nan=True))


if __name__ == "__main__":
    main()
