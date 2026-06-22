"""Kino-SFT trainer: LoRA + Kino-Projector supervised fine-tuning (spec §11 Stage 1).

Trains :class:`~kino_vla.vla.model.KinoVLA` on the filtered Hindsight-CoT dataset (spec §10) to
reproduce the privileged-grounded reflection: physical attribution + one atomic §5 recovery
primitive. A manual torch loop (not a high-level Trainer) so the non-standard Kino-Token soft
splice and the privileged-θ auxiliary loss are first-class. Resumable, seeded, and metrics are
logged to a tracked JSON (QA 5.2 learned-component rule); a checkpoint is "done" only when its
val metrics reproduce.

This is a real-dependency component (the trained VLA): it imports torch + the backbone and runs on
the GPU. The CPU surrogate cannot stand in (§0 hard rule).
"""

from __future__ import annotations

import json
import time
from pathlib import Path

import numpy as np
import torch

from kino_vla.utils.config import Config, load_config
from kino_vla.utils.seeding import seed_everything
from kino_vla.vla.dataset_build import (
    DatasetSplit,
    VlaExample,
    load_examples,
    load_nav_examples,
    split_examples,
)
from kino_vla.vla.model import KinoVLA


def _fit_projector_standardizer(model: KinoVLA, train: list[VlaExample], eps: float) -> None:
    """Install the proprio-window normalization from the train split (spec §4 shared stats)."""
    if model.projector is None:
        return
    # Fit ONLY on recovery examples (those carry privileged θ); nav examples' cruise proprio is a
    # different distribution, and including it shifts the normalization the recovery soft tokens
    # (and thus the M7 attribution) depend on. Nav still passes through the projector at this fit.
    fit_on = [ex for ex in train if ex.target_theta is not None] or train
    windows = np.concatenate([ex.proprio_window for ex in fit_on], axis=0)  # (sum_T, F)
    model.projector.set_standardizer(windows.mean(axis=0), windows.std(axis=0), eps=eps)


def _build_input_cache(model: KinoVLA, examples: list[VlaExample], n_images: int) -> dict:
    """Tokenize each example ONCE (CPU tensors) — reused across epochs (the processor is the
    CPU bottleneck; caching avoids re-encoding the image every epoch, ~Nx faster)."""
    cache: dict[str, object] = {}
    for ex in examples:
        cache[ex.sample_id] = model.build_inputs(
            ex.messages,
            list(ex.rgb[-n_images:]),
            target_text=ex.target_text,
            proprio_window=ex.proprio_window,
            target_theta=ex.target_theta,
            loss_span=getattr(ex, "loss_span", "completion"),
        )
    return cache


def evaluate_loss(model: KinoVLA, examples: list[VlaExample], cache: dict) -> float:
    """Mean completion CE over ``examples`` (no grad) — the val gate."""
    model.eval()
    total, n = 0.0, 0
    with torch.no_grad():
        for ex in examples:
            out = model.compute_loss(cache[ex.sample_id])
            total += float(out["lm_loss"])
            n += 1
    return total / max(1, n)


def train_sft(
    cfg: Config,
    dataset_dir: str | Path,
    out_dir: str | Path,
    *,
    nav_dataset_dir: str | Path | None = None,
) -> dict:
    """Run Kino-SFT; return the metrics dict (also written to ``out_dir/sft_metrics.json``).

    With ``nav_dataset_dir`` set, the RTX nav-SFT examples (route-around turn/waypoint decisions on
    real camera frames) are CO-TRAINED alongside the recovery examples in one adapter — the deployed
    planner uses one model for both recovery and 1 Hz nav (user directive 2026-06-21). Nav examples
    carry ``target_theta=None`` (θ loss skipped) + ``loss_span="action"``; the recovery split (so
    the M7 attribution data) is unchanged (nav is its own ``navigation`` operator stratum)."""
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    seed = int(cfg.train.seed)
    seed_everything(seed)
    torch.manual_seed(seed)

    route = str(cfg.get("route", "latent"))
    n_images = int(cfg.data.get("n_images", 1))
    # The prompt vocab (§5 library / category vocabulary) + taxonomy live in the data config; the
    # vla config (cfg) carries only model/projector/training knobs.
    prompt_cfg = load_config("data/hindsight.yaml")
    proprio_detail = str(cfg.data.get("proprio_detail", "binned"))
    examples = load_examples(
        dataset_dir, prompt_cfg, route=route, n_images=n_images, proprio_detail=proprio_detail
    )
    limit = cfg.data.get("limit")
    if limit is not None:
        examples = examples[: int(limit)]  # smoke-run cap (does not change the seed/split logic)
    n_recovery = len(examples)
    if nav_dataset_dir is not None:
        nav_examples = load_nav_examples(nav_dataset_dir)
        examples = examples + nav_examples
        print(f"[sft] co-training: {n_recovery} recovery + {len(nav_examples)} nav examples")
    split = split_examples(
        examples,
        seed=int(cfg.data.split_seed),
        val_frac=float(cfg.data.val_frac),
        test_frac=float(cfg.data.test_frac),
    )
    # OVERSAMPLE the nav TURN examples in TRAIN (after the split, so no frame leaks train→test): the
    # Turn primitive is a small fraction of all <Action>s (recovery + nav-waypoint dominate), so the
    # model collapses to "always waypoint" and never turns (held-out: 0/8 turns). Repeating balances.
    ot = int(cfg.data.get("nav_turn_oversample", 1))
    if ot > 1:
        extra = [
            e
            for e in split.train
            if e.operator_name == "navigation" and e.primitive_truth == "turn"
        ]
        split.train.extend(extra * (ot - 1))
        print(f"[sft] oversampled {len(extra)} nav-turn train examples ×{ot}", flush=True)
    (out / "split.json").write_text(
        json.dumps(
            {
                k: [e.sample_id for e in v]
                for k, v in {"train": split.train, "val": split.val, "test": split.test}.items()
            }
        )
    )

    device = "cuda" if torch.cuda.is_available() else "cpu"
    dtype = torch.bfloat16 if device == "cuda" else torch.float32
    model = KinoVLA.from_pretrained(cfg, device=device, dtype=dtype)
    _fit_projector_standardizer(model, split.train, float(cfg.projector.get("std_eps", 1e-3)))
    cache = _build_input_cache(model, split.train + split.val, n_images)

    opt = torch.optim.AdamW(
        model.trainable_parameters(),
        lr=float(cfg.train.lr),
        weight_decay=float(cfg.train.weight_decay),
    )
    accum = int(cfg.train.grad_accum)
    epochs = int(cfg.train.epochs)
    rng = np.random.default_rng(seed)
    history: list[dict] = []
    best_val = float("inf")
    t0 = time.perf_counter()

    for epoch in range(epochs):
        model.train()
        order = rng.permutation(len(split.train))
        running = 0.0
        opt.zero_grad()
        for i, idx in enumerate(order):
            ex = split.train[idx]
            out_dict = model.compute_loss(cache[ex.sample_id])
            (out_dict["loss"] / accum).backward()
            running += float(out_dict["loss"])
            if (i + 1) % accum == 0 or (i + 1) == len(order):
                torch.nn.utils.clip_grad_norm_(
                    model.trainable_parameters(), float(cfg.train.grad_clip)
                )
                opt.step()
                opt.zero_grad()
        train_loss = running / max(1, len(order))
        val_loss = evaluate_loss(model, split.val, cache) if split.val else float("nan")
        rec = {"epoch": epoch, "train_loss": train_loss, "val_loss": val_loss}
        history.append(rec)
        print(f"[sft] epoch {epoch} train {train_loss:.4f} val {val_loss:.4f}", flush=True)
        if split.val and val_loss <= best_val:
            best_val = val_loss
            model.save_adapter(out / "adapter_best")

    model.save_adapter(out / "adapter_last")
    metrics = {
        "seed": seed,
        "route": route,
        "n_train": len(split.train),
        "n_val": len(split.val),
        "n_test": len(split.test),
        "epochs": epochs,
        "best_val_loss": best_val,
        "history": history,
        "wall_time_s": time.perf_counter() - t0,
        "model_id": str(cfg.model.model_id),
    }
    (out / "sft_metrics.json").write_text(json.dumps(metrics, indent=2))
    return metrics


def load_split_for_eval(cfg: Config, dataset_dir: str | Path) -> DatasetSplit:
    """Rebuild the exact train/val/test split used by training (same seed + fractions)."""
    route = str(cfg.get("route", "latent"))
    prompt_cfg = load_config("data/hindsight.yaml")
    examples = load_examples(
        dataset_dir,
        prompt_cfg,
        route=route,
        n_images=int(cfg.data.get("n_images", 1)),
        proprio_detail=str(cfg.data.get("proprio_detail", "binned")),
    )
    return split_examples(
        examples,
        seed=int(cfg.data.split_seed),
        val_frac=float(cfg.data.val_frac),
        test_frac=float(cfg.data.test_frac),
    )
