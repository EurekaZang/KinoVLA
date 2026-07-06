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
from collections import defaultdict
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


def balance_train(
    train: list[VlaExample],
    *,
    nav_weight: float = 1.0,
    max_factor: int = 20,
    nav_turn_frac: float = 1.0,
) -> list[VlaExample]:
    """Class-balanced TRAIN re-sampling that fixes the nav-``Turn`` collapse at the data level.

    The deployed model collapses to "always waypoint" because ``Turn`` is a tiny fraction of all
    ``<Action>``s (recovery + nav-waypoint dominate ⇒ held-out 0/8 turns, #43). This balances the
    NAVIGATION stratum: ``waypoint`` → ``target`` and ``turn`` → ``target * nav_turn_frac``
    (``nav_turn_frac=1.0`` ⇒ parity, the round-1 default; <1 damps OVER-turning, the #44 round-2
    lever — the model turned 27/31 because parity over-weighted turning vs deploy frequency). The
    whole nav stratum is scaled to ``nav_weight`` × the recovery count. RECOVERY examples are kept
    VERBATIM — their per-primitive distribution (and thus the M7 attribution ceiling, 0.974) is
    never perturbed. Deterministic (sorted tile, capped at ``max_factor``) ⇒ reproducible from seed.

    Pure (no torch / no model) ⇒ unit-testable on the CI machine.
    """
    recovery = [e for e in train if e.operator_name != "navigation"]
    nav = [e for e in train if e.operator_name == "navigation"]
    if not nav:
        return list(train)
    by_kind: dict[str, list[VlaExample]] = defaultdict(list)
    for e in nav:
        by_kind[e.primitive_truth].append(e)  # "waypoint" | "turn"
    n_kinds = len(by_kind)
    budget = round(float(nav_weight) * len(recovery)) if recovery else len(nav)
    # waypoint → target; turn → target*nav_turn_frac (≤1 damps over-turning). target never below the
    # largest observed kind (no data lost at frac=1). Recovery distribution is untouched.
    target = max(max(len(v) for v in by_kind.values()), budget // max(1, n_kinds))
    out = list(recovery)
    for kind in sorted(by_kind):
        exs = sorted(by_kind[kind], key=lambda e: e.sample_id)
        k_target = max(1, round(target * (float(nav_turn_frac) if kind == "turn" else 1.0)))
        reps = min(int(max_factor), max(1, -(-k_target // len(exs))))  # ceil-div, capped
        out.extend((exs * reps)[:k_target])
    return out


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
    carry ``target_theta=None`` (θ loss skipped) + ``loss_span="completion"`` (the active-perception
    REASONING is trained, #43); the recovery split (so the M7 attribution data) is unchanged (nav is
    its own ``navigation`` operator stratum). ``balance_train`` then lifts nav-Turn to parity."""
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
    # Class-balance the TRAIN split AFTER the split (so no frame leaks train→val/test): lift the
    # nav-Turn class to parity with nav-waypoint and weight the nav stratum vs recovery, WITHOUT
    # perturbing the recovery per-primitive distribution (protects the M7 attribution ceiling). This
    # replaces the ad-hoc nav_turn_oversample band-aid (#43, data-level fix — no runtime plugin).
    if any(e.operator_name == "navigation" for e in split.train):
        n_before = len(split.train)
        # in-place list assignment (DatasetSplit is a frozen dataclass ⇒ cannot rebind .train)
        split.train[:] = balance_train(
            split.train,
            nav_weight=float(cfg.data.get("nav_weight", 1.0)),
            max_factor=int(cfg.data.get("nav_balance_max_factor", 20)),
            nav_turn_frac=float(cfg.data.get("nav_turn_frac", 1.0)),
        )
        n_turn = sum(
            e.operator_name == "navigation" and e.primitive_truth == "turn" for e in split.train
        )
        n_wp = sum(
            e.operator_name == "navigation" and e.primitive_truth == "waypoint" for e in split.train
        )
        print(
            f"[sft] class-balanced train {n_before}->{len(split.train)} "
            f"(nav turn={n_turn} waypoint={n_wp})",
            flush=True,
        )
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
