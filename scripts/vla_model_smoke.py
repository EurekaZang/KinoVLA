#!/usr/bin/env python
"""Smoke-test the real KinoVLA wrapper against the Qwen3-VL-4B weights (model integration check).

Validates the risky integration before the full SFT run: load Qwen3-VL-4B + LoRA + the Kino-
Projector, build a multimodal training example (RGB + the <kino_tokens> soft-token splice), run a
forward (loss finite), backprop into LoRA + projector, then generate and parse (parse-or-reject).

    python scripts/vla_model_smoke.py [--config vla/sft.yaml] [--route latent|text]
"""

from __future__ import annotations

import argparse

import numpy as np
import torch

from kino_vla.data.schema import CoTAnnotation, RecoveryPrimitive, Snapshot
from kino_vla.data.taxonomy import FailureTaxonomy
from kino_vla.utils.config import load_config
from kino_vla.vla.model import KinoVLA
from kino_vla.vla.output import parse_vla_decision
from kino_vla.vla.prompt import build_messages, context_from_snapshot, format_target


def _snapshot() -> Snapshot:
    rng = np.random.default_rng(0)
    w = np.zeros((25, 11), dtype=np.float32)
    w[:, 7] = 0.6  # slip channel sustained-high (low_friction signature)
    rgb = rng.uniform(0.4, 0.9, size=(5, 72, 96, 3)).astype(np.float32)
    return Snapshot(
        operator_name="O1_mu_field",
        appearance_class="ice_sheet",
        t=2.0,
        pose_xy=np.array([3.0, 0.0]),
        heading=0.0,
        rgb=rgb,
        depth=np.ones((5, 72, 96), dtype=np.float32),
        proprio_window=w,
        prior_outputs=[],
        privileged_theta={"mu": 0.1},
        monitor_channel="slip",
    )


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="vla/sft.yaml")
    ap.add_argument("--route", default="latent", choices=["latent", "text"])
    args = ap.parse_args()

    cfg = load_config(args.config, {"route": args.route})
    pcfg = load_config("data/hindsight.yaml")  # §5 vocab for the prompt
    tax = FailureTaxonomy(pcfg)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"loading {cfg.model.model_id} on {device} (route={args.route}) ...", flush=True)
    model = KinoVLA.from_pretrained(
        cfg, device=device, dtype=torch.bfloat16 if device == "cuda" else torch.float32
    )
    if model.projector is not None:
        model.projector.set_standardizer(np.zeros(11), np.ones(11))
    n_train = sum(p.numel() for p in model.trainable_parameters())
    print(f"trainable params: {n_train / 1e6:.2f}M", flush=True)

    snap = _snapshot()
    ctx = context_from_snapshot(snap, route=args.route)
    messages = build_messages(ctx, pcfg, route=args.route, n_images=int(cfg.data.n_images))
    ann = CoTAnnotation(
        thought="Sustained high slip on a reflective surface — low friction.",
        attribution="low_friction",
        primitive=RecoveryPrimitive("Set_Constraint", {"max_speed": 0.4, "stiffness": 0.5}),
        attribution_raw="low_friction",
        raw_text="",
    )
    target = format_target(ann)

    # 1) forward + backward (SFT step)
    inputs = model.build_inputs(
        messages,
        list(snap.rgb[-int(cfg.data.n_images) :]),
        target_text=target,
        proprio_window=snap.proprio_window,
        target_theta=[0.1, 0.0, 1.0, 1.0],
    )
    out = model.compute_loss(inputs)
    print(
        f"forward: loss={float(out['loss']):.4f} lm={float(out['lm_loss']):.4f} "
        f"theta={float(out['theta_loss']):.4f}",
        flush=True,
    )
    assert torch.isfinite(out["loss"]), "loss is not finite"
    out["loss"].backward()
    g = [p.grad for p in model.trainable_parameters() if p.grad is not None]
    assert g and any(float(x.abs().sum()) > 0 for x in g), "no gradient flowed to trainable params"
    print(f"backward OK: {len(g)} tensors have grad", flush=True)

    # 2) generate + parse (inference path)
    text = model.generate(
        messages,
        list(snap.rgb[-int(cfg.data.n_images) :]),
        proprio_window=snap.proprio_window,
        max_new_tokens=128,
        temperature=0.0,
    )
    print(f"generated:\n{text}\n", flush=True)
    parsed = parse_vla_decision(text, synonyms=tax.synonyms, valid_categories=tax.valid_categories)
    print(
        f"parse ok={parsed.ok} attribution={parsed.attribution} "
        f"primitive={parsed.primitive_name} code={parsed.reject_code}"
    )
    print("SMOKE OK")


if __name__ == "__main__":
    main()
