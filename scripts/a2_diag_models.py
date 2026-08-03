#!/usr/bin/env python
"""A2 diagnostic: do B5-unshaped and B5-conflict actually generate DIFFERENT decisions?

The A2 run showed them identical (both 0.600 O4), which contradicts E2 (0.267 vs 1.00). This loads
both adapters and prints the RAW generation for a handful of matched items (yellow O4 + brown O2 +
a held-out appearance), so we can see whether the adapters diverge at the text level or the corpus
snapshot is driving both to the same answer.
"""

from __future__ import annotations

import argparse


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--corpus", default="outputs/eval/a0/corpus")
    ap.add_argument("--config", default="vla/sft.yaml")
    ap.add_argument("--n", type=int, default=3)
    args = ap.parse_args()

    import torch

    from kino_vla.data.taxonomy import FailureTaxonomy
    from kino_vla.eval.a2_headline import load_matched_corpus
    from kino_vla.utils.config import load_config
    from kino_vla.vla.model import KinoVLA
    from kino_vla.vla.planner import ModelVlaPolicy

    pcfg = load_config("data/hindsight.yaml")
    tax = FailureTaxonomy(pcfg)
    items = load_matched_corpus(args.corpus)

    # pick: yellow O4 (train app), gray_tape O4 (held-out), brown O2
    def pick(op: str, app: str) -> list:
        return [it for it in items if it.operator == op and it.appearance_id == app][: args.n]

    probe = pick("O4_tether", "yellow_board") + pick("O4_tether", "gray_tape") + pick(
        "O2_compliance", "brown_mud"
    )
    vcfg = load_config(args.config, {"route": "latent"})
    device = "cuda" if torch.cuda.is_available() else "cpu"

    outputs: dict[str, list] = {}
    for tag, adapter in (
        ("B5-unshaped", "outputs/vla/sft_latent/adapter_best"),
        ("B5-conflict", "outputs/eval/e2/b5_conflict/adapter_best"),
    ):
        model = KinoVLA.from_pretrained(vcfg, device=device, adapter_dir=adapter)
        model.eval()
        # capture the projector's first soft-token norm to prove the adapter/projector loaded
        pj_norm = float(
            torch.stack([p.detach().float().norm() for p in model.projector.parameters()]).sum()
        )
        pol = ModelVlaPolicy(model, pcfg, tax, route="latent",
                             n_images=int(vcfg.data.get("n_images", 1)), temperature=0.0,
                             proprio_detail=str(vcfg.data.get("proprio_detail", "binned")))
        rows = []
        for it in probe:
            raw = model.generate(
                *_prep(it, pol, vcfg), proprio_window=it.snapshot.proprio_window, temperature=0.0
            )
            dec = pol.decide(it.snapshot)
            raw1 = raw[:120].replace("\n", " ")
            rows.append((it.operator, it.appearance_id, dec.attribution, raw1))
        outputs[tag] = rows
        print(f"\n=== {tag}  (projector param-norm sum={pj_norm:.3f}) ===", flush=True)
        for op, app, attr, raw in rows:
            print(f"  {op[:3]} {app[:14]:14s} attr={attr!s:20s} raw='{raw}'", flush=True)
        del model
        torch.cuda.empty_cache()

    # divergence summary
    u, c = outputs["B5-unshaped"], outputs["B5-conflict"]
    same = sum(1 for a, b in zip(u, c, strict=True) if a[2] == b[2])
    print(f"\nidentical attribution on {same}/{len(u)} probe items "
          f"({'IDENTICAL ⇒ suspect bug' if same == len(u) else 'DIVERGE ⇒ models differ'})")


def _prep(it: object, pol: object, vcfg: object) -> tuple:
    """Rebuild the (messages, images) the policy would pass to generate (mirrors ModelVlaPolicy)."""
    from kino_vla.vla.prompt import build_messages, context_from_snapshot

    ctx = context_from_snapshot(it.snapshot, route="latent", proprio_detail="binned")
    n_img = int(vcfg.data.get("n_images", 1))
    messages = build_messages(ctx, pol._cfg, route="latent", n_images=n_img)
    n_images = int(vcfg.data.get("n_images", 1))
    images = list(it.snapshot.rgb[-n_images:]) if it.snapshot.rgb.size else []
    return messages, images


if __name__ == "__main__":
    main()
