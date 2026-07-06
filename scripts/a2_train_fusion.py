#!/usr/bin/env python
"""Train B-F (the A2 no-VLM fusion baseline): real CLIP(RGB) ⊕ proprio-stats → MLP over categories.

Trained on the SAME snapshots B5 saw (the augmented SFT set), so B-F vs B5 is a controlled "do you
need the VLM?" comparison on the identical data. Eval is via ``FusionBaselinePolicy`` on the frozen
A2 corpus (scripts/a2_eval.py --agents ...,B-F).

Run:  KINOVLA_MODEL_ID=... env -u PYTHONPATH ~/miniconda3/envs/kinovla/bin/python \
        scripts/a2_train_fusion.py --train outputs/eval/e2/sft_augmented
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path


def main() -> None:
    ap = argparse.ArgumentParser(description="Train the B-F CLIP+proprio fusion baseline")
    ap.add_argument("--train", default="outputs/eval/e2/sft_augmented")
    ap.add_argument("--out", default="outputs/eval/a2/b_fusion/bf.pt")
    ap.add_argument("--hidden", type=int, default=256)
    ap.add_argument("--epochs", type=int, default=120)
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()

    from kino_vla.eval.fusion_baseline import train_fusion

    info = train_fusion(
        args.train, args.out, hidden=args.hidden, epochs=args.epochs, seed=args.seed
    )
    print(f"[b-f] trained on {info['n_train']} snapshots; categories={info['categories']}")
    print(f"[b-f] train_acc={info['train_acc']:.3f}  wrote {args.out}")
    Path(args.out).with_suffix(".card.json").write_text(json.dumps(info, indent=2))


if __name__ == "__main__":
    main()
