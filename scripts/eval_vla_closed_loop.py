#!/usr/bin/env python
"""Closed-loop success evaluation (the M7 exit-3 metric: DPO > SFT on held-out scenarios).

Usage:
    python scripts/eval_vla_closed_loop.py --policy model --adapter outputs/vla/sft/adapter_best \
        --backend surrogate --out outputs/vla/closed_loop_sft.json
    # ...then again with the DPO adapter, and compare the success rates.

Runs the planner over the Suite-Sem closed-loop scenarios and reports the success rate (reached the
goal without falling / dead-looping). Comparing the SFT and DPO adapters' success rates is the
exit-3 check. ``--policy stub`` is the CI/oracle reference.
"""

from __future__ import annotations

import argparse
import json

from kino_vla.data.taxonomy import FailureTaxonomy
from kino_vla.utils.config import load_config
from kino_vla.vla import scenarios as S
from kino_vla.vla.planner import StubVlaPolicy
from kino_vla.vla.rollout import run_vla_rollout


def main() -> None:
    ap = argparse.ArgumentParser(description="Closed-loop success eval (exit-3)")
    # policy = the spec §12 baseline: stub (oracle/CI), model (B5 latent / B4 text / B3 vision-only
    # via route+proprio_detail), or fsm (B2 rule-FSM, cause-blind Backstep+detour).
    ap.add_argument("--policy", choices=["stub", "model", "fsm"], default="stub")
    ap.add_argument("--adapter", default=None)
    ap.add_argument("--config", default="vla/dpo.yaml")
    ap.add_argument("--backend", default="surrogate", choices=["surrogate", "isaac"])
    ap.add_argument("--route", default=None, choices=["latent", "text"], help="route arm")
    ap.add_argument(
        "--proprio-detail",
        default="binned",
        choices=["binned", "scalar", "none"],
        help="text route: 'none' = B3 vision-only (no proprioception)",
    )
    ap.add_argument("--temperature", type=float, default=0.0)
    ap.add_argument("--repeats", type=int, default=3)
    ap.add_argument("--out", default="outputs/vla/closed_loop.json")
    args = ap.parse_args()

    cfg = load_config(args.config, {"route": args.route} if args.route else None)
    pcfg = load_config("data/hindsight.yaml")
    tax = FailureTaxonomy(pcfg)

    policy = None
    fsm_baseline = args.policy == "fsm"
    if args.policy == "model":
        import torch

        from kino_vla.vla.model import KinoVLA
        from kino_vla.vla.planner import ModelVlaPolicy

        device = "cuda" if torch.cuda.is_available() else "cpu"
        model = KinoVLA.from_pretrained(cfg, device=device, adapter_dir=args.adapter)
        model.eval()
        policy = ModelVlaPolicy(
            model,
            pcfg,
            tax,
            route=str(cfg.get("route", "latent")),
            n_images=int(cfg.data.get("n_images", 1)),
            temperature=args.temperature,
            proprio_detail=args.proprio_detail,
        )
    elif args.policy == "stub":
        policy = StubVlaPolicy(pcfg, tax)

    rows = []
    n_succ = 0
    n_tot = 0
    scenarios = S.surrogate_scenarios() if args.backend == "surrogate" else S.all_scenarios()
    for scn in scenarios:
        for rep in range(args.repeats):
            r = run_vla_rollout(
                scn, policy, seed=rep, backend=args.backend, fsm_baseline=fsm_baseline
            )
            rows.append({"scenario": scn.name, "rep": rep, **r.to_dict()})
            n_succ += int(r.success)
            n_tot += 1
    summary = {
        "policy": args.policy,
        "adapter": args.adapter,
        "backend": args.backend,
        "success_rate": n_succ / max(1, n_tot),
        "n": n_tot,
        "rows": rows,
    }
    import os

    os.makedirs(os.path.dirname(args.out) or ".", exist_ok=True)
    with open(args.out, "w") as f:
        json.dump(summary, f, indent=2)
    print(f"closed-loop success: {n_succ}/{n_tot} = {summary['success_rate']:.3f}  ({args.policy})")


if __name__ == "__main__":
    main()
