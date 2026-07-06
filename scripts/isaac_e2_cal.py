#!/usr/bin/env python
"""E2 Suite-Cal parity (real Go2): B1 ≈ B5 on the A-class operators (the fair-opponent panel).

On the six A-class scenarios (O1_A/O2_A/O5_A/O10_A/O6_A/O11_A) where low-level adaptation suffices
and no semantic attribution is needed, the strongest pure-proprio B1, the cause-blind FSM B2, and
the vision+proprio agent B5 should all reach the goal — parity proves B1 is a FAIR opponent, so the
Suite-Sem gap (the three-row ablation) is the real cost of proprioceptive blindness on the semantic
class, not a strawman artifact. Closed-loop reach success; no use_map/CBF (A-class is crossable).

Run:  KINOVLA_MODEL_ID=/home/eureka/models/Qwen3-VL-4B-Instruct HF_HUB_OFFLINE=1 \
        python scripts/isaac_e2_cal.py --headless --repeats 3
"""

from __future__ import annotations

import argparse
import json
import os
import sys

import numpy as np


def main() -> int:
    ap = argparse.ArgumentParser(description="E2 Suite-Cal parity (A-class)")
    from isaaclab.app import AppLauncher

    AppLauncher.add_app_launcher_args(ap)
    ap.add_argument("--config", default="eval/e2.yaml")
    ap.add_argument("--repeats", type=int, default=3)
    ap.add_argument("--out", default="outputs/eval/e2/suite_cal.json")
    args = ap.parse_args()
    app = AppLauncher(args).app  # noqa: F841 (keeps the sim app alive)

    import torch

    from kino_vla.data.taxonomy import FailureTaxonomy
    from kino_vla.eval.proprio_baseline import ProprioBaselinePolicy
    from kino_vla.monitor.learned_monitor import load_deployed_monitor
    from kino_vla.sim.isaac_policy_backend import IsaacPolicyBackend
    from kino_vla.utils.config import REPO_ROOT, load_config
    from kino_vla.vla import scenarios as S
    from kino_vla.vla.model import KinoVLA
    from kino_vla.vla.planner import ModelVlaPolicy
    from kino_vla.vla.rollout import run_closed_loop

    cfg = load_config(args.config).to_dict()
    pcfg = load_config("data/hindsight.yaml")
    tax = FailureTaxonomy(pcfg)
    backend = IsaacPolicyBackend(load_config("sim/go2_skeleton.yaml"), np.array([0.0, 0.0]), 0.0)
    monitor = load_deployed_monitor(backend.dt)

    b1 = ProprioBaselinePolicy.from_deployed(
        pcfg, tax, model_path=str(cfg["b1"]["model_path"]), device="cpu"
    )
    vcfg = load_config(str(cfg["b5"]["config"]), {"route": str(cfg["b5"]["route"])})
    print("[e2.cal] loading B5 agent …", flush=True)
    model = KinoVLA.from_pretrained(vcfg, device="cuda", adapter_dir=str(cfg["b5"]["adapter_dir"]))
    model.eval()
    torch.set_grad_enabled(False)
    b5 = ModelVlaPolicy(
        model, pcfg, tax, route=str(vcfg.get("route", "latent")),
        n_images=int(vcfg.data.get("n_images", 1)), temperature=0.0,
        proprio_detail=str(vcfg.data.get("proprio_detail", "binned")),
    )

    def run_one(scn, *, policy=None, fsm=False, seed=0):
        backend._start_pos = np.asarray(scn.start_xy, dtype=np.float64)
        backend._start_heading = float(scn.start_heading)
        return run_closed_loop(
            backend, scn, policy, monitor_cfg="monitor/rule_v0_isaac.yaml",
            fsm_cfg="recovery/fsm_isaac.yaml", seed=seed, fsm_baseline=fsm, monitor=monitor,
        )

    policies = [("B1", dict(policy=b1)), ("B2_fsm", dict(fsm=True)), ("B5", dict(policy=b5))]
    res = {name: {"success": 0, "n": 0, "rows": []} for name, _ in policies}
    for scn in S.suite_cal_scenarios(0.0):
        for name, kw in policies:
            for rep in range(args.repeats):
                r = run_one(scn, seed=rep, **kw)
                res[name]["success"] += int(r.success)
                res[name]["n"] += 1
                res[name]["rows"].append({"scn": scn.name, "rep": rep, "success": r.success})
                print(f"[e2.cal] {name:6} {scn.name:20} rep{rep}: success={r.success}", flush=True)
    for name in res:
        res[name]["success_rate"] = res[name]["success"] / max(1, res[name]["n"])

    parity = abs(res["B1"]["success_rate"] - res["B5"]["success_rate"])
    report = {"repeats": args.repeats, "results": res,
              "parity_abs_diff": round(parity, 3),
              "parity_tol": float(cfg["accept"]["cal_parity_tol"]),
              "PARITY": bool(parity <= float(cfg["accept"]["cal_parity_tol"]))}
    out = REPO_ROOT / args.out
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(report, indent=2))
    print("\n=== Suite-Cal parity (A-class) ===")
    for name in ("B1", "B2_fsm", "B5"):
        print(f"{name:8} success_rate={res[name]['success_rate']:.3f} (n={res[name]['n']})")
    print(f"parity |B1-B5|={parity:.3f} (tol {report['parity_tol']}) -> "
          f"{'PARITY' if report['PARITY'] else 'NO-PARITY'}")
    sys.stdout.flush()
    os._exit(0)


if __name__ == "__main__":
    main()
