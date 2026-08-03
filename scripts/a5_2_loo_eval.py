#!/usr/bin/env python
"""A5.2 — leave-one-operator-out generalization (Paper-A §4 A5.2, serves C4).

The LOO agent is retrained with operator family O8 (invisible_obstacle) EXCLUDED from all training
(``b5_loo_o8``; clean LOO — the hindsight base has no O8 either). It is then evaluated ZERO-SHOT on
the held-out O8 corpus snapshots. The C4 generalization metrics (spec §4 A5.2):
  (i) open-vocabulary attribution: does it name invisible_obstacle (never seen)?
  (ii) ADMISSIBLE-SET MEMBERSHIP of the chosen primitive (the metric that matters when the class
       name is novel): primitive ∈ {Backstep, Replan_Waypoint, Update_Topology} (O8 admissible);
  (iii) abstention rate (fraction that parse-fails or attributes `nominal`).
Control row: b5_conflict_bi (SAW O8) on the same snapshots. C4 claim: even on a fully
unseen family, a reasoner picks an ADMISSIBLE primitive (it need not name the novel class).

Run:  env -u PYTHONPATH HF_HUB_OFFLINE=1 KINOVLA_MODEL_ID=… \\
        ~/miniconda3/envs/kinovla/bin/python scripts/a5_2_loo_eval.py
"""

from __future__ import annotations

import argparse
import json
import math
import os
import sys

import numpy as np

O8_ADMISSIBLE = {
    "Backstep",
    "Replan_Waypoint",
    "Update_Topology",
}  # invisible_obstacle admissible set


def wilson(k: int, n: int, z: float = 1.96) -> tuple[float, float]:
    if n == 0:
        return (0.0, 0.0)
    p = k / n
    denom = 1.0 + z * z / n
    c = (p + z * z / (2 * n)) / denom
    h = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / denom
    return (round(max(0.0, c - h), 3), round(min(1.0, c + h), 3))


def main() -> int:
    ap = argparse.ArgumentParser(description="A5.2 LOO-operator eval (zero-shot on held-out O8)")
    from isaaclab.app import AppLauncher

    AppLauncher.add_app_launcher_args(ap)
    ap.add_argument("--loo-adapter", default="outputs/eval/a5/b5_loo_o8/adapter_best")
    ap.add_argument(
        "--ctrl-adapter",
        default="outputs/eval/a3/b5_conflict_bi/adapter_best",
        help="control: an adapter that SAW O8 (b5_conflict_bi)",
    )
    ap.add_argument("--config", default="vla/sft.yaml")
    ap.add_argument("--corpus-dirs", default="outputs/eval/a0/corpus,outputs/eval/a3/corpus_t3")
    ap.add_argument("--out", default="outputs/eval/a5/a5_2_loo.json")
    args = ap.parse_args()
    app = AppLauncher(args).app  # noqa: F841

    import torch

    from kino_vla.data.taxonomy import FailureTaxonomy
    from kino_vla.utils.config import load_config
    from kino_vla.vla.dataset_build import _snapshot_from_record
    from kino_vla.vla.model import KinoVLA
    from kino_vla.vla.planner import ModelVlaPolicy

    pcfg = load_config("data/hindsight.yaml")
    tax = FailureTaxonomy(pcfg)
    vcfg = load_config(args.config)
    ni = int(vcfg.data.get("n_images", 1))
    pdet = str(vcfg.data.get("proprio_detail", "binned"))
    dev = "cuda" if torch.cuda.is_available() else "cpu"

    # load the O8 held-out snapshots
    snaps = []
    for d in args.corpus_dirs.split(","):
        from pathlib import Path

        sp = Path(d, "samples.jsonl")
        npz = np.load(Path(d, "frames.npz"))
        if not sp.exists():
            continue
        for ln in sp.read_text().splitlines():
            if not ln:
                continue
            r = json.loads(ln)
            if r["snapshot"]["operator_name"] != "O8_invisible_collider":
                continue
            sid = r["sample_id"]
            if f"{sid}__rgb" not in npz:
                continue
            frames = {k: npz[f"{sid}__{k}"] for k in ("rgb", "depth", "proprio")}
            snaps.append((sid, _snapshot_from_record(r, frames)))
    print(f"[a5.2] {len(snaps)} O8 held-out snapshots", flush=True)

    def eval_adapter(adapter):
        m = KinoVLA.from_pretrained(vcfg, device=dev, adapter_dir=adapter)
        m.eval()
        pol = ModelVlaPolicy(
            m, pcfg, tax, route="latent", n_images=ni, temperature=0.0, proprio_detail=pdet
        )
        rows = []
        for sid, snap in snaps:
            dec = pol.decide(snap)
            parsed = bool(dec.ok and dec.annotation is not None)
            attr = dec.attribution if parsed else None
            prim = dec.primitive_name if parsed else None
            attr_ok = parsed and attr == "invisible_obstacle"
            prim_feasible = parsed and prim in O8_ADMISSIBLE
            joint_ok = attr_ok and prim_feasible
            abstain = (not parsed) or (attr in (None, "nominal"))
            rows.append(
                {
                    "sid": sid,
                    "parsed": parsed,
                    "attribution": attr,
                    "primitive": prim,
                    "attr_ok": attr_ok,
                    "prim_feasible": prim_feasible,
                    "joint_ok": joint_ok,
                    "abstain": abstain,
                }
            )
        del m
        torch.cuda.empty_cache()
        return rows

    out = {}
    for label, adp in (("loo_o8", args.loo_adapter), ("ctrl_saw_o8", args.ctrl_adapter)):
        rows = eval_adapter(adp)
        n = len(rows)
        kp = sum(1 for r in rows if r["prim_feasible"])
        ka = sum(1 for r in rows if r["attr_ok"])
        kj = sum(1 for r in rows if r["joint_ok"])
        kab = sum(1 for r in rows if r["abstain"])
        out[label] = {
            "adapter": adp,
            "n": n,
            "attr_named_invisible": round(ka / max(1, n), 3),
            "attr_ci": wilson(ka, n),
            "prim_in_admissible": round(kp / max(1, n), 3),
            "prim_ci": wilson(kp, n),
            "joint_ok": round(kj / max(1, n), 3),
            "abstain_rate": round(kab / max(1, n), 3),
            "rows": rows,
        }
        print(
            f"[a5.2] {label:14} attr_named={ka}/{n} ({ka / n:.2f}) | "
            f"prim∈admissible={kp}/{n} ({kp / n:.2f}) | abstain={kab}/{n}",
            flush=True,
        )

    from kino_vla.utils.config import REPO_ROOT

    Path(REPO_ROOT / args.out).parent.mkdir(parents=True, exist_ok=True)
    (REPO_ROOT / args.out).write_text(json.dumps(out, indent=2))
    print(f"\n[OK] wrote {args.out}")
    sys.stdout.flush()
    os._exit(0)


if __name__ == "__main__":
    main()
