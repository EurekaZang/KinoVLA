#!/usr/bin/env python
"""A6 eval — agent decision-flip vs θ* (A6.2) + the O2 modality-dependence panel (A6.3). Serves C4.

A6.2: on the O10 floor-sweep snapshots, the agent roster outputs `continue` (decay mild) vs an
INTERVENTION (Switch_Gait/Set_Constraint — decay severe). The decision-flip point (floor where
continue→intervention) is the agent's LEARNED boundary; compare to A6.1's privileged θ*.

A6.3: on O2_A_nominal (k_c=6, certified proprio-ambiguous mild mud), the agent outputs `continue`
(vision confirms mud ⇒ benign) where the proprio-only detector fires (E4 §3.6: O2 is the
sole residual A-class fire at deb 25). One panel: on ambiguous terrain, "when to intervene" is
a cross-modal question the proprio-only layer cannot answer (R8).

Run:  env -u PYTHONPATH HF_HUB_OFFLINE=1 KINOVLA_MODEL_ID=… \\
        ~/miniconda3/envs/kinovla/bin/python scripts/a6_eval.py
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from collections import defaultdict

import numpy as np

INTERVENTION_PRIMS = {
    "Switch_Gait",
    "Set_Constraint",
    "Hold_and_Request",
    "Backstep",
    "Update_Topology",
    "Replan_Waypoint",
    "Adjust_Posture",
}


def main() -> int:
    ap = argparse.ArgumentParser(description="A6.2 decision-flip + A6.3 O2 panel")
    from isaaclab.app import AppLauncher

    AppLauncher.add_app_launcher_args(ap)
    ap.add_argument("--o10-sweep", default="outputs/eval/a6/corpus_o10_sweep")
    ap.add_argument("--corpus", default="outputs/eval/a0/corpus")
    ap.add_argument(
        "--adapters",
        default=(
            "B5-unshaped=outputs/vla/sft_latent/adapter_best,"
            "B5-conflict-bi=outputs/eval/a3/b5_conflict_bi/adapter_best,"
            "zero-shot=none"
        ),
    )
    ap.add_argument("--config", default="vla/sft.yaml")
    ap.add_argument("--out", default="outputs/eval/a6/a6_eval.json")
    args = ap.parse_args()
    app = AppLauncher(args).app  # noqa: F841

    import torch

    from kino_vla.data.taxonomy import FailureTaxonomy
    from kino_vla.utils.config import REPO_ROOT, load_config
    from kino_vla.vla.dataset_build import _snapshot_from_record
    from kino_vla.vla.model import KinoVLA
    from kino_vla.vla.planner import ModelVlaPolicy

    pcfg = load_config("data/hindsight.yaml")
    tax = FailureTaxonomy(pcfg)
    vcfg = load_config(args.config)
    ni = int(vcfg.data.get("n_images", 1))
    pdet = str(vcfg.data.get("proprio_detail", "binned"))
    dev = "cuda" if torch.cuda.is_available() else "cpu"
    roster = [
        (p.split("=")[0], (None if p.split("=", 1)[1] == "none" else p.split("=", 1)[1]))
        for p in args.adapters.split(",")
        if "=" in p
    ]
    models = {}
    for name, adp in roster:
        m = KinoVLA.from_pretrained(vcfg, device=dev, adapter_dir=adp)
        m.eval()
        models[name] = ModelVlaPolicy(
            m, pcfg, tax, route="latent", n_images=ni, temperature=0.0, proprio_detail=pdet
        )

    def load_snaps(d, op_filter=None):
        from pathlib import Path

        sp = Path(d, "samples.jsonl")
        npz = np.load(Path(d, "frames.npz"))
        if not sp.exists():
            return []
        out = []
        for ln in sp.read_text().splitlines():
            if not ln:
                continue
            r = json.loads(ln)
            if op_filter and r["snapshot"]["operator_name"] != op_filter:
                continue
            sid = r["sample_id"]
            if f"{sid}__rgb" not in npz:
                continue
            frames = {k: npz[f"{sid}__{k}"] for k in ("rgb", "depth", "proprio")}
            out.append((r, _snapshot_from_record(r, frames)))
        return out

    def decide(pol, r, snap):
        dec = pol.decide(snap)
        parsed = bool(dec.ok and dec.annotation is not None)
        attr = dec.attribution if parsed else None
        prim = dec.primitive_name if parsed else None
        is_continue = parsed and (
            prim is None or prim not in INTERVENTION_PRIMS or attr in (None, "nominal")
        )
        return {"attribution": attr, "primitive": prim, "parsed": parsed, "continue": is_continue}

    # ---- A6.2: O10 floor-sweep decision-flip (roster) ----
    o10 = load_snaps(args.o10_sweep)
    print(f"[a6.2] {len(o10)} O10-sweep snapshots (onset t≈5 ⇒ floor-differentiated)", flush=True)
    flip_by_agent = {}
    for name, pol in models.items():
        by_floor = defaultdict(list)
        for r, snap in o10:
            by_floor[r["a6_floor"]].append(decide(pol, r, snap))
        flip_rows = []
        for floor in sorted(by_floor, reverse=True):  # mild → severe
            rows = by_floor[floor]
            p_cont = sum(1 for x in rows if x["continue"]) / max(1, len(rows))
            flip_rows.append(
                {
                    "floor": floor,
                    "n": len(rows),
                    "p_continue": round(p_cont, 3),
                    "p_intervene": round(1 - p_cont, 3),
                    "sample_attr": rows[0]["attribution"] if rows else None,
                }
            )
            print(
                f"[a6.2] {name:14} floor={floor:<5g} P(continue)={p_cont:.2f} "
                f"P(intervene)={1 - p_cont:.2f} attr={rows[0]['attribution'] if rows else None}",
                flush=True,
            )
        flip_by_agent[name] = flip_rows

    # ---- A6.3: O2_A_nominal panel (continue where proprio detector fires) — roster ----
    o2a = load_snaps(args.corpus, op_filter="O2_compliance")
    o2_panel = {}
    for name, pol in models.items():
        o2_rows = [decide(pol, r, s) for r, s in o2a]
        p_cont_o2 = sum(1 for x in o2_rows if x["continue"]) / max(1, len(o2_rows))
        o2_panel[name] = {
            "n": len(o2_rows),
            "p_continue": round(p_cont_o2, 3),
            "sample_attr": o2_rows[0]["attribution"] if o2_rows else None,
        }
        print(
            f"[a6.3] {name:14} O2_A_nominal P(continue)={p_cont_o2:.2f} (n {len(o2_rows)}) "
            f"attr={o2_rows[0]['attribution'] if o2_rows else None}",
            flush=True,
        )

    out = {
        "roster": [name for name, _ in roster],
        "a6_2_decision_flip": {
            "by_agent": flip_by_agent,
            "note": "each agent's learned continue/intervene flip vs A6.1 θ*",
        },
        "a6_3_o2_panel": {
            "by_agent": o2_panel,
            "note": "O2_A_nominal: agent continues (vision confirms mud, benign) where "
            "the proprio-only detector necessarily fires (E4 §3.6 / A1.5)",
        },
    }
    (REPO_ROOT / args.out).write_text(json.dumps(out, indent=2))
    print(f"\n[OK] wrote {args.out}")
    sys.stdout.flush()
    os._exit(0)


if __name__ == "__main__":
    main()
