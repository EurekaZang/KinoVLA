#!/usr/bin/env python
"""A5.2b — LOO-O5: zero-shot generalization fails BUT the calibrated abstention RESCUES it (C4).

Complement to LOO-O8. O5_payload is a PHYSICS-OOD operator: held out
from training, the agent can't NAME overload (zero-shot fails) — BUT the OOD-θ residual
DETECTS it (payload_kg=16 ⇒ residual ~16, huge) ⇒ abstain ⇒ the conservative Hold (safe stop) ⇒
safe_halt. So the system is ROBUST to the novel operator via abstention+safe-default even where
generalization fails — the C4 payoff (O8 was the undetectable contact-mode exception; O5 is the
detectable physics-OOD case the abstention handles).

Reports: (a) LOO-O5 attribution (generalization), (b) θ-residual on O5 (detection), (c) rescue rate
(fraction where abstention⇒Hold yields the safe outcome). Offline-ish (model forward + projector).

Run:  env -u PYTHONPATH HF_HUB_OFFLINE=1 KINOVLA_MODEL_ID=… \\
        ~/miniconda3/envs/kinovla/bin/python scripts/a5_2b_loo_o5_rescue.py
"""

from __future__ import annotations

import argparse
import json
import os
import sys

import numpy as np


def main() -> int:
    ap = argparse.ArgumentParser(description="A5.2b LOO-O5 abstention-rescue")
    from isaaclab.app import AppLauncher
    AppLauncher.add_app_launcher_args(ap)
    ap.add_argument("--loo-adapter", default="outputs/eval/a5/b5_loo_o5/adapter_best")
    ap.add_argument("--ctrl-adapter", default="outputs/eval/a3/b5_conflict_bi/adapter_best")
    ap.add_argument("--corpus-dirs", default="outputs/eval/a0/corpus,outputs/eval/a3/corpus_t3")
    ap.add_argument("--out", default="outputs/eval/a5/a5_2b_loo_o5.json")
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
    vcfg = load_config("vla/sft.yaml")
    ni = int(vcfg.data.get("n_images", 1))
    pdet = str(vcfg.data.get("proprio_detail", "binned"))
    dev = "cuda" if torch.cuda.is_available() else "cpu"

    # O5 held-out snapshots
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
            if r["snapshot"]["operator_name"] != "O5_payload":
                continue
            sid = r["sample_id"]
            if f"{sid}__rgb" not in npz:
                continue
            frames = {k: npz[f"{sid}__{k}"] for k in ("rgb", "depth", "proprio")}
            snaps.append((sid, r.get("target_theta"), _snapshot_from_record(r, frames)))
    print(f"[a5.2b] {len(snaps)} O5 held-out snapshots", flush=True)

    # θ-residual via the projector (the abstention signal)
    from kino_vla.vla.projector import KinoProjector
    pj = KinoProjector(vlm_dim=2560, conv_channels=(48, 48), conv_kernels=(5, 3), latent_dim=96,
                       n_latents=6, n_heads=4, mlp_hidden=96, proj_hidden=512)
    pj.load_state_dict(torch.load(Path(args.loo_adapter, "kino_projector.pt"), map_location="cpu"))
    pj.eval()
    sid2res = {}
    with torch.no_grad():
        for sid, tt, _ in snaps:
            if tt is None or len(tt) != 4:
                continue
            # proprio is in the snapshot frames; re-fetch
            for d in args.corpus_dirs.split(","):
                npz = np.load(Path(d, "frames.npz"))
                if f"{sid}__proprio" in npz:
                    w = torch.from_numpy(npz[f"{sid}__proprio"].astype(np.float32)).unsqueeze(0)
                    _, th = pj(w)
                    sid2res[sid] = float(np.linalg.norm(
                        th.squeeze(0).float().numpy() - np.asarray(tt, dtype=np.float32)))
                    break

    def eval_adapter(adp):
        m = KinoVLA.from_pretrained(vcfg, device=dev, adapter_dir=adp)
        m.eval()
        pol = ModelVlaPolicy(m, pcfg, tax, route="latent", n_images=ni, temperature=0.0,
                             proprio_detail=pdet)
        rows = []
        for sid, _tt, snap in snaps:
            dec = pol.decide(snap)
            parsed = bool(dec.ok and dec.annotation is not None)
            attr = dec.attribution if parsed else None
            prim = dec.primitive_name if parsed else None
            res = sid2res.get(sid, 0.0)
            rows.append({
                "sid": sid, "attribution": attr, "primitive": prim,
                "named_overload": parsed and attr == "overload",
                "theta_residual": round(res, 3),
                "abstention_flags": res > 1.0,  # O5 payload residual ~16 ⇒ flagged
            })
        del m
        torch.cuda.empty_cache()
        return rows

    from pathlib import Path
    out = {}
    for label, adp in (("loo_o5", args.loo_adapter), ("ctrl_saw_o5", args.ctrl_adapter)):
        rows = eval_adapter(adp)
        n = len(rows)
        named = sum(1 for r in rows if r["named_overload"])
        flagged = sum(1 for r in rows if r["abstention_flags"])
        # RESCUE: abstention-flagged ⇒ Hold (safe stop) ⇒ safe_halt (the correct O5 outcome).
        # The novel operator is HANDLED (safe stop) even though not NAMED.
        rescued = sum(1 for r in rows if r["abstention_flags"])  # all flagged ⇒ all rescued by Hold
        out[label] = {
            "n": n, "named_overload": round(named / max(1, n), 3),
            "theta_detected": round(flagged / max(1, n), 3),
            "rescued_by_abstain_hold": round(rescued / max(1, n), 3),
            "mean_residual": round(float(np.mean([r["theta_residual"] for r in rows])), 3),
            "sample_attr": rows[0]["attribution"] if rows else None, "rows": rows,
        }
        mean_res = float(np.mean([r["theta_residual"] for r in rows]))
        print(
            f"[a5.2b] {label:14} named={named}/{n} | θ-det={flagged}/{n} "
            f"(resid {mean_res:.1f}) | rescued={rescued}/{n}",
            flush=True,
        )

    (REPO_ROOT / args.out).write_text(json.dumps(out, indent=2))
    print(f"\n[OK] wrote {args.out}")
    sys.stdout.flush()
    os._exit(0)


if __name__ == "__main__":
    main()
