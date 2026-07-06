#!/usr/bin/env python
"""A5.4 — OOD-θ residual as a calibrated abstention signal (Paper-A §4 A5.4, serves C4).

The KinoProjector's privileged-θ regression head (``theta_head``) predicts the snapshot's privileged
physics (mu, payload_kg, effort_scale, support_ratio) from the 500 ms proprio window. Its prediction
RESIDUAL is an anomaly score on the proprio channel: small for in-distribution physics the projector
learned, large for physics outside the training-θ range (extrapolation failure). This script:

  1. runs the trained projector on all corpus + T3 snapshots ⇒ per-snapshot residual ||θ̂ − θ||;
  2. ABSTENTION UTILITY: AUROC of the residual for detecting each agent's attribution errors
     (a usable abstention signal ⇒ abstain on high-residual snapshots to avoid errors);
  3. OOD CALIBRATION: residual vs distance of θ from the training-θ centroid (monotonicity).

Projector-only (CPU, ~0.2M); no Isaac, no VLM generate. Ref: m7_route_ablation._theta_mae.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import torch

TARGET_NAMES = ("mu", "payload_kg", "effort_scale", "support_ratio")


def auroc(score: np.ndarray, label: np.ndarray) -> float:
    """Mann-Whitney U AUROC (score higher ⇒ label=1 more likely). label ∈ {0,1}."""
    label = label.astype(int)
    n1, n0 = int(label.sum()), int((1 - label).sum())
    if n1 == 0 or n0 == 0:
        return float("nan")
    order = np.argsort(score)
    ranks = np.empty_like(order, dtype=float)
    ranks[order] = np.arange(1, len(score) + 1)
    # average ranks for ties
    s = score[order]
    i = 0
    while i < len(s):
        j = i
        while j + 1 < len(s) and s[j + 1] == s[i]:
            j += 1
        if j > i:
            ranks[order[i : j + 1]] = (i + 1 + j + 1) / 2.0
        i = j + 1
    sum_ranks_pos = ranks[label == 1].sum()
    return float((sum_ranks_pos - n1 * (n1 + 1) / 2.0) / (n1 * n0))


def load_snapshots(dirs: list[str]) -> list[dict]:
    snaps = []
    for d in dirs:
        sp = Path(d, "samples.jsonl")
        npz = np.load(Path(d, "frames.npz"))
        if not sp.exists():
            continue
        for ln in sp.read_text().splitlines():
            if not ln:
                continue
            r = json.loads(ln)
            sid = r["sample_id"]
            if f"{sid}__proprio" not in npz:
                continue
            tt = r.get("target_theta")
            if tt is None or len(tt) != 4:
                continue
            snaps.append(
                {
                    "sid": sid,
                    "operator": r["snapshot"]["operator_name"],
                    "cell": r.get("taxonomy_cell", "?"),
                    "split": r.get("appearance_split", "train"),
                    "proprio": npz[f"{sid}__proprio"].astype(np.float32),
                    "target_theta": np.asarray(tt, dtype=np.float32),
                }
            )
    return snaps


def main() -> None:
    ap = argparse.ArgumentParser(description="A5.4 OOD-θ residual abstention signal (offline, CPU)")
    ap.add_argument(
        "--adapter",
        default="outputs/eval/a3/b5_conflict_bi/adapter_best",
        help="adapter dir with kino_projector.pt (the θ head)",
    )
    ap.add_argument("--a3-dir", default="outputs/eval/a3")
    ap.add_argument("--corpus-dirs", default="outputs/eval/a0/corpus,outputs/eval/a3/corpus_t3")
    ap.add_argument("--out", default="outputs/eval/a5")
    args = ap.parse_args()
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)

    from kino_vla.vla.projector import KinoProjector

    pj = KinoProjector(
        vlm_dim=2560,
        conv_channels=(48, 48),
        conv_kernels=(5, 3),
        latent_dim=96,
        n_latents=6,
        n_heads=4,
        mlp_hidden=96,
        proj_hidden=512,
    )
    pt = Path(args.adapter, "kino_projector.pt")
    pj.load_state_dict(torch.load(pt, map_location="cpu"))
    pj.eval()

    snaps = load_snapshots(args.corpus_dirs.split(","))
    print(f"[a5.4] {len(snaps)} snapshots | projector {pt}", flush=True)
    with torch.no_grad():
        for s in snaps:
            w = torch.from_numpy(s["proprio"]).unsqueeze(0)
            _soft, theta = pj(w)
            s["theta_pred"] = theta.squeeze(0).float().cpu().numpy()
            s["residual_l2"] = float(np.linalg.norm(s["theta_pred"] - s["target_theta"]))
            s["residual_per_target"] = np.abs(s["theta_pred"] - s["target_theta"]).tolist()

    # training-θ centroid + per-dim range (from train-split snapshots)
    train_tt = np.array([s["target_theta"] for s in snaps if s["split"] == "train"])
    centroid = train_tt.mean(axis=0)
    tmin, tmax = train_tt.min(axis=0), train_tt.max(axis=0)
    for s in snaps:
        d = np.maximum(0.0, np.maximum(s["target_theta"] - tmax, tmin - s["target_theta"]))
        s["ood_distance"] = float(np.linalg.norm(d))  # 0 inside the train range, growing outside
        s["centroid_dist"] = float(np.linalg.norm(s["target_theta"] - centroid))

    # ---- 1. OOD calibration: residual vs centroid distance (monotonicity) + OOD-distance ----
    cd = np.array([s["centroid_dist"] for s in snaps])
    res = np.array([s["residual_l2"] for s in snaps])
    ood = np.array([s["ood_distance"] for s in snaps])
    bins = np.quantile(cd, np.linspace(0, 1, 6))
    binned = []
    for i in range(len(bins) - 1):
        m = (cd >= bins[i]) & (cd <= bins[i + 1])
        if m.sum() > 0:
            binned.append(
                {
                    "centroid_dist_bin": round(float(bins[i]), 3),
                    "mean_residual": round(float(res[m].mean()), 4),
                    "n": int(m.sum()),
                }
            )
    # Spearman-like monotonicity: corr(rank cd, rank res)
    rank_cd = cd.argsort().argsort().astype(float)
    rank_res = res.argsort().argsort().astype(float)
    rho = float(np.corrcoef(rank_cd, rank_res)[0, 1])
    # AUROC: residual detecting OOD (ood_distance > 0 ⇒ outside train range)
    ood_label = (ood > 1e-6).astype(int)
    ood_auroc = (
        auroc(res, ood_label)
        if ood_label.sum() > 0 and ood_label.sum() < len(ood)
        else float("nan")
    )

    print(
        f"[a5.4] residual vs θ-centroid-distance Spearman ρ={rho:.3f} | "
        f"AUROC(residual detects OOD-θ outside train range)={ood_auroc:.3f} "
        f"(n_ood={int(ood_label.sum())}/{len(ood)})",
        flush=True,
    )
    print("[a5.4] residual by centroid-distance bin (monotone ⇒ calibrated OOD score):")
    for b in binned:
        print(
            f"    dist≥{b['centroid_dist_bin']:.2f}: resid {b['mean_residual']:.3f} (n {b['n']})"
        )

    # ---- 2. ABSTENTION UTILITY: AUROC of residual for each agent's attribution errors ----
    a3dir = Path(args.a3_dir)
    agents = {
        "B1": "B1",
        "B_F": "B-F",
        "B5_unshaped": "B5-unshaped",
        "B5_conflict": "B5-conflict",
        "B5_conflict_bi": "B5-conflict-bi",
        "B_V": "B-V",
        "B_T": "B-T",
        "zero_shot": "zero-shot",
    }
    sid2res = {s["sid"]: s["residual_l2"] for s in snaps}
    abstention: dict[str, dict] = {}
    print("\n[a5.4] ABSTENTION UTILITY — AUROC(residual ⇒ attribution error), per agent:")
    for fname, agent in agents.items():
        f = a3dir / f"per_item_{fname}.json"
        if not f.exists():
            continue
        rows = json.loads(f.read_text())
        score, label = [], []
        for r in rows:
            if r["sid"] not in sid2res or not r.get("parsed"):
                continue
            score.append(sid2res[r["sid"]])
            label.append(0 if r.get("attr_ok") else 1)  # 1 = error ⇒ abstain
        score, label = np.array(score), np.array(label)
        au = auroc(score, label)
        abstention[agent] = {"auroc": round(au, 3), "n": len(score), "n_err": int(label.sum())}
        print(f"    {agent:<16} AUROC={au:.3f}  (n={len(score)}, errors={int(label.sum())})")

    result = {
        "adapter": args.adapter,
        "n_snapshots": len(snaps),
        "ood_calibration": {
            "spearman_rho_centroid": round(rho, 3),
            "auroc_detect_ood_theta": round(ood_auroc, 3),
            "n_outside_train_range": int(ood_label.sum()),
            "residual_by_bin": binned,
        },
        "abstention_utility": abstention,
        "mean_residual_per_target": {
            TARGET_NAMES[i]: round(float(np.mean([s["residual_per_target"][i] for s in snaps])), 4)
            for i in range(4)
        },
    }
    (out / "a5_4_ood_theta.json").write_text(json.dumps(result, indent=2))
    print(f"\n[OK] wrote {out / 'a5_4_ood_theta.json'}")


if __name__ == "__main__":
    main()
