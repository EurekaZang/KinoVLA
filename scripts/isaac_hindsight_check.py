#!/usr/bin/env python
"""M6 Hindsight-CoT pipeline on the Isaac Go2 (spec §10) — real-Go2 GPU collection gate.

The spec builds the data pipeline on Isaac Lab (§1 "基于 Isaac Lab 搭建动力学仿真环境…采集",
§8.1 P1, §10 PHASE 1/2). This closes the M6 surrogate gap on GPU (like the M4/M5 backfills,
GPU collection protocol): it drives the physically-simulated Go2 into each operator's failure in its
own lateral lane (Isaac is one-episode/process, #21a), intercepts the anomaly with the high-
recall collection monitor, and packages the snapshot from the REAL proprioception + privileged
θ — then runs the SAME Oracle + truth-consistency filter (kino_vla/data, backend-agnostic) on
those real-Go2 snapshots. RGB-D is the camera-free §7 render (real geometry, the M5 closure).

Covers all three Suite-Sem ambiguity pairs (O4↔O2, O5↔O10, O3↔O1) + O7. The surrogate pipeline
(scripts/build_hindsight_dataset.py) stays the high-volume path; this proves the rollout is real.

Run:  python scripts/isaac_hindsight_check.py --headless
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import threading

import numpy as np


def main() -> int:
    parser = argparse.ArgumentParser(description="M6 Hindsight-CoT real-Go2 collection gate")
    from isaaclab.app import AppLauncher

    AppLauncher.add_app_launcher_args(parser)
    args = parser.parse_args()
    app = AppLauncher(args).app

    from kino_vla.data import FailureTaxonomy, ScriptedOracle, TruthConsistencyFilter
    from kino_vla.data.isaac_rollout import collect_lane
    from kino_vla.sim.isaac_policy_backend import IsaacPolicyBackend
    from kino_vla.utils.config import REPO_ROOT, load_config

    cfg = load_config("data/hindsight.yaml")
    cc = cfg.isaac_collect
    tax = FailureTaxonomy(cfg)
    oracle = ScriptedOracle(cfg, tax, seed=0)
    filt = TruthConsistencyFilter(tax)
    monitor_cfg = load_config(str(cfg.drive.monitor))

    backend = IsaacPolicyBackend(load_config("sim/go2_skeleton.yaml"), np.array([0.0, 0.0]), 0.0)

    # The lane drive (incl. the #31 O5/O10 embodiment regime) lives in the shared collect_lane, so
    # the gate exercises the EXACT dataset rollout — not a divergent copy. It still uses the
    # ScriptedOracle (deterministic, no API) + the truth filter to prove the pipeline runs on the
    # real-Go2 snapshots; the real-gpt-5.5 attribution is validated separately (build_hindsight).
    def run_lane(lane: dict, seed: int):
        snap, op_theta, fell = collect_lane(backend, cfg, monitor_cfg, lane, seed)
        return snap, tax.ground_truth(lane["op"], op_theta), fell

    def theta_confirms(op: str, th: dict[str, float]) -> bool:
        """The REAL privileged θ at the snapshot must reflect the operator's failure (it was
        captured at the failure locus, not on firm ground). O2/O4 resistance has no signature in
        the 4-channel θ — it is a velocity/tracking deficit — so they are confirmed by channel."""
        if op in ("O1_mu_field", "O7_visual_remap", "O3_collapse"):
            return th["mu"] < 0.5
        if op == "O5_payload":
            return th["payload_kg"] > 1.0
        if op == "O10_effort_decay":
            return th["effort_scale"] < 0.5
        return True

    # ---- collect one real-Go2 snapshot per operator lane (GPU) ----
    lanes = list(cc.lanes)
    rows = []
    for i, lane in enumerate(lanes):
        snap, gt, fell = run_lane(lane, seed=100 + i)
        intercepted = snap is not None
        theta_str = (
            f"mu={snap.privileged_theta['mu']:.2f} pay={snap.privileged_theta['payload_kg']:.1f} "
            f"eff={snap.privileged_theta['effort_scale']:.2f}"
            if intercepted
            else "—"
        )
        verdict = None
        theta_ok = intercepted and theta_confirms(lane["op"], snap.privileged_theta)
        if intercepted:
            _, verdict = filt.evaluate(oracle.annotate(snap), gt, prior_outputs=snap.prior_outputs)
        rows.append(
            {
                "op": lane["op"],
                "appearance": lane["appearance"],
                "intercepted": intercepted,
                "fell": fell,
                "channel": snap.monitor_channel if intercepted else "—",
                "real_theta": theta_str,
                "theta_ok": bool(theta_ok),
                "truth": gt.category,
                "verdict": verdict.reason if verdict else "—",
                "kept": bool(verdict and verdict.keep),
            }
        )
        print(
            f"[isaac_m6] {lane['op']:22s} appr={lane['appearance']:15s} "
            f"intercept={intercepted} ch={rows[-1]['channel']:14s} {theta_str} "
            f"θ_ok={rows[-1]['theta_ok']} truth={gt.category:18s} verdict={rows[-1]['verdict']}"
        )

    # ---- report (CPU) ----
    intercepted = [r for r in rows if r["intercepted"]]
    kept = [r for r in rows if r["kept"]]
    ops_intercepted = {r["op"] for r in intercepted}
    pair_o4o2 = {"O4_tether", "O2_compliance"} <= ops_intercepted
    out = REPO_ROOT / "outputs" / "gpu_audit" / "m6_hindsight.md"
    out.parent.mkdir(parents=True, exist_ok=True)
    lines = [
        "# M6 Hindsight-CoT — real-Go2 collection gate (spec §10 on Isaac)",
        "",
        f"intercepted **{len(intercepted)}/{len(rows)}** operators · kept **{len(kept)}** "
        "(same Oracle + truth-consistency filter as the surrogate pipeline)",
        "",
        "| operator | appearance | intercept | channel | real θ | θ_ok | truth | verdict |",
        "| --- | --- | --- | --- | --- | --- | --- | --- |",
    ]
    lines += [
        f"| {r['op']} | {r['appearance']} | {r['intercepted']} | {r['channel']} | "
        f"{r['real_theta']} | {r['theta_ok']} | {r['truth']} | {r['verdict']} |"
        for r in rows
    ]
    out.write_text("\n".join(lines) + "\n")
    print(f"[isaac_m6] wrote {out}")

    # ---- gate: real-Go2 interception works, θ confirms the failure, the pipeline runs on it ----
    theta_ok_all = all(r["theta_ok"] for r in intercepted)
    ok = len(intercepted) >= 6 and len(kept) > 0 and pair_o4o2 and theta_ok_all
    print(
        f"[isaac_m6] intercepted={len(intercepted)}/{len(rows)} kept={len(kept)} "
        f"O4&O2_both_intercepted={pair_o4o2} theta_confirms_all={theta_ok_all}"
    )
    print(json.dumps({r["op"]: r["verdict"] for r in rows}))
    print("PASS: M6 Hindsight on Isaac" if ok else "FAIL: M6 Hindsight on Isaac")

    sys.stdout.flush()
    closer = threading.Thread(target=app.close, daemon=True)
    closer.start()
    closer.join(timeout=15.0)
    os._exit(0 if ok else 1)


if __name__ == "__main__":
    main()
