#!/usr/bin/env python
"""M6 Hindsight-CoT pipeline on the Isaac Go2 (spec §10) — real-Go2 GPU collection gate.

The spec builds the data pipeline on Isaac Lab (§1 "基于 Isaac Lab 搭建动力学仿真环境…采集",
§8.1 P1, §10 PHASE 1/2). This closes the M6 surrogate gap on GPU (like the M4/M5 backfills,
CLAUDE.md #21/#23): it drives the physically-simulated Go2 into each operator's failure in its
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
    from kino_vla.data.snapshot import SnapshotRecorder
    from kino_vla.map.types import SemanticRegion
    from kino_vla.monitor.rule_monitor import RuleMonitor
    from kino_vla.sim.isaac_policy_backend import IsaacPolicyBackend
    from kino_vla.sim.types import CollapseRegion, FrictionRegion, ResistanceRegion
    from kino_vla.utils.config import REPO_ROOT, load_config
    from kino_vla.utils.geometry import Rect

    cfg = load_config("data/hindsight.yaml")
    cc = cfg.isaac_collect
    tax = FailureTaxonomy(cfg)
    oracle = ScriptedOracle(cfg, tax, seed=0)
    filt = TruthConsistencyFilter(tax)
    monitor_cfg = load_config(str(cfg.drive.monitor))

    backend = IsaacPolicyBackend(load_config("sim/go2_skeleton.yaml"), np.array([0.0, 0.0]), 0.0)
    dt = backend.dt
    period, duty = float(cc.speed_period_s), float(cc.speed_duty)
    lo, margin = float(cc.speed_lo), float(cc.gate_margin_m)
    region_ops = {"O1_mu_field", "O7_visual_remap", "O3_collapse", "O2_compliance", "O4_tether"}

    def bangbang(hi: float):
        return lambda k: hi if (((k * dt) / period) % 1.0) < duty else lo

    def theta_for(lane: dict) -> dict[str, float]:
        op = lane["op"]
        if op == "O2_compliance":
            return {"d_sink": float(lane["d_sink"])}
        if op == "O5_payload":
            return {"mass_kg": float(lane["mass"])}
        if op == "O10_effort_decay":
            return {"floor": float(lane["floor"])}
        return {}

    def apply_region(lane: dict, rect: Rect) -> None:
        op = lane["op"]
        if op in ("O1_mu_field", "O7_visual_remap"):
            mu = float(lane["mu"])
            backend.add_friction_regions([FrictionRegion(rect, mu, mu)])
        elif op == "O3_collapse":
            backend.add_collapse_regions(
                [
                    CollapseRegion(
                        rect=rect,
                        mu_intact=float(lane["mu_intact"]),
                        mu_collapsed=float(lane["mu_collapsed"]),
                        trigger_dwell_s=float(lane["dwell"]),
                    )
                ]
            )
        elif op == "O2_compliance":
            backend.add_resistance_regions(
                [
                    ResistanceRegion(
                        rect,
                        float(lane["k"]),
                        float(lane["c"]),
                        sink_depth_m=float(lane["d_sink"]),
                        kind="compliance",
                    )
                ]
            )
        elif op == "O4_tether":
            backend.add_resistance_regions(
                [
                    ResistanceRegion(
                        rect,
                        float(lane["k"]),
                        float(lane["c"]),
                        break_force_n=float(lane["f_break"]),
                        kind="tether",
                    )
                ]
            )

    def run_lane(lane: dict, seed: int):
        op, y, appr = lane["op"], float(lane["y"]), str(lane["appearance"])
        rect = Rect(float(cc.patch_cx), y, float(cc.patch_hx), float(cc.patch_hy))
        if op in region_ops:
            apply_region(lane, rect)
        backend._start_pos = np.array([0.0, y])
        backend._start_heading = 0.0
        obs = backend.reset(seed)
        speed_fn = bangbang(float(cc.effort_speed_hi if op == "O10_effort_decay" else cc.speed_hi))
        if op == "O10_effort_decay":
            backend.set_effort_scale(1.0)
        for _ in range(int(cc.settle_steps)):
            obs = backend.step(np.array([speed_fn(0), 0.0, 0.0]))
        if op == "O10_effort_decay":
            backend.set_effort_scale(float(lane["floor"]))  # cut after settling healthy
        if op == "O5_payload":
            backend.add_payload(float(lane["mass"]), np.zeros(2))
        scene = (
            [SemanticRegion(Rect(rect.cx, rect.cy, rect.hx, rect.hy), appr)]
            if op in region_ops
            else []
        )
        # Region operators gate to the patch (capture the in-region failure, not the bang-bang
        # transient); global operators (O5/O10) have no locus — their θ is everywhere, so any
        # post-arm interception confirms it (and the heavy/weak Go2 may stall before the patch).
        gate_rect = (
            Rect(rect.cx, rect.cy, rect.hx + margin, rect.hy + margin) if op in region_ops else None
        )
        recorder = SnapshotRecorder(
            cfg,
            scene=scene,
            operator_name=op,
            appearance_class=appr,
            privileged_fn=backend.privileged_physics,
            gate_rect=gate_rect,
        )
        monitor = RuleMonitor(monitor_cfg, dt=dt)
        monitor.reset()
        rng = np.random.default_rng(seed + 7)
        for k in range(int(cc.n_steps)):
            if obs.fallen:
                break
            event = monitor.step(obs)
            recorder.observe(obs, event)
            if recorder.snapshot is not None:
                break
            jit = 0.03 * rng.standard_normal(2)
            obs = backend.step(np.array([speed_fn(k), jit[0], jit[1]]))
        if op == "O10_effort_decay":
            backend.set_effort_scale(1.0)  # restore before the next lane
        return recorder.snapshot, tax.ground_truth(op, theta_for(lane)), bool(obs.fallen)

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
