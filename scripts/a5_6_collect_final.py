#!/usr/bin/env python
"""Collect the untouched A5.6 final-appearance validation corpus in Isaac.

The collection is additive: it writes dedicated A5.6 main/T3 corpora and never changes the frozen
A0/A3 corpora.  Appearance IDs, colors, sample counts, seeds, and the deployed gate are frozen in
``configs/eval/c4_final_appearances.yaml`` before this script is run.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import subprocess
import sys
from collections import deque
from pathlib import Path
from typing import Any

import numpy as np


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _git_commit() -> str:
    try:
        return (
            subprocess.check_output(
                ["git", "rev-parse", "--short", "HEAD"], stderr=subprocess.DEVNULL
            )
            .decode()
            .strip()
        )
    except Exception:  # noqa: BLE001
        return "unknown"


def main() -> int:
    parser = argparse.ArgumentParser(description="A5.6 untouched final-appearance collection")
    from isaaclab.app import AppLauncher

    AppLauncher.add_app_launcher_args(parser)
    parser.add_argument("--config", default="eval/c4_final_appearances.yaml")
    parser.add_argument("--main-out", default="outputs/eval/a5/c4_final_main")
    parser.add_argument("--t3-out", default="outputs/eval/a5/c4_final_t3")
    parser.add_argument("--quick", action="store_true")
    parser.add_argument("--steps", type=int, default=300)
    parser.add_argument("--binding-t", type=int, default=100)
    parser.add_argument("--arm", type=float, default=0.2)
    parser.add_argument("--lane-y", type=float, default=4.0)
    parser.add_argument("--cruise", type=float, default=0.6)
    parser.add_argument("--speed-hi", type=float, default=1.0)
    parser.add_argument("--speed-lo", type=float, default=0.1)
    parser.add_argument("--speed-period", type=float, default=1.0)
    parser.add_argument("--speed-duty", type=float, default=0.5)
    args = parser.parse_args()
    app = AppLauncher(args).app  # noqa: F841

    from kino_vla.data.snapshot import SnapshotRecorder
    from kino_vla.eval.oracle_trigger import OracleTrigger
    from kino_vla.eval.registry import load_registry
    from kino_vla.map.rgbd import register_appearance_colors
    from kino_vla.map.types import SemanticRegion
    from kino_vla.sim.isaac_policy_backend import IsaacPolicyBackend
    from kino_vla.sim.operators import OperatorStack
    from kino_vla.utils.config import REPO_ROOT, load_config
    from kino_vla.vla import scenarios as S

    config = load_config(args.config).to_dict()
    config_path = REPO_ROOT / "configs" / args.config
    provenance_key = "method_manifest" if "method_manifest" in config else "gate_config"
    provenance_path = (
        REPO_ROOT / str(config[provenance_key])
        if provenance_key == "method_manifest"
        else REPO_ROOT / "configs" / str(config[provenance_key])
    )
    appearances = {
        semantic: [dict(item) for item in entries]
        for semantic, entries in config["classes"].items()
    }
    register_appearance_colors(
        {
            str(item["id"]): tuple(float(value) for value in item["rgb"])
            for entries in appearances.values()
            for item in entries
        }
    )
    registry = load_registry()
    hcfg = load_config("data/hindsight.yaml")
    backend = IsaacPolicyBackend(
        load_config("sim/go2_skeleton.yaml"), np.array([0.0, args.lane_y]), 0.0
    )
    dt = backend.dt
    seed_base = int(config["seed_base"])
    sample_prefix = str(config.get("sample_prefix", "a5c4f"))

    main_dir = REPO_ROOT / args.main_out
    t3_dir = REPO_ROOT / args.t3_out
    for directory in (main_dir, t3_dir):
        directory.mkdir(parents=True, exist_ok=True)
        sample_path = directory / "samples.jsonl"
        if sample_path.exists():
            sample_path.unlink()

    frames: dict[str, dict[str, np.ndarray]] = {"main": {}, "t3": {}}
    records: dict[str, list[dict[str, Any]]] = {"main": [], "t3": []}
    counts = {"attempted": 0, "captured": 0}

    def read60() -> np.ndarray:
        observation = getattr(backend, "_obs", None)
        obs48 = (
            np.zeros(48, np.float32)
            if observation is None
            else np.asarray(observation[0].detach().cpu().numpy(), np.float32)
        )
        try:
            tau = np.asarray(
                backend._robot.data.applied_torque[0].detach().cpu().numpy(), np.float32
            )
        except Exception:  # noqa: BLE001
            tau = np.zeros(12, np.float32)
        return np.concatenate([obs48, tau])

    def collect(
        *,
        group: str,
        scenario_name: str,
        scenario: Any,
        spec: Any,
        appearance_id: str,
        seed: int,
        mu: float | None = None,
        direction: str | None = None,
    ) -> None:
        counts["attempted"] += 1
        scene = SemanticRegion(rect=scenario.scene_region.rect, appearance_class=appearance_id)
        operators = OperatorStack([scenario.operator])
        backend._start_pos = np.array([0.0, args.lane_y])
        backend._start_heading = 0.0
        obs = backend.deep_reset(seed)
        operators.on_reset(backend)
        oracle = OracleTrigger.for_scenario(scenario, dt=dt, arm_delay_s=args.arm)
        recorder = SnapshotRecorder(
            hcfg,
            scene=[scene],
            operator_name=spec.operator,
            appearance_class=appearance_id,
            privileged_fn=backend.privileged_physics,
            gate_rect=None,
        )
        binding = deque(maxlen=int(args.binding_t))
        for _ in range(args.steps):
            binding.append(read60())
            transformed = operators.transform_obs(obs)
            recorder.observe(transformed, oracle.step(transformed))
            if recorder.snapshot is not None:
                break
            operators.on_step(backend, obs.t)
            speed = args.cruise
            if group == "t3":
                fraction = (obs.t % args.speed_period) / args.speed_period
                speed = args.speed_hi if fraction < args.speed_duty else args.speed_lo
            obs = backend.step(np.array([speed, 0.0, 0.0]))
        snapshot = recorder.snapshot
        if snapshot is None:
            print(f"[a5.6] NO FIRE {scenario_name}/{appearance_id}/s{seed}", flush=True)
            return
        counts["captured"] += 1
        suffix = f"_mu{mu:g}" if mu is not None else ""
        sid = f"{sample_prefix}_{scenario_name}_{appearance_id}{suffix}_s{seed}"
        binding_array = np.asarray(list(binding), np.float32)
        target_mu = float(mu) if mu is not None else float(snapshot.privileged_theta.get("mu", 0.0))
        annotation = None
        if not spec.is_nominal and spec.canonical_recovery.primitive != "continue":
            annotation = {
                "thought": f"privileged final validation: {spec.true_category}",
                "attribution": spec.true_category,
                "attribution_raw": spec.true_category,
                "action": {
                    "primitive": spec.canonical_recovery.primitive,
                    "params": dict(spec.canonical_recovery.params),
                },
            }
        record = {
            "sample_id": sid,
            "taxonomy_cell": spec.taxonomy_cell,
            "appearance_id": appearance_id,
            "appearance_split": "final",
            "seed": seed,
            "pair_id": "O4_tether|O2_compliance" if scenario_name.startswith("matched_") else None,
            "ambiguity_pair": (
                "O4_tether|O2_compliance" if scenario_name.startswith("matched_") else None
            ),
            "success_criterion": spec.success_criterion,
            "admissible_recovery_set": sorted(spec.admissible_recovery_set),
            "snapshot": {**snapshot.to_meta(), "binding_shape": list(binding_array.shape)},
            "ground_truth": {
                "category": spec.true_category,
                "ab_class": spec.ab_class,
                "theta": {**dict(spec.theta), "mu": target_mu},
            },
            "annotation": annotation,
            "verdict": {
                "keep": True,
                "reason": "A5.6 untouched final validation",
                "detail": scenario_name,
            },
            "target_theta": [
                target_mu,
                float(snapshot.privileged_theta.get("payload_kg", 0.0)),
                float(snapshot.privileged_theta.get("effort_scale", 1.0)),
                float(snapshot.privileged_theta.get("support_ratio", 1.0)),
            ],
        }
        if group == "t3":
            record["a3_mu"] = target_mu
            record["a3_direction"] = direction or "looks_safe"
        records[group].append(record)
        out_dir = main_dir if group == "main" else t3_dir
        with (out_dir / "samples.jsonl").open("a") as handle:
            handle.write(json.dumps(record) + "\n")
        frames[group].update(
            {
                f"{sid}__rgb": snapshot.rgb.astype(np.float32),
                f"{sid}__depth": snapshot.depth.astype(np.float32),
                f"{sid}__proprio": snapshot.proprio_window.astype(np.float32),
                f"{sid}__binding": binding_array,
            }
        )

    builders = {
        "matched_O4": (lambda: S.o4_tether_matched(args.lane_y), "adhesion"),
        "matched_O2": (lambda: S.o2_compliance_matched(args.lane_y), "compliant_terrain"),
        "O1_ice": (lambda: S.o1_ice(args.lane_y), "low_friction"),
        "O1_A_nominal": (lambda: S.o1_ice_A(args.lane_y), "low_friction"),
        "O2_A_nominal": (lambda: S.o2_compliance_A(args.lane_y), "compliant_terrain"),
        "O6_push_A": (lambda: S.o6_push_A(args.lane_y), "solid_ground"),
        "O5_payload_B": (lambda: S.o5_payload(args.lane_y), "solid_ground"),
        "O10_decay_B": (lambda: S.o10_effort_decay(args.lane_y), "solid_ground"),
        "O8_invisible": (lambda: S.o8_invisible(args.lane_y), "solid_ground"),
    }
    main_counts = config["collection"]["main_seed_counts"]
    for scenario_name, (builder, semantic) in builders.items():
        if scenario_name not in main_counts:
            continue
        spec = registry[scenario_name]
        n_seeds = 1 if args.quick else int(main_counts[scenario_name])
        selected_appearances = appearances[semantic][:1] if args.quick else appearances[semantic]
        for appearance in selected_appearances:
            for offset in range(n_seeds):
                collect(
                    group="main",
                    scenario_name=scenario_name,
                    scenario=builder(),
                    spec=spec,
                    appearance_id=str(appearance["id"]),
                    seed=seed_base + offset,
                )
        print(f"[a5.6] main {scenario_name}: total captured={counts['captured']}", flush=True)

    spec = registry["O7_looks_safe"]
    mus = (
        [float(config["collection"]["o7_mu_sweep"][0])]
        if args.quick
        else [float(value) for value in config["collection"]["o7_mu_sweep"]]
    )
    n_t3_seeds = 1 if args.quick else int(config["collection"]["o7_seeds_per_cell"])
    selected_solid = appearances["solid_ground"][:1] if args.quick else appearances["solid_ground"]
    for mu in mus:
        for appearance in selected_solid:
            for offset in range(n_t3_seeds):
                collect(
                    group="t3",
                    scenario_name="O7_looks_safe",
                    scenario=S.o7_visual_remap(
                        args.lane_y,
                        mu_s=mu,
                        mu_d=mu,
                        appearance_class=str(appearance["id"]),
                    ),
                    spec=spec,
                    appearance_id=str(appearance["id"]),
                    seed=seed_base + 100 + offset,
                    mu=mu,
                    direction="looks_safe",
                )
        print(f"[a5.6] T3 mu={mu:g}: total captured={counts['captured']}", flush=True)

    reverse_count = int(config["collection"].get("reverse_seeds_per_cell", 0))
    if reverse_count:
        spec = registry["O7_reverse"]
        selected_hazards = (
            appearances["hazard_decal_benign"][:1]
            if args.quick
            else appearances["hazard_decal_benign"]
        )
        n_reverse = 1 if args.quick else reverse_count
        for appearance in selected_hazards:
            for offset in range(n_reverse):
                collect(
                    group="t3",
                    scenario_name="O7_reverse",
                    scenario=S.o7_visual_remap_reverse(
                        args.lane_y, appearance_class=str(appearance["id"])
                    ),
                    spec=spec,
                    appearance_id=str(appearance["id"]),
                    seed=seed_base + 200 + offset,
                    direction="reverse",
                )
        print(f"[a5.6] T3 reverse: total captured={counts['captured']}", flush=True)

    np.savez_compressed(main_dir / "frames.npz", **frames["main"])
    np.savez_compressed(t3_dir / "frames.npz", **frames["t3"])
    card = {
        "experiment": "A5.6-final",
        "commit": _git_commit(),
        "config": args.config,
        "config_sha256": _sha256(config_path),
        provenance_key: config[provenance_key],
        f"{provenance_key}_sha256": _sha256(provenance_path),
        "seed_base": seed_base,
        "attempted": counts["attempted"],
        "captured": counts["captured"],
        "main_snapshots": len(records["main"]),
        "t3_snapshots": len(records["t3"]),
        "appearance_split": "final",
        "determinism": "deep_reset(A0.1) + fixed lane_y",
        "quick": bool(args.quick),
    }
    for directory in (main_dir, t3_dir):
        (directory / "collection_card.json").write_text(json.dumps(card, indent=2) + "\n")
    expected_quick = len(main_counts) + 1 + int(reverse_count > 0)
    expected = (
        expected_quick if args.quick else int(config["collection"]["expected_total_snapshots"])
    )
    if counts["captured"] != expected:
        print(f"[a5.6] ERROR captured {counts['captured']} != expected {expected}", flush=True)
        sys.stdout.flush()
        os._exit(2)
    print(f"[a5.6] DONE {counts['captured']}/{expected} snapshots", flush=True)
    sys.stdout.flush()
    os._exit(0)


if __name__ == "__main__":
    main()
