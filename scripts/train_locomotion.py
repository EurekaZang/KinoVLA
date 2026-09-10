"""M2 low-level locomotion: train a Go2 flat velocity-tracking policy (RSL-RL PPO).

Replaces the M1 kinematic root drive with a real
trained policy. Mirrors IsaacLab's ``scripts/reinforcement_learning/rsl_rl/train.py``
wiring (env_cfg -> gym.make -> RslRlVecEnvWrapper -> OnPolicyRunner.learn) but is
in-repo and config-driven (``configs/locomotion/go2_flat_ppo.yaml``) so the policy
is reproducible from one config + seed (QA 5.2, learned components). Domain
randomization (friction buckets, base mass, periodic push) is applied on top of the
stock flat env so the policy survives the O1 ice patch and resists pushes.

The exported TorchScript actor (``<out_dir>/policy.pt``, plus the per-run copy) is
loaded open-loop by ``kino_vla.sim.isaac_policy_backend`` to drive the Go2 from
Sport-Client velocity commands.

Usage (GPU machine, headless):
    python scripts/train_locomotion.py --headless
    python scripts/train_locomotion.py --headless --num_envs 1024 --max_iterations 300
"""

from __future__ import annotations

import argparse
import hashlib
import importlib.util
import json
import os
import shutil
import sys
from datetime import UTC, datetime
from pathlib import Path


def _isaac_available() -> bool:
    return importlib.util.find_spec("isaaclab") is not None


def _sha256(path: str | Path) -> str:
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def main() -> int:
    parser = argparse.ArgumentParser(description="Train Go2 flat velocity-tracking policy")
    parser.add_argument("--config", default="locomotion/go2_flat_ppo.yaml")
    parser.add_argument("--num_envs", type=int, default=None, help="override config num_envs")
    parser.add_argument("--max_iterations", type=int, default=None, help="override config iters")
    parser.add_argument("--seed", type=int, default=None, help="override config seed")

    if not _isaac_available():
        print("SKIP: isaaclab not importable on this machine (CPU-only dev tier).")
        return 0

    # AppLauncher owns --headless/--device and must start the app before any other
    # isaaclab import (isaac-lab-dev convention).
    from isaaclab.app import AppLauncher

    AppLauncher.add_app_launcher_args(parser)
    args = parser.parse_args()
    app_launcher = AppLauncher(args)
    simulation_app = app_launcher.app  # noqa: F841 (keeps the app alive)

    import gymnasium as gym
    import isaaclab_tasks  # noqa: F401  (registers the velocity tasks)
    import isaaclab_tasks.manager_based.locomotion.velocity.mdp as mdp
    import torch
    from isaaclab.managers import EventTermCfg as EventTerm
    from isaaclab.managers import RewardTermCfg as RewTerm
    from isaaclab.managers import SceneEntityCfg
    from isaaclab_rl.rsl_rl import (
        RslRlVecEnvWrapper,
        export_policy_as_jit,
        export_policy_as_onnx,
    )
    from isaaclab_tasks.manager_based.locomotion.velocity.config.go2.agents.rsl_rl_ppo_cfg import (
        UnitreeGo2FlatPPORunnerCfg,
    )
    from isaaclab_tasks.manager_based.locomotion.velocity.config.go2.flat_env_cfg import (
        UnitreeGo2FlatEnvCfg,
    )
    from rsl_rl.runners import OnPolicyRunner

    from kino_vla.utils.config import CONFIGS_DIR, REPO_ROOT, load_config
    from kino_vla.utils.seeding import seed_everything

    cfg = load_config(args.config)
    config_path = Path(args.config)
    if not config_path.is_absolute():
        config_path = CONFIGS_DIR / config_path
    config_path = config_path.resolve()
    num_envs = args.num_envs if args.num_envs is not None else int(cfg.num_envs)
    max_iter = args.max_iterations if args.max_iterations is not None else int(cfg.max_iterations)
    seed = args.seed if args.seed is not None else int(cfg.seed)
    device = str(cfg.device)
    seed_everything(seed)

    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True

    # --- env cfg with KinoVLA domain randomization -------------------------------
    env_cfg = UnitreeGo2FlatEnvCfg()
    env_cfg.scene.num_envs = num_envs
    env_cfg.sim.device = device
    env_cfg.seed = seed
    dr = cfg.dr
    fric = env_cfg.events.physics_material.params
    fric["static_friction_range"] = tuple(dr.static_friction_range)
    fric["dynamic_friction_range"] = tuple(dr.dynamic_friction_range)
    fric["restitution_range"] = tuple(dr.restitution_range)
    env_cfg.events.add_base_mass.params["mass_distribution_params"] = tuple(dr.add_base_mass_range)
    if bool(dr.enable_push):
        pv = float(dr.push_velocity_range[1])
        env_cfg.events.push_robot = EventTerm(
            func=mdp.push_by_setting_velocity,
            mode="interval",
            interval_range_s=tuple(dr.push_interval_s),
            params={
                "asset_cfg": SceneEntityCfg("robot"),
                "velocity_range": {"x": (-pv, pv), "y": (-pv, pv)},
            },
        )

    # Reward shaping: kill the skating gait (feet sliding in contact -> friction-robust
    # low crawl) so the trained policy plants its feet and genuinely slips on ice.
    env_cfg.rewards.feet_slide = RewTerm(
        func=mdp.feet_slide,
        weight=float(cfg.rewards.feet_slide_weight),
        params={
            "sensor_cfg": SceneEntityCfg("contact_forces", body_names=".*_foot"),
            "asset_cfg": SceneEntityCfg("robot", body_names=".*_foot"),
        },
    )
    env_cfg.rewards.feet_air_time.weight = float(cfg.rewards.feet_air_time_weight)
    optional_reward_weights = {
        "track_lin_vel_xy_exp": cfg.rewards.get("track_lin_vel_xy_weight", None),
        "track_ang_vel_z_exp": cfg.rewards.get("track_ang_vel_z_weight", None),
        "lin_vel_z_l2": cfg.rewards.get("lin_vel_z_weight", None),
        "ang_vel_xy_l2": cfg.rewards.get("ang_vel_xy_weight", None),
        "action_rate_l2": cfg.rewards.get("action_rate_weight", None),
        "flat_orientation_l2": cfg.rewards.get("flat_orientation_weight", None),
    }
    for term_name, weight in optional_reward_weights.items():
        if weight is not None:
            getattr(env_cfg.rewards, term_name).weight = float(weight)

    # In-place yaw-turn command curriculum (user directive): train the SUSTAINED yaw command the
    # deployed backend issues. heading_command=False ⇒ ang_vel_z is sampled and HELD across the
    # resample window (the stock heading_command=True decays it to ~0 as the base aligns, so a
    # sustained in-place spin was never trained); widen ang_vel_z to the backend yaw clamp
    # (max_yaw_rate_radps=1.5). No obs-shape change (the command stays a 3-vector), so the trained
    # actor is drop-in for the inference backend. configs/locomotion/go2_flat_ppo.yaml :: command.
    cmdc = cfg.get("command", None)
    if cmdc is not None:
        bv = env_cfg.commands.base_velocity
        bv.heading_command = bool(cmdc.heading_command)
        bv.rel_standing_envs = float(cmdc.rel_standing_envs)
        bv.ranges.lin_vel_x = tuple(cmdc.lin_vel_x)
        bv.ranges.lin_vel_y = tuple(cmdc.lin_vel_y)
        bv.ranges.ang_vel_z = tuple(cmdc.ang_vel_z)
        print(
            f"[train] cmd curriculum: heading_command={bv.heading_command} "
            f"ang_vel_z={tuple(cmdc.ang_vel_z)} lin_x={tuple(cmdc.lin_vel_x)} "
            f"lin_y={tuple(cmdc.lin_vel_y)} rel_standing={bv.rel_standing_envs}"
        )

    # --- agent cfg ---------------------------------------------------------------
    agent_cfg = UnitreeGo2FlatPPORunnerCfg()
    agent_cfg.max_iterations = max_iter
    agent_cfg.seed = seed
    agent_cfg.device = device
    agent_cfg.experiment_name = str(cfg.experiment_name)
    agent_cfg.run_name = f"seed{seed}"

    out_dir = REPO_ROOT / str(cfg.out_dir)
    log_dir = out_dir / f"{cfg.experiment_name}_seed{seed}"
    os.makedirs(log_dir, exist_ok=True)
    config_snapshot = log_dir / "config_snapshot.yaml"
    if config_snapshot.exists():
        raise FileExistsError(f"refusing to overwrite training config snapshot: {config_snapshot}")
    shutil.copyfile(config_path, config_snapshot)

    # --- build, train ------------------------------------------------------------
    env = gym.make(str(cfg.task), cfg=env_cfg, render_mode=None)
    env = RslRlVecEnvWrapper(env, clip_actions=agent_cfg.clip_actions)
    runner = OnPolicyRunner(env, agent_cfg.to_dict(), log_dir=str(log_dir), device=device)
    print(f"[train] task={cfg.task} num_envs={num_envs} max_iter={max_iter} seed={seed}")
    print(
        f"[train] DR: static_fric={tuple(dr.static_friction_range)} "
        f"dyn_fric={tuple(dr.dynamic_friction_range)} push={bool(dr.enable_push)}"
    )
    runner.learn(num_learning_iterations=max_iter, init_at_random_ep_len=True)

    # --- save + export the inference actor (mirrors play.py) ----------------------
    final_ckpt = str(log_dir / "model_final.pt")
    runner.save(final_ckpt)
    policy_nn = runner.alg.policy
    normalizer = getattr(policy_nn, "actor_obs_normalizer", None)
    export_dir = str(log_dir / "exported")
    export_policy_as_jit(policy_nn, normalizer=normalizer, path=export_dir, filename="policy.pt")
    export_policy_as_onnx(policy_nn, normalizer=normalizer, path=export_dir, filename="policy.onnx")
    # Stable path the backend loads regardless of run/seed.
    stable_policy = out_dir / str(cfg.export_filename)
    if stable_policy.exists():
        raise FileExistsError(f"refusing to overwrite stable policy: {stable_policy}")
    shutil.copyfile(os.path.join(export_dir, "policy.pt"), stable_policy)

    reward_contract = {
        "feet_slide": float(env_cfg.rewards.feet_slide.weight),
        "feet_air_time": float(env_cfg.rewards.feet_air_time.weight),
    }
    for term_name in optional_reward_weights:
        reward_contract[term_name] = float(getattr(env_cfg.rewards, term_name).weight)
    training_manifest = {
        "schema_version": "kinovla.locomotion-training-manifest.v1",
        "created_utc": datetime.now(UTC).isoformat(),
        "passed": True,
        "task": str(cfg.task),
        "experiment_name": str(cfg.experiment_name),
        "seed": seed,
        "num_envs": num_envs,
        "max_iterations": max_iter,
        "device": device,
        "command_contract": cfg.command.to_dict() if cfg.get("command", None) is not None else None,
        "domain_randomization": cfg.dr.to_dict(),
        "reward_contract": reward_contract,
        "artifacts": {
            "config_snapshot": {
                "path": str(config_snapshot.resolve()),
                "sha256": _sha256(config_snapshot),
            },
            "training_script": {
                "path": str(Path(__file__).resolve()),
                "sha256": _sha256(Path(__file__).resolve()),
            },
            "final_checkpoint": {
                "path": str(Path(final_ckpt).resolve()),
                "sha256": _sha256(final_ckpt),
            },
            "exported_policy": {
                "path": str(Path(export_dir, "policy.pt").resolve()),
                "sha256": _sha256(Path(export_dir, "policy.pt")),
            },
            "exported_onnx": {
                "path": str(Path(export_dir, "policy.onnx").resolve()),
                "sha256": _sha256(Path(export_dir, "policy.onnx")),
            },
            "stable_policy": {
                "path": str(stable_policy.resolve()),
                "sha256": _sha256(stable_policy),
            },
        },
        "evidence_boundary": {
            "counts_as_a0_a7_evidence": False,
            "realistic_a0_a7_readiness": "0/8",
            "requires_post_training_route_gates": True,
        },
    }
    training_manifest_path = out_dir / "training_manifest.json"
    training_manifest_path.write_text(
        json.dumps(training_manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )

    print(f"[train] final checkpoint: {final_ckpt}")
    print(f"[train] exported JIT policy: {os.path.join(export_dir, 'policy.pt')}")
    print(f"[train] stable policy path: {stable_policy}")
    print(f"[train] manifest: {training_manifest_path}")
    print("PASS: locomotion training")

    # Isaac Sim 5.1 close() busy-spins on this headless setup; the artifacts above
    # are already on disk, so flush and force-exit.
    sys.stdout.flush()
    os._exit(0)


if __name__ == "__main__":
    sys.exit(main())
