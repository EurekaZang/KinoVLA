"""Isaac Lab Go2 backend driven by the trained velocity-tracking policy (M2).

This replaces the M1 kinematic root drive (CLAUDE.md Section 6 #5/#7): the Go2 is
physically simulated and walked by the RSL-RL policy trained in
``scripts/train_locomotion.py`` (exported JIT at ``outputs/locomotion/policy.pt``).

The Go2 runs inside the *same* ``ManagerBasedRLEnv`` (the flat velocity task) it was
trained in, so the physics, the 48-dim observation, and the JointPositionAction
scale/offset are bit-for-bit identical to training — a hand-built SimulationContext
did not reproduce the env's contact/solver fidelity and the policy degenerated to a
crawl. Domain randomization is switched off for inference (fixed friction/mass, no
obs noise, no command resampling); Sport-Client velocity commands are written
straight onto the policy's ``base_velocity`` command term.

Slip is *measured* from physics: a load-bearing foot (net contact force above a
threshold) that slides horizontally is slipping, so ``slip_ratio`` rises on the O1
ice patch because the real PhysX feet really slide. Operator O1's μ is read back from
the spawned PhysX material (θ-application, QA 5.2a). GPU-only; one episode per process
at M2.
"""

from __future__ import annotations

import math

import numpy as np

from kino_vla.sim.types import (
    BlockingRegion,
    CollapseRegion,
    FrictionRegion,
    Obs,
    SupportLossRegion,
)
from kino_vla.utils.config import REPO_ROOT, Config
from kino_vla.utils.geometry import wrap_angle


def _read_material_friction(prim_path: str) -> tuple[float, float]:
    """Read (static, dynamic) friction back from the spawned physics material."""
    import omni.usd
    from pxr import Usd

    stage = omni.usd.get_context().get_stage()
    root = stage.GetPrimAtPath(prim_path)
    for prim in Usd.PrimRange(root):
        static_attr = prim.GetAttribute("physics:staticFriction")
        dynamic_attr = prim.GetAttribute("physics:dynamicFriction")
        if static_attr.IsValid() and dynamic_attr.IsValid():
            return float(static_attr.Get()), float(dynamic_attr.Get())
    raise RuntimeError(f"no physics material attributes found under {prim_path}")


class IsaacPolicyBackend:
    """Go2 in the flat velocity env, walked by the trained RSL-RL policy (M2 deliverable)."""

    def __init__(
        self,
        cfg: Config,
        start_pos: np.ndarray,
        start_heading: float,
        record_cam: bool = False,
    ) -> None:
        import isaaclab.sim as sim_utils
        import torch
        from isaaclab.envs import ManagerBasedRLEnv
        from isaaclab_tasks.manager_based.locomotion.velocity.config.go2.flat_env_cfg import (
            UnitreeGo2FlatEnvCfg,
        )

        self._cfg = cfg
        self._torch = torch
        self._sim_utils = sim_utils
        self._device = str(cfg.device)
        self._action_scale = float(cfg.action_scale)
        self._start_pos = np.asarray(start_pos, dtype=np.float64).copy()
        self._start_heading = float(start_heading)
        self._regions: list[FrictionRegion] = []
        self._n_patches = 0
        self._record_cam = bool(record_cam)
        self._camera = None

        env_cfg = UnitreeGo2FlatEnvCfg()
        env_cfg.scene.num_envs = 1
        env_cfg.sim.device = self._device
        env_cfg.seed = 0
        # Inference: no domain randomization, no obs noise, no command resampling.
        env_cfg.observations.policy.enable_corruption = False
        env_cfg.events.add_base_mass = None
        env_cfg.events.physics_material.params["static_friction_range"] = (0.8, 0.8)
        env_cfg.events.physics_material.params["dynamic_friction_range"] = (0.6, 0.6)
        # reset_base randomization is irrelevant: _teleport_to_start overrides it.
        env_cfg.commands.base_velocity.heading_command = False
        env_cfg.commands.base_velocity.rel_standing_envs = 0.0  # never zero our command
        env_cfg.commands.base_velocity.resampling_time_range = (1.0e9, 1.0e9)
        env_cfg.episode_length_s = 1.0e6  # never truncate mid-demo
        self.dt = float(env_cfg.sim.dt * env_cfg.decimation)

        if self._record_cam:
            self._add_record_camera(env_cfg, cfg)

        self._env = ManagerBasedRLEnv(cfg=env_cfg)
        self._robot = self._env.scene["robot"]
        self._contact = self._env.scene["contact_forces"]
        self._cmd_term = self._env.command_manager.get_term("base_velocity")
        if self._record_cam:
            self._camera = self._env.scene["record_cam"]
            eye = self._torch.tensor([[3.0, -4.5, 7.5]], device=self._device)
            target = self._torch.tensor([[3.2, 0.3, 0.2]], device=self._device)
            self._camera.set_world_poses_from_view(eye, target)

        policy_path = REPO_ROOT / str(cfg.policy_path)
        if not policy_path.exists():
            raise FileNotFoundError(
                f"trained policy not found at {policy_path}; run scripts/train_locomotion.py"
            )
        self._policy = torch.jit.load(str(policy_path)).to(self._device).eval()

        self._foot_ids, foot_names = self._robot.find_bodies(".*_foot")
        self._contact_foot_ids, contact_foot_names = self._contact.find_bodies(".*_foot")
        # find_bodies sorts by name, so the articulation and sensor foot lists align.
        assert foot_names == contact_foot_names, (foot_names, contact_foot_names)
        print(f"[isaac] policy backend (env-driven): {len(self._foot_ids)} feet {foot_names}")
        self._spawn_z = float(self._robot.data.default_root_state[0, 2].cpu())
        try:
            self.mass_kg = float(self._robot.root_physx_view.get_masses().sum())
        except AttributeError:
            self.mass_kg = float(cfg.mass_fallback_kg)
        self._obs = None
        self._init_state()

    def _init_state(self) -> None:
        self._t = 0.0
        self._cmd_prev = np.zeros(3)
        self._slip = 0.0
        self._effort = 0.0
        self._fallen = False

    # ------------------------------------------------------------ episode API

    def reset(self, seed: int) -> Obs:
        obs_dict, _ = self._env.reset(seed=seed)
        self._obs = obs_dict["policy"]
        self._init_state()
        self._teleport_to_start()
        # Settle the stance with a zero command before the episode starts.
        for _ in range(int(self._cfg.settle_steps)):
            self._policy_step(np.zeros(3))
        self._init_state()
        return self._make_obs()

    def _teleport_to_start(self) -> None:
        root_state = self._robot.data.default_root_state.clone()
        env_origin = self._env.scene.env_origins[0]
        root_state[0, 0] = float(self._start_pos[0]) + float(env_origin[0])
        root_state[0, 1] = float(self._start_pos[1]) + float(env_origin[1])
        root_state[0, 2] = self._spawn_z
        half = self._start_heading / 2.0
        root_state[0, 3:7] = root_state.new_tensor([math.cos(half), 0.0, 0.0, math.sin(half)])
        root_state[0, 7:] = 0.0
        self._robot.write_root_state_to_sim(root_state)
        self._robot.reset()
        self._robot.update(self.dt)

    def step(self, cmd_vel: np.ndarray) -> Obs:
        if self._fallen:
            return self._make_obs()
        cmd = np.asarray(cmd_vel, dtype=np.float64).copy()
        cmd[:2] = np.clip(cmd[:2], -float(self._cfg.max_speed_mps), float(self._cfg.max_speed_mps))
        cmd[2] = float(
            np.clip(
                cmd[2], -float(self._cfg.max_yaw_rate_radps), float(self._cfg.max_yaw_rate_radps)
            )
        )
        self._cmd_prev = cmd
        terminated = self._policy_step(cmd)
        self._t += self.dt

        obs = self._make_obs()
        fell = (
            terminated
            or obs.base_height < float(self._cfg.fall_check.min_base_height_m)
            or obs.tilt > float(self._cfg.fall_check.max_tilt_rad)
        )
        if fell:
            self._fallen = True
            obs = self._make_obs()
        return obs

    def _policy_step(self, cmd: np.ndarray) -> bool:
        """Inject the command, run the actor, step the env one control step. Returns terminated."""
        torch = self._torch
        self._cmd_term.vel_command_b[:, 0] = float(cmd[0])
        self._cmd_term.vel_command_b[:, 1] = float(cmd[1])
        self._cmd_term.vel_command_b[:, 2] = float(cmd[2])
        # Refresh the obs so the policy sees the just-injected command.
        self._obs = self._env.observation_manager.compute()["policy"]
        with torch.no_grad():
            action = self._policy(self._obs)
        obs_dict, _, terminated, _truncated, _ = self._env.step(action)
        self._obs = obs_dict["policy"]
        self._slip = self._measure_slip()
        self._effort = self._measure_effort()
        return bool(terminated[0].item())

    def _measure_slip(self) -> float:
        """Measured contact slip: horizontal speed of *load-bearing* feet, normalized (spec §3).

        A foot is in stance when its net contact force exceeds a threshold (it bears
        load) — this is gait-height-independent, unlike a foot-z heuristic. A planted
        stance foot has ~0 horizontal speed on good ground; on ice the load-bearing
        foot slides, so the normalized horizontal speed is a genuine slip signal.
        """
        if not self._foot_ids:
            return 0.0
        forces = self._contact.data.net_forces_w[0, self._contact_foot_ids]  # (F, 3)
        foot_vel = self._robot.data.body_lin_vel_w[0, self._foot_ids]  # (F, 3), same order
        in_contact = self._torch.linalg.norm(forces, dim=1) > float(
            self._cfg.slip_force_threshold_n
        )
        n_contact = int(in_contact.sum().item())
        if n_contact == 0:
            return 0.0
        horiz_speed = self._torch.linalg.norm(foot_vel[:, :2], dim=1)
        slip = float((horiz_speed * in_contact).sum().item()) / n_contact
        return float(np.clip(slip / float(self._cfg.slip_ref_speed_mps), 0.0, 1.0))

    def _measure_effort(self) -> float:
        """Measured actuator-effort saturation above a floor (O5/O10 channel on Isaac)."""
        tau = self._torch.abs(self._robot.data.applied_torque[0])
        sat = float((tau.mean() / float(self._cfg.effort_limit_nm)).item())
        floor = float(self._cfg.effort_sat_floor)
        return float(np.clip((sat - floor) / max(1e-6, 1.0 - floor), 0.0, 1.0))

    def _make_obs(self) -> Obs:
        data = self._robot.data
        env_origin = self._env.scene.env_origins[0].cpu().numpy()
        pos_w = data.root_pos_w[0].cpu().numpy()
        pos = (pos_w[:2] - env_origin[:2]).astype(np.float64)  # report in the start frame
        quat = data.root_quat_w[0].cpu().numpy()
        w, x, y, z = (float(quat[0]), float(quat[1]), float(quat[2]), float(quat[3]))
        heading = wrap_angle(math.atan2(2.0 * (w * z + x * y), 1.0 - 2.0 * (y * y + z * z)))
        cos_tilt = float(np.clip(1.0 - 2.0 * (x * x + y * y), -1.0, 1.0))
        vel_b = data.root_lin_vel_b[0].cpu().numpy()
        return Obs(
            t=self._t,
            pos=pos,
            heading=heading,
            vel_body=vel_b[:2].astype(np.float64),
            yaw_rate=float(data.root_ang_vel_b[0, 2].cpu()),
            cmd_prev=self._cmd_prev.copy(),
            slip_ratio=self._slip,
            base_height=float(pos_w[2] - env_origin[2]),
            tilt=math.acos(cos_tilt),
            fallen=self._fallen,
            effort_ratio=self._effort,
        )

    # ------------------------------------------------------------ recording

    def _add_record_camera(self, env_cfg: object, cfg: Config) -> None:
        """Add a fixed wide-shot RGB camera to the scene (passive; for video recording)."""
        from isaaclab.sensors import CameraCfg

        env_cfg.scene.record_cam = CameraCfg(
            prim_path="/World/record_cam",
            update_period=0.0,
            height=int(cfg.cam_height),
            width=int(cfg.cam_width),
            data_types=["rgb"],
            spawn=self._sim_utils.PinholeCameraCfg(focal_length=20.0, clipping_range=(0.1, 1.0e4)),
        )

    def capture_rgb(self) -> np.ndarray | None:
        """Latest camera RGB as an ``(H, W, 3)`` uint8 array (None if not recording)."""
        if self._camera is None:
            return None
        self._camera.update(self.dt)
        rgb = self._camera.data.output["rgb"][0][..., :3]
        if rgb.dtype.is_floating_point:
            rgb = rgb.clamp(0.0, 1.0) * 255.0
        return rgb.to(self._torch.uint8).cpu().numpy()

    # ------------------------------------------------------------ operator API

    def add_friction_regions(self, regions: list[FrictionRegion]) -> None:
        """Spawn one thin static collider plate per region with its PhysX material (O1)."""
        env_origin = self._env.scene.env_origins[0].cpu().numpy()
        for region in regions:
            prim_path = f"/World/patch_{self._n_patches}"
            self._n_patches += 1
            thickness = float(self._cfg.patch_thickness_m)
            plate_cfg = self._sim_utils.CuboidCfg(
                size=(2.0 * region.rect.hx, 2.0 * region.rect.hy, thickness),
                collision_props=self._sim_utils.CollisionPropertiesCfg(),
                physics_material=self._sim_utils.RigidBodyMaterialCfg(
                    static_friction=region.mu_s,
                    dynamic_friction=region.mu_d,
                    restitution=region.restitution,
                    friction_combine_mode="min",
                ),
                visual_material=self._sim_utils.PreviewSurfaceCfg(diffuse_color=(0.6, 0.8, 1.0)),
            )
            plate_cfg.func(
                prim_path,
                plate_cfg,
                translation=(
                    region.rect.cx + float(env_origin[0]),
                    region.rect.cy + float(env_origin[1]),
                    float(env_origin[2]) + thickness / 2.0,
                ),
            )
            mu_s_applied, mu_d_applied = _read_material_friction(prim_path)
            print(
                f"[isaac] O1 patch material applied: set mu_d={region.mu_d:.3f} "
                f"readback mu_d={mu_d_applied:.3f}"
            )
            self._regions.append(
                FrictionRegion(
                    rect=region.rect,
                    mu_s=mu_s_applied,
                    mu_d=mu_d_applied,
                    restitution=region.restitution,
                )
            )

    def friction_at(self, pos: np.ndarray) -> float:
        for region in self._regions:
            if region.rect.contains(pos):
                return region.mu_d
        return float(self._cfg.mu_nominal)

    def apply_push(self, impulse_xy_ns: np.ndarray, yaw_impulse_nms: float) -> None:
        """Apply a real base-velocity impulse to the physically-simulated Go2 (O6)."""
        root_state = self._robot.data.root_state_w.clone()
        dv = np.asarray(impulse_xy_ns, dtype=np.float64) / self.mass_kg
        root_state[0, 7] += float(dv[0])
        root_state[0, 8] += float(dv[1])
        root_state[0, 12] += float(yaw_impulse_nms) / float(self._cfg.yaw_inertia_kgm2)
        self._robot.write_root_state_to_sim(root_state)

    # M2 operator hooks whose PhysX mechanism lands post-M2 (surrogate-validated now).
    def _isaac_deferred(self, name: str) -> None:
        raise NotImplementedError(
            f"{name} has no Isaac mechanism at M2 (validated on the surrogate); "
            "the Isaac demo exercises O1/O6 only."
        )

    def add_collapse_regions(self, regions: list[CollapseRegion]) -> None:
        self._isaac_deferred("O3 add_collapse_regions")

    def add_blocking_regions(self, regions: list[BlockingRegion]) -> None:
        self._isaac_deferred("O8 add_blocking_regions")

    def add_support_loss_regions(self, regions: list[SupportLossRegion]) -> None:
        self._isaac_deferred("O9 add_support_loss_regions")

    def add_payload(self, mass_kg: float, com_offset_m: np.ndarray) -> None:
        self._isaac_deferred("O5 add_payload")

    def set_effort_scale(self, scale: float) -> None:
        self._isaac_deferred("O10 set_effort_scale")

    def set_reflex(self, active: bool) -> None:  # noqa: B027
        # Reflex stance on the real robot would crouch/widen; the M2 survival gate runs
        # on the surrogate, so this is a no-op placeholder on the Isaac path.
        pass
