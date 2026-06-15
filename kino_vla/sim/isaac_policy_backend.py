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
    ResistanceRegion,
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


def _set_material_friction(prim_path: str, mu_s: float, mu_d: float) -> None:
    """Mutate a spawned plate's PhysX friction at runtime (O3 collapse collider-swap)."""
    import omni.usd
    from pxr import Usd

    stage = omni.usd.get_context().get_stage()
    root = stage.GetPrimAtPath(prim_path)
    for prim in Usd.PrimRange(root):
        static_attr = prim.GetAttribute("physics:staticFriction")
        dynamic_attr = prim.GetAttribute("physics:dynamicFriction")
        if static_attr.IsValid() and dynamic_attr.IsValid():
            static_attr.Set(float(mu_s))
            dynamic_attr.Set(float(mu_d))
            return
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
        self._resistance: list[dict] = []
        self._collapse: list[dict] = []  # O3 trigger-and-swap collider states
        self._blocking: list[BlockingRegion] = []  # O8 invisible colliders
        self._support: list[SupportLossRegion] = []  # O9 high-centering regions
        self._payload_kg = 0.0  # O5 attached payload
        self._effort_scale = 1.0  # O10 actuator-effort fraction
        self._nominal_effort: dict = {}  # actuator effort limits before O10 scaling
        self._effort_limit_nm = float(cfg.effort_limit_nm)  # current (O10-scaled) torque cap
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
        # Trunk body for the O2/O4 resistance/tether external wrench (spec §8.2 Axis I/II).
        base_ids, base_names = self._robot.find_bodies("base")
        self._base_id = base_ids if base_ids else [0]
        print(
            f"[isaac] policy backend (env-driven): {len(self._foot_ids)} feet {foot_names}; "
            f"wrench body {base_names or '[body0]'}"
        )
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
        for state in getattr(self, "_resistance", []):
            state["path_len"] = 0.0
            state["broken"] = False
            state["inside"] = False

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
        # O2/O4 tangential-resistance/tether wrench on the trunk (re-applied each control
        # step; persists across the env's physics substeps via write_data_to_sim).
        self._apply_resistance_wrench()
        self._update_collapse()  # O3: swap intact→collapsed friction on dwell
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
        """Measured actuator-effort saturation above a floor (O5/O10 channel on Isaac).

        Saturation is the applied torque relative to the *current* effort cap, which O10
        scales down — so a decayed budget drives the same locomotion torque demand toward
        the cap and the saturation reading rises (it would fall if measured against the
        fixed nominal cap, since the cap itself clamps the torque)."""
        tau = self._torch.abs(self._robot.data.applied_torque[0])
        sat = float((tau.mean()).item()) / max(1e-6, self._effort_limit_nm)
        floor = float(self._cfg.effort_sat_floor)
        return float(np.clip((sat - floor) / max(1e-6, 1.0 - floor), 0.0, 1.0))

    def _measure_support(self) -> float:
        """Fraction of feet bearing load (O9 high-centering channel): 1.0 = all four planted."""
        if not self._contact_foot_ids:
            return 1.0
        forces = self._contact.data.net_forces_w[0, self._contact_foot_ids]
        thr = float(self._cfg.slip_force_threshold_n)
        in_contact = self._torch.linalg.norm(forces, dim=1) > thr
        return float(int(in_contact.sum().item()) / len(self._contact_foot_ids))

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
            support_ratio=self._measure_support(),
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

    def privileged_physics(self) -> dict[str, float]:
        """God's-eye physics truth at the current step — the Kino-Tokens M4 regression
        target on the real Go2 (spec §4): μ at the CoM ground projection (O1/O3), attached
        payload (O5), actuator-effort fraction (O10), and measured foot-support (O9)."""
        env_origin = self._env.scene.env_origins[0].cpu().numpy()
        pos = self._robot.data.root_pos_w[0].cpu().numpy()[:2] - env_origin[:2]
        return {
            "mu": self.friction_at(pos),
            "payload_kg": float(self._payload_kg),
            "effort_scale": float(self._effort_scale),
            "support_ratio": self._measure_support(),
        }

    def friction_at(self, pos: np.ndarray) -> float:
        for region in self._regions:
            if region.rect.contains(pos):
                return region.mu_d
        for st in self._collapse:
            if st["region"].contains(pos):
                return st["mu_collapsed"] if st["collapsed"] else st["mu_intact"]
        return float(self._cfg.mu_nominal)

    def apply_push(self, impulse_xy_ns: np.ndarray, yaw_impulse_nms: float) -> None:
        """Apply a real base-velocity impulse to the physically-simulated Go2 (O6)."""
        root_state = self._robot.data.root_state_w.clone()
        dv = np.asarray(impulse_xy_ns, dtype=np.float64) / self.mass_kg
        root_state[0, 7] += float(dv[0])
        root_state[0, 8] += float(dv[1])
        root_state[0, 12] += float(yaw_impulse_nms) / float(self._cfg.yaw_inertia_kgm2)
        self._robot.write_root_state_to_sim(root_state)

    def add_collapse_regions(self, regions: list[CollapseRegion]) -> None:
        """Spawn an intact-μ plate per region; its PhysX friction is swapped to the
        collapsed value at runtime once the Go2 dwells on it (O3 collider-swap + hysteresis)."""
        env_origin = self._env.scene.env_origins[0].cpu().numpy()
        for region in regions:
            prim_path = f"/World/collapse_{self._n_patches}"
            self._n_patches += 1
            thickness = float(self._cfg.patch_thickness_m)
            plate_cfg = self._sim_utils.CuboidCfg(
                size=(2.0 * region.rect.hx, 2.0 * region.rect.hy, thickness),
                collision_props=self._sim_utils.CollisionPropertiesCfg(),
                physics_material=self._sim_utils.RigidBodyMaterialCfg(
                    static_friction=region.mu_intact,
                    dynamic_friction=region.mu_intact,
                    friction_combine_mode="min",
                ),
                visual_material=self._sim_utils.PreviewSurfaceCfg(diffuse_color=(0.7, 0.9, 1.0)),
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
            mu_s_applied, _ = _read_material_friction(prim_path)
            print(f"[isaac] O3 collapse plate: intact mu={mu_s_applied:.3f} (swaps on dwell)")
            self._collapse.append(
                {
                    "region": region.rect,
                    "prim_path": prim_path,
                    "mu_intact": region.mu_intact,
                    "mu_collapsed": region.mu_collapsed,
                    "trigger_dwell_s": region.trigger_dwell_s,
                    "dwell": 0.0,
                    "collapsed": False,
                }
            )

    def _update_collapse(self) -> None:
        """Swap intact→collapsed friction once the Go2 has dwelled past the threshold (O3)."""
        if not self._collapse:
            return
        env_origin = self._env.scene.env_origins[0].cpu().numpy()
        pos = self._robot.data.root_pos_w[0].cpu().numpy()[:2] - env_origin[:2]
        for st in self._collapse:
            if st["collapsed"] or not st["region"].contains(pos):
                continue
            st["dwell"] += self.dt
            if st["dwell"] >= st["trigger_dwell_s"]:
                _set_material_friction(st["prim_path"], st["mu_collapsed"], st["mu_collapsed"])
                st["collapsed"] = True
                print(f"[isaac] O3 collapse triggered: mu -> {st['mu_collapsed']:.3f}")

    def add_blocking_regions(self, regions: list[BlockingRegion]) -> None:
        """Spawn a tall collision wall per region — a real PhysX collider the Go2 cannot
        cross (O8 invisible collider; the visual is faint, render-suppression is cosmetic)."""
        env_origin = self._env.scene.env_origins[0].cpu().numpy()
        h_wall = float(getattr(self._cfg, "wall_height_m", 0.8))
        for region in regions:
            prim_path = f"/World/wall_{self._n_patches}"
            self._n_patches += 1
            wall_cfg = self._sim_utils.CuboidCfg(
                size=(2.0 * region.rect.hx, 2.0 * region.rect.hy, h_wall),
                collision_props=self._sim_utils.CollisionPropertiesCfg(),
                physics_material=self._sim_utils.RigidBodyMaterialCfg(
                    static_friction=1.0, dynamic_friction=1.0
                ),
                visual_material=self._sim_utils.PreviewSurfaceCfg(
                    diffuse_color=(0.9, 0.9, 0.9), opacity=0.15
                ),
            )
            wall_cfg.func(
                prim_path,
                wall_cfg,
                translation=(
                    region.rect.cx + float(env_origin[0]),
                    region.rect.cy + float(env_origin[1]),
                    float(env_origin[2]) + h_wall / 2.0,
                ),
            )
            print(f"[isaac] O8 collider wall at ({region.rect.cx:.2f}, {region.rect.cy:.2f})")
            self._blocking.append(region)

    def add_support_loss_regions(
        self, regions: list[SupportLossRegion], height_m: float | None = None
    ) -> None:
        """Spawn a low ridge per region: the Go2 high-centers (belly grounds, feet partially
        clear), so the measured foot-support fraction drops (O9 high-centering).

        ``height_m`` overrides the config ridge height for this call (taller lip ⇒ more feet
        unloaded ⇒ lower support); ``None`` keeps ``cfg.ridge_height_m`` so existing callers
        (the O9 operator, the M2 gate) are unchanged. The M4 gate uses it to span a range of
        graded support levels for the support-channel regression."""
        env_origin = self._env.scene.env_origins[0].cpu().numpy()
        h_ridge = float(
            height_m if height_m is not None else getattr(self._cfg, "ridge_height_m", 0.18)
        )
        for region in regions:
            prim_path = f"/World/ridge_{self._n_patches}"
            self._n_patches += 1
            # A narrow ridge along y so the mid-body grounds while feet straddle/clear it.
            ridge_cfg = self._sim_utils.CuboidCfg(
                size=(2.0 * region.rect.hx, 2.0 * region.rect.hy, h_ridge),
                collision_props=self._sim_utils.CollisionPropertiesCfg(),
                physics_material=self._sim_utils.RigidBodyMaterialCfg(
                    static_friction=0.8, dynamic_friction=0.8
                ),
                visual_material=self._sim_utils.PreviewSurfaceCfg(diffuse_color=(0.8, 0.6, 0.4)),
            )
            ridge_cfg.func(
                prim_path,
                ridge_cfg,
                translation=(
                    region.rect.cx + float(env_origin[0]),
                    region.rect.cy + float(env_origin[1]),
                    float(env_origin[2]) + h_ridge / 2.0,
                ),
            )
            print(f"[isaac] O9 high-centering ridge at ({region.rect.cx:.2f},{region.rect.cy:.2f})")
            self._support.append(region)

    def add_resistance_regions(self, regions: list[ResistanceRegion]) -> None:
        """Register O2/O4 tangential-resistance regions applied as a base external wrench.

        Spec §8.2 names a D6 spring-damper for O4 and compliant contact for O2; on the
        physically-simulated Go2 we apply the *equivalent* Hooke's-law restoring/viscous
        wrench (F = k·s + c·|v|, opposing motion, with O4's break force) directly to the
        trunk via the PhysX external-force API. This is numerically robust (no runtime joint
        prim creation/destruction) and produces the real measurable effect — a tracking-error
        / velocity deficit the monitor and the M5 map see. The wrench is applied in
        ``_policy_step`` whenever the trunk is inside a region (see deviation #20).
        """
        for r in regions:
            self._resistance.append(
                {"region": r, "path_len": 0.0, "broken": False, "inside": False}
            )

    def _apply_resistance_wrench(self) -> None:
        """Apply the O2/O4 base wrench for the region underfoot, else clear it (PhysX)."""
        if not self._resistance:
            return
        torch = self._torch
        env_origin = self._env.scene.env_origins[0].cpu().numpy()
        pos_w = self._robot.data.root_pos_w[0].cpu().numpy()
        pos = pos_w[:2] - env_origin[:2]
        vel_b = self._robot.data.root_lin_vel_b[0].cpu().numpy()[:2]
        speed = float(np.linalg.norm(vel_b))
        force_b = np.zeros(3)
        for state in self._resistance:
            region = state["region"]
            if not region.rect.contains(pos):
                state["inside"] = False
                continue
            if not state["inside"]:
                state["inside"] = True  # path length accrues from region entry
            s_eff = max(0.0, state["path_len"] - region.slack_length_m)
            mag = region.stiffness_n_per_m * s_eff + region.damping_ns_per_m * speed
            if mag > region.break_force_n:
                state["broken"] = True
            if not state["broken"] and speed > 1e-6:
                # Oppose the body-frame planar velocity (Hooke's-law + viscous drag).
                force_b[:2] += -(vel_b / speed) * mag
            state["path_len"] += speed * self.dt
        forces = torch.from_numpy(force_b.reshape(1, 1, 3).astype(np.float32)).to(
            self._device
        )  # (env=1, body=1, 3)
        torques = torch.zeros((1, 1, 3), dtype=torch.float32, device=self._device)
        # is_global=False ⇒ the wrench is expressed in the trunk body frame (matches vel_b).
        # Older Isaac Lab lacks the kwarg (defaulting to local-frame), so fall back to it.
        try:
            self._robot.set_external_force_and_torque(
                forces, torques, body_ids=self._base_id, is_global=False
            )
        except TypeError:
            self._robot.set_external_force_and_torque(forces, torques, body_ids=self._base_id)

    def add_payload(self, mass_kg: float, com_offset_m: np.ndarray) -> None:
        """Add rigidly-attached mass to the trunk via the PhysX mass API (O5 payload)."""
        view = self._robot.root_physx_view
        masses = view.get_masses().clone()  # (num_instances, num_links), CPU
        base = self._base_id[0]
        masses[0, base] += float(mass_kg)
        try:
            view.set_masses(masses, self._torch.tensor([0]))
        except TypeError:
            view.set_masses(masses)
        self._payload_kg += float(mass_kg)
        self.mass_kg = float(masses.sum())
        print(f"[isaac] O5 payload: +{mass_kg:.2f} kg on trunk; total mass {self.mass_kg:.2f} kg")

    def set_effort_scale(self, scale: float) -> None:
        """Scale every actuator's effort limit (O10 effort-decay); 1.0 restores nominal."""
        scale = float(scale)
        # Scale every torque-cap attribute the actuator clamps against. Go2 uses an explicit
        # DCMotor actuator whose compute() clips to effort_limit AND shapes torque by
        # saturation_effort, so both must shrink for the decay to bind.
        for name, act in self._robot.actuators.items():
            if name not in self._nominal_effort:
                self._nominal_effort[name] = {
                    attr: getattr(act, attr).clone()
                    for attr in ("effort_limit", "saturation_effort")
                    if hasattr(act, attr) and hasattr(getattr(act, attr), "clone")
                }
            for attr, nominal in self._nominal_effort[name].items():
                setattr(act, attr, nominal * scale)
        self._effort_scale = scale
        self._effort_limit_nm = float(self._cfg.effort_limit_nm) * scale

    def set_reflex(self, active: bool) -> None:  # noqa: B027
        # Reflex stance on the real robot would crouch/widen; the M2 survival gate runs
        # on the surrogate, so this is a no-op placeholder on the Isaac path.
        pass
