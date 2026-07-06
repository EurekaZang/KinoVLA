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
from kino_vla.utils.geometry import Rect, wrap_angle


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
        perception_cam: bool = False,
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
        self._last_deep_reset_removed = 0  # A0.1: prims removed by the last deep_reset (diagnostic)
        self._record_cam = bool(record_cam)
        self._camera = None
        self._perception_cam_on = bool(perception_cam)
        self._perception_cam = None
        self._perception_eye: np.ndarray | None = None
        self._perception_target: np.ndarray | None = None

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
        if self._perception_cam_on:
            self._add_perception_camera(env_cfg, cfg)

        self._env = ManagerBasedRLEnv(cfg=env_cfg)
        self._robot = self._env.scene["robot"]
        self._contact = self._env.scene["contact_forces"]
        self._cmd_term = self._env.command_manager.get_term("base_velocity")
        if self._record_cam:
            self._camera = self._env.scene["record_cam"]
            eye = self._torch.tensor([[3.0, -4.5, 7.5]], device=self._device)
            target = self._torch.tensor([[3.2, 0.3, 0.2]], device=self._device)
            self._camera.set_world_poses_from_view(eye, target)
        if self._perception_cam_on:
            self._perception_cam = self._env.scene["perception_cam"]
            self._disable_tonemap()

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
        self._setup_posture_controller()
        self._obs = None
        self._init_state()

    def _setup_posture_controller(self) -> None:
        """Build the closed-loop body-height controller (#41): the leg-extension residual direction
        in the policy's action space (thigh/calf joints), so a commanded posture is physically
        tracked by the real Go2. Discovers the thigh/calf joint indices (the action order follows
        the articulation joint order for the flat env's JointPositionAction) and assembles the flex
        vector ``calf_gain`` on calves + ``thigh_gain`` on thighs, scaled by the global ``sign``."""
        torch = self._torch
        p = self._cfg.posture
        n_joints = int(self._robot.data.default_joint_pos.shape[1])
        thigh_ids, _ = self._robot.find_joints(".*thigh.*")
        calf_ids, _ = self._robot.find_joints(".*calf.*")
        flex = torch.zeros(n_joints, dtype=torch.float32, device=self._device)
        if thigh_ids:
            flex[thigh_ids] = float(p.thigh_gain)
        if calf_ids:
            flex[calf_ids] = float(p.calf_gain)
        self._flex_dir = float(p.sign) * flex  # (n_joints,) action-space RAISE direction
        self._posture_alpha_gain = float(p.alpha_gain)
        self._posture_alpha_max = float(p.alpha_max)
        self._posture_release_decay = float(p.release_decay)
        self._posture_speed_scale = float(p.speed_scale)
        self._brace_height_m = float(p.brace_height_m)
        print(f"[isaac] posture controller: {len(thigh_ids)} thigh + {len(calf_ids)} calf joints")

    def _init_state(self) -> None:
        self._t = 0.0
        self._cmd_prev = np.zeros(3)
        self._slip = 0.0
        self._effort = 0.0
        self._fallen = False
        # Closed-loop posture state (#41): None ⇒ nominal trot (policy owns the body).
        self._posture_target: float | None = None
        self._posture_stiffness = 1.0
        self._reflex_posture_active = False
        self._posture_alpha = 0.0  # I-control flex residual magnitude (carried across steps)
        for state in getattr(self, "_resistance", []):
            state["path_len"] = 0.0
            state["broken"] = False
            state["inside"] = False
            state["entry"] = None
            state["pen_prev"] = 0.0
            state["max_pen"] = 0.0
            state["max_grip"] = 0.0
            state["broke_this_step"] = False
            state["phase"] = "none"

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

    # ------------------------------------------------- A0.1 determinism-grade reset

    def deep_reset(self, seed: int) -> Obs:
        """Determinism-grade reset (A0.1): scrub EVERY operator residue before the standard
        reset, so lane N is independent of every operator that ran in lanes 0..N-1 — the
        #51/#52 operator-ORDER PhysX residual behind the E1 C2ST confound, the E2 Suite-Cal
        non-reproducibility, and the E4 closed-loop order-sensitivity.

        Additive + opt-in: the deployed :meth:`reset` is byte-for-byte untouched (red-line
        discipline); only the A0 determinism/collection harness calls this. It (1) zeros the
        persistent PhysX external-wrench buffer (``_apply_resistance_wrench`` early-returns
        when ``_resistance`` is empty, so it never self-clears on reset), (2) restores the
        trunk mass (O5) and actuator effort caps (O10), (3) despawns every operator collider/
        visual/material prim so the accumulating collider set can no longer change PhysX
        broadphase/warm-start ORDER across lanes, (4) empties the region containers, (5)
        flushes the contact-sensor buffers, then runs the standard env reset + settle on
        now-clean ground. See A实验/A0.md §A0.1.
        """
        self._zero_external_wrench()  # explicit: the wrench path early-returns when _resistance=[]
        self.clear_payload()  # O5 → nominal trunk mass
        self.set_effort_scale(1.0)  # O10 → nominal actuator caps
        n_removed = self._despawn_operator_prims()  # O1/O3/O7/O8/O9 colliders + visual plates
        self._regions = []
        self._resistance = []
        self._collapse = []
        self._blocking = []
        self._support = []
        contact_reset = getattr(self._contact, "reset", None)
        if callable(contact_reset):
            try:
                contact_reset()  # flush contact-sensor net-force buffers
            except (RuntimeError, TypeError, ValueError):  # surface nothing; best-effort flush
                pass
        self._last_deep_reset_removed = int(n_removed)
        return self.reset(seed)

    def _zero_external_wrench(self) -> None:
        """Zero the persistent PhysX external force/torque buffer on the trunk (O2/O4 wrench)."""
        torch = self._torch
        z = torch.zeros((1, 1, 3), dtype=torch.float32, device=self._device)
        try:
            self._robot.set_external_force_and_torque(z, z, body_ids=self._base_id, is_global=False)
        except TypeError:
            self._robot.set_external_force_and_torque(z, z, body_ids=self._base_id)

    def _despawn_operator_prims(self) -> int:
        """Delete every operator-spawned prim under ``/World`` and reset the patch counter.

        Operator colliders/visuals are spawned at ``/World/{patch,collapse,wall,ridge,vplate,
        patchtex}_<n>`` (+ the textured-patch material at ``/World/Looks/patchtex_<n>``), with
        ``_n_patches`` the monotonic name counter. Nothing despawns them today, so they pile up
        in the shared PhysX scene across lanes and reorder broadphase/warm-start — the dominant
        determinism residue. Removing them + zeroing the counter restores a first-lane-clean
        stage. Returns the number of prims removed."""
        import omni.usd

        stage = omni.usd.get_context().get_stage()
        prefixes = ("patch_", "collapse_", "wall_", "ridge_", "vplate_", "patchtex_")
        removed = 0
        world = stage.GetPrimAtPath("/World")
        if world.IsValid():
            for child in list(world.GetChildren()):
                if child.GetName().startswith(prefixes) and stage.RemovePrim(child.GetPath()):
                    removed += 1
        looks = stage.GetPrimAtPath("/World/Looks")
        if looks.IsValid():
            for child in list(looks.GetChildren()):
                if child.GetName().startswith("patchtex_") and stage.RemovePrim(child.GetPath()):
                    removed += 1
        self._n_patches = 0
        return removed

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
        speed_factor = self._posture_speed_factor()  # slow forward speed while holding a posture
        self._cmd_term.vel_command_b[:, 0] = float(cmd[0]) * speed_factor
        self._cmd_term.vel_command_b[:, 1] = float(cmd[1]) * speed_factor
        self._cmd_term.vel_command_b[:, 2] = float(cmd[2])
        # Refresh the obs so the policy sees the just-injected command.
        self._obs = self._env.observation_manager.compute()["policy"]
        with torch.no_grad():
            action = self._policy(self._obs)
        # Closed-loop posture: ADD the leg-extension residual so the trunk physically tracks the
        # commanded height (#41) — the dog-executed response to a posture/gait/Set_Constraint
        # primitive (and the real §6.8 brace). Zero residual ⇒ the nominal trot is untouched.
        action = action + self._posture_residual()
        # O2/O4 tangential-resistance/tether wrench on the trunk (re-applied each control
        # step; persists across the env's physics substeps via write_data_to_sim).
        self._apply_resistance_wrench()
        self._update_collapse()  # O3: swap intact→collapsed friction on dwell
        if self._perception_cam_on:
            self._auto_aim_perception()  # aim the ground-perception cam BEFORE the render
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

    def aim_record_camera(self, eye_xyz: np.ndarray, target_xyz: np.ndarray) -> None:
        """Re-aim the record camera (eye, target in the start frame) — M5 perception (spec §7)."""
        if self._camera is None:
            raise RuntimeError("record camera not enabled (construct with record_cam=True)")
        origin = self._env.scene.env_origins[0]
        eye = (
            self._torch.tensor(
                [[float(eye_xyz[0]), float(eye_xyz[1]), float(eye_xyz[2])]], device=self._device
            )
            + origin
        )
        target = (
            self._torch.tensor(
                [[float(target_xyz[0]), float(target_xyz[1]), float(target_xyz[2])]],
                device=self._device,
            )
            + origin
        )
        self._camera.set_world_poses_from_view(eye, target)

    def add_visual_plate(
        self, center_xy: np.ndarray, rgb: tuple[float, float, float], half_size_m: float = 0.8
    ) -> str:
        """Spawn a flat colour-rendered material plate (M5 real-perception encoder, spec §7).

        A thin static cuboid with a PreviewSurface diffuse colour, so the record camera sees
        the *rendered* material — the pixel appearance encoder consumes real RGB instead of a
        class label. Returns the prim path."""
        env_origin = self._env.scene.env_origins[0].cpu().numpy()
        prim_path = f"/World/vplate_{self._n_patches}"
        self._n_patches += 1
        thickness = 0.02
        plate_cfg = self._sim_utils.CuboidCfg(
            size=(2.0 * half_size_m, 2.0 * half_size_m, thickness),
            collision_props=self._sim_utils.CollisionPropertiesCfg(),
            physics_material=self._sim_utils.RigidBodyMaterialCfg(
                static_friction=0.8, dynamic_friction=0.8
            ),
            visual_material=self._sim_utils.PreviewSurfaceCfg(
                diffuse_color=tuple(float(c) for c in rgb), roughness=0.9
            ),
        )
        plate_cfg.func(
            prim_path,
            plate_cfg,
            translation=(
                float(center_xy[0]) + float(env_origin[0]),
                float(center_xy[1]) + float(env_origin[1]),
                float(env_origin[2]) + thickness / 2.0,
            ),
        )
        return prim_path

    # ------------------------------------------------------------ §7 live perception camera

    def _add_perception_camera(self, env_cfg: object, cfg: Config) -> None:
        """Add a robot-following down-looking RGB-D + semantic-seg camera (spec §7 perception).

        Real pixels (rgb), real depth (distance_to_image_plane) and per-pixel semantic ids
        (semantic_segmentation, raw ids — colorize off) so a failure region is CLIP-labelled from
        real pixels and back-projected to odometry-frame geometry via the measured depth."""
        from isaaclab.sensors import CameraCfg

        env_cfg.scene.perception_cam = CameraCfg(
            prim_path="/World/perception_cam",
            update_period=0.0,
            height=int(cfg.get("perception_cam_height", 180)),
            width=int(cfg.get("perception_cam_width", 240)),
            data_types=["rgb", "distance_to_image_plane", "semantic_segmentation"],
            colorize_semantic_segmentation=False,  # raw uint32 ids + idToLabels mapping
            spawn=self._sim_utils.PinholeCameraCfg(
                focal_length=float(cfg.get("perception_focal_mm", 18.0)),
                clipping_range=(0.05, 30.0),
            ),
        )

    def _disable_tonemap(self) -> None:
        """Albedo-faithful render so CLIP sees the texture, not a tonemapped frame (#28)."""
        import carb

        s = carb.settings.get_settings()
        for key in ("/rtx/post/tonemap/enabled", "/rtx/post/histogram/enabled"):
            s.set_bool(key, False)

    def aim_perception_camera(self, eye_xyz: np.ndarray, target_xyz: np.ndarray) -> None:
        """Re-aim the perception camera (eye, target in the start frame)."""
        if self._perception_cam is None:
            raise RuntimeError("perception camera not enabled (construct with perception_cam=True)")
        origin = self._env.scene.env_origins[0]
        eye = self._torch.tensor([[float(v) for v in eye_xyz]], device=self._device) + origin
        target = self._torch.tensor([[float(v) for v in target_xyz]], device=self._device) + origin
        self._perception_cam.set_world_poses_from_view(eye, target)
        self._perception_eye = np.asarray(eye_xyz, dtype=np.float64)
        self._perception_target = np.asarray(target_xyz, dtype=np.float64)

    def _auto_aim_perception(self) -> None:
        """Aim the ground-perception camera ahead of the robot (behind+above looking forward-down,
        the vantage the Stage-A probe validated), in the start/odometry frame."""
        data = self._robot.data
        env_origin = self._env.scene.env_origins[0].cpu().numpy()
        pos = data.root_pos_w[0].cpu().numpy()[:2] - env_origin[:2]
        q = data.root_quat_w[0].cpu().numpy()  # (w, x, y, z)
        yaw = math.atan2(2.0 * (q[0] * q[3] + q[1] * q[2]), 1.0 - 2.0 * (q[2] ** 2 + q[3] ** 2))
        fwd = np.array([math.cos(yaw), math.sin(yaw)])
        eye = np.array([pos[0] - 0.8 * fwd[0], pos[1] - 0.8 * fwd[1], 1.2])
        target = np.array([pos[0] + 1.6 * fwd[0], pos[1] + 1.6 * fwd[1], 0.0])
        self.aim_perception_camera(eye, target)

    def add_textured_patch(
        self, rect: Rect, material_name: str, semantic_label: str, *, lift_m: float = 0.03
    ) -> str:
        """Spawn a flat textured + semantically-tagged quad at a world Rect so the perception
        camera sees a real surface to CLIP-segment + depth-back-project (§7)."""
        import omni.usd
        from PIL import Image
        from pxr import Gf, Sdf, UsdGeom, UsdShade

        from kino_vla.map.clip_segmentation import material_texture

        stage = omni.usd.get_context().get_stage()
        env_origin = self._env.scene.env_origins[0].cpu().numpy()
        tex_dir = REPO_ROOT / "outputs" / "map" / "rtx_patch_tex"
        tex_dir.mkdir(parents=True, exist_ok=True)
        name = f"patchtex_{self._n_patches}"
        self._n_patches += 1
        png = str(tex_dir / f"{name}_{material_name}.png")
        Image.fromarray(
            (material_texture(material_name, seed=0, size=256) * 255).astype("uint8")
        ).save(png)

        mtl = UsdShade.Material.Define(stage, f"/World/Looks/{name}")
        pbr = UsdShade.Shader.Define(stage, f"/World/Looks/{name}/PBR")
        pbr.CreateIdAttr("UsdPreviewSurface")
        pbr.CreateInput("roughness", Sdf.ValueTypeNames.Float).Set(0.9)
        pbr.CreateInput("metallic", Sdf.ValueTypeNames.Float).Set(0.0)
        reader = UsdShade.Shader.Define(stage, f"/World/Looks/{name}/stReader")
        reader.CreateIdAttr("UsdPrimvarReader_float2")
        reader.CreateInput("varname", Sdf.ValueTypeNames.Token).Set("st")
        tex = UsdShade.Shader.Define(stage, f"/World/Looks/{name}/diffuseTex")
        tex.CreateIdAttr("UsdUVTexture")
        tex.CreateInput("file", Sdf.ValueTypeNames.Asset).Set(png)
        tex.CreateInput("st", Sdf.ValueTypeNames.Float2).ConnectToSource(
            reader.ConnectableAPI(), "result"
        )
        for w in ("wrapS", "wrapT"):
            tex.CreateInput(w, Sdf.ValueTypeNames.Token).Set("repeat")
        tex.CreateOutput("rgb", Sdf.ValueTypeNames.Float3)
        pbr.CreateInput("diffuseColor", Sdf.ValueTypeNames.Color3f).ConnectToSource(
            tex.ConnectableAPI(), "rgb"
        )
        mtl.CreateSurfaceOutput().ConnectToSource(pbr.ConnectableAPI(), "surface")

        prim_path = f"/World/{name}"
        mesh = UsdGeom.Mesh.Define(stage, prim_path)
        cx, cy = float(rect.cx + env_origin[0]), float(rect.cy + env_origin[1])
        z = float(env_origin[2]) + float(lift_m)
        hx, hy = float(rect.hx), float(rect.hy)
        mesh.CreatePointsAttr(
            [
                Gf.Vec3f(cx - hx, cy - hy, z),
                Gf.Vec3f(cx + hx, cy - hy, z),
                Gf.Vec3f(cx + hx, cy + hy, z),
                Gf.Vec3f(cx - hx, cy + hy, z),
            ]
        )
        mesh.CreateFaceVertexCountsAttr([4])
        mesh.CreateFaceVertexIndicesAttr([0, 1, 2, 3])
        mesh.CreateNormalsAttr([Gf.Vec3f(0, 0, 1)] * 4)
        mesh.SetNormalsInterpolation("vertex")
        mesh.CreateDoubleSidedAttr(True)
        mesh.CreateSubdivisionSchemeAttr("none")
        st = UsdGeom.PrimvarsAPI(mesh).CreatePrimvar(
            "st", Sdf.ValueTypeNames.TexCoord2fArray, UsdGeom.Tokens.faceVarying
        )
        st.Set([(0, 0), (1, 0), (1, 1), (0, 1)])
        UsdShade.MaterialBindingAPI(mesh.GetPrim()).Bind(mtl)
        try:
            from isaacsim.core.utils.semantics import add_update_semantics
        except ImportError:  # older Isaac
            from omni.isaac.core.utils.semantics import add_update_semantics
        add_update_semantics(mesh.GetPrim(), semantic_label=semantic_label, type_label="class")
        return prim_path

    def capture_perception(self) -> dict | None:
        """Real rgb + depth + semantic-seg (+ intrinsics, world pose, env origin) for §7
        back-projection. Returns numpy arrays, or None if the perception camera is off."""
        if self._perception_cam is None:
            return None
        cam = self._perception_cam
        cam.update(self.dt)
        out = cam.data.output

        def _np(x: object) -> np.ndarray:
            return x[0].detach().cpu().numpy()

        seg = np.squeeze(_np(out["semantic_segmentation"])).astype(np.int64)
        return {
            "rgb": _np(out["rgb"])[..., :3],
            "depth": np.squeeze(_np(out["distance_to_image_plane"])).astype(np.float64),
            "seg": seg,
            "id_to_labels": cam.data.info[0]["semantic_segmentation"]["idToLabels"],
            "K": _np(cam.data.intrinsic_matrices),
            "pos": cam.data.pos_w[0].detach().cpu().numpy(),
            "quat_ros": cam.data.quat_w_ros[0].detach().cpu().numpy(),
            "quat_world": cam.data.quat_w_world[0].detach().cpu().numpy(),
            "quat_opengl": cam.data.quat_w_opengl[0].detach().cpu().numpy(),
            "env_origin": self._env.scene.env_origins[0].detach().cpu().numpy(),
            "eye": self._perception_eye,
            "target": self._perception_target,
        }

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
        # A4.1 two-phase consequence telemetry: the max adhesive grip + penetration reached this
        # episode, whether the tether tore (the catapult mechanism), and the current grip phase.
        # Privileged ground truth for labeling M(s,ℓ) outcomes — the AGENT never sees this.
        tether = self._tether_summary(pos)
        return {
            "mu": self.friction_at(pos),
            "payload_kg": float(self._payload_kg),
            "effort_scale": float(self._effort_scale),
            "support_ratio": self._measure_support(),
            "tether_grip_n": tether["grip_n"],
            "tether_pen_m": tether["pen_m"],
            "tether_broken": tether["broken"],
            "tether_phase": tether["phase_code"],
        }

    def _tether_summary(self, pos: np.ndarray) -> dict[str, float]:
        """Max adhesive grip/penetration underfoot + break/phase (A4.1 privileged telemetry)."""
        _PHASE_CODE = {"none": 0.0, "compliance": 0.0, "plateau": 1.0, "ramp": 2.0, "spring": 3.0}
        grip_n = 0.0
        pen_m = 0.0
        broken = 0.0
        phase_code = 0.0
        for state in self._resistance:
            region = state["region"]
            if region.kind == "compliance" or not region.rect.contains(pos):
                continue
            if state["max_grip"] > grip_n:
                grip_n = state["max_grip"]
            if state["max_pen"] > pen_m:
                pen_m = state["max_pen"]
            if state["broken"]:
                broken = 1.0
            phase_code = max(phase_code, _PHASE_CODE.get(state["phase"], 0.0))
        return {"grip_n": grip_n, "pen_m": pen_m, "broken": broken, "phase_code": phase_code}

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
                {
                    "region": r,
                    "path_len": 0.0,
                    "broken": False,
                    "inside": False,
                    "entry": None,
                    "pen_prev": 0.0,
                    # A4.1 two-phase telemetry (privileged ground truth for consequence labeling):
                    # the max grip [N] + penetration [m] reached this episode, whether the tether
                    # tore this step (catapult mechanism), and the current phase. Reset each reset.
                    "max_pen": 0.0,
                    "max_grip": 0.0,
                    "broke_this_step": False,
                    "phase": "none",
                }
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
            state["broke_this_step"] = False  # per-step catapult marker (privileged ground truth)
            if not region.rect.contains(pos):
                state["inside"] = False
                state["entry"] = None  # left the patch ⇒ the grip resets (the foot peels off)
                state["pen_prev"] = 0.0
                state["phase"] = "none"
                continue
            state["inside"] = True
            if region.kind == "compliance":
                # Soft-ground DRAG FIELD (mud): bounded constant + viscous drag, crossable (§6 #38);
                # stiffness_n_per_m is the constant drag [N] for the compliance kind.
                mag = region.stiffness_n_per_m + region.damping_ns_per_m * speed
                state["phase"] = "compliance"
            else:
                # O4 adhesive GRIP (entry-point form, HEADING-INDEPENDENT): the hold grows with the
                # distance from where the dog ENTERED. It resists going DEEPER (push-through STALLS)
                # and peels (peel_factor) backing toward entry, so back-off escapes whatever way
                # the dog faces.
                if state["entry"] is None:
                    state["entry"] = pos.copy()
                pen = float(np.linalg.norm(pos - state["entry"]))
                moving_out = pen < state["pen_prev"] - 1.0e-4
                state["pen_prev"] = pen
                if region.p0_m > 0.0:
                    # A4.1 TWO-PHASE delayed-divergence (experiments_design.md §4 A4.1): a constant
                    # PLATEAU grip = force_offset_n (≡ O2 compliance's k_c drag, byte-identical, for
                    # pen≤p0_m) then a linear RAMP force_offset_n + k2·(pen−p0) beyond.
                    # where attribution happens (C2ST-indistinguishable from O2 by construction;
                    # A1.3 re-certifies it); the ramp is the CONSEQUENCE region. f_break fires on
                    # ramp grip ⇒ finite tears under forward lean (catapult energy release); inf +
                    # high k2 grows without bound (immobilization).
                    grip = region.force_offset_n + region.k2_n_per_m * max(0.0, pen - region.p0_m)
                    state["phase"] = "plateau" if pen <= region.p0_m else "ramp"
                else:
                    # #49 peel-plateau (E1 calibration; default inf/0 ⇒ unshaped, byte-identical).
                    grip = region.stiffness_n_per_m * max(0.0, pen - region.slack_length_m)
                    if grip > region.break_force_n:
                        state["broken"] = True
                    grip = min(grip, region.force_cap_n) + region.force_offset_n
                    state["phase"] = "spring"
                if not state["broken"] and grip > region.break_force_n:
                    state["broken"] = True
                    state["broke_this_step"] = True  # the catapult mechanism (privileged truth)
                mag = grip * (region.peel_factor if moving_out else 1.0)
                mag += region.damping_ns_per_m * speed
                if not state["broken"]:
                    if pen > state["max_pen"]:
                        state["max_pen"] = pen
                    if grip > state["max_grip"]:
                        state["max_grip"] = grip
            if not state["broken"] and speed > 1e-6:
                force_b[:2] += -(vel_b / speed) * mag
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

    def clear_payload(self) -> None:
        """Strip any attached O5 payload, restoring the trunk's nominal mass.

        ``reset()`` does NOT restore PhysX masses, so a payload added in one lane persists into
        the next (#22). The M6 data collection drives many O5 lanes (and O5 mixed with other ops)
        in one process, so each lane must start payload-free to log the mass it actually set —
        otherwise the cumulative ``add_payload`` (+=) compounds across lanes. No-op when unloaded;
        independent of the M4 token-gate's deliberate ascending-payload sweep (which never clears).
        """
        if self._payload_kg == 0.0:
            return
        view = self._robot.root_physx_view
        masses = view.get_masses().clone()
        base = self._base_id[0]
        masses[0, base] -= float(self._payload_kg)
        try:
            view.set_masses(masses, self._torch.tensor([0]))
        except TypeError:
            view.set_masses(masses)
        self._payload_kg = 0.0
        self.mass_kg = float(masses.sum())

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

    def set_reflex(self, active: bool) -> None:
        """Engage the §6.8 fallback Reflex: crouch the trunk to the brace height (#41). Previously a
        no-op on Isaac — now a real physical crouch via the closed-loop posture controller, so the
        robot's actual support/capture stance matches the brace mode the shield adjudicated against.
        Takes priority over a planner-commanded posture while the fallback is active."""
        self._reflex_posture_active = bool(active)

    def set_posture(self, height_m: float | None, stiffness: float = 1.0) -> None:
        """Command a target trunk height the Go2 physically tracks (#41) — the dog-executed response
        to Switch_Gait / Adjust_Posture / Set_Constraint. ``None`` releases to the nominal trot (the
        residual decays out). ``stiffness`` scales the I-control gain (a stiffer hold tracks faster
        and holds tighter). The reflex brace (set_reflex) overrides this while active."""
        self._posture_target = None if height_m is None else float(height_m)
        self._posture_stiffness = float(stiffness)

    def _effective_posture_target(self) -> float | None:
        """The height the controller drives to this step: the brace crouch when the shield-fallback
        Reflex is active (safety wins), else the planner-commanded posture, else None (nominal)."""
        if self._reflex_posture_active:
            return self._brace_height_m
        return self._posture_target

    def _posture_residual(self) -> object:
        """The flex residual (n_joints,) to ADD to the policy action this step so the trunk tracks
        the commanded height (I-control on the MEASURED height; #41). Returns a zero/decaying
        residual when no posture is active, so release is smooth. Closed-loop ⇒ precise regardless
        of the exact leg kinematics; ``_flex_dir``'s sign defines which way raises the trunk."""
        target = self._effective_posture_target()
        if target is None:
            self._posture_alpha *= self._posture_release_decay  # smooth return to the policy pose
            return self._posture_alpha * self._flex_dir
        env_origin_z = float(self._env.scene.env_origins[0, 2].cpu())
        base_h = float(self._robot.data.root_pos_w[0, 2].cpu()) - env_origin_z
        stiff = float(np.clip(self._posture_stiffness, 0.1, 2.0))
        gain = self._posture_alpha_gain * stiff * self.dt
        self._posture_alpha = float(
            np.clip(
                self._posture_alpha + gain * (target - base_h),
                -self._posture_alpha_max,
                self._posture_alpha_max,
            )
        )
        return self._posture_alpha * self._flex_dir

    def _posture_speed_factor(self) -> float:
        """Cap forward speed while actively holding a posture (blended-residual stability)."""
        active = self._effective_posture_target() is not None
        return self._posture_speed_scale if active else 1.0
