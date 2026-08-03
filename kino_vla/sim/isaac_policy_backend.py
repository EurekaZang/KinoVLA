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
import os

import numpy as np

from kino_vla.sim.actuator import actuator_sample
from kino_vla.sim.adhesion import (
    FootAdhesionConfig,
    FootAdhesionState,
    FootAdhesionUpdate,
    step_foot_adhesion,
)
from kino_vla.sim.collapse import (
    CollapseDamageState,
    collapsed_cell_indices,
    damage_update_to_dict,
    step_collapse_damage,
    support_cell_layout,
)
from kino_vla.sim.payload import combine_with_cuboid_payload
from kino_vla.sim.proprio_pipeline import RawProprioPacket
from kino_vla.sim.terrain_materials import TerrainAppearanceBinding, bind_omnipbr_material
from kino_vla.sim.terramechanics import (
    ContinuousSupportMesh,
    FootTerrainState,
    FootTerrainUpdate,
    FootTerramechanicsConfig,
    SinkRampSpec,
    continuous_support_mesh,
    longitudinal_sink_profile,
    step_foot_terramechanics,
)
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


INFERENCE_RESET_SCHEMA_VERSION = "kinofail.isaac-go2-inference-reset.v1"


def _freeze_inference_joint_reset(env_cfg: object) -> dict[str, object]:
    """Make the effective Go2 joint-reset contract explicit and immutable.

    Isaac Lab's generic velocity-task parent uses a 0.5--1.5x pose range, while the current Go2
    specialization already overrides it to 1.0--1.0.  Record the effective pre-freeze value so
    an audit can distinguish a real intervention from a no-op assertion of the Go2 default.
    """
    events = getattr(env_cfg, "events", None)
    reset = getattr(events, "reset_robot_joints", None)
    params = getattr(reset, "params", None)
    if not isinstance(params, dict):
        raise RuntimeError("Isaac Go2 config lacks a mutable reset_robot_joints event")
    original_position_range = tuple(float(value) for value in params["position_range"])
    original_velocity_range = tuple(float(value) for value in params["velocity_range"])
    params["position_range"] = (1.0, 1.0)
    params["velocity_range"] = (0.0, 0.0)
    return {
        "schema_version": INFERENCE_RESET_SCHEMA_VERSION,
        "joint_position_rule": "exact_default_joint_positions",
        "joint_velocity_rule": "zero_joint_velocity",
        "position_scale_range": [1.0, 1.0],
        "velocity_scale_range": [0.0, 0.0],
        "observed_pre_freeze_position_scale_range": list(original_position_range),
        "observed_pre_freeze_velocity_scale_range": list(original_velocity_range),
        "seed_may_randomize_initial_joints": False,
        "explicit_scheduled_nuisance_required_for_future_pose_perturbation": True,
    }


def _contact_local_soil_offset_update(
    existing_offset_m: np.ndarray,
    x_m: np.ndarray,
    y_m: np.ndarray,
    *,
    center_xy_m: tuple[float, float],
    sinkage_m: float,
    radius_xy_m: tuple[float, float],
) -> np.ndarray:
    """Accumulate one smooth foot-local depression without marking the operator rectangle."""
    if sinkage_m <= 0.0 or min(radius_xy_m) <= 0.0:
        raise ValueError("contact-local soil deformation dimensions must be positive")
    existing = np.asarray(existing_offset_m, dtype=np.float64)
    x = np.asarray(x_m, dtype=np.float64)
    y = np.asarray(y_m, dtype=np.float64)
    if existing.shape != x.shape or x.shape != y.shape:
        raise ValueError("soil offset and coordinate arrays must have equal shape")
    rx, ry = (float(value) for value in radius_xy_m)
    rho_sq = ((x - float(center_xy_m[0])) / rx) ** 2 + (
        (y - float(center_xy_m[1])) / ry
    ) ** 2
    compact_support = np.clip(1.0 - rho_sq, 0.0, 1.0) ** 2
    requested = -float(sinkage_m) * compact_support
    return np.minimum(existing, requested)


def _coplanar_segmented_ground_layout(region: Rect, half_extent_m: float) -> tuple[Rect, ...]:
    """Partition a square ground plane into four surroundings plus one operator cell.

    The rectangles meet exactly at the operator boundary and never overlap in area.  This lets
    PhysX assign a different material to O1 without raising a plate or introducing a second
    coplanar collider beneath the region.
    """
    half_extent = float(half_extent_m)
    if half_extent <= 0.0:
        raise ValueError("segmented ground half extent must be positive")
    left = float(region.cx - region.hx)
    right = float(region.cx + region.hx)
    bottom = float(region.cy - region.hy)
    top = float(region.cy + region.hy)
    if not (
        -half_extent < left < right < half_extent
        and -half_extent < bottom < top < half_extent
    ):
        raise ValueError("operator region must lie inside the segmented ground extent")
    return (
        Rect(
            cx=0.5 * (-half_extent + left),
            cy=0.0,
            hx=0.5 * (left + half_extent),
            hy=half_extent,
        ),
        Rect(
            cx=0.5 * (right + half_extent),
            cy=0.0,
            hx=0.5 * (half_extent - right),
            hy=half_extent,
        ),
        Rect(
            cx=region.cx,
            cy=0.5 * (-half_extent + bottom),
            hx=region.hx,
            hy=0.5 * (bottom + half_extent),
        ),
        Rect(
            cx=region.cx,
            cy=0.5 * (top + half_extent),
            hx=region.hx,
            hy=0.5 * (half_extent - top),
        ),
        region,
    )


def _triggered_collapse_offset_update(
    existing_offset_m: np.ndarray,
    x_m: np.ndarray,
    y_m: np.ndarray,
    *,
    center_xy_m: tuple[float, float],
    region: Rect,
    drop_m: float,
) -> np.ndarray:
    """Create one compact, asymmetric post-trigger basin without a rectangular boundary."""
    if drop_m <= 0.0 or min(region.hx, region.hy) <= 0.0:
        raise ValueError("collapse visual dimensions must be positive")
    existing = np.asarray(existing_offset_m, dtype=np.float64)
    x = np.asarray(x_m, dtype=np.float64)
    y = np.asarray(y_m, dtype=np.float64)
    if existing.shape != x.shape or x.shape != y.shape:
        raise ValueError("collapse offset and coordinate arrays must have equal shape")
    center_x = float(np.clip(center_xy_m[0], region.cx - 0.25 * region.hx, region.cx + 0.25 * region.hx))
    center_y = float(np.clip(center_xy_m[1], region.cy - 0.25 * region.hy, region.cy + 0.25 * region.hy))
    dx = (x - center_x) / (0.92 * region.hx)
    dy = (y - center_y) / (0.92 * region.hy)
    theta = np.arctan2(dy, dx)
    phase = 0.73 * float(region.cx) - 0.41 * float(region.cy)
    irregular_radius = (
        1.0
        + 0.11 * np.sin(3.0 * theta + phase)
        + 0.065 * np.sin(5.0 * theta - 0.7 * phase)
    )
    rho = np.sqrt(dx * dx + dy * dy) / irregular_radius
    compact_support = np.clip(1.0 - rho, 0.0, 1.0) ** 0.62
    # The slight directional modulation prevents a rotationally symmetric sink bowl while keeping
    # the deepest displacement bounded by the physically authored catch-bed drop.
    modulation = np.clip(0.91 + 0.09 * np.cos(theta - 0.4), 0.82, 1.0)
    requested = -float(drop_m) * compact_support * modulation
    return np.minimum(existing, requested)


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


def _set_collision_enabled(prim_path: str, enabled: bool) -> list[str]:
    """Set collision state on every authored collider below a prim and return changed paths."""
    import omni.usd
    from pxr import Usd, UsdPhysics

    stage = omni.usd.get_context().get_stage()
    root = stage.GetPrimAtPath(prim_path)
    if not root.IsValid():
        return []
    changed: list[str] = []
    for prim in Usd.PrimRange(root):
        if not prim.HasAPI(UsdPhysics.CollisionAPI):
            continue
        api = UsdPhysics.CollisionAPI(prim)
        attr = api.GetCollisionEnabledAttr()
        previous = bool(attr.Get()) if attr.IsValid() and attr.HasAuthoredValueOpinion() else True
        if previous != enabled:
            api.CreateCollisionEnabledAttr(bool(enabled)).Set(bool(enabled))
            changed.append(str(prim.GetPath()))
    return changed


def _set_prim_visible(prim_path: str, visible: bool) -> None:
    """Toggle the rendered subtree while leaving collision state under separate control."""
    import omni.usd
    from pxr import UsdGeom

    prim = omni.usd.get_context().get_stage().GetPrimAtPath(prim_path)
    if not prim.IsValid() or not prim.IsA(UsdGeom.Imageable):
        return
    imageable = UsdGeom.Imageable(prim)
    if visible:
        imageable.MakeVisible()
    else:
        imageable.MakeInvisible()


def body_command_to_world_xy(command_body_xy: np.ndarray, heading_rad: float) -> np.ndarray:
    """Rotate a planar body-frame command into the episode/world frame."""
    command = np.asarray(command_body_xy, dtype=np.float64)
    if command.shape != (2,):
        raise ValueError("body-frame planar command must have shape (2,)")
    c, s = math.cos(float(heading_rad)), math.sin(float(heading_rad))
    return np.array(
        [c * command[0] - s * command[1], s * command[0] + c * command[1]],
        dtype=np.float64,
    )


class IsaacPolicyBackend:
    """Go2 in the flat velocity env, walked by the trained RSL-RL policy (M2 deliverable)."""

    def __init__(
        self,
        cfg: Config,
        start_pos: np.ndarray,
        start_heading: float,
        record_cam: bool = False,
        perception_cam: bool = False,
        front_cam: bool = False,
    ) -> None:
        import isaaclab.sim as sim_utils
        import torch
        from isaaclab.envs import ManagerBasedRLEnv
        from isaaclab.sensors import ImuCfg
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
        self._friction_prim_paths: list[tuple[Rect, str]] = []
        self._resistance: list[dict] = []
        self._collapse: list[dict] = []  # O3 legacy swap or realistic damage/topology states
        self._blocking: list[dict[str, object]] = []  # O8 obstacle geometry/readback
        self._support: list[dict] = []  # O9 geometry + measured foot/belly contact telemetry
        self._payload_kg = 0.0  # O5 attached payload
        self._payload_visuals: list[dict[str, object]] = []
        self._effort_scale = 1.0  # O10 actuator-effort fraction
        self._nominal_effort: dict = {}  # actuator effort limits before O10 scaling
        self._effort_limit_nm = float(cfg.effort_limit_nm)  # current (O10-scaled) torque cap
        self._n_patches = 0
        self._disabled_default_ground_colliders: list[str] = []
        self._realistic_scene_floor_path: str | None = None
        self._realistic_scene_nominal_friction: tuple[float, float] | None = None
        self._disabled_realistic_scene_floor_colliders: list[str] = []
        self._realistic_scene_route_appearance_path: str | None = None
        self._realistic_scene_route_appearance_hidden = False
        self._realistic_scene_route_original_points: list[tuple[float, float, float]] | None = None
        self._realistic_scene_route_visual_offsets: np.ndarray | None = None
        self._realistic_scene_route_visual_deformation: dict[str, object] | None = None
        self._realistic_scene_friction_topology: dict[str, object] | None = None
        self._disabled_default_sky_lights: list[str] = []
        self._collapse_ground_segmented = False
        self._terrain_appearance: TerrainAppearanceBinding | None = None
        self._last_deep_reset_removed = 0  # A0.1: prims removed by the last deep_reset (diagnostic)
        self._record_cam = bool(record_cam)
        self._camera = None
        self._perception_cam_on = bool(perception_cam)
        self._perception_cam = None
        self._perception_eye: np.ndarray | None = None
        self._perception_target: np.ndarray | None = None
        self._perception_manual_aim = False
        self._perception_mount_offset_m = np.array(
            [
                float(cfg.get("perception_cam_x_m", 0.335)),
                float(cfg.get("perception_cam_y_m", 0.0)),
                float(cfg.get("perception_cam_z_m", 0.065)),
            ],
            dtype=np.float32,
        )
        self._perception_pitch_down_rad = float(cfg.get("perception_cam_pitch_down_rad", 0.52))
        self._perception_depth_faults = []
        self._perception_frame_id = 0
        self._perception_pose_history: list[dict[str, object]] = []
        self._front_cam_on = bool(front_cam)
        if self._record_cam and self._front_cam_on:
            raise ValueError(
                "Isaac Sim 5.1 initializes one stable Camera sensor per environment; "
                "record_cam and front_cam must be rendered as separate same-seed replays"
            )
        self._front_camera = None
        self._front_camera_offset_m = np.array(
            [
                float(cfg.get("front_cam_x_m", 0.335)),
                float(cfg.get("front_cam_y_m", 0.0)),
                float(cfg.get("front_cam_z_m", 0.065)),
            ],
            dtype=np.float32,
        )
        self._front_camera_pitch_down_rad = float(cfg.get("front_cam_pitch_down_rad", 0.0))
        self._foot_adhesion_cfg: FootAdhesionConfig | None = None
        self._foot_adhesion_states: list[FootAdhesionState] = []
        self._foot_adhesion_updates: list[FootAdhesionUpdate] = []
        self._foot_adhesion_contacts_n: list[float] = []
        self._foot_adhesion_terminal_mode = False
        self._foot_terrain_cfg: FootTerramechanicsConfig | None = None
        self._foot_terrain_states: list[FootTerrainState] = []
        self._foot_terrain_updates: list[FootTerrainUpdate] = []
        self._foot_terrain_contacts_n: list[float] = []
        self._foot_terrain_paths: list[str] = []
        self._foot_terrain_footprints: list[dict[str, object]] = []
        self._foot_terrain_matched_control = False
        self._foot_terrain_topology: dict[str, object] = {}

        env_cfg = UnitreeGo2FlatEnvCfg()
        # Isaac Lab >=2.3 exposes backend-dependent sections as ``PresetCfg`` objects when
        # configs are instantiated directly (outside Hydra).  Kino-Fail is explicitly a PhysX
        # benchmark, so resolve those presets here before touching event fields or constructing
        # ``ManagerBasedRLEnv``.  Older Isaac Lab releases already expose the concrete configs.
        if hasattr(env_cfg.events, "physx"):
            env_cfg.events = env_cfg.events.physx
        physics_cfg = getattr(env_cfg.sim, "physics", None)
        if physics_cfg is not None and hasattr(physics_cfg, "physx"):
            env_cfg.sim.physics = physics_cfg.physx
        env_cfg.scene.num_envs = 1
        env_cfg.scene.kinofail_imu = ImuCfg(
            prim_path="{ENV_REGEX_NS}/Robot/base",
            update_period=0.0,
            gravity_bias=(0.0, 0.0, 9.81),
        )
        env_cfg.sim.device = self._device
        env_cfg.seed = 0
        # Inference: no domain randomization, no obs noise, no command resampling.
        env_cfg.observations.policy.enable_corruption = False
        env_cfg.events.add_base_mass = None
        self._inference_reset_contract = _freeze_inference_joint_reset(env_cfg)
        env_cfg.events.physics_material.params["static_friction_range"] = (0.8, 0.8)
        env_cfg.events.physics_material.params["dynamic_friction_range"] = (0.6, 0.6)
        # reset_base randomization is irrelevant: _teleport_to_start overrides it.
        env_cfg.commands.base_velocity.heading_command = False
        env_cfg.commands.base_velocity.debug_vis = False
        env_cfg.commands.base_velocity.rel_standing_envs = 0.0  # never zero our command
        env_cfg.commands.base_velocity.resampling_time_range = (1.0e9, 1.0e9)
        env_cfg.episode_length_s = 1.0e6  # never truncate mid-demo
        env_cfg.sim.render.enable_translucency = True
        # The training task treats every chassis contact as an illegal-contact termination and
        # auto-resets inside ``env.step``.  That destroys the very state O9 must measure: a robot
        # can be high-centered with its belly load-bearing while still upright.  Kino-Fail owns
        # failure termination via measured height/tilt below, so retain the contact instead of
        # silently replacing it with a freshly reset sample before telemetry is read.
        env_cfg.terminations.base_contact = None
        self.dt = float(env_cfg.sim.dt * env_cfg.decimation)

        if self._record_cam or self._front_cam_on:
            self._add_record_camera(env_cfg, cfg)
        if self._perception_cam_on:
            self._add_perception_camera(env_cfg, cfg)

        self._env = ManagerBasedRLEnv(cfg=env_cfg)
        print("[isaac] ManagerBasedRLEnv initialized", flush=True)
        self._robot = self._env.scene["robot"]
        self._contact = self._env.scene["contact_forces"]
        self._imu = self._env.scene["kinofail_imu"]
        self._cmd_term = self._env.command_manager.get_term("base_velocity")
        if self._record_cam:
            self._camera = self._env.scene["record_cam"]
            eye = self._torch.tensor([[3.0, -4.5, 7.5]], device=self._device)
            target = self._torch.tensor([[3.2, 0.3, 0.2]], device=self._device)
            self._camera.set_world_poses_from_view(eye, target)
        if self._perception_cam_on:
            self._perception_cam = self._env.scene["perception_cam"]
            self._disable_tonemap()
            self._sync_perception_camera_pose()
        if self._front_cam_on:
            # Reuse the repository's proven record_cam Replicator graph, then drive it with the
            # exact Go2 base-frame transform.  A second sensor key is unstable in Isaac Sim 5.1.
            self._front_camera = self._env.scene["record_cam"]
            self._sync_front_camera_pose()
            print("[isaac] body-fixed front camera initialized", flush=True)

        policy_path = REPO_ROOT / str(cfg.policy_path)
        print(f"[isaac] loading policy {policy_path}", flush=True)
        if not policy_path.exists():
            raise FileNotFoundError(
                f"trained policy not found at {policy_path}; run scripts/train_locomotion.py"
            )
        self._policy = torch.jit.load(str(policy_path)).to(self._device).eval()
        print("[isaac] policy loaded", flush=True)

        self._foot_ids, foot_names = self._robot.find_bodies(".*_foot")
        self._foot_names = list(foot_names)
        self._contact_foot_ids, contact_foot_names = self._contact.find_bodies(".*_foot")
        # find_bodies sorts by name, so the articulation and sensor foot lists align.
        assert foot_names == contact_foot_names, (foot_names, contact_foot_names)
        self._contact_base_ids, self._contact_base_names = self._contact.find_bodies("base")
        self._contact_all_ids, self._contact_all_names = self._contact.find_bodies(".*")
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
        view = self._robot.root_physx_view
        base = self._base_id[0]
        self._nominal_base_mass = view.get_masses()[0, base].clone()
        self._nominal_base_com = view.get_coms()[0, base].clone()
        self._nominal_base_inertia = view.get_inertias()[0, base].clone()
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
        n_joints = int(self._robot.data.applied_torque.shape[1])
        self._actuator_measurement_steps = 0
        self._actuator_abs_energy_j = 0.0
        self._actuator_peak_utilization = np.zeros(n_joints, dtype=np.float64)
        self._actuator_peak_power_w = np.zeros(n_joints, dtype=np.float64)
        self._actuator_utilization_p90_sum = 0.0
        self._actuator_binding_fraction_sum = 0.0
        self._actuator_power_sum_w = 0.0
        self._actuator_last_sample: dict[str, object] | None = None
        self._push_pulse: dict[str, object] | None = None
        self._push_last_telemetry: dict[str, object] = {
            "mode": "inactive",
            "active": False,
            "complete": False,
            "applied_steps": 0,
            "commanded_linear_impulse_xy_ns": [0.0, 0.0],
            "commanded_torque_impulse_xyz_nms": [0.0, 0.0, 0.0],
            "peak_force_n": 0.0,
            "peak_torque_nm": 0.0,
        }
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
        for state in getattr(self, "_support", []):
            state["min_measured_support"] = 1.0
            state["max_belly_contact_force_n"] = 0.0
            state["belly_contact_steps"] = 0
            state["consecutive_belly_contact_steps"] = 0
            state["max_consecutive_belly_contact_steps"] = 0
            state["belly_contact_force_sum_n"] = 0.0
            state["mean_belly_contact_force_n"] = 0.0
            state["belly_contact_duty_cycle"] = 0.0
            state["max_base_height_m"] = 0.0
            state["measurement_steps"] = 0
            state["max_nonfoot_contact_by_body_n"] = {
                name: 0.0 for name in getattr(self, "_contact_all_names", []) if "foot" not in name
            }
        if self._foot_adhesion_cfg is not None and hasattr(self, "_foot_ids"):
            self._foot_adhesion_states = [FootAdhesionState() for _ in self._foot_ids]
            self._foot_adhesion_updates = []
            self._foot_adhesion_contacts_n = [0.0 for _ in self._foot_ids]
            self._foot_adhesion_terminal_mode = False
        if self._foot_terrain_cfg is not None and hasattr(self, "_foot_ids"):
            self._foot_terrain_states = [FootTerrainState() for _ in self._foot_ids]
            self._foot_terrain_updates = []
            self._foot_terrain_contacts_n = [0.0 for _ in self._foot_ids]

    # ------------------------------------------------------------ episode API

    def reset(self, seed: int, *, preserve_settle_telemetry: bool = False) -> Obs:
        obs_dict, _ = self._env.reset(seed=seed)
        self._obs = obs_dict["policy"]
        self._init_state()
        self._teleport_to_start()
        # Settle the stance with a zero command before the episode starts.
        for _ in range(int(self._cfg.settle_steps)):
            self._policy_step(np.zeros(3))
        if not preserve_settle_telemetry:
            self._init_state()
        return self._make_obs()

    def inference_reset_contract(self) -> dict[str, object]:
        """Return the frozen reset contract recorded by formal collectors."""
        return dict(self._inference_reset_contract)

    # ------------------------------------------------- A0.1 determinism-grade reset

    def deep_reset(self, seed: int) -> Obs:
        """Determinism-grade reset (A0.1): scrub EVERY operator residue before the standard
        reset, so lane N is independent of every operator that ran in lanes 0..N-1 — the
        #51/#52 operator-ORDER PhysX residual behind the E1 C2ST confound, the E2 Suite-Cal
        non-reproducibility, and the E4 closed-loop order-sensitivity.

        Additive + opt-in: the deployed :meth:`reset` is byte-for-byte untouched (red-line
        discipline); only the A0 determinism/collection harness calls this. It (1) zeros the
        persistent PhysX external-wrench buffer, (2) restores the
        trunk mass (O5) and actuator effort caps (O10), (3) despawns every operator collider/
        visual/material prim so the accumulating collider set can no longer change PhysX
        broadphase/warm-start ORDER across lanes, (4) empties the region containers, (5)
        flushes the contact-sensor buffers, then runs the standard env reset + settle on
        now-clean ground. See A实验/A0.md §A0.1.
        """
        self._zero_external_wrench()  # explicit: the wrench path early-returns when _resistance=[]
        self.clear_foot_adhesion()
        self.clear_foot_compliance()
        self.clear_payload()  # O5 → nominal trunk mass
        self.set_effort_scale(1.0)  # O10 → nominal actuator caps
        self._restore_default_ground_collision()
        self._restore_realistic_scene_floor_collision()
        self._restore_realistic_scene_route_appearance()
        n_removed = self._despawn_operator_prims()  # O1/O3/O7/O8/O9 colliders + visual plates
        self._regions = []
        self._friction_prim_paths = []
        self._resistance = []
        self._collapse = []
        self._blocking = []
        self._support = []
        self._collapse_ground_segmented = False
        self._realistic_scene_friction_topology = None
        if self._realistic_scene_floor_path is not None:
            self._disable_default_ground_collision_for_segmented_surface()
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
        prefixes = (
            "patch_",
            "collapse_",
            "wall_",
            "ridge_",
            "vplate_",
            "patchtex_",
            "terrain_visual_",
            "collapse_ground_",
            "collapse_cell_",
            "collapse_catch_",
            "collapse_fragment_",
            "compliance_ground_",
            "compliance_support_",
            "compliance_bed_",
            "compliance_ramp_",
            "soil_footprint_",
            "payload_visual_",
            "friction_ground_",
            "friction_patch_",
        )
        removed = 0
        world = stage.GetPrimAtPath("/World")
        if world.IsValid():
            for child in list(world.GetChildren()):
                if child.GetName().startswith(prefixes) and stage.RemovePrim(child.GetPath()):
                    removed += 1
        looks = stage.GetPrimAtPath("/World/Looks")
        if looks.IsValid():
            for child in list(looks.GetChildren()):
                if child.GetName().startswith(("patchtex_", "kino_terrain_")) and stage.RemovePrim(
                    child.GetPath()
                ):
                    removed += 1
        self._n_patches = 0
        self._terrain_appearance = None
        self._perception_depth_faults = []
        self._perception_frame_id = 0
        self._perception_manual_aim = False
        self._perception_pose_history = []
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

    def step_terminal_sensing(self, cmd_vel: np.ndarray | None = None) -> Obs:
        """Advance one real control step after a failure without clearing its terminal label.

        ``step`` intentionally latches a detected fall and returns cached observations on
        later calls.  That is the right episode-control contract, but confirmatory
        attribution collection also needs a short, timestamped post-failure sensor window.
        The underlying Isaac Lab task has chassis-contact termination disabled and an
        effectively infinite timeout, so calling ``_policy_step`` here advances PhysX,
        RTX sensors, IMU, articulation state, and operator forces without an automatic
        environment reset.  A zero velocity command is the default (the locomotion policy
        holds posture rather than continuing the route), while ``fallen`` remains latched.
        """
        if not self._fallen:
            raise RuntimeError("terminal sensing is only valid after a latched failure")
        command = (
            np.zeros(3, dtype=np.float64)
            if cmd_vel is None
            else np.asarray(cmd_vel, dtype=np.float64).copy()
        )
        if command.shape != (3,):
            raise ValueError("terminal sensing command must have shape (3,)")
        command[:2] = np.clip(
            command[:2],
            -float(self._cfg.max_speed_mps),
            float(self._cfg.max_speed_mps),
        )
        command[2] = float(
            np.clip(
                command[2],
                -float(self._cfg.max_yaw_rate_radps),
                float(self._cfg.max_yaw_rate_radps),
            )
        )
        self._cmd_prev = command
        terminated = self._policy_step(command)
        if terminated:
            raise RuntimeError(
                "Isaac Lab auto-termination fired during terminal sensing; "
                "post-failure state may have been reset"
            )
        self._t += self.dt
        # Failure is an episode-level fact.  Do not relabel a fallen robot as nominal
        # if it happens to right itself during the diagnostic observation window.
        self._fallen = True
        return self._make_obs()

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
        self._apply_base_wrenches()
        self._apply_foot_terrain_forces()
        command_world_xy = body_command_to_world_xy(cmd[:2], self._make_obs().heading)
        self._apply_foot_adhesion_forces(command_world_xy=command_world_xy)
        self._update_collapse()  # O3: measured support damage or legacy dwell proxy
        if self._perception_cam_on and not self._perception_manual_aim:
            self._sync_perception_camera_pose()  # rigid Go2-front extrinsics before rendering
        if self._front_cam_on:
            self._sync_front_camera_pose()  # rigid Go2-base extrinsics; no target/auto-aim
        self._sync_payload_visual_poses()
        obs_dict, _, terminated, _truncated, _ = self._env.step(action)
        self._obs = obs_dict["policy"]
        self._sync_payload_visual_poses()
        self._update_high_centering_telemetry()
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
        sample = actuator_sample(
            self._robot.data.applied_torque[0].detach().cpu().numpy(),
            self._robot.data.joint_vel[0].detach().cpu().numpy(),
            self._effort_limit_nm,
        )
        self._actuator_measurement_steps += 1
        self._actuator_abs_energy_j += sample.total_abs_mechanical_power_w * self.dt
        self._actuator_peak_utilization = np.maximum(
            self._actuator_peak_utilization, sample.torque_utilization
        )
        self._actuator_peak_power_w = np.maximum(
            self._actuator_peak_power_w, sample.mechanical_power_w
        )
        self._actuator_utilization_p90_sum += sample.utilization_p90
        self._actuator_binding_fraction_sum += sample.binding_joint_fraction
        self._actuator_power_sum_w += sample.total_abs_mechanical_power_w
        self._actuator_last_sample = {
            "torque_utilization": sample.torque_utilization.tolist(),
            "mechanical_power_w": sample.mechanical_power_w.tolist(),
            "utilization_mean": sample.utilization_mean,
            "utilization_p90": sample.utilization_p90,
            "utilization_max": sample.utilization_max,
            "binding_joint_fraction": sample.binding_joint_fraction,
            "total_abs_mechanical_power_w": sample.total_abs_mechanical_power_w,
        }
        floor = float(self._cfg.effort_sat_floor)
        return float(np.clip((sat - floor) / max(1e-6, 1.0 - floor), 0.0, 1.0))

    def actuator_telemetry(self) -> dict[str, object]:
        """Return joint-level torque utilization and mechanical energy diagnostics."""
        steps = max(self._actuator_measurement_steps, 1)
        actuator_limits = {}
        for name, actuator in self._robot.actuators.items():
            actuator_limits[name] = {}
            for public_name, internal_name in (
                ("effort_limit", "effort_limit"),
                ("saturation_effort", "_saturation_effort"),
                ("velocity_limit", "velocity_limit"),
            ):
                if not hasattr(actuator, internal_name):
                    continue
                value = getattr(actuator, internal_name)
                if hasattr(value, "min"):
                    minimum = float(value.min().item())
                    maximum = float(value.max().item())
                else:
                    minimum = maximum = float(value)
                actuator_limits[name][public_name] = {
                    "min": minimum,
                    "max": maximum,
                }
        return {
            "joint_names": list(self._robot.joint_names),
            "effort_scale": float(self._effort_scale),
            "effort_limit_nm": float(self._effort_limit_nm),
            "measurement_steps": int(self._actuator_measurement_steps),
            "peak_torque_utilization_by_joint": self._actuator_peak_utilization.tolist(),
            "peak_mechanical_power_w_by_joint": self._actuator_peak_power_w.tolist(),
            "mean_utilization_p90": self._actuator_utilization_p90_sum / steps,
            "mean_binding_joint_fraction": self._actuator_binding_fraction_sum / steps,
            "mean_total_abs_mechanical_power_w": self._actuator_power_sum_w / steps,
            "total_abs_mechanical_energy_j": self._actuator_abs_energy_j,
            "last_sample": self._actuator_last_sample,
            "actuator_limit_readback": actuator_limits,
            "legacy_effort_ratio": float(self._effort),
            "legacy_effort_note": (
                "legacy channel thresholds mean joint torque at 70% of the nominal cap; "
                "use joint telemetry for realistic O5/O10 evidence"
            ),
        }

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

    def raw_proprio_packet(self) -> RawProprioPacket:
        """Return timestamped Isaac IMU plus ideal articulation odometry before O11 faults."""
        truth = self._make_obs()
        imu = self._imu.data
        return RawProprioPacket(
            t=float(truth.t),
            imu_projected_gravity_b=(
                imu.projected_gravity_b[0].detach().cpu().numpy().astype(np.float64)
            ),
            imu_ang_vel_b_radps=(
                imu.ang_vel_b[0].detach().cpu().numpy().astype(np.float64)
            ),
            imu_lin_acc_b_mps2=(
                imu.lin_acc_b[0].detach().cpu().numpy().astype(np.float64)
            ),
            odom_pos_xy_m=truth.pos.copy(),
            odom_heading_rad=float(truth.heading),
            odom_vel_body_mps=truth.vel_body.copy(),
        )

    # ------------------------------------------------------------ recording

    def _add_record_camera(self, env_cfg: object, cfg: Config) -> None:
        """Add a fixed wide-shot RGB camera to the scene (passive; for video recording)."""
        from isaaclab.sensors import CameraCfg

        focal_length = float(cfg.get("front_cam_focal_mm", 20.0)) if self._front_cam_on else 20.0
        horizontal_aperture = (
            float(cfg.get("front_cam_aperture_mm", 20.955))
            if self._front_cam_on
            else 20.955
        )
        env_cfg.scene.record_cam = CameraCfg(
            prim_path="/World/record_cam",
            update_period=0.0,
            # The same sensor is reused as the body-fixed front camera.  Without this flag its
            # RGB render follows manually updated poses while ``data.pos_w``/quaternions remain at
            # the initialization pose, making calibration provenance internally inconsistent.
            update_latest_camera_pose=True,
            height=int(cfg.cam_height),
            width=int(cfg.cam_width),
            data_types=["rgb"],
            spawn=self._sim_utils.PinholeCameraCfg(
                focal_length=focal_length,
                horizontal_aperture=horizontal_aperture,
                clipping_range=(0.1, 1.0e4),
            ),
        )

    def capture_rgb(self, *, force_recompute: bool = False) -> np.ndarray | None:
        """Latest camera RGB as an ``(H, W, 3)`` uint8 array (None if not recording)."""
        if self._camera is None:
            return None
        self._camera.update(self.dt, force_recompute=force_recompute)
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

    def _sync_front_camera_pose(self) -> None:
        """Apply the calibrated body-frame mount through the measured full base quaternion."""
        if self._front_camera is None:
            return
        from isaaclab.utils.math import quat_apply, quat_mul

        base_pos = self._robot.data.root_pos_w[0:1]
        base_quat = self._robot.data.root_quat_w[0:1]
        local_offset = self._torch.tensor(
            self._front_camera_offset_m.reshape(1, 3),
            dtype=base_pos.dtype,
            device=self._device,
        )
        camera_pos = base_pos + quat_apply(base_quat, local_offset)
        half_pitch = 0.5 * self._front_camera_pitch_down_rad
        local_pitch = self._torch.tensor(
            [[math.cos(half_pitch), 0.0, math.sin(half_pitch), 0.0]],
            dtype=base_quat.dtype,
            device=self._device,
        )
        camera_quat = quat_mul(base_quat, local_pitch)
        # "world" convention means camera forward=+X/up=+Z, matching the Go2 base axes.
        self._front_camera.set_world_poses(camera_pos, camera_quat, convention="world")

    def capture_front_camera(self) -> dict | None:
        """Return body-fixed RGB plus calibration and the measured world pose."""
        if self._front_camera is None:
            return None
        cam = self._front_camera
        self._sync_front_camera_pose()
        cam.update(self.dt, force_recompute=True)
        out = cam.data.output

        def _np(value: object) -> np.ndarray:
            return value[0].detach().cpu().numpy()

        rgb = _np(out["rgb"])[..., :3]
        if np.issubdtype(rgb.dtype, np.floating):
            rgb = (np.clip(rgb, 0.0, 1.0) * 255.0).astype(np.uint8)
        else:
            rgb = rgb.astype(np.uint8, copy=False)
        return {
            "rgb": rgb,
            # Do not manufacture a zero-depth channel.  This RTX vertical slice is explicitly RGB;
            # depth will be re-enabled only after the Isaac 5.1 articulated-camera annotator bug is
            # resolved and validated.
            "depth": None,
            "K": _np(cam.data.intrinsic_matrices),
            "pos": cam.data.pos_w[0].detach().cpu().numpy(),
            "quat_ros": cam.data.quat_w_ros[0].detach().cpu().numpy(),
            "quat_world": cam.data.quat_w_world[0].detach().cpu().numpy(),
            "env_origin": self._env.scene.env_origins[0].detach().cpu().numpy(),
        }

    def load_realistic_scene(self, episode_usd: str) -> str:
        """Reference a compiled scene and select exactly one authoritative ground collider.

        Legacy Kino-authored rooms use Isaac's known-flat ground and a visual-only finish.  New
        EmbodiedGen packages contain a Kino-owned ``Collision/Floor`` proxy; for those packages the
        default ground collider is disabled so two nearly coplanar contact surfaces cannot coexist.
        """
        import omni.usd
        from pxr import UsdPhysics

        prim_path = "/World/KinoIndoor"
        cfg = self._sim_utils.UsdFileCfg(usd_path=str(episode_usd))
        env_origin = self._env.scene.env_origins[0].cpu().numpy()
        cfg.func(
            prim_path,
            cfg,
            translation=(float(env_origin[0]), float(env_origin[1]), float(env_origin[2])),
        )
        stage = omni.usd.get_context().get_stage()
        self.hide_default_ground_visual()
        self._disable_default_sky_light_for_realistic_scene()
        if not stage.GetPrimAtPath(prim_path).IsValid():
            raise RuntimeError(f"compiled scene did not load at {prim_path}")
        custom_floor_path = f"{prim_path}/Collision/Floor"
        custom_floor = stage.GetPrimAtPath(custom_floor_path)
        if custom_floor.IsValid() and custom_floor.HasAPI(UsdPhysics.CollisionAPI):
            self._realistic_scene_floor_path = custom_floor_path
            self._realistic_scene_nominal_friction = _read_material_friction(custom_floor_path)
            self._disable_default_ground_collision_for_segmented_surface()
        else:
            self._realistic_scene_floor_path = None
            self._realistic_scene_nominal_friction = None
        route_appearance_candidates = (
            f"{prim_path}/Appearance/RouteSurface",
            f"{prim_path}/AppearanceV4/RenderOnlyRouteSurface",
        )
        self._realistic_scene_route_appearance_path = next(
            (
                candidate
                for candidate in route_appearance_candidates
                if stage.GetPrimAtPath(candidate).IsValid()
            ),
            None,
        )
        self._realistic_scene_route_appearance_hidden = False
        self._realistic_scene_route_original_points = None
        self._realistic_scene_route_visual_offsets = None
        self._realistic_scene_route_visual_deformation = None
        self._realistic_scene_friction_topology = None
        return prim_path

    def _disable_default_sky_light_for_realistic_scene(self) -> int:
        """Let a loaded realistic scene own the environment map and illumination."""
        import omni.usd
        from pxr import UsdGeom, UsdLux

        stage = omni.usd.get_context().get_stage()
        changed: list[str] = []
        for path in ("/World/skyLight", "/World/defaultLight"):
            prim = stage.GetPrimAtPath(path)
            if not prim.IsValid() or not prim.IsA(UsdLux.DomeLight):
                continue
            UsdGeom.Imageable(prim).MakeInvisible()
            changed.append(path)
        self._disabled_default_sky_lights = changed
        return len(changed)

    def hide_default_ground_visual(self) -> int:
        """Hide Isaac's black/white grid while retaining its collision and PhysX material."""
        import omni.usd
        from pxr import Usd, UsdGeom

        stage = omni.usd.get_context().get_stage()
        hidden = 0
        for root_path in ("/World/ground", "/World/defaultGroundPlane"):
            root = stage.GetPrimAtPath(root_path)
            if not root.IsValid():
                continue
            for prim in Usd.PrimRange(root):
                if prim.IsA(UsdGeom.Imageable):
                    UsdGeom.Imageable(prim).MakeInvisible()
                    hidden += 1
        return hidden

    def _disable_default_ground_collision_for_segmented_surface(self) -> int:
        """Replace the monolithic Isaac plane with an explicitly segmented O3 support surface."""
        if self._disabled_default_ground_colliders:
            return len(self._disabled_default_ground_colliders)
        changed: list[str] = []
        for root_path in ("/World/ground", "/World/defaultGroundPlane"):
            changed.extend(_set_collision_enabled(root_path, False))
        if not changed:
            raise RuntimeError(
                "O3 realistic topology requires disabling the default monolithic ground, "
                "but no ground collider was found"
            )
        self._disabled_default_ground_colliders = changed
        self.hide_default_ground_visual()
        return len(changed)

    def _restore_default_ground_collision(self) -> None:
        """Restore colliders disabled by the segmented O3 surface during a deep reset."""
        for prim_path in self._disabled_default_ground_colliders:
            _set_collision_enabled(prim_path, True)
        self._disabled_default_ground_colliders = []

    def _disable_realistic_scene_floor_for_operator_topology(self) -> int:
        """Disable the Kino scene floor while O2/O3 author replacement support topology.

        Loading a realistic scene already disables Isaac's default plane and keeps the
        scene's ``Collision/Floor`` as the sole nominal support.  O2/O3 must replace that
        custom floor too; otherwise their lowered bed or removed cells sit on top of an
        unchanged hidden collider and the requested physical anomaly never occurs.
        """
        if self._disabled_realistic_scene_floor_colliders:
            return len(self._disabled_realistic_scene_floor_colliders)
        if self._realistic_scene_floor_path is None:
            return 0
        changed = _set_collision_enabled(self._realistic_scene_floor_path, False)
        if not changed:
            raise RuntimeError(
                "segmented realistic terrain requires disabling the Kino scene floor, "
                "but its collider was missing or already disabled"
            )
        self._disabled_realistic_scene_floor_colliders = changed
        return len(changed)

    def _nominal_ground_friction_readback(self) -> tuple[float, float]:
        """Return the exact static/dynamic pair that segmented terrain must preserve."""
        if self._realistic_scene_nominal_friction is not None:
            return self._realistic_scene_nominal_friction
        for path in ("/World/ground", "/World/defaultGroundPlane"):
            try:
                return _read_material_friction(path)
            except RuntimeError:
                continue
        return (
            float(self._cfg.mu_nominal),
            float(self._cfg.get("mu_nominal_dynamic", 0.6)),
        )

    def nominal_ground_friction(self) -> tuple[float, float]:
        """Expose the loaded scene's nominal material for matched operator controls.

        A nominal counterfactual for a segmented terrain operator must install the same
        collision topology as its anomaly lane.  Returning the material read directly from
        the loaded scene avoids silently substituting a configuration default for that control.
        """
        return self._nominal_ground_friction_readback()

    def foot_contact_snapshot(
        self,
        region: Rect | None = None,
        *,
        region_margin_m: float = 0.0,
        load_threshold_n: float = 1.0,
    ) -> dict[str, object]:
        """Return measured foot positions/normal loads for counterfactual contact audits."""
        env_origin = self._env.scene.env_origins[0].detach().cpu().numpy()
        positions = self._robot.data.body_pos_w[0, self._foot_ids].detach().cpu().numpy()
        positions = positions - env_origin.reshape(1, 3)
        contact = self._contact.data.net_forces_w[0, self._contact_foot_ids]
        normal_force = np.maximum(
            contact[:, 2].detach().cpu().numpy().astype(np.float64), 0.0
        )
        margin = float(region_margin_m)
        feet: list[dict[str, object]] = []
        for name, position, force in zip(
            self._foot_names, positions, normal_force, strict=True
        ):
            inside = False
            if region is not None:
                inside = (
                    abs(float(position[0]) - region.cx) <= region.hx + margin
                    and abs(float(position[1]) - region.cy) <= region.hy + margin
                )
            feet.append(
                {
                    "name": name,
                    "position_xyz_m": [float(value) for value in position],
                    "normal_force_n": float(force),
                    "inside_region_with_margin": inside,
                    "load_bearing_in_region": inside and float(force) >= load_threshold_n,
                }
            )
        return {
            "region_margin_m": margin,
            "load_threshold_n": float(load_threshold_n),
            "any_load_bearing_in_region": any(
                bool(foot["load_bearing_in_region"]) for foot in feet
            ),
            "feet": feet,
        }

    def _restore_realistic_scene_floor_collision(self) -> None:
        """Restore the nominal Kino scene floor after removing O2/O3 operator topology."""
        for prim_path in self._disabled_realistic_scene_floor_colliders:
            _set_collision_enabled(prim_path, True)
        self._disabled_realistic_scene_floor_colliders = []

    def _hide_realistic_scene_route_appearance_for_operator_topology(self) -> bool:
        """Expose O2/O3 replacement geometry instead of a stale flat render overlay."""
        if self._realistic_scene_route_appearance_hidden:
            return True
        if self._realistic_scene_route_appearance_path is None:
            return False
        _set_prim_visible(self._realistic_scene_route_appearance_path, False)
        self._realistic_scene_route_appearance_hidden = True
        return True

    def _realistic_scene_has_continuous_opaque_ground(self) -> bool:
        if self._realistic_scene_route_appearance_path is None:
            return False
        import omni.usd
        from pxr import UsdGeom

        prim = omni.usd.get_context().get_stage().GetPrimAtPath(
            self._realistic_scene_route_appearance_path
        )
        marker = prim.GetAttribute("kino:opaqueCompositedGround")
        return bool(
            prim.IsValid()
            and prim.IsA(UsdGeom.Mesh)
            and marker.IsValid()
            and marker.Get() is True
        )

    def _initialize_contact_local_ground_deformation(
        self,
        config: FootTerramechanicsConfig,
    ) -> dict[str, object]:
        """Keep the initial surface nominal and prepare load-triggered local soil imprints."""
        if not self._realistic_scene_has_continuous_opaque_ground():
            return {"applied": False, "reason": "continuous_opaque_ground_unavailable"}
        import omni.usd
        from pxr import Sdf, UsdGeom

        path = str(self._realistic_scene_route_appearance_path)
        stage = omni.usd.get_context().get_stage()
        mesh = UsdGeom.Mesh(stage.GetPrimAtPath(path))
        points_attr = mesh.GetPointsAttr()
        points = list(points_attr.Get() or [])
        if not points:
            raise RuntimeError("continuous realistic ground has no authored points")
        if self._realistic_scene_route_original_points is not None:
            raise RuntimeError("continuous realistic ground deformation is already initialized")
        original = [(float(point[0]), float(point[1]), float(point[2])) for point in points]
        prim = mesh.GetPrim()
        prim.CreateAttribute("kino:o2VisualDeformationApplied", Sdf.ValueTypeNames.Bool).Set(True)
        prim.CreateAttribute("kino:o2VisualSurfacePredeformed", Sdf.ValueTypeNames.Bool).Set(
            False
        )
        self._realistic_scene_route_original_points = original
        self._realistic_scene_route_visual_offsets = np.zeros(len(original), dtype=np.float64)
        self._realistic_scene_route_visual_deformation = {
            "applied": True,
            "mode": "contact_local_load_triggered_imprints",
            "mesh_path": path,
            "vertex_count": len(original),
            "deformed_vertex_count": 0,
            "minimum_visual_offset_m": 0.0,
            "maximum_visual_offset_m": 0.0,
            "contact_local_imprint_count": 0,
            "surface_predeformed": False,
            "operator_region_has_no_visual_boundary": True,
            "operator_collision_geometry_visible": False,
        }
        return dict(self._realistic_scene_route_visual_deformation)

    def _apply_contact_local_ground_deformation(
        self,
        *,
        center_xy_m: tuple[float, float],
        sinkage_m: float,
    ) -> int:
        """Update the continuous PBR mesh only around a measured load-bearing foot contact."""
        original = self._realistic_scene_route_original_points
        offsets = self._realistic_scene_route_visual_offsets
        if (
            original is None
            or offsets is None
            or self._realistic_scene_route_appearance_path is None
        ):
            return 0
        x = np.asarray([point[0] for point in original], dtype=np.float64)
        y = np.asarray([point[1] for point in original], dtype=np.float64)
        sink_fraction = float(
            np.clip(
                sinkage_m
                / max(float(self._foot_terrain_cfg.max_sink_depth_m), 1.0e-6),
                0.0,
                1.0,
            )
        ) if self._foot_terrain_cfg is not None else 0.0
        updated = _contact_local_soil_offset_update(
            offsets,
            x,
            y,
            center_xy_m=center_xy_m,
            sinkage_m=float(sinkage_m),
            radius_xy_m=(0.075 + 0.015 * sink_fraction, 0.050 + 0.012 * sink_fraction),
        )
        changed = int(np.count_nonzero(updated < offsets - 1.0e-6))
        if changed == 0:
            return 0
        import omni.usd
        from pxr import Gf, Sdf, UsdGeom

        prim = omni.usd.get_context().get_stage().GetPrimAtPath(
            self._realistic_scene_route_appearance_path
        )
        if not prim.IsValid() or not prim.IsA(UsdGeom.Mesh):
            raise RuntimeError("contact-local soil mesh disappeared during O2 rollout")
        UsdGeom.Mesh(prim).GetPointsAttr().Set(
            [
                Gf.Vec3f(px, py, pz + float(dz))
                for (px, py, pz), dz in zip(original, updated, strict=True)
            ]
        )
        self._realistic_scene_route_visual_offsets = updated
        audit = self._realistic_scene_route_visual_deformation
        if audit is not None:
            audit["deformed_vertex_count"] = int(np.count_nonzero(updated < -1.0e-6))
            audit["minimum_visual_offset_m"] = float(updated.min())
            audit["maximum_visual_offset_m"] = float(updated.max())
            audit["contact_local_imprint_count"] = int(
                audit.get("contact_local_imprint_count", 0)
            ) + 1
        prim.CreateAttribute(
            "kino:o2ContactLocalDeformedVertexCount", Sdf.ValueTypeNames.Int
        ).Set(int(np.count_nonzero(updated < -1.0e-6)))
        prim.CreateAttribute("kino:o2VisualMinimumOffsetM", Sdf.ValueTypeNames.Double).Set(
            float(updated.min())
        )
        return changed

    def _restore_realistic_scene_route_appearance(self) -> None:
        """Restore the nominal PBR route surface after removing segmented operators."""
        if (
            getattr(self, "_realistic_scene_route_original_points", None) is not None
            and self._realistic_scene_route_appearance_path is not None
        ):
            import omni.usd
            from pxr import Gf, UsdGeom

            prim = omni.usd.get_context().get_stage().GetPrimAtPath(
                self._realistic_scene_route_appearance_path
            )
            if prim.IsValid() and prim.IsA(UsdGeom.Mesh):
                UsdGeom.Mesh(prim).GetPointsAttr().Set(
                    [
                        Gf.Vec3f(*point)
                        for point in self._realistic_scene_route_original_points or []
                    ]
                )
                marker = prim.GetAttribute("kino:o2VisualDeformationApplied")
                if marker.IsValid():
                    marker.Set(False)
                collapse_marker = prim.GetAttribute("kino:o3VisualDeformationApplied")
                if collapse_marker.IsValid():
                    collapse_marker.Set(False)
        self._realistic_scene_route_original_points = None
        self._realistic_scene_route_visual_offsets = None
        self._realistic_scene_route_visual_deformation = None
        if (
            self._realistic_scene_route_appearance_hidden
            and self._realistic_scene_route_appearance_path is not None
        ):
            _set_prim_visible(self._realistic_scene_route_appearance_path, True)
        self._realistic_scene_route_appearance_hidden = False

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
            # Isaac Lab defaults this to False: RGB/depth refresh while pos_w/quat_w stay stale.
            # O7 and waypoint back-projection require the pose from the exact rendered frame.
            update_latest_camera_pose=True,
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
        """Manually re-aim the camera for a diagnostic; production uses the Go2 rigid mount."""
        if self._perception_cam is None:
            raise RuntimeError("perception camera not enabled (construct with perception_cam=True)")
        origin = self._env.scene.env_origins[0]
        eye = self._torch.tensor([[float(v) for v in eye_xyz]], device=self._device) + origin
        target = self._torch.tensor([[float(v) for v in target_xyz]], device=self._device) + origin
        self._perception_cam.set_world_poses_from_view(eye, target)
        self._perception_eye = np.asarray(eye_xyz, dtype=np.float64)
        self._perception_target = np.asarray(target_xyz, dtype=np.float64)
        self._perception_manual_aim = True

    def _sync_perception_camera_pose(self) -> None:
        """Rigidly place the RGB-D optical centre at the Go2 front-camera mount."""
        if self._perception_cam is None:
            return
        from isaaclab.utils.math import quat_apply, quat_mul

        base_pos = self._robot.data.root_pos_w[0:1]
        base_quat = self._robot.data.root_quat_w[0:1]
        offset = self._torch.tensor(
            self._perception_mount_offset_m.reshape(1, 3),
            dtype=base_pos.dtype,
            device=self._device,
        )
        camera_pos = base_pos + quat_apply(base_quat, offset)
        half_pitch = 0.5 * self._perception_pitch_down_rad
        local_pitch = self._torch.tensor(
            [[math.cos(half_pitch), 0.0, math.sin(half_pitch), 0.0]],
            dtype=base_quat.dtype,
            device=self._device,
        )
        camera_quat = quat_mul(base_quat, local_pitch)
        self._perception_cam.set_world_poses(camera_pos, camera_quat, convention="world")
        forward_local = self._torch.tensor(
            [[1.0, 0.0, 0.0]], dtype=base_pos.dtype, device=self._device
        )
        forward_world = quat_apply(camera_quat, forward_local)
        env_origin = self._env.scene.env_origins[0:1]
        eye = (camera_pos - env_origin)[0].detach().cpu().numpy().astype(np.float64)
        forward = forward_world[0].detach().cpu().numpy().astype(np.float64)
        self._perception_eye = eye
        self._perception_target = eye + 2.0 * forward
        entry = {
            "time_s": float(getattr(self, "_t", 0.0)),
            "eye": eye.copy(),
            "quat_world": camera_quat[0].detach().cpu().numpy().astype(np.float64),
        }
        if self._perception_pose_history and abs(
            float(self._perception_pose_history[-1]["time_s"]) - float(entry["time_s"])
        ) < 1e-12:
            self._perception_pose_history[-1] = entry
        else:
            self._perception_pose_history.append(entry)
            self._perception_pose_history = self._perception_pose_history[-64:]

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
        # RTX may read this PNG asynchronously after USD binding.  A shared
        # filename lets concurrent Isaac processes overwrite the file while a
        # peer is reading it, which can expose a partial PNG and abort Kit.
        # Process-local storage changes only the asset namespace; pixels,
        # material parameters, geometry, and physics remain identical.
        tex_dir = (
            REPO_ROOT
            / "outputs"
            / "map"
            / "rtx_patch_tex"
            / f"process_{os.getpid()}"
        )
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
        if not self._perception_manual_aim:
            self._sync_perception_camera_pose()
        cam = self._perception_cam
        # A normal cached update can return the previous pose after a rigid mount write. Force the
        # annotator to recompute so RGB/depth and the pose below belong to the same camera frame.
        cam.update(self.dt, force_recompute=True)
        out = cam.data.output

        def _np(x: object) -> np.ndarray:
            return x[0].detach().cpu().numpy()

        seg = np.squeeze(_np(out["semantic_segmentation"])).astype(np.int64)
        pos_w = cam.data.pos_w[0].detach().cpu().numpy()
        quat_world = cam.data.quat_w_world[0].detach().cpu().numpy()
        qw, qx, qy, qz = (float(value) for value in quat_world)
        forward_world = np.array(
            [
                1.0 - 2.0 * (qy * qy + qz * qz),
                2.0 * (qx * qy + qw * qz),
                2.0 * (qx * qz - qw * qy),
            ],
            dtype=np.float64,
        )
        env_origin = self._env.scene.env_origins[0].detach().cpu().numpy()
        actual_eye = pos_w.astype(np.float64) - env_origin.astype(np.float64)
        matched_pose = None
        if self._perception_pose_history:
            matched_pose = min(
                reversed(self._perception_pose_history),
                key=lambda item: float(
                    np.linalg.norm(actual_eye - np.asarray(item["eye"], dtype=np.float64))
                ),
            )
        commanded_eye = None if matched_pose is None else np.asarray(matched_pose["eye"]).copy()
        pose_source_time_s = (
            float(self._t) if matched_pose is None else float(matched_pose["time_s"])
        )
        expected_quat = (
            None
            if matched_pose is None
            else np.asarray(matched_pose["quat_world"], dtype=np.float64)
        )
        angular_error = 0.0
        if expected_quat is not None:
            dot = float(abs(np.dot(expected_quat, quat_world.astype(np.float64))))
            angular_error = 2.0 * math.acos(float(np.clip(dot, -1.0, 1.0)))
        capture = {
            "rgb": _np(out["rgb"])[..., :3],
            "depth": np.squeeze(_np(out["distance_to_image_plane"])).astype(np.float64),
            "seg": seg,
            "id_to_labels": cam.data.info[0]["semantic_segmentation"]["idToLabels"],
            "K": _np(cam.data.intrinsic_matrices),
            "pos": pos_w,
            "quat_ros": cam.data.quat_w_ros[0].detach().cpu().numpy(),
            "quat_world": quat_world,
            "quat_opengl": cam.data.quat_w_opengl[0].detach().cpu().numpy(),
            "env_origin": env_origin,
            "eye": actual_eye,
            "target": actual_eye + 2.0 * forward_world,
            "commanded_eye": commanded_eye,
            "camera_pose_sync_error_m": (
                None
                if commanded_eye is None
                else float(np.linalg.norm(actual_eye - commanded_eye))
            ),
            "camera_orientation_sync_error_rad": angular_error,
            "camera_pose_source_timestamp_s": pose_source_time_s,
            "body_fixed": not self._perception_manual_aim,
            "pose_sync_method": (
                "rigid_base_transform_each_capture"
                if not self._perception_manual_aim
                else "manual_world_look_at"
            ),
            "base_to_camera_xyz_m": self._perception_mount_offset_m.copy(),
            "pitch_down_rad": self._perception_pitch_down_rad,
        }
        from kino_vla.sim.depth_pipeline import apply_timestamped_depth_faults

        transformed = apply_timestamped_depth_faults(
            capture,
            self._perception_depth_faults,
            capture_time_s=pose_source_time_s,
            frame_id=self._perception_frame_id,
            read_time_s=float(self._t),
        )
        self._perception_frame_id += 1
        return transformed

    def add_visual_depth_fault_region(
        self, rect: Rect, appearance_class: str, depth_bias_m: float
    ) -> str | None:
        """Render an O7 benign-looking patch and fault its real RTX depth pixels after capture."""
        if self._perception_cam is None:
            return None
        from kino_vla.sim.depth_pipeline import DepthFaultRegion

        semantic_class = f"kinofail_o7_depth_{len(self._perception_depth_faults)}"
        prim_path = None
        if self._terrain_appearance is not None:
            # Reuse the immediately preceding friction collider and its real OmniPBR material so
            # semantic mask, depth and contact physics refer to the same USD surface.
            for candidate_rect, candidate_path in reversed(self._friction_prim_paths):
                if candidate_rect == rect:
                    prim_path = candidate_path
                    break
        if prim_path is None:
            prim_path = self.add_textured_patch(rect, appearance_class, semantic_class)
        else:
            import omni.usd

            try:
                from isaacsim.core.utils.semantics import add_update_semantics
            except ImportError:
                from omni.isaac.core.utils.semantics import add_update_semantics
            prim = omni.usd.get_context().get_stage().GetPrimAtPath(prim_path)
            add_update_semantics(prim, semantic_label=semantic_class, type_label="class")
        self._perception_depth_faults.append(
            DepthFaultRegion(semantic_class=semantic_class, bias_m=float(depth_bias_m))
        )
        return prim_path

    # ------------------------------------------------------------ operator API

    def set_terrain_appearance(self, binding: TerrainAppearanceBinding | None) -> None:
        """Set the visual-only PBR binding used by subsequently spawned terrain operator prims.

        The binding is episode-scoped and is cleared by :meth:`deep_reset`.  It never mutates
        PhysX material attributes; O1/O3/O7/O9 keep full ownership of contact physics.
        """
        self._terrain_appearance = binding

    def add_pbr_visual_ground(
        self,
        rect: Rect,
        binding: TerrainAppearanceBinding,
        *,
        lift_m: float = 0.004,
    ) -> str:
        """Add a non-colliding PBR ground shell above the known physics surface.

        The tiny lift prevents z-fighting.  Contact remains on the independently authored ground
        or operator collider, so normal/displacement maps cannot silently alter dynamics.
        """
        import omni.usd
        from pxr import Gf, Sdf, UsdGeom

        stage = omni.usd.get_context().get_stage()
        env_origin = self._env.scene.env_origins[0].cpu().numpy()
        prim_path = f"/World/terrain_visual_{self._n_patches}"
        self._n_patches += 1
        cx = float(rect.cx + env_origin[0])
        cy = float(rect.cy + env_origin[1])
        z = float(env_origin[2]) + float(lift_m)
        hx, hy = float(rect.hx), float(rect.hy)
        mesh = UsdGeom.Mesh.Define(stage, prim_path)
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
        mesh.CreateNormalsAttr([Gf.Vec3f(0.0, 0.0, 1.0)])
        mesh.SetNormalsInterpolation("constant")
        mesh.CreateSubdivisionSchemeAttr("none")
        mesh.CreateDoubleSidedAttr(True)
        mesh.GetPrim().CreateAttribute("kino:collisionEnabled", Sdf.ValueTypeNames.Bool).Set(False)
        mesh.GetPrim().CreateAttribute("kino:semanticClass", Sdf.ValueTypeNames.String).Set(
            "traversable_terrain_visual"
        )
        bind_omnipbr_material(stage, prim_path, binding)
        return prim_path

    def _bind_episode_terrain_appearance(self, prim_path: str) -> None:
        binding = self._terrain_appearance
        if binding is None:
            return
        import omni.usd

        result = bind_omnipbr_material(
            omni.usd.get_context().get_stage(),
            prim_path,
            binding,
        )
        print(
            f"[isaac] terrain appearance {result['material_id']} / "
            f"{result['appearance_id']} -> {prim_path} (physics unchanged)"
        )

    def add_friction_regions(self, regions: list[FrictionRegion]) -> None:
        """Apply O1 friction while preserving a flat, continuous realistic PBR surface.

        Legacy scenes retain their thin-plate implementation.  A realistic scene with the
        ``kino:opaqueCompositedGround`` contract instead replaces both monolithic ground
        colliders with five invisible, coplanar collision cells: four nominal surroundings and
        one O1 cell.  The original PBR mesh remains visible and perfectly flat, so neither a blue
        plate nor a texture boundary leaks the anomaly label into RGB.
        """
        if self._realistic_scene_has_continuous_opaque_ground():
            self._add_realistic_friction_region(regions)
            return
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
            self._bind_episode_terrain_appearance(prim_path)
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
            self._friction_prim_paths.append((region.rect, prim_path))

    def _add_realistic_friction_region(self, regions: list[FrictionRegion]) -> None:
        if len(regions) != 1:
            raise ValueError("realistic O1 requires exactly one friction region per episode")
        if self._collapse_ground_segmented:
            raise RuntimeError(
                "only one segmented terrain topology operator is allowed per episode"
            )
        region = regions[0]
        disabled_default = self._disable_default_ground_collision_for_segmented_surface()
        disabled_scene = self._disable_realistic_scene_floor_for_operator_topology()
        self._collapse_ground_segmented = True
        group = self._n_patches
        self._n_patches += 1
        half_extent = 0.5 * float(self._cfg.get("collapse_ground_extent_m", 30.0))
        layout = _coplanar_segmented_ground_layout(region.rect, half_extent)
        nominal_mu_s, nominal_mu_d = self._nominal_ground_friction_readback()
        surrounding_paths: list[str] = []
        for index, rect in enumerate(layout[:-1]):
            path = self._spawn_collapse_surface_box(
                f"/World/friction_ground_{group}_{index}",
                rect,
                top_z_m=0.0,
                mu_s=nominal_mu_s,
                mu_d=nominal_mu_d,
                color=(0.42, 0.43, 0.40),
                bind_terrain_appearance=False,
            )
            _set_prim_visible(path, False)
            surrounding_paths.append(path)
        patch_path = self._spawn_collapse_surface_box(
            f"/World/friction_patch_{group}",
            layout[-1],
            top_z_m=0.0,
            mu_s=region.mu_s,
            mu_d=region.mu_d,
            color=(0.42, 0.43, 0.40),
            bind_terrain_appearance=False,
        )
        _set_prim_visible(patch_path, False)
        mu_s_applied, mu_d_applied = _read_material_friction(patch_path)
        applied = FrictionRegion(
            rect=region.rect,
            mu_s=mu_s_applied,
            mu_d=mu_d_applied,
            restitution=region.restitution,
        )
        self._regions.append(applied)
        self._friction_prim_paths.append((region.rect, patch_path))
        all_paths = [*surrounding_paths, patch_path]
        surrounding_mu_s, surrounding_mu_d = _read_material_friction(
            surrounding_paths[0]
        )
        self._realistic_scene_friction_topology = {
            "enabled": True,
            "mode": "coplanar_segmented_collision_under_continuous_pbr",
            "region": {
                "cx": region.rect.cx,
                "cy": region.rect.cy,
                "hx": region.rect.hx,
                "hy": region.rect.hy,
            },
            "requested_static_friction": float(region.mu_s),
            "requested_dynamic_friction": float(region.mu_d),
            "readback_static_friction": mu_s_applied,
            "readback_dynamic_friction": mu_d_applied,
            "nominal_requested_static_friction": nominal_mu_s,
            "nominal_requested_dynamic_friction": nominal_mu_d,
            "nominal_readback_static_friction": surrounding_mu_s,
            "nominal_readback_dynamic_friction": surrounding_mu_d,
            "isaac_default_ground_colliders_disabled": disabled_default,
            "realistic_scene_floor_colliders_disabled": disabled_scene,
            "surrounding_collision_paths": surrounding_paths,
            "operator_collision_path": patch_path,
            "collision_segment_count": len(all_paths),
            "collision_segment_visuals_hidden": True,
            "continuous_pbr_surface_visible": True,
            "surface_predeformed": False,
            "operator_region_has_no_visual_boundary": True,
            "operator_collision_geometry_visible": False,
        }
        print(
            f"[isaac] O1 realistic coplanar topology: requested mu_s/mu_d="
            f"{region.mu_s:.3f}/{region.mu_d:.3f}, readback="
            f"{mu_s_applied:.3f}/{mu_d_applied:.3f}, segments={len(all_paths)}, "
            f"base_colliders_disabled={disabled_default + disabled_scene}"
        )

    def friction_topology_telemetry(self) -> dict[str, object]:
        """Return O1 material readback and no-visual-leak topology assertions."""
        if self._realistic_scene_friction_topology is None:
            return {"enabled": False}
        return dict(self._realistic_scene_friction_topology)

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
        """Apply the legacy instantaneous velocity jump used by the frozen controlled core."""
        root_state = self._robot.data.root_state_w.clone()
        dv = np.asarray(impulse_xy_ns, dtype=np.float64) / self.mass_kg
        root_state[0, 7] += float(dv[0])
        root_state[0, 8] += float(dv[1])
        root_state[0, 12] += float(yaw_impulse_nms) / float(self._cfg.yaw_inertia_kgm2)
        self._robot.write_root_state_to_sim(root_state)

    def start_push_pulse(
        self,
        impulse_xy_ns: np.ndarray,
        yaw_impulse_nms: float,
        duration_s: float,
        application_point_body_m: np.ndarray,
    ) -> None:
        """Schedule a finite PhysX wrench at an explicit body-frame point (realistic O6)."""
        impulse = np.asarray(impulse_xy_ns, dtype=np.float64)
        point = np.asarray(application_point_body_m, dtype=np.float64)
        if impulse.shape != (2,):
            raise ValueError("push-pulse impulse must have shape (2,)")
        if point.shape != (3,):
            raise ValueError("push-pulse application point must have shape (3,)")
        if duration_s <= 0.0:
            raise ValueError("push-pulse duration must be positive")
        if self._push_pulse is not None:
            raise RuntimeError("a push pulse is already active")
        steps = max(1, int(round(float(duration_s) / self.dt)))
        effective_duration_s = steps * self.dt
        force_world_n = np.array(
            [impulse[0] / effective_duration_s, impulse[1] / effective_duration_s, 0.0],
            dtype=np.float64,
        )
        self._push_pulse = {
            "requested_impulse_xy_ns": impulse.copy(),
            "requested_yaw_impulse_nms": float(yaw_impulse_nms),
            "requested_duration_s": float(duration_s),
            "effective_duration_s": effective_duration_s,
            "application_point_body_m": point.copy(),
            "force_world_n": force_world_n,
            "extra_yaw_torque_nm": float(yaw_impulse_nms) / effective_duration_s,
            "target_steps": steps,
            "remaining_steps": steps,
            "applied_steps": 0,
            "commanded_linear_impulse_xy_ns": np.zeros(2, dtype=np.float64),
            "commanded_torque_impulse_xyz_nms": np.zeros(3, dtype=np.float64),
            "peak_force_n": 0.0,
            "peak_torque_nm": 0.0,
        }
        self._push_last_telemetry = self._serialize_push_pulse(self._push_pulse, active=True)

    @staticmethod
    def _serialize_push_pulse(pulse: dict[str, object], *, active: bool) -> dict[str, object]:
        return {
            "mode": "finite_physx_wrench_at_body_point",
            "active": active,
            "complete": int(pulse["remaining_steps"]) == 0,
            "requested_impulse_xy_ns": np.asarray(
                pulse["requested_impulse_xy_ns"], dtype=np.float64
            ).tolist(),
            "requested_yaw_impulse_nms": float(pulse["requested_yaw_impulse_nms"]),
            "requested_duration_s": float(pulse["requested_duration_s"]),
            "effective_duration_s": float(pulse["effective_duration_s"]),
            "application_point_body_m": np.asarray(
                pulse["application_point_body_m"], dtype=np.float64
            ).tolist(),
            "target_steps": int(pulse["target_steps"]),
            "remaining_steps": int(pulse["remaining_steps"]),
            "applied_steps": int(pulse["applied_steps"]),
            "commanded_linear_impulse_xy_ns": np.asarray(
                pulse["commanded_linear_impulse_xy_ns"], dtype=np.float64
            ).tolist(),
            "commanded_torque_impulse_xyz_nms": np.asarray(
                pulse["commanded_torque_impulse_xyz_nms"], dtype=np.float64
            ).tolist(),
            "peak_force_n": float(pulse["peak_force_n"]),
            "peak_torque_nm": float(pulse["peak_torque_nm"]),
        }

    def push_telemetry(self) -> dict[str, object]:
        """Return the requested and actually commanded finite-wrench impulse."""
        if self._push_pulse is not None:
            return self._serialize_push_pulse(self._push_pulse, active=True)
        return dict(self._push_last_telemetry)

    def add_collapse_regions(self, regions: list[CollapseRegion]) -> None:
        """Spawn legacy friction-swap plates or realistic impulse-damaged support cells."""
        env_origin = self._env.scene.env_origins[0].cpu().numpy()
        for region in regions:
            if region.damage_threshold_ns is not None and region.drop_m > 0.0:
                self._add_realistic_collapse_region(region)
                continue
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
            self._bind_episode_terrain_appearance(prim_path)
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
                    "mode": "legacy_dwell_friction_swap",
                }
            )

    def _spawn_collapse_surface_box(
        self,
        prim_path: str,
        rect: Rect,
        *,
        top_z_m: float,
        mu_s: float,
        mu_d: float,
        color: tuple[float, float, float],
        bind_terrain_appearance: bool = True,
    ) -> str:
        """Spawn one static box whose top face lies at a requested local ground height."""
        env_origin = self._env.scene.env_origins[0].cpu().numpy()
        thickness = max(float(self._cfg.patch_thickness_m), 0.012)
        config = self._sim_utils.CuboidCfg(
            size=(2.0 * rect.hx, 2.0 * rect.hy, thickness),
            collision_props=self._sim_utils.CollisionPropertiesCfg(),
            physics_material=self._sim_utils.RigidBodyMaterialCfg(
                static_friction=float(mu_s),
                dynamic_friction=float(mu_d),
                friction_combine_mode="min",
            ),
            visual_material=self._sim_utils.PreviewSurfaceCfg(diffuse_color=color, roughness=0.88),
        )
        config.func(
            prim_path,
            config,
            translation=(
                rect.cx + float(env_origin[0]),
                rect.cy + float(env_origin[1]),
                float(env_origin[2]) + top_z_m - thickness / 2.0,
            ),
        )
        if bind_terrain_appearance:
            self._bind_episode_terrain_appearance(prim_path)
        return prim_path

    def _add_realistic_collapse_region(self, region: CollapseRegion) -> None:
        """Split the ground, add a lower catch bed, and author removable support cells."""
        if self._collapse_ground_segmented:
            raise RuntimeError(
                "realistic O3 currently supports exactly one collapse region/episode"
            )
        disabled_default = self._disable_default_ground_collision_for_segmented_surface()
        disabled_scene = self._disable_realistic_scene_floor_for_operator_topology()
        disabled = disabled_default + disabled_scene
        continuous_visual_ground = self._realistic_scene_has_continuous_opaque_ground()
        route_appearance_hidden = (
            False
            if continuous_visual_ground
            else self._hide_realistic_scene_route_appearance_for_operator_topology()
        )
        self._collapse_ground_segmented = True
        group = self._n_patches
        self._n_patches += 1
        half_extent = 0.5 * float(self._cfg.get("collapse_ground_extent_m", 30.0))
        left = region.rect.cx - region.rect.hx
        right = region.rect.cx + region.rect.hx
        bottom = region.rect.cy - region.rect.hy
        top = region.rect.cy + region.rect.hy
        if not (
            -half_extent < left < right < half_extent
            and -half_extent < bottom < top < half_extent
        ):
            raise ValueError("collapse region must lie inside the segmented ground extent")
        surroundings = _coplanar_segmented_ground_layout(region.rect, half_extent)[:-1]
        nominal_mu_s, nominal_mu_d = self._nominal_ground_friction_readback()
        surrounding_paths: list[str] = []
        for index, rect in enumerate(surroundings):
            surrounding_paths.append(
                self._spawn_collapse_surface_box(
                    f"/World/collapse_ground_{group}_{index}",
                    rect,
                    top_z_m=0.0,
                    mu_s=nominal_mu_s,
                    mu_d=nominal_mu_d,
                    color=(0.42, 0.43, 0.40),
                    bind_terrain_appearance=not continuous_visual_ground,
                )
            )
            if continuous_visual_ground:
                _set_prim_visible(surrounding_paths[-1], False)
        catch_path = self._spawn_collapse_surface_box(
            f"/World/collapse_catch_{group}",
            region.rect,
            top_z_m=-region.drop_m,
            mu_s=region.mu_collapsed,
            mu_d=region.mu_collapsed,
            color=(0.19, 0.17, 0.14),
            bind_terrain_appearance=False,
        )
        if continuous_visual_ground:
            _set_prim_visible(catch_path, False)
        cells = support_cell_layout(
            region.rect,
            cells_xy=(4, 4),
            gap_m=0.0 if continuous_visual_ground else 0.006,
        )
        cell_paths = []
        for index, cell in enumerate(cells):
            cell_paths.append(
                self._spawn_collapse_surface_box(
                    f"/World/collapse_cell_{group}_{index}",
                    cell,
                    top_z_m=0.0,
                    mu_s=nominal_mu_s,
                    mu_d=nominal_mu_d,
                    color=(0.68, 0.70, 0.66),
                    bind_terrain_appearance=not continuous_visual_ground,
                )
            )
            if continuous_visual_ground:
                _set_prim_visible(cell_paths[-1], False)
        failed_indices = collapsed_cell_indices(
            cells, region=region.rect, residual_support=region.residual_support
        )
        mu_s_applied, mu_d_applied = _read_material_friction(cell_paths[0])
        continuous_visual = self._initialize_triggered_collapse_visual_deformation(
            region
        )
        self._collapse.append(
            {
                "mode": "impulse_triggered_support_topology",
                "region": region.rect,
                "mu_intact": mu_d_applied,
                "mu_collapsed": region.mu_collapsed,
                "damage_threshold_ns": float(region.damage_threshold_ns),
                "drop_m": region.drop_m,
                "residual_support": region.residual_support,
                "damage_state": CollapseDamageState(),
                "last_damage_update": None,
                "step_index": 0,
                "collapsed": False,
                "cell_paths": cell_paths,
                "cell_rects": cells,
                "failed_cell_indices": failed_indices,
                "disabled_cell_colliders": 0,
                "catch_path": catch_path,
                "group": group,
                "fracture_visual_paths": [],
                "fracture_source_cells": [],
                "visual_sync_trigger_step": None,
                "default_ground_colliders_disabled": disabled,
                "isaac_default_ground_colliders_disabled": disabled_default,
                "realistic_scene_floor_colliders_disabled": disabled_scene,
                "nominal_route_appearance_hidden": route_appearance_hidden,
                "surrounding_collision_paths": surrounding_paths,
                "collision_geometry_visible": not continuous_visual_ground,
                "continuous_visual_surface": continuous_visual,
                "support_cell_gap_m": 0.0 if continuous_visual_ground else 0.006,
                "nominal_requested_static_friction": nominal_mu_s,
                "nominal_requested_dynamic_friction": nominal_mu_d,
                "nominal_readback_static_friction": mu_s_applied,
                "nominal_readback_dynamic_friction": mu_d_applied,
            }
        )
        print(
            f"[isaac] O3 realistic support: mu={mu_s_applied:.3f}/{mu_d_applied:.3f}, "
            f"threshold={region.damage_threshold_ns:.1f} Ns, drop={region.drop_m:.3f} m, "
            f"residual={region.residual_support:.2f}, cells={len(cells)}"
        )

    def _initialize_triggered_collapse_visual_deformation(
        self, region: CollapseRegion
    ) -> dict[str, object]:
        """Keep O3 visually nominal until measured load triggers the topology transition."""
        if not self._realistic_scene_has_continuous_opaque_ground():
            return {"applied": False, "reason": "continuous_opaque_ground_unavailable"}
        import omni.usd
        from pxr import Sdf, UsdGeom

        path = str(self._realistic_scene_route_appearance_path)
        mesh = UsdGeom.Mesh(omni.usd.get_context().get_stage().GetPrimAtPath(path))
        points = list(mesh.GetPointsAttr().Get() or [])
        if not points:
            raise RuntimeError("continuous realistic ground has no authored points")
        if self._realistic_scene_route_original_points is not None:
            raise RuntimeError("continuous realistic ground deformation is already initialized")
        original = [(float(point[0]), float(point[1]), float(point[2])) for point in points]
        prim = mesh.GetPrim()
        prim.CreateAttribute("kino:o3VisualDeformationApplied", Sdf.ValueTypeNames.Bool).Set(True)
        prim.CreateAttribute("kino:o3VisualSurfacePredeformed", Sdf.ValueTypeNames.Bool).Set(False)
        self._realistic_scene_route_original_points = original
        self._realistic_scene_route_visual_offsets = np.zeros(len(original), dtype=np.float64)
        self._realistic_scene_route_visual_deformation = {
            "applied": True,
            "mode": "measured_impulse_triggered_irregular_pbr_collapse",
            "mesh_path": path,
            "vertex_count": len(original),
            "deformed_vertex_count": 0,
            "minimum_visual_offset_m": 0.0,
            "maximum_visual_offset_m": 0.0,
            "visual_triggered": False,
            "visual_trigger_step": None,
            "surface_predeformed": False,
            "operator_region_has_no_pretrigger_visual_boundary": True,
            "operator_collision_geometry_visible": False,
            "requested_drop_m": float(region.drop_m),
        }
        return dict(self._realistic_scene_route_visual_deformation)

    def _apply_triggered_collapse_visual_deformation(
        self,
        state: dict[str, object],
        *,
        trigger_center_xy_m: tuple[float, float],
        trigger_step: int,
    ) -> int:
        original = self._realistic_scene_route_original_points
        offsets = self._realistic_scene_route_visual_offsets
        path = self._realistic_scene_route_appearance_path
        if original is None or offsets is None or path is None:
            return 0
        x = np.asarray([point[0] for point in original], dtype=np.float64)
        y = np.asarray([point[1] for point in original], dtype=np.float64)
        updated = _triggered_collapse_offset_update(
            offsets,
            x,
            y,
            center_xy_m=trigger_center_xy_m,
            region=state["region"],
            drop_m=float(state["drop_m"]),
        )
        changed = int(np.count_nonzero(updated < offsets - 1.0e-6))
        if changed == 0:
            raise RuntimeError("O3 triggered but continuous collapse mesh had no affected vertices")
        import omni.usd
        from pxr import Gf, Sdf, UsdGeom

        prim = omni.usd.get_context().get_stage().GetPrimAtPath(path)
        if not prim.IsValid() or not prim.IsA(UsdGeom.Mesh):
            raise RuntimeError("continuous realistic ground disappeared during O3 trigger")
        UsdGeom.Mesh(prim).GetPointsAttr().Set(
            [
                Gf.Vec3f(px, py, pz + float(dz))
                for (px, py, pz), dz in zip(original, updated, strict=True)
            ]
        )
        self._realistic_scene_route_visual_offsets = updated
        audit = self._realistic_scene_route_visual_deformation
        if audit is not None:
            audit["deformed_vertex_count"] = int(np.count_nonzero(updated < -1.0e-6))
            audit["minimum_visual_offset_m"] = float(updated.min())
            audit["maximum_visual_offset_m"] = float(updated.max())
            audit["visual_triggered"] = True
            audit["visual_trigger_step"] = int(trigger_step)
            audit["trigger_center_xy_m"] = [
                float(trigger_center_xy_m[0]),
                float(trigger_center_xy_m[1]),
            ]
        prim.CreateAttribute("kino:o3VisualTriggerStep", Sdf.ValueTypeNames.Int).Set(
            int(trigger_step)
        )
        prim.CreateAttribute("kino:o3DeformedVertexCount", Sdf.ValueTypeNames.Int).Set(
            int(np.count_nonzero(updated < -1.0e-6))
        )
        prim.CreateAttribute("kino:o3VisualMinimumOffsetM", Sdf.ValueTypeNames.Double).Set(
            float(updated.min())
        )
        state["continuous_visual_surface"] = dict(audit or {})
        return changed

    def _spawn_collapse_fracture_visuals(self, state: dict[str, object]) -> list[str]:
        """Author two slumped triangular surface facets for every physically removed O3 cell."""
        import omni.usd
        from pxr import Gf, Sdf, UsdGeom, UsdShade

        stage = omni.usd.get_context().get_stage()
        env_origin = self._env.scene.env_origins[0].detach().cpu().numpy()
        group = int(state["group"])
        drop_m = float(state["drop_m"])
        paths: list[str] = []
        fallback_material_path = f"/World/Looks/collapse_fragment_{group}"
        fallback_material = UsdShade.Material.Define(stage, fallback_material_path)
        shader = UsdShade.Shader.Define(stage, f"{fallback_material_path}/PBR")
        shader.CreateIdAttr("UsdPreviewSurface")
        shader.CreateInput("diffuseColor", Sdf.ValueTypeNames.Color3f).Set(
            Gf.Vec3f(0.31, 0.25, 0.18)
        )
        shader.CreateInput("roughness", Sdf.ValueTypeNames.Float).Set(0.94)
        fallback_material.CreateSurfaceOutput().ConnectToSource(
            shader.ConnectableAPI(), "surface"
        )
        for cell_index in state["failed_cell_indices"]:
            cell = state["cell_rects"][cell_index]
            x0 = float(cell.cx - cell.hx + env_origin[0])
            x1 = float(cell.cx + cell.hx + env_origin[0])
            y0 = float(cell.cy - cell.hy + env_origin[1])
            y1 = float(cell.cy + cell.hy + env_origin[1])
            z0 = float(env_origin[2])
            pieces = (
                (
                    (x0, y0, z0 - 0.015),
                    (x1, y0, z0 - 0.48 * drop_m),
                    (x1, y1, z0 - 0.68 * drop_m),
                ),
                (
                    (x0, y0, z0 - 0.015),
                    (x1, y1, z0 - 0.68 * drop_m),
                    (x0, y1, z0 - 0.38 * drop_m),
                ),
            )
            for piece_index, points in enumerate(pieces):
                path = f"/World/collapse_fragment_{group}_{cell_index}_{piece_index}"
                mesh = UsdGeom.Mesh.Define(stage, path)
                mesh.CreatePointsAttr([Gf.Vec3f(*point) for point in points])
                mesh.CreateFaceVertexCountsAttr([3])
                mesh.CreateFaceVertexIndicesAttr([0, 1, 2])
                mesh.CreateSubdivisionSchemeAttr("none")
                mesh.CreateDoubleSidedAttr(True)
                prim = mesh.GetPrim()
                prim.CreateAttribute("kino:failureState", Sdf.ValueTypeNames.String).Set(
                    "fractured_after_impulse_trigger"
                )
                prim.CreateAttribute("kino:sourceCellIndex", Sdf.ValueTypeNames.Int).Set(
                    int(cell_index)
                )
                if self._terrain_appearance is None:
                    UsdShade.MaterialBindingAPI(prim).Bind(fallback_material)
                else:
                    bind_omnipbr_material(stage, path, self._terrain_appearance)
                paths.append(path)
        return paths

    def _update_collapse(self) -> None:
        """Advance either the controlled dwell proxy or measured per-foot damage state."""
        if not self._collapse:
            return
        env_origin = self._env.scene.env_origins[0].cpu().numpy()
        pos = self._robot.data.root_pos_w[0].cpu().numpy()[:2] - env_origin[:2]
        for st in self._collapse:
            if st["collapsed"]:
                continue
            if st["mode"] == "impulse_triggered_support_topology":
                foot_pos = self._robot.data.body_pos_w[0, self._foot_ids].detach().cpu().numpy()
                foot_pos = foot_pos[:, :2] - env_origin[:2].reshape(1, 2)
                contact = self._contact.data.net_forces_w[0, self._contact_foot_ids]
                normal_force = contact[:, 2].detach().cpu().numpy()
                update = step_collapse_damage(
                    st["damage_state"],
                    region=st["region"],
                    foot_xy_m=foot_pos,
                    foot_normal_force_n=normal_force,
                    dt_s=self.dt,
                    damage_threshold_ns=st["damage_threshold_ns"],
                    step_index=st["step_index"],
                )
                st["damage_state"] = update.state
                st["last_damage_update"] = damage_update_to_dict(update)
                st["step_index"] += 1
                if update.triggered_now:
                    inside = np.asarray(
                        [st["region"].contains(position) for position in foot_pos],
                        dtype=bool,
                    )
                    weights = np.maximum(normal_force, 0.0) * inside
                    if float(weights.sum()) > 1.0e-9:
                        trigger_center = tuple(
                            np.average(foot_pos, axis=0, weights=weights).tolist()
                        )
                    else:
                        trigger_center = (float(st["region"].cx), float(st["region"].cy))
                    changed = 0
                    for index in st["failed_cell_indices"]:
                        path = st["cell_paths"][index]
                        changed += len(_set_collision_enabled(path, False))
                        _set_prim_visible(path, False)
                    continuous_visual = st.get("continuous_visual_surface", {})
                    if continuous_visual.get("applied") is True:
                        self._apply_triggered_collapse_visual_deformation(
                            st,
                            trigger_center_xy_m=trigger_center,
                            trigger_step=int(update.state.trigger_step or 0),
                        )
                        fracture_paths = []
                    else:
                        fracture_paths = self._spawn_collapse_fracture_visuals(st)
                    st["disabled_cell_colliders"] = changed
                    st["fracture_visual_paths"] = fracture_paths
                    st["fracture_source_cells"] = list(st["failed_cell_indices"])
                    st["visual_sync_trigger_step"] = int(update.state.trigger_step or 0)
                    st["collapsed"] = True
                    print(
                        f"[isaac] O3 topology triggered at "
                        f"{update.state.normal_impulse_ns:.2f} Ns: "
                        f"disabled {len(st['failed_cell_indices'])} support cells / "
                        f"{changed} colliders; spawned {len(fracture_paths)} fracture facets; "
                        f"catch bed z=-{st['drop_m']:.3f} m"
                    )
                continue
            if not st["region"].contains(pos):
                continue
            st["dwell"] += self.dt
            if st["dwell"] >= st["trigger_dwell_s"]:
                _set_material_friction(st["prim_path"], st["mu_collapsed"], st["mu_collapsed"])
                st["collapsed"] = True
                print(f"[isaac] O3 collapse triggered: mu -> {st['mu_collapsed']:.3f}")

    def collapse_telemetry(self) -> dict[str, object]:
        """Return privileged O3 trigger and topology readback for runtime-manifest QA."""
        regions: list[dict[str, object]] = []
        for state in self._collapse:
            row: dict[str, object] = {
                "mode": state["mode"],
                "collapsed": state["collapsed"],
                "mu_intact": state["mu_intact"],
                "mu_collapsed": state["mu_collapsed"],
            }
            if state["mode"] == "impulse_triggered_support_topology":
                row.update(
                    {
                        "damage_threshold_ns": state["damage_threshold_ns"],
                        "drop_m": state["drop_m"],
                        "residual_support": state["residual_support"],
                        "failed_support_cells": len(state["failed_cell_indices"]),
                        "disabled_cell_colliders": state["disabled_cell_colliders"],
                        "fracture_visual_paths": list(state["fracture_visual_paths"]),
                        "fracture_visual_count": len(state["fracture_visual_paths"]),
                        "fracture_source_cells": list(state["fracture_source_cells"]),
                        "visual_sync_trigger_step": state["visual_sync_trigger_step"],
                        "default_ground_colliders_disabled": state[
                            "default_ground_colliders_disabled"
                        ],
                        "last_damage_update": state["last_damage_update"],
                        "collision_geometry_visible": state.get(
                            "collision_geometry_visible", True
                        ),
                        "support_cell_gap_m": state.get("support_cell_gap_m", 0.006),
                        "continuous_visual_surface": dict(
                            state.get("continuous_visual_surface", {})
                        ),
                        "nominal_requested_static_friction": state.get(
                            "nominal_requested_static_friction"
                        ),
                        "nominal_requested_dynamic_friction": state.get(
                            "nominal_requested_dynamic_friction"
                        ),
                        "nominal_readback_static_friction": state.get(
                            "nominal_readback_static_friction"
                        ),
                        "nominal_readback_dynamic_friction": state.get(
                            "nominal_readback_dynamic_friction"
                        ),
                    }
                )
            else:
                row["dwell_s"] = state["dwell"]
                row["trigger_dwell_s"] = state["trigger_dwell_s"]
            regions.append(row)
        return {"enabled": bool(regions), "regions": regions}

    def add_blocking_regions(self, regions: list[BlockingRegion]) -> None:
        """Spawn legacy hidden walls or rendered transparent/low-contrast O8 obstacles."""
        env_origin = self._env.scene.env_origins[0].cpu().numpy()
        for region in regions:
            prim_path = f"/World/wall_{self._n_patches}"
            self._n_patches += 1
            h_wall = (
                float(getattr(self._cfg, "wall_height_m", 0.8))
                if region.height_m is None
                else float(region.height_m)
            )
            if region.geometry_kind == "transparent_acrylic":
                visual_material = self._sim_utils.GlassMdlCfg(
                    glass_color=(0.94, 0.98, 1.0),
                    frosting_roughness=max(
                        0.0, min(0.12, 1.0 - region.optical_transmission)
                    ),
                    thin_walled=False,
                    glass_ior=1.49,
                )
            elif region.geometry_kind == "occluded_low_bar":
                visual_material = self._sim_utils.PreviewSurfaceCfg(
                    diffuse_color=(0.30, 0.29, 0.25), roughness=0.72
                )
            else:
                visual_material = self._sim_utils.PreviewSurfaceCfg(
                    diffuse_color=(0.9, 0.9, 0.9), opacity=0.0
                )
            wall_cfg = self._sim_utils.CuboidCfg(
                size=(2.0 * region.rect.hx, 2.0 * region.rect.hy, h_wall),
                collision_props=self._sim_utils.CollisionPropertiesCfg(
                    collision_enabled=region.collision_enabled
                ),
                physics_material=self._sim_utils.RigidBodyMaterialCfg(
                    static_friction=1.0, dynamic_friction=1.0
                ),
                visual_material=visual_material,
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
            print(
                f"[isaac] O8 {region.geometry_kind} at "
                f"({region.rect.cx:.2f}, {region.rect.cy:.2f}); "
                f"height={h_wall:.3f} m collision={region.collision_enabled}"
            )
            self._blocking.append(
                {
                    "region": region,
                    "prim_path": prim_path,
                    "height_m": h_wall,
                    "geometry_kind": region.geometry_kind,
                    "collision_requested": region.collision_enabled,
                }
            )

    def blocking_telemetry(self) -> dict[str, object]:
        """Read O8 collider state and visual-material binding from the composed USD."""
        import omni.usd
        from pxr import Usd, UsdPhysics, UsdShade

        stage = omni.usd.get_context().get_stage()
        rows = []
        for state in self._blocking:
            prim = stage.GetPrimAtPath(str(state["prim_path"]))
            collider_rows = []
            materials = []
            if prim.IsValid():
                for child in Usd.PrimRange(prim):
                    if child.HasAPI(UsdPhysics.CollisionAPI):
                        attr = UsdPhysics.CollisionAPI(child).GetCollisionEnabledAttr()
                        collider_rows.append(
                            {
                                "path": str(child.GetPath()),
                                "enabled": bool(attr.Get()) if attr.IsValid() else True,
                            }
                        )
                    binding = UsdShade.MaterialBindingAPI(child).GetDirectBinding()
                    path = str(binding.GetMaterialPath())
                    if path:
                        materials.append(path)
            material_parameters = []
            for path in sorted(set(materials)):
                shader = stage.GetPrimAtPath(f"{path}/Shader")
                material_parameters.append(
                    {
                        "material_path": path,
                        "shader_valid": shader.IsValid(),
                        "glass_ior": (
                            shader.GetAttribute("inputs:glass_ior").Get()
                            if shader.IsValid()
                            else None
                        ),
                        "frosting_roughness": (
                            shader.GetAttribute("inputs:frosting_roughness").Get()
                            if shader.IsValid()
                            else None
                        ),
                        "thin_walled": (
                            shader.GetAttribute("inputs:thin_walled").Get()
                            if shader.IsValid()
                            else None
                        ),
                    }
                )
            region = state["region"]
            rows.append(
                {
                    "prim_path": str(state["prim_path"]),
                    "prim_valid": prim.IsValid(),
                    "geometry_kind": str(state["geometry_kind"]),
                    "height_m": float(state["height_m"]),
                    "size_xyz_m": [
                        2.0 * float(region.rect.hx),
                        2.0 * float(region.rect.hy),
                        float(state["height_m"]),
                    ],
                    "collision_requested": bool(state["collision_requested"]),
                    "colliders": collider_rows,
                    "visual_material_paths": sorted(set(materials)),
                    "visual_material_parameters": material_parameters,
                    "optical_transmission_metadata": float(region.optical_transmission),
                }
            )
        return {"enabled": bool(rows), "obstacles": rows}

    def add_support_loss_regions(
        self,
        regions: list[SupportLossRegion],
        height_m: float | None = None,
        geometry_kind: str = "box_ridge",
    ) -> None:
        """Spawn a measured high-centering geometry; residual support is never forced in Isaac.

        ``height_m`` overrides the config ridge height for this call (taller lip ⇒ more feet
        unloaded ⇒ lower support); ``None`` keeps ``cfg.ridge_height_m`` so existing callers
        are unchanged. ``rounded_ridge`` authors a transverse cylinder; ``pallet_edge`` and the
        legacy ``box_ridge`` author a cuboid.  Foot support and belly contact are sensor readbacks,
        not values copied from ``SupportLossRegion.residual_support``."""
        if geometry_kind not in {
            "box_ridge",
            "rounded_ridge",
            "longitudinal_rounded_ridge",
            "central_pallet_runner",
            "pallet_edge",
        }:
            raise ValueError(f"unsupported O9 geometry {geometry_kind!r}")
        env_origin = self._env.scene.env_origins[0].cpu().numpy()
        h_ridge = float(
            height_m if height_m is not None else getattr(self._cfg, "ridge_height_m", 0.18)
        )
        for region in regions:
            prim_path = f"/World/ridge_{self._n_patches}"
            self._n_patches += 1
            common = {
                "collision_props": self._sim_utils.CollisionPropertiesCfg(),
                "physics_material": self._sim_utils.RigidBodyMaterialCfg(
                    static_friction=0.8, dynamic_friction=0.8
                ),
                "visual_material": self._sim_utils.PreviewSurfaceCfg(
                    diffuse_color=(0.36, 0.25, 0.14), roughness=0.86
                ),
            }
            if geometry_kind in {"rounded_ridge", "longitudinal_rounded_ridge"}:
                longitudinal = geometry_kind == "longitudinal_rounded_ridge"
                radius = float(region.rect.hy if longitudinal else region.rect.hx)
                ridge_cfg = self._sim_utils.CylinderCfg(
                    radius=radius,
                    height=2.0 * (region.rect.hx if longitudinal else region.rect.hy),
                    axis="X" if longitudinal else "Y",
                    **common,
                )
                center_z = h_ridge - radius
            else:
                ridge_cfg = self._sim_utils.CuboidCfg(
                    size=(2.0 * region.rect.hx, 2.0 * region.rect.hy, h_ridge),
                    **common,
                )
                center_z = h_ridge / 2.0
            ridge_cfg.func(
                prim_path,
                ridge_cfg,
                translation=(
                    region.rect.cx + float(env_origin[0]),
                    region.rect.cy + float(env_origin[1]),
                    float(env_origin[2]) + center_z,
                ),
            )
            self._bind_episode_terrain_appearance(prim_path)
            width_axis_half = (
                region.rect.hy
                if geometry_kind == "longitudinal_rounded_ridge"
                else region.rect.hx
            )
            print(
                f"[isaac] O9 {geometry_kind} at ({region.rect.cx:.2f},{region.rect.cy:.2f}) "
                f"height={h_ridge:.3f} width={2.0 * width_axis_half:.3f}"
            )
            self._support.append(
                {
                    "region": region.rect,
                    "prim_path": prim_path,
                    "geometry_kind": geometry_kind,
                    "ridge_height_m": h_ridge,
                    "ridge_width_m": 2.0 * width_axis_half,
                    "target_residual_support": region.residual_support,
                    "min_measured_support": 1.0,
                    "max_belly_contact_force_n": 0.0,
                    "belly_contact_steps": 0,
                    "consecutive_belly_contact_steps": 0,
                    "max_consecutive_belly_contact_steps": 0,
                    "belly_contact_force_sum_n": 0.0,
                    "mean_belly_contact_force_n": 0.0,
                    "belly_contact_duty_cycle": 0.0,
                    "max_base_height_m": 0.0,
                    "measurement_steps": 0,
                    "max_nonfoot_contact_by_body_n": {
                        name: 0.0 for name in self._contact_all_names if "foot" not in name
                    },
                }
            )

    def _update_high_centering_telemetry(self) -> None:
        if not self._support:
            return
        env_origin = self._env.scene.env_origins[0].detach().cpu().numpy()
        root_pos = self._robot.data.root_pos_w[0].detach().cpu().numpy() - env_origin
        foot_positions = self._robot.data.body_pos_w[0, self._foot_ids].detach().cpu().numpy()
        foot_positions -= env_origin.reshape(1, 3)
        support = self._measure_support()
        all_forces = self._contact.data.net_forces_w[0, self._contact_all_ids]
        all_force_norms = self._torch.linalg.norm(all_forces, dim=1).detach().cpu().numpy()
        history = getattr(self._contact.data, "net_forces_w_history", None)
        if history is not None:
            # ContactSensor is sampled at physics rate while this callback runs at policy rate.
            # Use its complete history window so a short chassis impulse cannot disappear merely
            # because it occurred one PhysX substep before the last sample.
            history_norms = self._torch.linalg.norm(
                history[0, :, self._contact_all_ids, :], dim=-1
            )
            all_force_norms = self._torch.maximum(
                self._torch.as_tensor(all_force_norms, device=history_norms.device),
                history_norms.max(dim=0).values,
            ).detach().cpu().numpy()
        nonfoot_force_by_name = {
            name: float(all_force_norms[index])
            for index, name in enumerate(self._contact_all_names)
            if "foot" not in name
        }
        belly_force = 0.0
        if self._contact_base_ids:
            if history is None:
                base_forces = self._contact.data.net_forces_w[0, self._contact_base_ids]
                belly_force = float(self._torch.linalg.norm(base_forces, dim=1).sum().item())
            else:
                base_history = history[0, :, self._contact_base_ids, :]
                belly_force = float(
                    self._torch.linalg.norm(base_history, dim=-1).sum(dim=1).max().item()
                )
        for state in self._support:
            region = state["region"]
            near_geometry = (
                abs(float(root_pos[0]) - region.cx) <= region.hx + 0.55
                and abs(float(root_pos[1]) - region.cy) <= region.hy + 0.35
            )
            engaged = near_geometry or region.contains(root_pos[:2]) or any(
                region.contains(position[:2]) for position in foot_positions
            )
            if not engaged:
                continue
            state["min_measured_support"] = min(state["min_measured_support"], support)
            state["max_belly_contact_force_n"] = max(
                state["max_belly_contact_force_n"], belly_force
            )
            if belly_force > 2.0:
                state["belly_contact_steps"] += 1
                state["consecutive_belly_contact_steps"] += 1
                state["belly_contact_force_sum_n"] += belly_force
                state["max_consecutive_belly_contact_steps"] = max(
                    state["max_consecutive_belly_contact_steps"],
                    state["consecutive_belly_contact_steps"],
                )
            else:
                state["consecutive_belly_contact_steps"] = 0
            state["max_base_height_m"] = max(state["max_base_height_m"], float(root_pos[2]))
            state["measurement_steps"] += 1
            state["mean_belly_contact_force_n"] = state["belly_contact_force_sum_n"] / max(
                state["belly_contact_steps"], 1
            )
            state["belly_contact_duty_cycle"] = state["belly_contact_steps"] / max(
                state["measurement_steps"], 1
            )
            for name, force_n in nonfoot_force_by_name.items():
                state["max_nonfoot_contact_by_body_n"][name] = max(
                    state["max_nonfoot_contact_by_body_n"][name], force_n
                )

    def high_centering_telemetry(self) -> dict[str, object]:
        """Return O9 geometry parameters and contact-sensor outcomes."""
        return {
            "enabled": bool(self._support),
            "regions": [
                {
                    key: value
                    for key, value in state.items()
                    if key not in {"region", "prim_path"}
                }
                | {
                    "region": {
                        "cx": state["region"].cx,
                        "cy": state["region"].cy,
                        "hx": state["region"].hx,
                        "hy": state["region"].hy,
                    },
                    "prim_path": state["prim_path"],
                }
                for state in self._support
            ],
        }

    # --------------------------------------------- realistic O2 foot-local terramechanics

    def add_foot_compliance(
        self,
        config: FootTerramechanicsConfig,
        *,
        matched_control: bool = False,
    ) -> None:
        """Install O2 or its topology-matched flat, nominal-material control."""
        if self._foot_adhesion_cfg is not None:
            raise RuntimeError("foot terramechanics and adhesion need an explicit force mixer")
        if self._collapse_ground_segmented:
            raise RuntimeError(
                "only one segmented terrain topology operator is allowed per episode"
            )
        disabled_default = self._disable_default_ground_collision_for_segmented_surface()
        disabled_scene = self._disable_realistic_scene_floor_for_operator_topology()
        disabled = disabled_default + disabled_scene
        continuous_visual_ground = self._realistic_scene_has_continuous_opaque_ground()
        route_appearance_hidden = (
            False
            if continuous_visual_ground
            else self._hide_realistic_scene_route_appearance_for_operator_topology()
        )
        self._collapse_ground_segmented = True
        group = self._n_patches
        self._n_patches += 1
        region = config.region
        half_extent = 0.5 * float(self._cfg.get("collapse_ground_extent_m", 30.0))
        left = region.cx - region.hx
        right = region.cx + region.hx
        bottom = region.cy - region.hy
        top = region.cy + region.hy
        if not (
            -half_extent < left < right < half_extent
            and -half_extent < bottom < top < half_extent
        ):
            raise ValueError("compliance region must lie inside the segmented ground extent")
        nominal_mu_s, nominal_mu_d = self._nominal_ground_friction_readback()
        support_path, support_mesh = self._spawn_continuous_compliance_support_mesh(
            config,
            group=group,
            half_extent_m=half_extent,
            matched_control=matched_control,
            mu_s=nominal_mu_s,
            mu_d=nominal_mu_d,
        )
        paths = [support_path]
        if continuous_visual_ground:
            _set_prim_visible(support_path, False)
        visual_deformation = self._initialize_contact_local_ground_deformation(config)
        self._foot_terrain_cfg = config
        self._foot_terrain_states = [FootTerrainState() for _ in self._foot_ids]
        self._foot_terrain_updates = []
        self._foot_terrain_contacts_n = [0.0 for _ in self._foot_ids]
        self._foot_terrain_paths = paths
        self._foot_terrain_footprints = []
        self._foot_terrain_matched_control = bool(matched_control)
        support_readback = _read_material_friction(support_path)
        self._foot_terrain_topology = {
            "mode": "continuous_topology_matched_o2_heightfield_under_continuous_pbr",
            "matched_control": bool(matched_control),
            "collision_segment_count": 1,
            "inter_prim_collision_seam_count": 0,
            "vertex_count": len(support_mesh.points_xyz_m),
            "quad_count": len(support_mesh.face_vertex_counts),
            "longitudinal_axis": support_mesh.longitudinal_axis,
            "longitudinal_coordinates_m": list(
                support_mesh.longitudinal_coordinates_m
            ),
            "lateral_coordinates_m": list(support_mesh.lateral_coordinates_m),
            "central_surface_state": (
                "flat_nominal_material"
                if matched_control
                else "continuous_lowered_soft_heightfield"
            ),
            "minimum_surface_height_m": support_mesh.minimum_height_m,
            "support_material_readback": {
                "static": support_readback[0],
                "dynamic": support_readback[1],
            },
            "nominal_material": {"static": nominal_mu_s, "dynamic": nominal_mu_d},
            "collision_geometry_visible": False if continuous_visual_ground else True,
            "operator_region_has_no_precontact_visual_boundary": continuous_visual_ground,
        }
        print(
            f"[isaac] O2 {'matched flat control' if matched_control else 'per-foot ramped terrain'}: "
            f"sink={config.max_sink_depth_m:.3f} m, "
            f"shear_retention={config.shear_retention:.2f}, "
            f"vertical_k={config.vertical_stiffness_n_per_m:.1f} N/m, "
            f"base_colliders_disabled={disabled} "
            f"(isaac_default={disabled_default}, realistic_scene={disabled_scene}), "
            f"nominal_route_appearance_hidden={route_appearance_hidden}, "
            f"continuous_visual_deformation={visual_deformation.get('applied', False)}"
        )

    def _spawn_continuous_compliance_support_mesh(
        self,
        config: FootTerramechanicsConfig,
        *,
        group: int,
        half_extent_m: float,
        matched_control: bool,
        mu_s: float,
        mu_d: float,
    ) -> tuple[str, ContinuousSupportMesh]:
        """Author one static O2 collision mesh shared topologically by both lanes."""
        import omni.usd
        from pxr import Gf, Sdf, UsdGeom, UsdPhysics

        support = continuous_support_mesh(
            config,
            half_extent_m=half_extent_m,
            matched_control=matched_control,
        )
        env_origin = self._env.scene.env_origins[0].detach().cpu().numpy()
        stage = omni.usd.get_context().get_stage()
        prim_path = f"/World/compliance_support_{group}"
        mesh = UsdGeom.Mesh.Define(stage, prim_path)
        mesh.CreatePointsAttr(
            [
                Gf.Vec3f(
                    float(x + env_origin[0]),
                    float(y + env_origin[1]),
                    float(z + env_origin[2]),
                )
                for x, y, z in support.points_xyz_m
            ]
        )
        mesh.CreateFaceVertexCountsAttr(list(support.face_vertex_counts))
        mesh.CreateFaceVertexIndicesAttr(list(support.face_vertex_indices))
        mesh.CreateSubdivisionSchemeAttr("none")
        mesh.CreateDoubleSidedAttr(True)
        prim = mesh.GetPrim()
        UsdPhysics.CollisionAPI.Apply(prim).CreateCollisionEnabledAttr(True)
        UsdPhysics.MeshCollisionAPI.Apply(prim).CreateApproximationAttr().Set("none")
        material = UsdPhysics.MaterialAPI.Apply(prim)
        material.CreateStaticFrictionAttr(float(mu_s))
        material.CreateDynamicFrictionAttr(float(mu_d))
        material.CreateRestitutionAttr(0.0)
        prim.CreateAttribute("kino:o2SupportMode", Sdf.ValueTypeNames.String).Set(
            "continuous_topology_matched_heightfield"
        )
        prim.CreateAttribute("kino:o2LongitudinalAxis", Sdf.ValueTypeNames.String).Set(
            support.longitudinal_axis
        )
        prim.CreateAttribute("kino:o2MatchedControl", Sdf.ValueTypeNames.Bool).Set(
            bool(matched_control)
        )
        return prim_path, support

    def _spawn_compliance_ramp_box(
        self,
        prim_path: str,
        ramp: SinkRampSpec,
        *,
        surface_offset_z_m: float,
        mu_s: float,
        mu_d: float,
    ) -> str:
        """Spawn a pitched static box whose top face realizes one continuous sink ramp."""
        env_origin = self._env.scene.env_origins[0].cpu().numpy()
        thickness = max(float(self._cfg.patch_thickness_m), 0.012)
        half_pitch = 0.5 * ramp.pitch_rad
        # Rotation about +Y gives dz/dx < 0.  Account for the rotated top-face centre so the
        # requested surface, rather than the cuboid centre, lands on the analytic profile.
        center_x = ramp.center_x_m - math.sin(ramp.pitch_rad) * thickness / 2.0
        center_z = (
            surface_offset_z_m
            + ramp.center_top_z_m
            - math.cos(ramp.pitch_rad) * thickness / 2.0
        )
        config = self._sim_utils.CuboidCfg(
            size=(ramp.slope_length_m, ramp.width_m, thickness),
            collision_props=self._sim_utils.CollisionPropertiesCfg(),
            physics_material=self._sim_utils.RigidBodyMaterialCfg(
                static_friction=float(mu_s),
                dynamic_friction=float(mu_d),
                friction_combine_mode="min",
            ),
            visual_material=self._sim_utils.PreviewSurfaceCfg(
                diffuse_color=(0.31, 0.24, 0.17), roughness=0.9
            ),
        )
        config.func(
            prim_path,
            config,
            translation=(
                center_x + float(env_origin[0]),
                ramp.center_y_m + float(env_origin[1]),
                center_z + float(env_origin[2]),
            ),
            orientation=(math.cos(half_pitch), 0.0, math.sin(half_pitch), 0.0),
        )
        self._bind_episode_terrain_appearance(prim_path)
        return prim_path

    def _zero_foot_terrain_forces(self) -> None:
        if not getattr(self, "_foot_ids", None):
            return
        zeros = self._torch.zeros(
            (1, len(self._foot_ids), 3), dtype=self._torch.float32, device=self._device
        )
        try:
            self._robot.set_external_force_and_torque(
                zeros, zeros, body_ids=self._foot_ids, is_global=True
            )
        except TypeError:
            self._robot.set_external_force_and_torque(zeros, zeros, body_ids=self._foot_ids)

    def clear_foot_compliance(self) -> None:
        """Remove per-foot soil forces; deep reset separately restores the monolithic ground."""
        if hasattr(self, "_robot"):
            self._zero_foot_terrain_forces()
        self._foot_terrain_cfg = None
        self._foot_terrain_states = []
        self._foot_terrain_updates = []
        self._foot_terrain_contacts_n = []
        self._foot_terrain_paths = []
        self._foot_terrain_footprints = []
        self._foot_terrain_matched_control = False
        self._foot_terrain_topology = {}

    def _spawn_soil_footprint_visual(
        self,
        *,
        foot_index: int,
        position_xyz_m: np.ndarray,
        contact_force_n: float,
        sinkage_m: float,
    ) -> None:
        """Persist one load-triggered disturbed-soil imprint at the measured foot location."""
        config = self._foot_terrain_cfg
        if config is None:
            return
        position = np.asarray(position_xyz_m, dtype=np.float64)
        minimum_spacing_m = float(self._cfg.get("o2_footprint_spacing_m", 0.065))
        for row in reversed(self._foot_terrain_footprints):
            if int(row["foot_index"]) != foot_index:
                continue
            previous = np.asarray(row["contact_position_xyz_m"], dtype=np.float64)
            if float(np.linalg.norm(position[:2] - previous[:2])) < minimum_spacing_m:
                if float(sinkage_m) > float(row["sinkage_m"]) + 1.0e-4:
                    changed = self._apply_contact_local_ground_deformation(
                        center_xy_m=(float(position[0]), float(position[1])),
                        sinkage_m=float(sinkage_m),
                    )
                    row["sinkage_m"] = float(sinkage_m)
                    row["contact_force_n"] = max(
                        float(row["contact_force_n"]), float(contact_force_n)
                    )
                    row["changed_vertex_count"] = int(
                        row.get("changed_vertex_count", 0)
                    ) + changed
                    row["last_update_time_s"] = float(self._t)
                return
            break
        if len(self._foot_terrain_footprints) >= 80:
            return
        sink_fraction = float(np.clip(sinkage_m / config.max_sink_depth_m, 0.0, 1.0))
        radius_m = 0.032 + 0.010 * sink_fraction
        top_z_m = min(float(config.surface_z_m) + 0.002, float(position[2]) - 0.012)
        changed_vertices = self._apply_contact_local_ground_deformation(
            center_xy_m=(float(position[0]), float(position[1])),
            sinkage_m=float(sinkage_m),
        )
        self._foot_terrain_footprints.append(
            {
                "visualization_mode": "continuous_pbr_mesh_contact_local_depression",
                "foot_index": foot_index,
                "foot_name": self._foot_names[foot_index],
                "contact_position_xyz_m": [float(value) for value in position],
                "contact_force_n": float(contact_force_n),
                "sinkage_m": float(sinkage_m),
                "radius_m": radius_m,
                "visual_top_z_m": top_z_m,
                "changed_vertex_count": changed_vertices,
                "spawn_time_s": float(self._t),
                "trigger": "measured_load_bearing_contact",
            }
        )

    def _apply_foot_terrain_forces(self) -> None:
        config = self._foot_terrain_cfg
        if config is None:
            return
        if self._foot_terrain_matched_control:
            self._zero_foot_terrain_forces()
            self._foot_terrain_updates = []
            return
        env_origin = self._env.scene.env_origins[0].detach().cpu().numpy()
        foot_pos = self._robot.data.body_pos_w[0, self._foot_ids].detach().cpu().numpy()
        foot_pos -= env_origin.reshape(1, 3)
        foot_vel = self._robot.data.body_lin_vel_w[0, self._foot_ids].detach().cpu().numpy()
        contact_vec = self._contact.data.net_forces_w[0, self._contact_foot_ids]
        contact_n = np.maximum(
            contact_vec[:, 2].detach().cpu().numpy().astype(np.float64), 0.0
        )
        updates: list[FootTerrainUpdate] = []
        next_states: list[FootTerrainState] = []
        forces = np.zeros((len(self._foot_ids), 3), dtype=np.float32)
        for index, state in enumerate(self._foot_terrain_states):
            update = step_foot_terramechanics(
                config,
                state,
                foot_pos_xyz_m=foot_pos[index],
                foot_vel_xyz_mps=foot_vel[index],
                normal_force_n=float(contact_n[index]),
                dt_s=self.dt,
            )
            forces[index] = update.force_world_n.astype(np.float32)
            updates.append(update)
            next_states.append(update.state)
            if update.load_bearing and update.sinkage_m > 0.003:
                self._spawn_soil_footprint_visual(
                    foot_index=index,
                    position_xyz_m=foot_pos[index],
                    contact_force_n=float(contact_n[index]),
                    sinkage_m=float(update.sinkage_m),
                )
        force_tensor = self._torch.from_numpy(forces.reshape(1, len(self._foot_ids), 3)).to(
            self._device
        )
        torque_tensor = self._torch.zeros_like(force_tensor)
        try:
            self._robot.set_external_force_and_torque(
                force_tensor, torque_tensor, body_ids=self._foot_ids, is_global=True
            )
        except TypeError:
            self._robot.set_external_force_and_torque(
                force_tensor, torque_tensor, body_ids=self._foot_ids
            )
        self._foot_terrain_states = next_states
        self._foot_terrain_updates = updates
        self._foot_terrain_contacts_n = [float(value) for value in contact_n]

    def foot_compliance_telemetry(self) -> dict[str, object]:
        """Return measured per-foot sinkage, load and dissipative shear work for O2 QA."""
        config = self._foot_terrain_cfg
        if config is None:
            return {"enabled": False, "feet": []}
        env_origin = self._env.scene.env_origins[0].detach().cpu().numpy()
        positions = self._robot.data.body_pos_w[0, self._foot_ids].detach().cpu().numpy()
        positions -= env_origin.reshape(1, 3)
        feet: list[dict[str, object]] = []
        for index, state in enumerate(self._foot_terrain_states):
            update = (
                self._foot_terrain_updates[index]
                if index < len(self._foot_terrain_updates)
                else None
            )
            feet.append(
                {
                    "name": self._foot_names[index],
                    "position_xyz_m": [float(value) for value in positions[index]],
                    "contact_force_n": (
                        self._foot_terrain_contacts_n[index]
                        if index < len(self._foot_terrain_contacts_n)
                        else 0.0
                    ),
                    "in_region": bool(update.in_region) if update else False,
                    "load_bearing": bool(update.load_bearing) if update else False,
                    "sinkage_m": float(update.sinkage_m) if update else 0.0,
                    "max_sinkage_m": state.max_sinkage_m,
                    "max_applied_force_n": state.max_applied_force_n,
                    "shear_work_j": state.shear_work_j,
                    "contact_steps": state.contact_steps,
                    "applied_force_world_n": list(state.last_force_world_n),
                }
            )
        return {
            "enabled": True,
            "mode": "per_foot_lowered_collision_bed_plus_shear_yield",
            "nominal_ground_friction": {
                "static": self._nominal_ground_friction_readback()[0],
                "dynamic": self._nominal_ground_friction_readback()[1],
                "readback_static": _read_material_friction(self._foot_terrain_paths[0])[0],
                "readback_dynamic": _read_material_friction(self._foot_terrain_paths[0])[1],
            },
            "max_sink_depth_m": config.max_sink_depth_m,
            "shear_retention": config.shear_retention,
            "vertical_stiffness_n_per_m": config.vertical_stiffness_n_per_m,
            "matched_control": self._foot_terrain_matched_control,
            "collision_topology": dict(self._foot_terrain_topology),
            "isaac_default_ground_colliders_disabled": list(
                self._disabled_default_ground_colliders
            ),
            "realistic_scene_floor_colliders_disabled": list(
                self._disabled_realistic_scene_floor_colliders
            ),
            "nominal_route_appearance_hidden": bool(
                self._realistic_scene_route_appearance_hidden
            ),
            "continuous_visual_surface": dict(
                self._realistic_scene_route_visual_deformation
                or {"applied": False, "reason": "legacy_or_unavailable"}
            ),
            "visual_footprint_count": len(self._foot_terrain_footprints),
            "visual_footprints": [dict(row) for row in self._foot_terrain_footprints],
            "feet": feet,
        }

    # --------------------------------------------- realistic O4 foot-local adhesion vertical slice

    def add_foot_adhesion(self, config: FootAdhesionConfig) -> None:
        """Enable explicit per-foot adhesive contacts for the current episode."""
        if self._foot_terrain_cfg is not None:
            raise RuntimeError("foot adhesion and foot terramechanics need an explicit force mixer")
        self._foot_adhesion_cfg = config
        self._foot_adhesion_states = [FootAdhesionState() for _ in self._foot_ids]
        self._foot_adhesion_updates = []
        self._foot_adhesion_contacts_n = [0.0 for _ in self._foot_ids]
        self._foot_adhesion_terminal_mode = False

    def _zero_foot_adhesion_forces(self) -> None:
        if not getattr(self, "_foot_ids", None):
            return
        zeros = self._torch.zeros(
            (1, len(self._foot_ids), 3), dtype=self._torch.float32, device=self._device
        )
        try:
            self._robot.set_external_force_and_torque(
                zeros, zeros, body_ids=self._foot_ids, is_global=True
            )
        except TypeError:
            self._robot.set_external_force_and_torque(zeros, zeros, body_ids=self._foot_ids)

    def clear_foot_adhesion(self) -> None:
        """Remove the realistic adhesion operator and clear PhysX's persistent foot forces."""
        if hasattr(self, "_robot"):
            self._zero_foot_adhesion_forces()
        self._foot_adhesion_cfg = None
        self._foot_adhesion_states = []
        self._foot_adhesion_updates = []
        self._foot_adhesion_contacts_n = []
        self._foot_adhesion_terminal_mode = False

    def _apply_foot_adhesion_forces(self, command_world_xy: np.ndarray) -> None:
        config = self._foot_adhesion_cfg
        if config is None:
            return
        env_origin = self._env.scene.env_origins[0].detach().cpu().numpy()
        foot_pos = self._robot.data.body_pos_w[0, self._foot_ids].detach().cpu().numpy()
        foot_pos -= env_origin.reshape(1, 3)
        foot_vel = self._robot.data.body_lin_vel_w[0, self._foot_ids].detach().cpu().numpy()
        contact_vec = self._contact.data.net_forces_w[0, self._contact_foot_ids]
        contact_n = self._torch.linalg.norm(contact_vec, dim=1).detach().cpu().numpy()
        active = sum(state.attached for state in self._foot_adhesion_states)
        commanded_progress = float(
            np.dot(np.asarray(command_world_xy, dtype=np.float64), config.unit_progress_axis_xy)
        )
        peel_requested = commanded_progress < -config.peel_velocity_threshold_mps
        if peel_requested:
            # A reversal starts the terminal phase of this localized patch encounter.  Keep it
            # terminal through the following zero-command posture-recovery window so another
            # swing foot cannot silently create a fresh bond before retreat begins.
            self._foot_adhesion_terminal_mode = True
        forces = np.zeros((len(self._foot_ids), 3), dtype=np.float32)
        updates: list[FootAdhesionUpdate] = []
        next_states: list[FootAdhesionState] = []
        for index, state in enumerate(self._foot_adhesion_states):
            # Reversal is a terminal peel phase for this patch encounter.  Existing bonds may
            # release, but a swing foot must not create a new bond while the robot is explicitly
            # backing out; otherwise attachment can hop from foot to foot and defeat the peel.
            allow_attach = (not self._foot_adhesion_terminal_mode) and (
                state.attached or active < config.max_active_feet
            )
            update = step_foot_adhesion(
                config,
                state,
                foot_pos[index],
                foot_vel[index],
                float(contact_n[index]),
                allow_attach=allow_attach,
                peel_requested=peel_requested,
            )
            if not state.attached and update.state.attached:
                active += 1
            if state.attached and not update.state.attached:
                active -= 1
            forces[index] = update.force_world_n.astype(np.float32)
            updates.append(update)
            next_states.append(update.state)
        force_tensor = self._torch.from_numpy(forces.reshape(1, len(self._foot_ids), 3)).to(
            self._device
        )
        torque_tensor = self._torch.zeros_like(force_tensor)
        try:
            self._robot.set_external_force_and_torque(
                force_tensor, torque_tensor, body_ids=self._foot_ids, is_global=True
            )
        except TypeError:
            self._robot.set_external_force_and_torque(
                force_tensor, torque_tensor, body_ids=self._foot_ids
            )
        self._foot_adhesion_states = next_states
        self._foot_adhesion_updates = updates
        self._foot_adhesion_contacts_n = [float(value) for value in contact_n]

    def adhesion_telemetry(self) -> dict[str, object]:
        """Privileged, JSON-serializable attachment telemetry for QA and consequence labels."""
        feet = []
        total_applied = 0.0
        for index, state in enumerate(self._foot_adhesion_states):
            applied_n = float(np.linalg.norm(state.force_world_n))
            total_applied += applied_n
            raw_force_n = (
                float(self._foot_adhesion_updates[index].raw_force_n)
                if index < len(self._foot_adhesion_updates)
                else 0.0
            )
            feet.append(
                {
                    "name": self._foot_names[index],
                    "attached": state.attached,
                    "terminal_release": state.terminal_release,
                    "broken": state.broken,
                    "phase": state.phase,
                    "event": state.last_event,
                    "anchor_xyz": list(state.anchor_xyz) if state.anchor_xyz is not None else None,
                    "contact_force_n": (
                        self._foot_adhesion_contacts_n[index]
                        if index < len(self._foot_adhesion_contacts_n)
                        else 0.0
                    ),
                    "applied_force_n": applied_n,
                    "raw_force_n": raw_force_n,
                    "max_force_n": state.max_force_n,
                    "max_extension_m": state.max_extension_m,
                    "attachment_count": state.attachment_count,
                }
            )
        return {
            "enabled": self._foot_adhesion_cfg is not None,
            "terminal_peel_mode": bool(self._foot_adhesion_terminal_mode),
            "active_feet": sum(state.attached for state in self._foot_adhesion_states),
            "released_feet": sum(state.terminal_release for state in self._foot_adhesion_states),
            "broken_feet": sum(state.broken for state in self._foot_adhesion_states),
            "total_applied_force_n": total_applied,
            "feet": feet,
        }

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

    def _resistance_force_body(self) -> np.ndarray:
        """Return the legacy O2/O4 resistance contribution in the base frame."""
        if not self._resistance:
            return np.zeros(3, dtype=np.float64)
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
        return force_b

    def _step_push_wrench_world(self) -> tuple[np.ndarray, np.ndarray]:
        """Advance the finite O6 pulse by one control step and return its world wrench."""
        pulse = self._push_pulse
        if pulse is None:
            return np.zeros(3, dtype=np.float64), np.zeros(3, dtype=np.float64)
        from isaaclab.utils.math import quat_apply

        point_body = self._torch.as_tensor(
            np.asarray(pulse["application_point_body_m"], dtype=np.float32),
            dtype=self._torch.float32,
            device=self._device,
        ).reshape(1, 3)
        point_world = (
            quat_apply(self._robot.data.root_quat_w[0:1], point_body)[0]
            .detach()
            .cpu()
            .numpy()
            .astype(np.float64)
        )
        force_world = np.asarray(pulse["force_world_n"], dtype=np.float64)
        torque_world = np.cross(point_world, force_world)
        torque_world[2] += float(pulse["extra_yaw_torque_nm"])
        pulse["applied_steps"] = int(pulse["applied_steps"]) + 1
        pulse["remaining_steps"] = int(pulse["remaining_steps"]) - 1
        pulse["commanded_linear_impulse_xy_ns"] = np.asarray(
            pulse["commanded_linear_impulse_xy_ns"], dtype=np.float64
        ) + force_world[:2] * self.dt
        pulse["commanded_torque_impulse_xyz_nms"] = np.asarray(
            pulse["commanded_torque_impulse_xyz_nms"], dtype=np.float64
        ) + torque_world * self.dt
        pulse["peak_force_n"] = max(
            float(pulse["peak_force_n"]), float(np.linalg.norm(force_world))
        )
        pulse["peak_torque_nm"] = max(
            float(pulse["peak_torque_nm"]), float(np.linalg.norm(torque_world))
        )
        active = int(pulse["remaining_steps"]) > 0
        self._push_last_telemetry = self._serialize_push_pulse(pulse, active=active)
        if not active:
            self._push_pulse = None
        return force_world, torque_world

    def _apply_base_wrenches(self) -> None:
        """Mix O2/O4 resistance and finite O6 pulse before one PhysX wrench write."""
        from isaaclab.utils.math import quat_apply, quat_apply_inverse

        torch = self._torch
        resistance_body = torch.from_numpy(
            self._resistance_force_body().reshape(1, 3).astype(np.float32)
        ).to(self._device)
        quat = self._robot.data.root_quat_w[0:1]
        force_world = quat_apply(quat, resistance_body)[0].detach().cpu().numpy()
        pulse_force_world, pulse_torque_world = self._step_push_wrench_world()
        force_world = np.asarray(force_world, dtype=np.float64) + pulse_force_world
        forces = torch.from_numpy(force_world.reshape(1, 1, 3).astype(np.float32)).to(
            self._device
        )
        torques = torch.from_numpy(
            pulse_torque_world.reshape(1, 1, 3).astype(np.float32)
        ).to(self._device)
        try:
            self._robot.set_external_force_and_torque(
                forces, torques, body_ids=self._base_id, is_global=True
            )
        except TypeError:
            local_forces = quat_apply_inverse(quat, forces[:, 0]).reshape(1, 1, 3)
            local_torques = quat_apply_inverse(quat, torques[:, 0]).reshape(1, 1, 3)
            self._robot.set_external_force_and_torque(
                local_forces, local_torques, body_ids=self._base_id
            )

    def add_payload(
        self,
        mass_kg: float,
        com_offset_m: np.ndarray,
        size_m: np.ndarray | None = None,
    ) -> None:
        """Compose a visible, rigidly attached cuboid into the base mass properties."""
        if mass_kg <= 0.0:
            raise ValueError("payload mass must be positive in the Isaac realistic path")
        payload_com = np.asarray(com_offset_m, dtype=np.float64)
        if payload_com.shape == (2,):
            payload_com = np.concatenate([payload_com, [0.14]])
        payload_size = np.asarray(
            (0.30, 0.20, 0.16) if size_m is None else size_m, dtype=np.float64
        )
        if payload_com.shape != (3,) or payload_size.shape != (3,):
            raise ValueError("payload CoM and size must be 3-vectors")
        view = self._robot.root_physx_view
        base = self._base_id[0]
        masses = view.get_masses().clone()
        coms = view.get_coms().clone()
        inertias = view.get_inertias().clone()
        combined = combine_with_cuboid_payload(
            base_mass_kg=float(masses[0, base]),
            base_com_m=coms[0, base, :3].cpu().numpy(),
            base_inertia_kg_m2=inertias[0, base].cpu().numpy().reshape(3, 3, order="F"),
            payload_mass_kg=float(mass_kg),
            payload_com_m=payload_com,
            payload_size_m=payload_size,
        )
        masses[0, base] = combined.mass_kg
        coms[0, base, :3] = self._torch.as_tensor(
            combined.com_m, dtype=coms.dtype, device=coms.device
        )
        inertias[0, base] = self._torch.as_tensor(
            combined.inertia_kg_m2.reshape(9, order="F"),
            dtype=inertias.dtype,
            device=inertias.device,
        )
        indices = self._torch.tensor([0])
        view.set_masses(masses, indices)
        view.set_coms(coms, indices)
        view.set_inertias(inertias, indices)
        self._payload_kg += float(mass_kg)
        self.mass_kg = float(masses.sum())
        self._spawn_payload_visual(payload_com, payload_size, float(mass_kg))
        print(
            f"[isaac] O5 rigid payload: +{mass_kg:.2f} kg, local CoM="
            f"{payload_com.tolist()}, size={payload_size.tolist()}, total={self.mass_kg:.2f} kg"
        )

    def _spawn_payload_visual(
        self, payload_com_m: np.ndarray, payload_size_m: np.ndarray, mass_kg: float
    ) -> None:
        path = f"/World/payload_visual_{self._n_patches}"
        self._n_patches += 1
        cfg = self._sim_utils.CuboidCfg(
            size=tuple(float(value) for value in payload_size_m),
            visual_material=self._sim_utils.PreviewSurfaceCfg(
                diffuse_color=(0.16, 0.24, 0.38), metallic=0.12, roughness=0.62
            ),
        )
        cfg.func(path, cfg, translation=(0.0, 0.0, 0.0), orientation=(1.0, 0.0, 0.0, 0.0))
        self._payload_visuals.append(
            {
                "prim_path": path,
                "payload_com_m": np.asarray(payload_com_m, dtype=np.float64).copy(),
                "size_m": np.asarray(payload_size_m, dtype=np.float64).copy(),
                "mass_kg": float(mass_kg),
            }
        )
        self._sync_payload_visual_poses()

    def _sync_payload_visual_poses(self) -> None:
        """Make collision-free payload visuals follow the base's full six-DoF pose."""
        if not self._payload_visuals:
            return
        import omni.usd
        from pxr import Gf

        base_pos = self._robot.data.root_pos_w[0].detach().cpu().numpy()
        quat = self._robot.data.root_quat_w[0].detach().cpu().numpy()
        w = float(quat[0])
        vector = np.asarray(quat[1:4], dtype=np.float64)
        stage = omni.usd.get_context().get_stage()
        for row in self._payload_visuals:
            local = np.asarray(row["payload_com_m"], dtype=np.float64)
            rotated = local + 2.0 * np.cross(vector, np.cross(vector, local) + w * local)
            world = base_pos + rotated
            prim = stage.GetPrimAtPath(str(row["prim_path"]))
            if not prim.IsValid():
                continue
            prim.GetAttribute("xformOp:translate").Set(
                Gf.Vec3d(float(world[0]), float(world[1]), float(world[2]))
            )
            prim.GetAttribute("xformOp:orient").Set(
                Gf.Quatd(w, Gf.Vec3d(float(vector[0]), float(vector[1]), float(vector[2])))
            )

    def payload_telemetry(self) -> dict[str, object]:
        """Read back O5 mass, CoM, inertia and authored visible-load metadata."""
        view = self._robot.root_physx_view
        base = self._base_id[0]
        return {
            "payload_kg": float(self._payload_kg),
            "base_mass_kg": float(view.get_masses()[0, base]),
            "base_com_pose": view.get_coms()[0, base].cpu().numpy().tolist(),
            "base_inertia_kg_m2": view.get_inertias()[0, base]
            .cpu()
            .numpy()
            .reshape(3, 3, order="F")
            .tolist(),
            "nominal_base_mass_kg": float(self._nominal_base_mass),
            "nominal_base_com_pose": self._nominal_base_com.cpu().numpy().tolist(),
            "nominal_base_inertia_kg_m2": self._nominal_base_inertia
            .cpu()
            .numpy()
            .reshape(3, 3, order="F")
            .tolist(),
            "visuals": [
                {
                    "prim_path": str(row["prim_path"]),
                    "mass_kg": float(row["mass_kg"]),
                    "payload_com_m": np.asarray(row["payload_com_m"]).tolist(),
                    "size_m": np.asarray(row["size_m"]).tolist(),
                }
                for row in self._payload_visuals
            ],
        }

    def clear_payload(self) -> None:
        """Strip any attached O5 payload, restoring the trunk's nominal mass.

        ``reset()`` does NOT restore PhysX masses, so a payload added in one lane persists into
        the next (#22). The M6 data collection drives many O5 lanes (and O5 mixed with other ops)
        in one process, so each lane must start payload-free to log the mass it actually set —
        otherwise the cumulative ``add_payload`` (+=) compounds across lanes. No-op when unloaded;
        independent of the M4 token-gate's deliberate ascending-payload sweep (which never clears).
        """
        if self._payload_kg == 0.0 and not self._payload_visuals:
            return
        view = self._robot.root_physx_view
        masses = view.get_masses().clone()
        coms = view.get_coms().clone()
        inertias = view.get_inertias().clone()
        base = self._base_id[0]
        masses[0, base] = self._nominal_base_mass
        coms[0, base] = self._nominal_base_com
        inertias[0, base] = self._nominal_base_inertia
        indices = self._torch.tensor([0])
        view.set_masses(masses, indices)
        view.set_coms(coms, indices)
        view.set_inertias(inertias, indices)
        import omni.usd

        stage = omni.usd.get_context().get_stage()
        for row in self._payload_visuals:
            stage.RemovePrim(str(row["prim_path"]))
        self._payload_visuals = []
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
                self._nominal_effort[name] = {}
                for attr in ("effort_limit", "_saturation_effort"):
                    if not hasattr(act, attr):
                        continue
                    value = getattr(act, attr)
                    self._nominal_effort[name][attr] = (
                        value.clone() if hasattr(value, "clone") else float(value)
                    )
            for attr, nominal in self._nominal_effort[name].items():
                setattr(act, attr, nominal * scale)
            # DCMotor caches the velocity at which the continuous effort cap
            # intersects its saturation torque-speed line.  Refreshing this value is
            # essential: changing the two caps without it leaves a stale envelope.
            if all(
                hasattr(act, attr)
                for attr in ("velocity_limit", "effort_limit", "_saturation_effort")
            ):
                act._vel_at_effort_lim = act.velocity_limit * (  # noqa: SLF001
                    1 + act.effort_limit / act._saturation_effort  # noqa: SLF001
                )
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
