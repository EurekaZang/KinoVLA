"""Isaac Lab Go2 backend for the walking skeleton — M1 kinematic root drive.

GPU-only and AUTHORED BLIND on the CPU dev laptop: requires a running Isaac app
(``scripts/run_demo.py`` launches ``AppLauncher`` before importing this module)
and manual verification on the RTX 5090 machine (CLAUDE.md Section 6 issues
#1/#4/#5). Written against the Isaac Lab 2.x API, mirroring scripts/stand_go2.py.

M1 has no locomotion policy (that is M2's scope), so the Go2 base is driven by
writing root velocities computed by the *same* friction-limited traction model
as the surrogate, while the legs hold the default stance pose. What is real
Isaac here — and what the GPU-gated test verifies — is the scene, the Go2 asset,
and operator O1's PhysX material patch: the μ the traction model consumes is
read back from the spawned material prim, not trusted from the config
(θ-application, QA 5.2a). One episode per process at M1 (spawned patch prims
persist across resets).
"""

from __future__ import annotations

import math

import numpy as np

from kino_vla.sim.traction import traction_step
from kino_vla.sim.types import FrictionRegion, Obs
from kino_vla.utils.config import Config
from kino_vla.utils.geometry import body_to_world, wrap_angle


def _read_material_friction(prim_path: str) -> tuple[float, float]:
    """Read (static, dynamic) friction back from the spawned physics material.

    Searches the prim subtree for the UsdPhysics material attributes so the value
    asserted in tests is what PhysX actually received, not what we asked for.
    """
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


class IsaacKinematicBackend:
    """Go2 in Isaac Lab driven by traction-model root-velocity writes (M1 stub tier)."""

    def __init__(self, cfg: Config, start_pos: np.ndarray, start_heading: float) -> None:
        import isaaclab.sim as sim_utils
        from isaaclab.assets import Articulation
        from isaaclab.sim import SimulationContext
        from isaaclab_assets import UNITREE_GO2_CFG

        self._cfg = cfg
        self._sim_utils = sim_utils
        self.dt = 1.0 / float(cfg.control_hz)
        self._physics_dt = float(cfg.physics_dt)
        self._decimation = int(round(self.dt / self._physics_dt))
        self._start_pos = np.asarray(start_pos, dtype=np.float64).copy()
        self._start_heading = float(start_heading)
        self._regions: list[FrictionRegion] = []
        self._n_patches = 0

        self._sim = SimulationContext(
            sim_utils.SimulationCfg(dt=self._physics_dt, device=str(cfg.device))
        )
        ground_material = sim_utils.RigidBodyMaterialCfg(
            static_friction=float(cfg.mu_nominal),
            dynamic_friction=float(cfg.mu_nominal),
        )
        ground_cfg = sim_utils.GroundPlaneCfg(physics_material=ground_material)
        ground_cfg.func("/World/ground", ground_cfg)
        light_cfg = sim_utils.DomeLightCfg(intensity=2000.0)
        light_cfg.func("/World/light", light_cfg)
        self._robot = Articulation(UNITREE_GO2_CFG.replace(prim_path="/World/Go2"))
        self._sim.reset()

        try:
            self.mass_kg = float(self._robot.root_physx_view.get_masses().sum())
        except AttributeError:  # API drift guard — verify on the GPU machine
            self.mass_kg = float(cfg.mass_fallback_kg)

        self._t = 0.0
        self._cmd_prev = np.zeros(3)
        self._slip = 0.0
        self._fallen = False

    # ------------------------------------------------------------ operator API

    def add_friction_regions(self, regions: list[FrictionRegion]) -> None:
        """Spawn one thin static collider plate per region with its PhysX material."""
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
                    friction_combine_mode="min",  # the patch must dominate foot materials
                ),
                visual_material=self._sim_utils.PreviewSurfaceCfg(diffuse_color=(0.6, 0.8, 1.0)),
            )
            plate_cfg.func(
                prim_path,
                plate_cfg,
                translation=(region.rect.cx, region.rect.cy, thickness / 2.0),
            )
            mu_s_applied, mu_d_applied = _read_material_friction(prim_path)
            print(
                f"[isaac] O1 patch material applied: set mu_d={region.mu_d:.3f} "
                f"readback mu_d={mu_d_applied:.3f}"
            )
            # friction_at serves the readback value — the θ actually in the sim.
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
        root_state = self._robot.data.root_state_w.clone()
        dv = np.asarray(impulse_xy_ns, dtype=np.float64) / self.mass_kg
        root_state[0, 7] += float(dv[0])
        root_state[0, 8] += float(dv[1])
        root_state[0, 12] += float(yaw_impulse_nms) / float(self._cfg.yaw_inertia_kgm2)
        self._robot.write_root_state_to_sim(root_state)

    # ------------------------------------------------------------ episode API

    def reset(self, seed: int) -> Obs:
        # Physics is deterministic for a fixed seed/scene; the seed parameter is
        # part of the protocol for backends with internal noise (surrogate).
        root_state = self._robot.data.default_root_state.clone()
        root_state[0, 0] = float(self._start_pos[0])
        root_state[0, 1] = float(self._start_pos[1])
        half = self._start_heading / 2.0
        root_state[0, 3:7] = root_state.new_tensor(
            [math.cos(half), 0.0, 0.0, math.sin(half)]
        )  # (w, x, y, z) yaw-only
        root_state[0, 7:13] = 0.0
        self._robot.write_root_state_to_sim(root_state)
        self._robot.write_joint_state_to_sim(
            self._robot.data.default_joint_pos.clone(),
            self._robot.data.default_joint_vel.clone(),
        )
        self._robot.reset()
        self._t = 0.0
        self._cmd_prev = np.zeros(3)
        self._slip = 0.0
        self._fallen = False
        return self._make_obs()

    def step(self, cmd_vel: np.ndarray) -> Obs:
        if self._fallen:
            return self._make_obs()
        cmd = np.asarray(cmd_vel, dtype=np.float64).copy()
        max_v = float(self._cfg.max_speed_mps)
        max_w = float(self._cfg.max_yaw_rate_radps)
        cmd[:2] = np.clip(cmd[:2], -max_v, max_v)
        cmd[2] = float(np.clip(cmd[2], -max_w, max_w))
        self._cmd_prev = cmd

        for _ in range(self._decimation):
            obs_now = self._make_obs()
            mu = self.friction_at(obs_now.pos)
            result = traction_step(
                obs_now.vel_body,
                cmd[:2],
                mu,
                tau_track_s=float(self._cfg.tau_track_s),
                gait_demand_per_speed=float(self._cfg.gait_demand_per_speed),
                gravity=float(self._cfg.gravity),
            )
            self._slip = result.slip_ratio
            vel_body = obs_now.vel_body + result.accel_body * self._physics_dt
            yaw_accel = (cmd[2] - obs_now.yaw_rate) / float(self._cfg.tau_yaw_s) * result.authority
            yaw_rate = obs_now.yaw_rate + yaw_accel * self._physics_dt
            vel_world = body_to_world(vel_body, obs_now.heading)

            root_state = self._robot.data.root_state_w.clone()
            root_state[0, 7] = float(vel_world[0])
            root_state[0, 8] = float(vel_world[1])
            root_state[0, 10:12] = 0.0  # no commanded roll/pitch rates
            root_state[0, 12] = float(yaw_rate)
            self._robot.write_root_state_to_sim(root_state)
            self._robot.set_joint_position_target(self._robot.data.default_joint_pos.clone())
            self._robot.write_data_to_sim()
            self._sim.step()
            self._robot.update(self._physics_dt)
            self._t += self._physics_dt

        obs = self._make_obs()
        if obs.base_height < float(self._cfg.fall_check.min_base_height_m) or obs.tilt > float(
            self._cfg.fall_check.max_tilt_rad
        ):
            self._fallen = True
            obs = self._make_obs()
        return obs

    def _make_obs(self) -> Obs:
        data = self._robot.data
        pos_w = data.root_pos_w[0].cpu().numpy()
        quat = data.root_quat_w[0].cpu().numpy()  # (w, x, y, z)
        w, x, y, z = (float(v) for v in quat)
        heading = wrap_angle(math.atan2(2.0 * (w * z + x * y), 1.0 - 2.0 * (y * y + z * z)))
        cos_tilt = float(np.clip(1.0 - 2.0 * (x * x + y * y), -1.0, 1.0))
        vel_b = data.root_lin_vel_b[0].cpu().numpy()
        yaw_rate = float(data.root_ang_vel_b[0, 2].cpu())
        return Obs(
            t=self._t,
            pos=np.asarray(pos_w[:2], dtype=np.float64),
            heading=heading,
            vel_body=np.asarray(vel_b[:2], dtype=np.float64),
            yaw_rate=yaw_rate,
            cmd_prev=self._cmd_prev.copy(),
            slip_ratio=self._slip,
            base_height=float(pos_w[2]),
            tilt=math.acos(cos_tilt),
            fallen=self._fallen,
        )
