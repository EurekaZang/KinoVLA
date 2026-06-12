"""CPU surrogate locomotion backend for the walking skeleton.

A planar point-robot (unicycle with lateral velocity) that tracks Sport-Client
velocity commands through the shared friction-limited traction model. It exists
so the M1 demo, its assertions, and the operator gates run end-to-end on machines
without a GPU (dev laptop, CI) — the Isaac Lab backend is the deliverable and
replaces it wherever a GPU is present (CLAUDE.md Section 6 issue #4).

Failure modeling: sustained slip at speed accumulates an instability integrator;
crossing the configured threshold is scored as a fall. This makes the demo's
"robot did not fall" assertion falsifiable on CPU (tests prove a non-recovering
policy falls on the ice patch with the same seed).
"""

from __future__ import annotations

import numpy as np

from kino_vla.sim.traction import traction_step
from kino_vla.sim.types import FrictionRegion, Obs
from kino_vla.utils.config import Config
from kino_vla.utils.geometry import body_to_world, world_to_body, wrap_angle
from kino_vla.utils.seeding import rng


class SurrogateBackend:
    """Deterministic seeded planar robot implementing ``LocomotionBackend``."""

    def __init__(self, cfg: Config, start_pos: np.ndarray, start_heading: float) -> None:
        self._cfg = cfg
        self.dt = 1.0 / float(cfg.control_hz)
        self.mass_kg = float(cfg.mass_kg)
        self._start_pos = np.asarray(start_pos, dtype=np.float64).copy()
        self._start_heading = float(start_heading)
        self._regions: list[FrictionRegion] = []
        self._rng = rng(0)
        self._init_state()

    def _init_state(self) -> None:
        self._t = 0.0
        self._pos = self._start_pos.copy()
        self._heading = self._start_heading
        self._vel_world = np.zeros(2)
        self._yaw_rate = 0.0
        self._cmd_prev = np.zeros(3)
        self._slip = 0.0
        self._instability = 0.0
        self._fallen = False

    def reset(self, seed: int) -> Obs:
        self._rng = rng(seed)
        self._regions = []
        self._init_state()
        return self._obs()

    def add_friction_regions(self, regions: list[FrictionRegion]) -> None:
        self._regions.extend(regions)

    def friction_at(self, pos: np.ndarray) -> float:
        for region in self._regions:
            if region.rect.contains(pos):
                return region.mu_d
        return float(self._cfg.mu_nominal)

    def apply_push(self, impulse_xy_ns: np.ndarray, yaw_impulse_nms: float) -> None:
        self._vel_world = self._vel_world + np.asarray(impulse_xy_ns, dtype=np.float64) / (
            self.mass_kg
        )
        self._yaw_rate += float(yaw_impulse_nms) / float(self._cfg.yaw_inertia_kgm2)

    def step(self, cmd_vel: np.ndarray) -> Obs:
        if self._fallen:
            return self._obs()
        cmd = np.asarray(cmd_vel, dtype=np.float64).copy()
        max_v = float(self._cfg.max_speed_mps)
        max_w = float(self._cfg.max_yaw_rate_radps)
        cmd[:2] = np.clip(cmd[:2], -max_v, max_v)
        cmd[2] = float(np.clip(cmd[2], -max_w, max_w))
        self._cmd_prev = cmd

        mu = self.friction_at(self._pos)
        vel_body = world_to_body(self._vel_world, self._heading)
        result = traction_step(
            vel_body,
            cmd[:2],
            mu,
            tau_track_s=float(self._cfg.tau_track_s),
            gait_demand_per_speed=float(self._cfg.gait_demand_per_speed),
            gravity=float(self._cfg.gravity),
        )
        self._slip = result.slip_ratio
        vel_body = vel_body + result.accel_body * self.dt
        # Lateral skid: seeded noise proportional to slip (ice drift), body-frame y.
        vel_body[1] += (
            float(self._cfg.skid_noise_std)
            * self._slip
            * np.sqrt(self.dt)
            * self._rng.standard_normal()
        )
        # Yaw tracking shares the traction authority: a slipping robot also steers poorly.
        yaw_accel = (cmd[2] - self._yaw_rate) / float(self._cfg.tau_yaw_s) * result.authority
        self._yaw_rate += yaw_accel * self.dt
        self._heading = wrap_angle(self._heading + self._yaw_rate * self.dt)
        self._vel_world = body_to_world(vel_body, self._heading)
        self._pos = self._pos + self._vel_world * self.dt
        self._t += self.dt

        speed = float(np.linalg.norm(vel_body))
        fall = self._cfg.fall
        if self._slip > 0.0 and speed > float(fall.min_speed_mps):
            self._instability += float(fall.instability_gain) * self._slip * speed * self.dt
        else:
            self._instability = max(0.0, self._instability - float(fall.decay_per_s) * self.dt)
        if self._instability > float(fall.threshold):
            self._fallen = True
        return self._obs()

    def _obs(self) -> Obs:
        return Obs(
            t=self._t,
            pos=self._pos.copy(),
            heading=self._heading,
            vel_body=world_to_body(self._vel_world, self._heading),
            yaw_rate=self._yaw_rate,
            cmd_prev=self._cmd_prev.copy(),
            slip_ratio=self._slip,
            base_height=float(self._cfg.base_height_m),
            tilt=0.0,
            fallen=self._fallen,
        )
