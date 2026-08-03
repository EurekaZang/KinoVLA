"""CPU surrogate locomotion backend for the walking skeleton.

A planar point-robot (unicycle with lateral velocity) that tracks Sport-Client
velocity commands through the shared friction-limited traction model. It exists
so the M1/M2 demo, its assertions, and the operator gates run end-to-end on
machines without a GPU (dev laptop, CI) — the Isaac Lab backend is the deliverable
and replaces it wherever a GPU is present (CLAUDE.md Section 6 issue #4).

M2 generalizes the single friction budget into two: contact friction (O1/O3) and
actuator effort (O5/O10), plus geometric effects — impassable colliders (O8),
foot-support loss (O9) — and a Reflex stance that trades cruising for stability.
Failure modeling: sustained slip *or* a post-push velocity excursion the stance
cannot capture accumulates an instability integrator; crossing the configured
threshold is a fall. The Reflex stance widens/lowers (raising the threshold) and
drops the gait's self-generated demand, so it measurably extends survival time.
"""

from __future__ import annotations

import numpy as np

from kino_vla.sim.traction import traction_step
from kino_vla.sim.types import (
    BlockingRegion,
    CollapseRegion,
    FrictionRegion,
    Obs,
    ResistanceRegion,
    SupportLossRegion,
)
from kino_vla.utils.config import Config
from kino_vla.utils.geometry import body_to_world, world_to_body, wrap_angle
from kino_vla.utils.seeding import rng


class SurrogateBackend:
    """Deterministic seeded planar robot implementing ``LocomotionBackend``."""

    def __init__(self, cfg: Config, start_pos: np.ndarray, start_heading: float) -> None:
        self._cfg = cfg
        self.dt = 1.0 / float(cfg.control_hz)
        self._base_mass_kg = float(cfg.mass_kg)
        self.mass_kg = self._base_mass_kg
        self._effort_budget_nominal = float(cfg.effort_budget_mps2)
        self._start_pos = np.asarray(start_pos, dtype=np.float64).copy()
        self._start_heading = float(start_heading)
        self._regions: list[FrictionRegion] = []
        self._collapse: list[_CollapseState] = []
        self._blocking: list[BlockingRegion] = []
        self._support: list[SupportLossRegion] = []
        self._resistance: list[_ResistanceState] = []
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
        self._effort = 0.0
        self._support_ratio = 1.0
        self._instability = 0.0
        self._fallen = False
        # Operator-set dynamics state (cleared on reset; mass/regions re-added by ops).
        self.mass_kg = self._base_mass_kg
        self._effort_scale = 1.0
        self._payload_com_demand = 0.0
        self._reflex_active = False
        self._base_height_offset = 0.0  # O2 compliance sink (lowers measured base height)
        # Commanded body posture (the closed-loop height controller, #41). None ⇒ nominal trot.
        self._posture_target: float | None = None
        self._posture_stiffness = 1.0
        self._posture_height = float(self._cfg.base_height_m)  # current tracked body height [m]
        for state in self._resistance:
            state.reset()

    def reset(self, seed: int) -> Obs:
        self._rng = rng(seed)
        self._regions = []
        self._collapse = []
        self._blocking = []
        self._support = []
        self._resistance = []
        self._init_state()
        return self._obs()

    # ------------------------------------------------------------ operator API

    def add_friction_regions(self, regions: list[FrictionRegion]) -> None:
        self._regions.extend(regions)

    def add_collapse_regions(self, regions: list[CollapseRegion]) -> None:
        self._collapse.extend(_CollapseState(region=r) for r in regions)

    def add_blocking_regions(self, regions: list[BlockingRegion]) -> None:
        self._blocking.extend(regions)

    def add_support_loss_regions(
        self,
        regions: list[SupportLossRegion],
        height_m: float | None = None,
        geometry_kind: str = "box_ridge",
    ) -> None:
        del height_m, geometry_kind  # geometry is outside the planar surrogate's scope
        self._support.extend(regions)

    def add_resistance_regions(self, regions: list[ResistanceRegion]) -> None:
        """Append tangential-resistance regions (operators O2 compliance / O4 tether)."""
        self._resistance.extend(_ResistanceState(region=r) for r in regions)

    def add_payload(
        self,
        mass_kg: float,
        com_offset_m: np.ndarray,
        size_m: np.ndarray | None = None,
    ) -> None:
        del size_m
        self.mass_kg = self.mass_kg + float(mass_kg)
        offset = float(np.linalg.norm(np.asarray(com_offset_m, dtype=np.float64)[:2]))
        self._payload_com_demand += float(self._cfg.payload_com_demand_gain) * offset

    def set_effort_scale(self, scale: float) -> None:
        self._effort_scale = float(scale)

    def set_reflex(self, active: bool) -> None:
        """Engage the Reflex damping/widen/lower stance (spec §6.8 fallback)."""
        self._reflex_active = bool(active)

    def set_posture(self, height_m: float | None, stiffness: float = 1.0) -> None:
        """Command a body-height posture the model tracks closed-loop (#41), so a Switch_Gait /
        Adjust_Posture / Set_Constraint primitive produces a real, measurable body-height response
        here too (the scaffolding mirror of the Isaac articulation controller). ``None`` ⇒ release
        to the nominal trot. ``stiffness`` sets the convergence rate (a stiffer hold tracks faster).
        """
        self._posture_target = None if height_m is None else float(height_m)
        self._posture_stiffness = float(stiffness)

    def friction_at(self, pos: np.ndarray) -> float:
        for region in self._regions:
            if region.rect.contains(pos):
                return region.mu_d
        for state in self._collapse:
            if state.region.rect.contains(pos):
                return state.region.mu_collapsed if state.collapsed else state.region.mu_intact
        return float(self._cfg.mu_nominal)

    def privileged_physics(self) -> dict[str, float]:
        """God's-eye physical truth at the current step (spec §4 distillation target).

        The canonical, operator-agnostic physics vector the Kino-Tokens extractor (M4)
        regresses from a 500 ms proprioception window — the privileged supervision of
        the teacher-student distillation. All four channels are sampled at the robot's
        *current* state, so time-/location-varying operators (O3 collapse drops μ
        mid-crossing, O10 decays effort, O9 is region-gated) are supervised correctly:

        - ``mu``: effective dynamic friction at the CoM ground projection (O1/O3).
        - ``payload_kg``: extra rigidly-attached mass above the base (O5).
        - ``effort_scale``: actuator-effort budget fraction in (0, 1] (O10).
        - ``support_ratio``: fraction of nominal foot support in (0, 1] (O9).
        """
        return {
            "mu": self.friction_at(self._pos),
            "payload_kg": float(self.mass_kg - self._base_mass_kg),
            "effort_scale": float(self._effort_scale),
            "support_ratio": float(self._support_at(self._pos)),
        }

    def apply_push(self, impulse_xy_ns: np.ndarray, yaw_impulse_nms: float) -> None:
        self._vel_world = self._vel_world + np.asarray(impulse_xy_ns, dtype=np.float64) / (
            self.mass_kg
        )
        self._yaw_rate += float(yaw_impulse_nms) / float(self._cfg.yaw_inertia_kgm2)

    def start_push_pulse(
        self,
        impulse_xy_ns: np.ndarray,
        yaw_impulse_nms: float,
        duration_s: float,
        application_point_body_m: np.ndarray,
    ) -> None:
        """CPU fallback preserves total impulse; point/duration fidelity is Isaac-only."""
        if duration_s <= 0.0:
            raise ValueError("push-pulse duration must be positive")
        point = np.asarray(application_point_body_m, dtype=np.float64)
        if point.shape != (3,):
            raise ValueError("push-pulse application point must have shape (3,)")
        self.apply_push(impulse_xy_ns, yaw_impulse_nms)

    # ------------------------------------------------------------ dynamics

    def step(self, cmd_vel: np.ndarray) -> Obs:
        if self._fallen:
            return self._obs()
        cmd = np.asarray(cmd_vel, dtype=np.float64).copy()
        max_v = float(self._cfg.max_speed_mps)
        max_w = float(self._cfg.max_yaw_rate_radps)
        cmd[:2] = np.clip(cmd[:2], -max_v, max_v)
        cmd[2] = float(np.clip(cmd[2], -max_w, max_w))
        self._cmd_prev = cmd

        # Closed-loop posture tracking (#41): drive the body height toward the commanded target
        # (or relax to nominal when released), at a rate set by the hold stiffness.
        target = (
            self._posture_target
            if self._posture_target is not None
            else float(self._cfg.base_height_m)
        )
        rate = float(np.clip(self._posture_stiffness, 0.1, 1.0)) * float(
            self._cfg.get("posture_track_rate", 0.5)
        )
        self._posture_height += rate * (target - self._posture_height)

        self._update_collapse_triggers()
        mu = self.friction_at(self._pos)
        support = self._support_at(self._pos)
        self._support_ratio = support
        reflex = self._reflex_active
        gait_scale = float(self._cfg.reflex.gait_demand_scale) if reflex else 1.0
        # Effort budget in accel units: O10 scales the actuator force (effort_scale),
        # O5 raises the mass it must drive (base/total) — both shrink the achievable
        # accel and saturate torque, which is the constructive O5↔O10 ambiguity.
        effort_budget = (
            self._effort_budget_nominal * self._effort_scale * (self._base_mass_kg / self.mass_kg)
        )

        vel_body = world_to_body(self._vel_world, self._heading)
        result = traction_step(
            vel_body,
            cmd[:2],
            mu,
            tau_track_s=float(self._cfg.tau_track_s),
            gait_demand_per_speed=float(self._cfg.gait_demand_per_speed) * gait_scale,
            gravity=float(self._cfg.gravity),
            effort_budget=effort_budget,
            extra_demand=self._payload_com_demand,
        )
        self._slip = result.slip_ratio
        self._effort = result.effort_ratio
        # Foot-support loss (O9) scales delivered authority below the friction/effort cap.
        accel_body = result.accel_body * support
        vel_body = vel_body + accel_body * self.dt
        # Support-loss drag (O9): feet that cannot grip cannot sustain the gait, so a
        # beached robot bleeds speed instead of coasting through — this is what turns
        # the high-centering region into a measurable stall (tracking-error spike).
        if support < 1.0:
            drag_factor = max(
                0.0, 1.0 - (1.0 - support) * float(self._cfg.gait_demand_per_speed) * self.dt
            )
            vel_body = vel_body * drag_factor
        # Tangential resistance (O2 compliance sink / O4 elastic tether): a displacement-
        # dependent decelerating force opposing motion (spec §8.2). The two operators share
        # this force law with matched coefficients, so their proprioceptive traces are
        # identical (the constructive ambiguity pair, P4) — only the visual map separates them.
        vel_body = self._apply_resistance(vel_body)
        # Lateral skid: seeded noise proportional to slip (ice drift), body-frame y.
        vel_body[1] += (
            float(self._cfg.skid_noise_std)
            * self._slip
            * np.sqrt(self.dt)
            * self._rng.standard_normal()
        )
        # Yaw tracking shares the (support-scaled) traction authority.
        yaw_auth = result.authority * support
        yaw_accel = (cmd[2] - self._yaw_rate) / float(self._cfg.tau_yaw_s) * yaw_auth
        self._yaw_rate += yaw_accel * self.dt
        self._heading = wrap_angle(self._heading + self._yaw_rate * self.dt)
        self._vel_world = body_to_world(vel_body, self._heading)

        proposed = self._pos + self._vel_world * self.dt
        proposed, blocked = self._apply_blocking(self._pos, proposed)
        if blocked:
            # Measured velocity reflects the hard stop, so the command/measured gap
            # (tracking error) is what reveals the invisible wall (O8).
            self._vel_world = (proposed - self._pos) / self.dt
        self._pos = proposed
        self._t += self.dt

        self._update_instability(world_to_body(self._vel_world, self._heading))
        return self._obs()

    def _apply_resistance(self, vel_body: np.ndarray) -> np.ndarray:
        """Decelerate by the tangential resistance of any region underfoot (O2 mud / O4 tether).

        O2 compliance is a soft-ground DRAG FIELD: ``F = drag + c*|v|`` — bounded, distance-
        independent, crossable at reduced speed (real mud is drag, not an elastic trap). O4 tether
        is the elastic spring ``F = k*(s−L_0) + c*|v|`` that grows with displacement and can snap.
        The force opposes the current velocity (capped so a single step can't reverse it) and
        accumulates a sink offset (O2). On exit the per-region path/anchor state resets.
        """
        sink = 0.0
        for state in self._resistance:
            region = state.region
            inside = region.rect.contains(self._pos)
            if not inside:
                state.reset()
                continue
            speed = float(np.linalg.norm(vel_body))
            if region.kind == "compliance":
                # Soft-ground drag field (mud): BOUNDED constant + viscous drag, NOT a spring; no
                # penetration growth so it is crossable (§6 #38). stiffness_n_per_m is the drag [N].
                force = region.stiffness_n_per_m + region.damping_ns_per_m * speed
            else:
                # O4 adhesive GRIP (Bug-1, entry-point form — HEADING-INDEPENDENT): the hold grows
                # with the distance from where the dog ENTERED the patch (how deep). It resists
                # motion that goes DEEPER (away from the entry) at full strength (STALLS), and
                # peels (peel_factor) when the dog moves back TOWARD the entry — so back-off escapes
                # whatever way the dog is facing (the body-frame form broke when it turned). A LOW-
                # break tether still tears under forward load (distance==path_len there → M5 gate).
                if state.entry is None:
                    state.entry = self._pos.copy()
                pen = float(np.linalg.norm(self._pos - state.entry))
                moving_out = pen < state.pen_prev - 1.0e-4
                state.pen_prev = pen
                if region.p0_m > 0.0:
                    # A4.1 two-phase delayed-divergence (mirrors IsaacPolicyBackend byte-for-byte):
                    # plateau grip = force_offset_n (≡ O2 k_c drag) for pen≤p0, then a linear ramp
                    # force_offset_n + k2·(pen−p0) beyond; f_break fires on the ramp grip
                    # finite, immobilization if inf + high k2). Default 0/0 ⇒ the #49 path below.
                    grip = region.force_offset_n + region.k2_n_per_m * max(0.0, pen - region.p0_m)
                else:
                    grip = region.stiffness_n_per_m * max(0.0, pen - region.slack_length_m)
                    if grip > region.break_force_n:
                        state.broken = True
                    # #49 peel-plateau: bound the spring grip + constant pre-load (default inf/0 ⇒
                    # unchanged). Break is on the RAW grip (a real tether tears under spring load).
                    grip = min(grip, region.force_cap_n) + region.force_offset_n
                if not state.broken and grip > region.break_force_n:
                    state.broken = True
                force = grip * (region.peel_factor if moving_out else 1.0)
                force += region.damping_ns_per_m * speed
            if state.broken:
                # Tether snapped under forward load (O4 low-break): no further resistance.
                sink = max(sink, region.sink_depth_m)
                continue
            if speed > 1e-9:
                decel = force / self.mass_kg
                dv = min(decel * self.dt, speed)
                vel_body = vel_body - (vel_body / speed) * dv
            sink = max(sink, region.sink_depth_m)
        self._base_height_offset = sink
        return vel_body

    def _support_at(self, pos: np.ndarray) -> float:
        support = 1.0
        for region in self._support:
            if region.rect.contains(pos):
                support = min(support, region.residual_support)
        return support

    def _update_collapse_triggers(self) -> None:
        for state in self._collapse:
            if state.collapsed:
                continue
            if state.region.rect.contains(self._pos):
                if state.region.damage_threshold_ns is not None:
                    # CPU scaffolding has no feet.  Use total static support impulse as a monotonic
                    # proxy; Isaac's publication path integrates measured per-foot normal forces.
                    state.damage_ns += self.mass_kg * float(self._cfg.gravity) * self.dt
                    threshold_reached = state.damage_ns >= state.region.damage_threshold_ns
                else:
                    state.dwell += self.dt
                    threshold_reached = state.dwell >= state.region.trigger_dwell_s
                if threshold_reached:
                    state.collapsed = True

    def _apply_blocking(self, cur: np.ndarray, proposed: np.ndarray) -> tuple[np.ndarray, bool]:
        """Hard-stop motion that would cross into a blocking region (O8 invisible wall)."""
        blocked = False
        out = proposed.copy()
        for region in self._blocking:
            if not region.collision_enabled:
                continue
            rect = region.rect
            if rect.contains(cur):
                continue  # already inside (shouldn't happen): don't trap
            if rect.contains(out):
                blocked = True
                # Clamp the axis with the larger penetration to the rect boundary.
                dx = out[0] - cur[0]
                dy = out[1] - cur[1]
                if abs(dx) >= abs(dy) and dx != 0.0:
                    edge = rect.cx - np.sign(dx) * rect.hx
                    out[0] = edge - np.sign(dx) * 1e-3
                elif dy != 0.0:
                    edge = rect.cy - np.sign(dy) * rect.hy
                    out[1] = edge - np.sign(dy) * 1e-3
        return out, blocked

    def _update_instability(self, vel_body: np.ndarray) -> None:
        speed = float(np.linalg.norm(vel_body))
        fall = self._cfg.fall
        threshold = float(fall.threshold)
        if self._reflex_active:
            threshold *= float(self._cfg.reflex.stability_factor)
        # Capture limit: the speed the current stance can brake within its support
        # polygon (proxy for 0-step capturability, spec §6.2). A push past it threatens
        # a topple even on dry ground; the Reflex stance widens it.
        v_cap = float(fall.capture_speed_mps)
        if self._reflex_active:
            v_cap *= float(self._cfg.reflex.capture_factor)
        excite = 0.0
        if self._slip > 0.0 and speed > float(fall.min_speed_mps):
            excite += float(fall.instability_gain) * self._slip * speed
        if speed > v_cap:
            excite += float(fall.capture_gain) * (speed - v_cap)
        if excite > 0.0:
            self._instability += excite * self.dt
        else:
            self._instability = max(0.0, self._instability - float(fall.decay_per_s) * self.dt)
        if self._instability > threshold:
            self._fallen = True

    def _obs(self) -> Obs:
        return Obs(
            t=self._t,
            pos=self._pos.copy(),
            heading=self._heading,
            vel_body=world_to_body(self._vel_world, self._heading),
            yaw_rate=self._yaw_rate,
            cmd_prev=self._cmd_prev.copy(),
            slip_ratio=self._slip,
            # When a posture is actively commanded the height controller dominates (the dog holds
            # the commanded height, #41); otherwise the passive O2 compliance sink applies.
            base_height=(
                self._posture_height
                if self._posture_target is not None
                else float(self._cfg.base_height_m) - self._base_height_offset
            ),
            tilt=0.0,
            fallen=self._fallen,
            effort_ratio=self._effort,
            support_ratio=self._support_ratio,
        )


class _CollapseState:
    """Mutable legacy-dwell or load-damage trigger state for O3 collapse regions."""

    __slots__ = ("region", "dwell", "damage_ns", "collapsed")

    def __init__(self, region: CollapseRegion) -> None:
        self.region = region
        self.dwell = 0.0
        self.damage_ns = 0.0
        self.collapsed = False


class _ResistanceState:
    """Mutable per-region state for O2/O4 resistance regions (Bug-1: entry-point grip)."""

    __slots__ = ("region", "path_len", "broken", "entry", "pen_prev")

    def __init__(self, region: ResistanceRegion) -> None:
        self.region = region
        self.path_len = 0.0
        self.broken = False
        self.entry = None  # world entry point into the patch (set on first contact)
        self.pen_prev = 0.0  # last distance-from-entry (to detect backing-out vs going-deeper)

    def reset(self) -> None:
        self.path_len = 0.0
        self.broken = False
        self.entry = None
        self.pen_prev = 0.0
