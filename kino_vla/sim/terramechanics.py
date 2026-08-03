"""Auditable per-foot reduced terramechanics for realistic O2 soft terrain."""

from __future__ import annotations

import math
from dataclasses import dataclass, replace

import numpy as np

from kino_vla.utils.geometry import Rect


@dataclass(frozen=True)
class FootTerramechanicsConfig:
    """Calibratable soft-patch geometry and shear-contact parameters."""

    region: Rect
    max_sink_depth_m: float
    shear_retention: float
    vertical_stiffness_n_per_m: float
    shear_velocity_scale_mps: float = 0.12
    contact_threshold_n: float = 2.0
    surface_z_m: float = 0.0
    longitudinal_axis: str = "x"

    def __post_init__(self) -> None:
        if self.max_sink_depth_m <= 0.0:
            raise ValueError("max_sink_depth_m must be positive")
        if not 0.0 <= self.shear_retention <= 1.0:
            raise ValueError("shear_retention must be in [0, 1]")
        if self.vertical_stiffness_n_per_m <= 0.0:
            raise ValueError("vertical_stiffness_n_per_m must be positive")
        if self.shear_velocity_scale_mps <= 0.0 or self.contact_threshold_n < 0.0:
            raise ValueError("velocity scale must be positive and contact threshold non-negative")
        if self.longitudinal_axis not in {"x", "y"}:
            raise ValueError("longitudinal_axis must be 'x' or 'y'")


@dataclass(frozen=True)
class ContinuousSupportMesh:
    """One topology-matched collision surface for nominal and compliant O2 lanes."""

    points_xyz_m: tuple[tuple[float, float, float], ...]
    face_vertex_counts: tuple[int, ...]
    face_vertex_indices: tuple[int, ...]
    longitudinal_coordinates_m: tuple[float, ...]
    lateral_coordinates_m: tuple[float, ...]
    longitudinal_axis: str
    matched_control: bool
    minimum_height_m: float


def continuous_support_mesh(
    config: FootTerramechanicsConfig,
    *,
    half_extent_m: float,
    matched_control: bool,
    longitudinal_transition_m: float | None = None,
    lateral_transition_m: float | None = None,
) -> ContinuousSupportMesh:
    """Build a single continuous height-field mesh with no inter-prim collision seams.

    Both lanes share the exact same vertices and faces.  The nominal lane keeps every vertex
    coplanar; the anomaly lane lowers only the interior using separable piecewise-linear
    longitudinal and lateral shoulders.  The surface therefore has neither a collider seam nor
    a rigid vertical lip at the operator boundary.
    """

    half_extent = float(half_extent_m)
    if half_extent <= 0.0:
        raise ValueError("support mesh half extent must be positive")
    region = config.region
    if config.longitudinal_axis == "x":
        long_center, long_half = float(region.cx), float(region.hx)
        lat_center, lat_half = float(region.cy), float(region.hy)
    else:
        long_center, long_half = float(region.cy), float(region.hy)
        lat_center, lat_half = float(region.cx), float(region.hx)
    if not (
        -half_extent < long_center - long_half < long_center + long_half < half_extent
        and -half_extent < lat_center - lat_half < lat_center + lat_half < half_extent
    ):
        raise ValueError("compliance region must lie inside the support mesh extent")

    long_transition = float(
        longitudinal_transition_m
        if longitudinal_transition_m is not None
        else min(0.30, 0.30 * long_half)
    )
    lat_transition = float(
        lateral_transition_m
        if lateral_transition_m is not None
        else min(0.12, 0.25 * lat_half)
    )
    if not 0.0 < 2.0 * long_transition < 2.0 * long_half:
        raise ValueError("longitudinal transitions must leave a non-empty compliant core")
    if not 0.0 < 2.0 * lat_transition < 2.0 * lat_half:
        raise ValueError("lateral transitions must leave a non-empty compliant core")

    long_coordinates = (
        -half_extent,
        long_center - long_half,
        long_center - long_half + long_transition,
        long_center + long_half - long_transition,
        long_center + long_half,
        half_extent,
    )
    lat_coordinates = (
        -half_extent,
        lat_center - lat_half,
        lat_center - lat_half + lat_transition,
        lat_center + lat_half - lat_transition,
        lat_center + lat_half,
        half_extent,
    )
    long_weights = (0.0, 0.0, 1.0, 1.0, 0.0, 0.0)
    lat_weights = (0.0, 0.0, 1.0, 1.0, 0.0, 0.0)
    points: list[tuple[float, float, float]] = []
    for lateral, lateral_weight in zip(lat_coordinates, lat_weights, strict=True):
        for longitudinal, longitudinal_weight in zip(
            long_coordinates, long_weights, strict=True
        ):
            sink = (
                0.0
                if matched_control
                else float(config.max_sink_depth_m)
                * longitudinal_weight
                * lateral_weight
            )
            if config.longitudinal_axis == "x":
                points.append((longitudinal, lateral, float(config.surface_z_m) - sink))
            else:
                points.append((lateral, longitudinal, float(config.surface_z_m) - sink))

    n_long = len(long_coordinates)
    n_lat = len(lat_coordinates)
    indices: list[int] = []
    for lat_index in range(n_lat - 1):
        for long_index in range(n_long - 1):
            lower_left = lat_index * n_long + long_index
            if config.longitudinal_axis == "x":
                indices.extend(
                    (
                        lower_left,
                        lower_left + 1,
                        lower_left + n_long + 1,
                        lower_left + n_long,
                    )
                )
            else:
                indices.extend(
                    (
                        lower_left,
                        lower_left + n_long,
                        lower_left + n_long + 1,
                        lower_left + 1,
                    )
                )
    return ContinuousSupportMesh(
        points_xyz_m=tuple(points),
        face_vertex_counts=(4,) * ((n_long - 1) * (n_lat - 1)),
        face_vertex_indices=tuple(indices),
        longitudinal_coordinates_m=tuple(long_coordinates),
        lateral_coordinates_m=tuple(lat_coordinates),
        longitudinal_axis=config.longitudinal_axis,
        matched_control=bool(matched_control),
        minimum_height_m=min(point[2] for point in points),
    )


@dataclass(frozen=True)
class SinkRampSpec:
    """One longitudinal top surface connecting firm ground to a lowered soil bed."""

    center_x_m: float
    center_y_m: float
    horizontal_length_m: float
    slope_length_m: float
    width_m: float
    center_top_z_m: float
    pitch_rad: float


@dataclass(frozen=True)
class SinkSurfaceProfile:
    """Entry ramp, flat deformable core and exit ramp without rigid vertical lips."""

    entry: SinkRampSpec
    core: Rect
    exit: SinkRampSpec
    bed_top_z_m: float


def longitudinal_sink_profile(
    region: Rect,
    *,
    sink_depth_m: float,
    transition_length_m: float = 0.30,
) -> SinkSurfaceProfile:
    """Construct a continuous piecewise-planar soft-patch collision profile.

    A lowered box with vertical boundaries is not a compliant terrain model: its exit wall can
    trip the robot independently of soil physics.  These two ramps make the collision height
    continuous while preserving a flat core where full sink depth is observable.
    """
    if sink_depth_m <= 0.0 or transition_length_m <= 0.0:
        raise ValueError("sink depth and transition length must be positive")
    if 2.0 * transition_length_m >= 2.0 * region.hx:
        raise ValueError("two transitions must leave a non-empty soft-terrain core")
    angle = math.atan2(sink_depth_m, transition_length_m)
    slope_length = math.hypot(transition_length_m, sink_depth_m)
    left = region.cx - region.hx
    right = region.cx + region.hx
    entry = SinkRampSpec(
        center_x_m=left + 0.5 * transition_length_m,
        center_y_m=region.cy,
        horizontal_length_m=transition_length_m,
        slope_length_m=slope_length,
        width_m=2.0 * region.hy,
        center_top_z_m=-0.5 * sink_depth_m,
        pitch_rad=angle,
    )
    exit_ramp = SinkRampSpec(
        center_x_m=right - 0.5 * transition_length_m,
        center_y_m=region.cy,
        horizontal_length_m=transition_length_m,
        slope_length_m=slope_length,
        width_m=2.0 * region.hy,
        center_top_z_m=-0.5 * sink_depth_m,
        pitch_rad=-angle,
    )
    core = Rect(
        cx=region.cx,
        cy=region.cy,
        hx=region.hx - transition_length_m,
        hy=region.hy,
    )
    return SinkSurfaceProfile(entry=entry, core=core, exit=exit_ramp, bed_top_z_m=-sink_depth_m)


@dataclass(frozen=True)
class FootTerrainState:
    """History carried independently for one named foot."""

    max_sinkage_m: float = 0.0
    max_applied_force_n: float = 0.0
    shear_work_j: float = 0.0
    contact_steps: int = 0
    last_force_world_n: tuple[float, float, float] = (0.0, 0.0, 0.0)


@dataclass(frozen=True)
class FootTerrainUpdate:
    """One measured contact update and the force to apply at the foot body."""

    state: FootTerrainState
    force_world_n: np.ndarray
    sinkage_m: float
    in_region: bool
    load_bearing: bool


def step_foot_terramechanics(
    config: FootTerramechanicsConfig,
    state: FootTerrainState,
    *,
    foot_pos_xyz_m: np.ndarray,
    foot_vel_xyz_mps: np.ndarray,
    normal_force_n: float,
    dt_s: float,
) -> FootTerrainUpdate:
    """Compute load-gated shear loss from measured per-foot pose, velocity and normal force.

    The collision patch supplies the real vertical drop.  This law supplies tangential soil
    yielding at the actual foot, with ``shear_retention=1`` meaning firm-ground traction and zero
    meaning maximum yielding.  No force is applied to a swing foot or a foot outside the patch.
    """
    position = np.asarray(foot_pos_xyz_m, dtype=np.float64)
    velocity = np.asarray(foot_vel_xyz_mps, dtype=np.float64)
    if position.shape != (3,) or velocity.shape != (3,):
        raise ValueError("foot position and velocity must have shape (3,)")
    if not np.isfinite(position).all() or not np.isfinite(velocity).all():
        raise ValueError("foot state must be finite")
    if not math.isfinite(normal_force_n) or dt_s <= 0.0:
        raise ValueError("normal force must be finite and dt_s positive")
    inside = config.region.contains(position[:2])
    loaded = inside and normal_force_n >= config.contact_threshold_n
    sinkage = (
        float(np.clip(config.surface_z_m - position[2], 0.0, config.max_sink_depth_m))
        if inside
        else 0.0
    )
    force = np.zeros(3, dtype=np.float64)
    if loaded:
        speed = float(np.linalg.norm(velocity[:2]))
        if speed > 1.0e-9:
            yielding = 1.0 - config.shear_retention
            magnitude = yielding * max(normal_force_n, 0.0) * math.tanh(
                speed / config.shear_velocity_scale_mps
            )
            force[:2] = -magnitude * velocity[:2] / speed
        shear_power_w = max(0.0, -float(np.dot(force[:2], velocity[:2])))
        next_state = replace(
            state,
            max_sinkage_m=max(state.max_sinkage_m, sinkage),
            max_applied_force_n=max(state.max_applied_force_n, float(np.linalg.norm(force))),
            shear_work_j=state.shear_work_j + shear_power_w * dt_s,
            contact_steps=state.contact_steps + 1,
            last_force_world_n=tuple(float(value) for value in force),
        )
    else:
        next_state = replace(
            state,
            max_sinkage_m=max(state.max_sinkage_m, sinkage),
            last_force_world_n=(0.0, 0.0, 0.0),
        )
    return FootTerrainUpdate(next_state, force, sinkage, inside, loaded)
