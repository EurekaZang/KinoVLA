from __future__ import annotations

import numpy as np
import pytest

from kino_vla.sim.terramechanics import (
    FootTerrainState,
    FootTerramechanicsConfig,
    continuous_support_mesh,
    longitudinal_sink_profile,
    step_foot_terramechanics,
)
from kino_vla.utils.geometry import Rect


def _config(retention: float = 0.32) -> FootTerramechanicsConfig:
    return FootTerramechanicsConfig(
        region=Rect(0.0, 0.0, 1.0, 1.0),
        max_sink_depth_m=0.11,
        shear_retention=retention,
        vertical_stiffness_n_per_m=240.0,
    )


def test_soft_contact_is_foot_local_load_gated_and_dissipative() -> None:
    update = step_foot_terramechanics(
        _config(),
        FootTerrainState(),
        foot_pos_xyz_m=np.array([0.1, 0.2, -0.06]),
        foot_vel_xyz_mps=np.array([0.3, 0.0, 0.0]),
        normal_force_n=50.0,
        dt_s=0.02,
    )
    assert update.in_region and update.load_bearing
    assert update.sinkage_m == pytest.approx(0.06)
    assert update.force_world_n[0] < 0.0
    assert np.dot(update.force_world_n, np.array([0.3, 0.0, 0.0])) < 0.0
    assert update.state.shear_work_j > 0.0
    assert update.state.max_applied_force_n > 0.0
    assert update.state.contact_steps == 1


def test_swing_foot_and_outside_foot_receive_no_magic_drag() -> None:
    for position, normal_force in (([0.0, 0.0, -0.04], 0.5), ([2.0, 0.0, 0.0], 100.0)):
        update = step_foot_terramechanics(
            _config(),
            FootTerrainState(),
            foot_pos_xyz_m=np.asarray(position),
            foot_vel_xyz_mps=np.array([0.5, 0.0, 0.0]),
            normal_force_n=normal_force,
            dt_s=0.02,
        )
        assert np.array_equal(update.force_world_n, np.zeros(3))
        assert update.state.shear_work_j == 0.0


def test_severity_monotonically_reduces_retained_shear_support() -> None:
    forces = []
    for retention in (0.8, 0.55, 0.32):
        update = step_foot_terramechanics(
            _config(retention),
            FootTerrainState(),
            foot_pos_xyz_m=np.array([0.0, 0.0, -0.05]),
            foot_vel_xyz_mps=np.array([0.25, 0.0, 0.0]),
            normal_force_n=45.0,
            dt_s=0.02,
        )
        forces.append(abs(update.force_world_n[0]))
    assert forces[0] < forces[1] < forces[2]


def test_sink_profile_is_continuous_and_leaves_flat_core() -> None:
    region = Rect(1.9, 0.0, 0.9, 0.8)
    profile = longitudinal_sink_profile(region, sink_depth_m=0.11, transition_length_m=0.30)
    assert profile.entry.pitch_rad > 0.0
    assert profile.exit.pitch_rad == pytest.approx(-profile.entry.pitch_rad)
    assert profile.entry.center_top_z_m == pytest.approx(-0.055)
    assert profile.bed_top_z_m == pytest.approx(-0.11)
    assert profile.core.hx == pytest.approx(0.6)
    half_run = 0.5 * profile.entry.horizontal_length_m
    entry_top = profile.entry.center_top_z_m + np.tan(profile.entry.pitch_rad) * half_run
    entry_bottom = profile.entry.center_top_z_m - np.tan(profile.entry.pitch_rad) * half_run
    assert entry_top == pytest.approx(0.0)
    assert entry_bottom == pytest.approx(profile.bed_top_z_m)


def test_sink_profile_rejects_transitions_that_consume_patch() -> None:
    with pytest.raises(ValueError, match="non-empty soft-terrain core"):
        longitudinal_sink_profile(
            Rect(0.0, 0.0, 0.2, 0.5), sink_depth_m=0.05, transition_length_m=0.20
        )


@pytest.mark.parametrize("axis", ["x", "y"])
def test_continuous_support_mesh_is_topology_matched_and_seam_free(axis: str) -> None:
    region = Rect(0.0, 0.86, 0.6, 0.45) if axis == "y" else Rect(0.86, 0.0, 0.45, 0.6)
    config = FootTerramechanicsConfig(
        region=region,
        max_sink_depth_m=0.11,
        shear_retention=0.32,
        vertical_stiffness_n_per_m=240.0,
        longitudinal_axis=axis,
    )
    nominal = continuous_support_mesh(config, half_extent_m=15.0, matched_control=True)
    anomaly = continuous_support_mesh(config, half_extent_m=15.0, matched_control=False)
    assert nominal.face_vertex_counts == anomaly.face_vertex_counts == (4,) * 25
    assert nominal.face_vertex_indices == anomaly.face_vertex_indices
    assert len(nominal.points_xyz_m) == len(anomaly.points_xyz_m) == 36
    assert {point[2] for point in nominal.points_xyz_m} == {0.0}
    assert anomaly.minimum_height_m == pytest.approx(-0.11)
    assert all(np.isfinite(point).all() for point in map(np.asarray, anomaly.points_xyz_m))
    assert nominal.longitudinal_axis == anomaly.longitudinal_axis == axis
    first_face = np.asarray(anomaly.face_vertex_indices[:4])
    face_points = np.asarray(anomaly.points_xyz_m)[first_face]
    normal = np.cross(face_points[1] - face_points[0], face_points[3] - face_points[0])
    assert normal[2] > 0.0


def test_continuous_support_mesh_has_zero_height_at_every_outer_operator_boundary() -> None:
    config = FootTerramechanicsConfig(
        region=Rect(0.0, 0.86, 0.6, 0.45),
        max_sink_depth_m=0.11,
        shear_retention=0.32,
        vertical_stiffness_n_per_m=240.0,
        longitudinal_axis="y",
    )
    support = continuous_support_mesh(config, half_extent_m=15.0, matched_control=False)
    points = np.asarray(support.points_xyz_m)
    boundary = (
        np.isclose(points[:, 0], config.region.cx - config.region.hx)
        | np.isclose(points[:, 0], config.region.cx + config.region.hx)
        | np.isclose(points[:, 1], config.region.cy - config.region.hy)
        | np.isclose(points[:, 1], config.region.cy + config.region.hy)
    )
    assert np.allclose(points[boundary, 2], 0.0)
