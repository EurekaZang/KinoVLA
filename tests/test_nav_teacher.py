"""Geometric route-around teacher (nav SFT labels) — pure-geometry unit tests."""

from __future__ import annotations

import math

import numpy as np

from kino_vla.utils.geometry import Rect, wrap_angle
from kino_vla.vla.nav_teacher import _seg_crosses_rect, next_nav_label, next_waypoint

PATCH = Rect(3.5, 0.0, 2.236, 2.236)  # the 5x tether patch (x∈[1.26,5.74], y∈[-2.24,2.24])
GOAL = np.array([7.5, 0.0])


def test_seg_crosses_rect_straight_through_vs_over_the_top():
    assert _seg_crosses_rect(np.array([0.0, 0.0]), np.array([7.5, 0.0]), PATCH)  # straight through
    assert not _seg_crosses_rect(
        np.array([0.0, 3.0]), np.array([7.5, 3.0]), PATCH
    )  # clears the top
    assert not _seg_crosses_rect(np.array([0.0, -3.0]), np.array([7.5, -3.0]), PATCH)  # below


def test_next_waypoint_routes_around_when_blocked():
    wp = next_waypoint(np.array([0.0, 0.0]), GOAL, PATCH, margin=0.5)
    assert not np.allclose(wp, GOAL)  # the straight line is blocked ⇒ first hop is a corner
    assert abs(wp[1]) > 2.0  # routed to a top/bottom corner, clear of the patch


def test_clear_straight_line_goes_to_goal():
    wp = next_waypoint(np.array([6.4, 3.0]), GOAL, PATCH, margin=0.5)
    assert np.allclose(wp, GOAL)  # already past + above the patch ⇒ head straight to the goal


def _mock_project(pose: np.ndarray, heading: float, hfov_deg: float = 30.0):
    """A toy forward-down camera: a ground point is 'in frame' iff within ±hfov of the heading and
    within range; maps the horizontal offset to a u in [0,1] (v fixed). Mirrors the FOV gate."""

    def proj(wp: np.ndarray):
        d = np.asarray(wp, dtype=np.float64) - pose
        rng = float(np.linalg.norm(d))
        off = math.degrees(wrap_angle(math.atan2(d[1], d[0]) - heading))
        if abs(off) <= hfov_deg and rng < 4.0:
            return np.array([0.5 + 0.5 * off / hfov_deg, 0.5])
        return None

    return proj


def test_label_is_turn_when_route_is_out_of_frame():
    # Backout pose facing +x straight into the patch; the route-around corner is ~90° to the side,
    # outside the camera FOV ⇒ the teacher must TURN (not pick a forbidden forward waypoint).
    pose, heading = np.array([0.8, 0.0]), 0.0
    label = next_nav_label(pose, heading, PATCH, GOAL, _mock_project(pose, heading), margin=0.5)
    assert label["kind"] == "turn"
    assert -120.0 <= label["yaw_deg"] <= 120.0  # clamp raised 90→120 (#44 round-2)
    assert abs(label["yaw_deg"]) > 30.0  # a real turn toward the side corner


def test_label_is_waypoint_when_route_is_in_view():
    # Skirting the top edge, heading toward the goal; the next hop is roughly ahead ⇒ a waypoint.
    pose, heading = np.array([3.0, 3.0]), math.atan2(GOAL[1] - 3.0, GOAL[0] - 3.0)
    label = next_nav_label(pose, heading, PATCH, GOAL, _mock_project(pose, heading), margin=0.5)
    assert label["kind"] == "waypoint"
    assert 0 <= label["point_px"][0] <= 1000 and 0 <= label["point_px"][1] <= 1000


# --- multi-patch (the DAgger scenario battery: two offset hazards, #43) -----------------------
P_TOP = Rect(3.5, 1.2, 1.0, 1.0)  # y ∈ [0.2, 2.2]
P_BOT = Rect(3.5, -1.2, 1.0, 1.0)  # y ∈ [-2.2, -0.2]; the +0.5 inflation overlaps over y≈0


def test_single_patch_list_matches_single_patch_rect():
    """Backward-compat: a one-element list routes identically to passing the bare Rect."""
    start = np.array([0.0, 0.0])
    assert np.allclose(
        next_waypoint(start, GOAL, PATCH, margin=0.5),
        next_waypoint(start, GOAL, [PATCH], margin=0.5),
    )


def test_next_waypoint_routes_around_two_offset_patches():
    """Two patches straddling the centreline block the straight line; the first hop clears BOTH."""
    start = np.array([0.0, 0.0])
    wp = next_waypoint(start, GOAL, [P_TOP, P_BOT], margin=0.5)
    assert not np.allclose(wp, GOAL)  # the middle is blocked by both inflated patches
    assert not _seg_crosses_rect(start, wp, P_TOP), "first hop clears the top hazard"
    assert not _seg_crosses_rect(start, wp, P_BOT), "first hop clears the bottom hazard"


def test_multi_patch_label_turns_when_route_out_of_frame():
    """Facing straight into the two-patch wall, the side route is out of the FOV ⇒ TURN."""
    pose, heading = np.array([0.8, 0.0]), 0.0
    label = next_nav_label(
        pose, heading, [P_TOP, P_BOT], GOAL, _mock_project(pose, heading), margin=0.5
    )
    assert label["kind"] == "turn"
    assert abs(label["yaw_deg"]) > 20.0
