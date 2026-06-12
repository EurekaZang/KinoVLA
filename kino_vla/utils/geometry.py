"""Small planar geometry helpers shared by the sim backends and the FSM planner."""

from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np


@dataclass(frozen=True)
class Rect:
    """Axis-aligned rectangle: center (cx, cy) and half-sizes (hx, hy), world frame [m]."""

    cx: float
    cy: float
    hx: float
    hy: float

    def contains(self, pos: np.ndarray) -> bool:
        return bool(
            abs(float(pos[0]) - self.cx) <= self.hx and abs(float(pos[1]) - self.cy) <= self.hy
        )


def wrap_angle(angle: float) -> float:
    """Wrap an angle to (-pi, pi]."""
    return float(math.atan2(math.sin(angle), math.cos(angle)))


def unit(vec: np.ndarray) -> np.ndarray:
    """Unit vector along ``vec``; zero vector maps to +x to keep callers deterministic."""
    norm = float(np.linalg.norm(vec))
    if norm == 0.0:
        return np.array([1.0, 0.0])
    return np.asarray(vec, dtype=np.float64) / norm


def rot90(vec: np.ndarray) -> np.ndarray:
    """Rotate a 2D vector by +90 degrees (counter-clockwise)."""
    return np.array([-float(vec[1]), float(vec[0])])


def world_to_body(vec_world: np.ndarray, heading: float) -> np.ndarray:
    """Rotate a world-frame 2D vector into the body frame of a robot with ``heading``."""
    c, s = math.cos(heading), math.sin(heading)
    return np.array(
        [
            c * float(vec_world[0]) + s * float(vec_world[1]),
            -s * float(vec_world[0]) + c * float(vec_world[1]),
        ]
    )


def body_to_world(vec_body: np.ndarray, heading: float) -> np.ndarray:
    """Rotate a body-frame 2D vector into the world frame."""
    c, s = math.cos(heading), math.sin(heading)
    return np.array(
        [
            c * float(vec_body[0]) - s * float(vec_body[1]),
            s * float(vec_body[0]) + c * float(vec_body[1]),
        ]
    )


def segment_hits_circle(p0: np.ndarray, p1: np.ndarray, center: np.ndarray, radius: float) -> bool:
    """True when the segment p0->p1 passes within ``radius`` of ``center``.

    An endpoint inside the circle counts as a hit (the FSM replans from positions
    that may sit inside the freshly marked avoid region).
    """
    d = np.asarray(p1, dtype=np.float64) - np.asarray(p0, dtype=np.float64)
    f = np.asarray(p0, dtype=np.float64) - np.asarray(center, dtype=np.float64)
    dd = float(d @ d)
    t = 0.0 if dd == 0.0 else float(np.clip(-(f @ d) / dd, 0.0, 1.0))
    closest = f + t * d
    return float(closest @ closest) <= radius * radius
