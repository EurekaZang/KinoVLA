"""Rigid-body mass-property composition used by the realistic O5 payload path."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass(frozen=True)
class CompositeMassProperties:
    """Mass, center of mass and inertia tensor expressed in the parent body frame."""

    mass_kg: float
    com_m: np.ndarray
    inertia_kg_m2: np.ndarray


def cuboid_inertia(mass_kg: float, size_m: np.ndarray) -> np.ndarray:
    """Return the centroidal inertia of an axis-aligned uniform cuboid."""
    size = np.asarray(size_m, dtype=np.float64)
    if mass_kg <= 0.0 or size.shape != (3,) or not np.isfinite(size).all():
        raise ValueError("mass must be positive and size_m must be a finite 3-vector")
    if np.any(size <= 0.0):
        raise ValueError("cuboid dimensions must be positive")
    x, y, z = size
    return np.diag(
        [
            mass_kg * (y * y + z * z) / 12.0,
            mass_kg * (x * x + z * z) / 12.0,
            mass_kg * (x * x + y * y) / 12.0,
        ]
    )


def _parallel_axis(mass_kg: float, displacement_m: np.ndarray) -> np.ndarray:
    d = np.asarray(displacement_m, dtype=np.float64)
    return mass_kg * (float(d @ d) * np.eye(3) - np.outer(d, d))


def combine_with_cuboid_payload(
    *,
    base_mass_kg: float,
    base_com_m: np.ndarray,
    base_inertia_kg_m2: np.ndarray,
    payload_mass_kg: float,
    payload_com_m: np.ndarray,
    payload_size_m: np.ndarray,
) -> CompositeMassProperties:
    """Compose a body and attached cuboid via the exact parallel-axis theorem."""
    base_com = np.asarray(base_com_m, dtype=np.float64)
    payload_com = np.asarray(payload_com_m, dtype=np.float64)
    base_inertia = np.asarray(base_inertia_kg_m2, dtype=np.float64)
    if base_mass_kg <= 0.0 or payload_mass_kg <= 0.0:
        raise ValueError("base and payload masses must be positive")
    if base_com.shape != (3,) or payload_com.shape != (3,):
        raise ValueError("base and payload CoM values must be 3-vectors")
    if base_inertia.shape != (3, 3):
        raise ValueError("base inertia must have shape (3, 3)")
    if not (
        np.isfinite(base_com).all()
        and np.isfinite(payload_com).all()
        and np.isfinite(base_inertia).all()
    ):
        raise ValueError("mass properties must be finite")
    if not np.allclose(base_inertia, base_inertia.T, atol=1.0e-10):
        raise ValueError("base inertia must be symmetric")
    total_mass = float(base_mass_kg + payload_mass_kg)
    total_com = (base_mass_kg * base_com + payload_mass_kg * payload_com) / total_mass
    total_inertia = (
        base_inertia
        + _parallel_axis(base_mass_kg, base_com - total_com)
        + cuboid_inertia(payload_mass_kg, np.asarray(payload_size_m, dtype=np.float64))
        + _parallel_axis(payload_mass_kg, payload_com - total_com)
    )
    eigenvalues = np.linalg.eigvalsh(total_inertia)
    if np.any(eigenvalues <= 0.0):
        raise ValueError("composite inertia must be positive definite")
    return CompositeMassProperties(total_mass, total_com, total_inertia)
