from __future__ import annotations

import numpy as np
import pytest

from kino_vla.sim.payload import combine_with_cuboid_payload, cuboid_inertia


def test_uniform_cuboid_inertia_matches_closed_form() -> None:
    inertia = cuboid_inertia(6.0, np.array([0.3, 0.2, 0.1]))
    assert np.diag(inertia) == pytest.approx([0.025, 0.05, 0.065])
    assert np.count_nonzero(inertia - np.diag(np.diag(inertia))) == 0


def test_offset_payload_shifts_com_and_adds_parallel_axis_inertia() -> None:
    combined = combine_with_cuboid_payload(
        base_mass_kg=12.0,
        base_com_m=np.zeros(3),
        base_inertia_kg_m2=np.diag([0.20, 0.30, 0.40]),
        payload_mass_kg=6.0,
        payload_com_m=np.array([0.18, -0.06, 0.12]),
        payload_size_m=np.array([0.30, 0.20, 0.10]),
    )
    assert combined.mass_kg == pytest.approx(18.0)
    assert combined.com_m == pytest.approx([0.06, -0.02, 0.04])
    assert np.trace(combined.inertia_kg_m2) > 0.90
    assert abs(combined.inertia_kg_m2[0, 1]) > 0.0
    assert np.linalg.eigvalsh(combined.inertia_kg_m2).min() > 0.0


def test_payload_model_rejects_nonphysical_dimensions() -> None:
    with pytest.raises(ValueError, match="dimensions"):
        cuboid_inertia(1.0, np.array([0.2, 0.0, 0.1]))
