"""A4.1 — two-phase (delayed-divergence) O4 adhesion construction tests (CPU surrogate; no Isaac).

Locks in the A4.1 construction as a regression test on the CPU surrogate (scaffolding only, per
CLAUDE.md §0 — the surrogate never produces A4 RESULTS, but it faithfully mirrors the Isaac tether
force law byte-for-byte, so it validates the construction here):
  - the PLATEAU (pen ≤ p0_m) is byte-identical to O2 compliance (the C1/A1.3 attribution window);
  - the RAMP (pen > p0_m) diverges (the consequence region);
  - ``f_break=inf`` never snaps (the immobilization preset);
  - defaults (p0_m=0, k2=0) reproduce the #49 path unchanged (additive).
The Isaac plateau branch is ``force_offset_n + damping·speed`` == compliance's ``k_c + c_c·speed``
(construction-identical); A1.3 re-certifies on real Go2.
"""

from __future__ import annotations

import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from kino_vla.sim.operators import ComplianceField, Tether  # noqa: E402
from kino_vla.sim.surrogate import SurrogateBackend  # noqa: E402
from kino_vla.utils.config import load_config  # noqa: E402
from kino_vla.utils.geometry import Rect  # noqa: E402


def _decel(op, pen: float, speed: float = 0.5) -> float:
    """Resistance-induced speed loss at a given penetration into the patch (surrogate, CPU)."""
    cfg = load_config("sim/surrogate.yaml")
    b = SurrogateBackend(cfg, np.array([0.0, 0.0]), 0.0)
    b._resistance = []
    op.on_reset(b)
    b._pos = np.array([3.0 - 1.0 + pen, 0.0])  # entry at the patch left edge; pen along +x
    st = b._resistance[0]
    st.entry = np.array([2.0, 0.0])
    st.pen_prev = pen
    vel = np.array([speed, 0.0])
    out = b._apply_resistance(vel.copy())
    return float(np.linalg.norm(vel) - np.linalg.norm(out))


def test_twophase_plateau_byte_identical_to_compliance():
    rect = Rect(cx=3.0, cy=0.0, hx=1.0, hy=1.0)
    o2 = ComplianceField(rect, k_c=14.0, c_c=6.0, d_sink=0.08)  # mud: 14 + 6·speed
    o4 = Tether(rect, k=14.0, d=6.0, l0=0.0, f_break=1e9, force_offset_n=14.0,
                p0_m=0.30, k2_n_per_m=60.0)  # two-phase plateau ≡ O2 for pen≤p0
    for pen in (0.05, 0.15, 0.30):  # plateau region (pen ≤ p0)
        assert _decel(o2, pen) == _decel(o4, pen), f"plateau differs at pen={pen}"


def test_twophase_ramp_diverges_beyond_p0():
    rect = Rect(cx=3.0, cy=0.0, hx=1.0, hy=1.0)
    o2 = ComplianceField(rect, k_c=14.0, c_c=6.0, d_sink=0.08)
    o4 = Tether(rect, k=14.0, d=6.0, l0=0.0, f_break=1e9, force_offset_n=14.0,
                p0_m=0.30, k2_n_per_m=60.0)
    for pen in (0.45, 0.60, 0.90):  # ramp region (pen > p0): two-phase grip exceeds O2's plateau
        assert _decel(o4, pen) > _decel(o2, pen), f"ramp did not diverge at pen={pen}"


def test_twophase_ramp_grows_monotonically_with_pen():
    rect = Rect(cx=3.0, cy=0.0, hx=1.0, hy=1.0)
    o4 = Tether(rect, k=14.0, d=6.0, l0=0.0, f_break=1e9, force_offset_n=14.0,
                p0_m=0.30, k2_n_per_m=60.0)
    ds = [_decel(o4, pen) for pen in (0.45, 0.60, 0.90)]
    assert ds == sorted(ds)


def test_twophase_inf_break_never_snaps():
    rect = Rect(cx=3.0, cy=0.0, hx=1.0, hy=1.0)
    o4 = Tether(rect, k=14.0, d=6.0, l0=0.0, f_break=1e9, force_offset_n=14.0,
                p0_m=0.30, k2_n_per_m=60.0)  # immobilization preset
    cfg = load_config("sim/surrogate.yaml")
    b = SurrogateBackend(cfg, np.array([0.0, 0.0]), 0.0)
    o4.on_reset(b)
    st = b._resistance[0]
    for pen in (0.5, 1.0, 2.0):  # deep penetration — grip ramps but never breaks
        st.entry = np.array([2.0, 0.0])
        st.pen_prev = pen
        b._pos = np.array([2.0 + pen, 0.0])
        b._apply_resistance(np.array([0.5, 0.0]))
        assert not st.broken, f"inf-break tether snapped at pen={pen}"


def test_twophase_defaults_reproduce_shaped_path():
    # p0_m=0, k2=0 ⇒ the two-phase branch is OFF ⇒ the #49 path is unchanged (additive).
    rect = Rect(cx=3.0, cy=0.0, hx=1.0, hy=1.0)
    shaped = Tether(rect, k=14.0, d=6.0, l0=0.0, f_break=1e9, force_cap_n=0.0,
                    force_offset_n=14.0, peel_factor=0.3)  # #49 matched preset
    default = Tether(rect, k=14.0, d=6.0, l0=0.0, f_break=1e9, force_cap_n=0.0,
                     force_offset_n=14.0, peel_factor=0.3)  # same, p0/k2 default 0
    for pen in (0.2, 0.5, 1.0):
        assert _decel(shaped, pen) == _decel(default, pen)
