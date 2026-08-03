"""A0.4 appearance-library tests (offline; no Isaac). The library is the pre-registered
shortcut-killer — ≥4 appearances/class, disjoint train/test splits, within-class colour variety,
cross-class separability of the canonical appearances (so C1 is unaffected)."""

from __future__ import annotations

import numpy as np

from kino_vla.eval.appearance_library import AppearanceLibrary, load_appearance_library
from kino_vla.map.rgbd import material_color


def _dist(a, b) -> float:
    return float(np.linalg.norm(np.asarray(a) - np.asarray(b)))


def test_library_loads_and_validates() -> None:
    lib = load_appearance_library(register=False)  # raises on any inconsistency
    assert lib.appearances(), "empty library"


def test_min_appearances_per_class() -> None:
    lib = AppearanceLibrary.load(register=False)
    assert len(lib.appearances("adhesion")) >= 4
    assert len(lib.appearances("compliant_terrain")) >= 4
    assert len(lib.appearances("low_friction")) >= 2


def test_splits_disjoint_and_both_populated() -> None:
    lib = AppearanceLibrary.load(register=False)
    for c in ("adhesion", "compliant_terrain", "low_friction", "solid_ground"):
        train = {a.id for a in lib.appearances(c, "train")}
        test = {a.id for a in lib.appearances(c, "test")}
        assert train and test, f"{c}: a split is empty"
        assert train.isdisjoint(test), f"{c}: train/test overlap"


def test_canonical_appearances_are_train_and_cross_separable() -> None:
    """The C1 canonical appearances (one/class) must be in train AND far apart in colour, so the
    O4↔O2 vision certificate (CLIP AUC=1.0) is unaffected by the library."""
    lib = AppearanceLibrary.load(register=False)
    cans = {c: lib.canonical(c) for c in ("adhesion", "compliant_terrain", "low_friction")}
    for cid in cans.values():
        assert lib.split(cid) == "train", f"canonical {cid} not in train"
    adh, mud, ice = (lib.color(cans[c]) for c in ("adhesion", "compliant_terrain", "low_friction"))
    assert _dist(adh, mud) > 0.4 and _dist(adh, ice) > 0.4 and _dist(mud, ice) > 0.4


def test_within_class_colour_variety() -> None:
    """Within a class, appearances must be visually distinct — that is what defeats the exact-colour
    shortcut ('yellow ⇒ adhesion')."""
    lib = AppearanceLibrary.load(register=False)
    for c in ("adhesion", "compliant_terrain"):
        cols = [a.rgb for a in lib.appearances(c)]
        pairs = [(i, j) for i in range(len(cols)) for j in range(i + 1, len(cols))]
        dmin = min(_dist(cols[i], cols[j]) for i, j in pairs)
        assert dmin > 0.03, f"{c}: appearances too close (min pairwise dist {dmin:.3f})"


def test_render_registration_overrides_hash() -> None:
    """After registration, a held-out appearance id renders its controlled colour, not a hash."""
    lib = AppearanceLibrary.load(register=True)
    for a in lib.appearances():
        assert material_color(a.id) == a.rgb, f"{a.id}: render colour not registered"


def test_benign_decals_map_to_nominal() -> None:
    """The A3.2 reverse-probe decals are hazard-COLOURED but the physics is benign (nominal)."""
    lib = AppearanceLibrary.load(register=False)
    decals = lib.appearances("hazard_decal_benign")
    assert len(decals) >= 2
    # a benign yellow decal must look like the adhesion canonical (same colour ⇒ the deception)
    yellow = next(a for a in decals if "yellow" in a.id)
    assert _dist(yellow.rgb, lib.color(lib.canonical("adhesion"))) < 0.05
