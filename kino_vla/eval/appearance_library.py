"""A0.4 — Procedural appearance-library loader (Paper-A §2 A0.4).

Loads ``configs/eval/appearance_library.yaml`` — the pre-registered ≥4-appearances-per-class library
with frozen train/test splits — into typed :class:`Appearance` records, and registers each
appearance's controlled colour into the §7 renderer (``kino_vla.map.rgbd.material_color``) so a
held-out appearance (e.g. ``gray_tape`` for adhesion) renders distinctly instead of a hash fallback.

The library is the shortcut-killer: an agent trained on the ``train`` appearances and tested on the
disjoint ``test`` appearances demonstrably learned the CONCEPT (physics category), not the colour.
``semantic_class`` ties every appearance to the taxonomy category and is NEVER a render input.
"""

from __future__ import annotations

from dataclasses import dataclass

from kino_vla.map.rgbd import register_appearance_colors
from kino_vla.utils.config import load_config

# Per-class minimum appearance counts (spec §2 A0.4: adhesive/mud ≥4, ice 2–3, + benign decals).
_MIN_PER_CLASS: dict[str, int] = {
    "adhesion": 4,
    "compliant_terrain": 4,
    "low_friction": 2,
    "solid_ground": 3,
    "hazard_decal_benign": 2,
}


@dataclass(frozen=True)
class Appearance:
    """One procedural appearance: a render colour + texture base, tagged to a class + split."""

    id: str
    rgb: tuple[float, float, float]
    base_material: str
    split: str  # "train" | "test"
    semantic_class: str


class AppearanceLibraryError(ValueError):
    """Raised when the appearance library is malformed (bad split, count, or colour)."""


class AppearanceLibrary:
    """The parsed A0.4 appearance library with class/split accessors + render registration."""

    def __init__(
        self, appearances: list[Appearance], canonical: dict[str, str], meta: dict
    ) -> None:
        self._appearances = {a.id: a for a in appearances}
        self._canonical = dict(canonical)
        self.meta = dict(meta)
        if len(self._appearances) != len(appearances):
            raise AppearanceLibraryError("duplicate appearance id in the library")

    @classmethod
    def load(
        cls, path: str = "eval/appearance_library.yaml", *, register: bool = True
    ) -> AppearanceLibrary:
        d = load_config(path).to_dict()
        appearances: list[Appearance] = []
        for sem_class, entries in d["classes"].items():
            for e in entries:
                appearances.append(
                    Appearance(
                        id=str(e["id"]),
                        rgb=tuple(float(c) for c in e["rgb"]),
                        base_material=str(e["base_material"]),
                        split=str(e["split"]),
                        semantic_class=str(sem_class),
                    )
                )
        lib = cls(appearances, d.get("meta", {}).get("canonical", {}), d.get("meta", {}))
        if register:
            lib.register_render_colors()
        return lib

    # ------------------------------------------------------------------- accessors
    def __getitem__(self, appearance_id: str) -> Appearance:
        return self._appearances[appearance_id]

    def color(self, appearance_id: str) -> tuple[float, float, float]:
        return self._appearances[appearance_id].rgb

    def semantic_class(self, appearance_id: str) -> str:
        return self._appearances[appearance_id].semantic_class

    def split(self, appearance_id: str) -> str:
        return self._appearances[appearance_id].split

    def classes(self) -> list[str]:
        return sorted({a.semantic_class for a in self._appearances.values()})

    def appearances(
        self, semantic_class: str | None = None, split: str | None = None
    ) -> list[Appearance]:
        out = list(self._appearances.values())
        if semantic_class is not None:
            out = [a for a in out if a.semantic_class == semantic_class]
        if split is not None:
            out = [a for a in out if a.split == split]
        return sorted(out, key=lambda a: a.id)

    def canonical(self, semantic_class: str) -> str:
        return self._canonical[semantic_class]

    def register_render_colors(self) -> None:
        """Register every appearance's colour into the §7 renderer (so it paints distinctly)."""
        register_appearance_colors({a.id: a.rgb for a in self._appearances.values()})

    # ------------------------------------------------------------------- validation
    def validate(self) -> list[str]:
        """Return a list of issues ([] ⇒ valid): per-class counts, disjoint splits, colours in
        range, canonicals present in train."""
        issues: list[str] = []
        for a in self._appearances.values():
            if a.split not in ("train", "test"):
                issues.append(f"{a.id}: bad split {a.split!r}")
            if not all(0.0 <= c <= 1.0 for c in a.rgb):
                issues.append(f"{a.id}: rgb {a.rgb} outside [0,1]")
        for sem_class, need in _MIN_PER_CLASS.items():
            got = self.appearances(sem_class)
            if len(got) < need:
                issues.append(f"{sem_class}: {len(got)} appearances < required {need}")
            # a class used for generalisation needs BOTH a train and a held-out test appearance
            if sem_class != "hazard_decal_benign":
                splits = {a.split for a in got}
                if got and ("train" not in splits or "test" not in splits):
                    issues.append(f"{sem_class}: split not both-populated ({sorted(splits)})")
        for sem_class, cid in self._canonical.items():
            if cid not in self._appearances:
                issues.append(f"canonical {cid!r} for {sem_class} not in library")
            elif self._appearances[cid].split != "train":
                issues.append(f"canonical {cid!r} must be in the train split")
        return issues


def load_appearance_library(
    path: str = "eval/appearance_library.yaml", *, register: bool = True
) -> AppearanceLibrary:
    """Load + validate the A0.4 appearance library; raise on any inconsistency."""
    lib = AppearanceLibrary.load(path, register=register)
    issues = lib.validate()
    if issues:
        raise AppearanceLibraryError("appearance library invalid:\n  " + "\n  ".join(issues))
    return lib
