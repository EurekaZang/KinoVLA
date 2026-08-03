#!/usr/bin/env python3
"""Scale-v7 collector plus one frozen, scene-independent held-out lighting intervention."""

from __future__ import annotations

import hashlib
import importlib.util
import types
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
V1 = ROOT / "scripts/isaac_collect_kinofail_realistic_pair_v1.py"
V4 = ROOT / "scripts/isaac_collect_kinofail_realistic_pair_v4.py"
EXPECTED_V1_SHA256 = "3f1b86cfda574dc4185375a6653e7b8ba71be37710d28041033e1643de0f820a"
EXPECTED_V4_SHA256 = "7f8956a162ffd2cda6a5207efefc1cd87805df9d48881072f555c13dc17985b0"


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _load_implementation():
    if _sha256(V1) != EXPECTED_V1_SHA256 or _sha256(V4) != EXPECTED_V4_SHA256:
        raise RuntimeError("A5 lighting collector dependency hash mismatch")
    source = V1.read_text(encoding="utf-8")
    patches = (
        ("    capture_stride = 20\n", "    capture_stride = 5\n"),
        (
            "        scene_prim = backend.load_realistic_scene(str(episode))\n",
            "        scene_prim = backend.load_realistic_scene(str(episode))\n"
            "        _apply_scheduled_lighting(representative)\n",
        ),
        (
            '            "capture_stride_control_steps": capture_stride,\n',
            '            "capture_stride_control_steps": capture_stride,\n'
            '            "lighting_readback": dict(_LIGHTING_READBACK),\n',
        ),
    )
    for old, new in patches:
        if source.count(old) != 1:
            raise RuntimeError(f"A5 lighting patch point is not unique: {old!r}")
        source = source.replace(old, new)
    module = types.ModuleType("kinofail_scale_v7_a5_lighting")
    module.__file__ = str(V1)
    module.__package__ = "scripts"
    exec(compile(source, str(V1), "exec"), module.__dict__)
    return module


def _load_v4():
    spec = importlib.util.spec_from_file_location("kinofail_a5_lighting_v4_frozen", V4)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"cannot load {V4}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _install_lighting_function(implementation) -> None:
    def apply_scheduled_lighting(record):
        import omni.usd

        profile = str(record.get("lighting_profile", ""))
        scale = float(record.get("lighting_intensity_scale", 1.0))
        if profile != "heldout_dim_055_v1" or abs(scale - 0.55) > 1.0e-12:
            raise RuntimeError(f"unauthorized lighting intervention: {profile} {scale}")
        stage = omni.usd.get_context().get_stage()
        changed = []
        for prim in stage.Traverse():
            attribute = prim.GetAttribute("inputs:intensity")
            if not attribute.IsValid() or not attribute.HasAuthoredValueOpinion():
                continue
            value = attribute.Get()
            if not isinstance(value, (int, float)) or float(value) <= 0.0:
                continue
            before = float(value)
            after = before * scale
            attribute.Set(after)
            changed.append({"prim_path": str(prim.GetPath()), "before": before, "after": after})
        if not changed:
            raise RuntimeError("lighting intervention found no authored positive light intensity")
        implementation._LIGHTING_READBACK = {
            "profile": profile,
            "visual_intervention_only": True,
            "global_intensity_scale": scale,
            "changed_light_count": len(changed),
            "all_readbacks_match": all(
                abs(row["after"] - scale * row["before"]) <= 1.0e-6 for row in changed
            ),
            "lights": changed,
        }
        if not implementation._LIGHTING_READBACK["all_readbacks_match"]:
            raise RuntimeError("lighting intensity readback mismatch")

    implementation._LIGHTING_READBACK = {}
    implementation._apply_scheduled_lighting = apply_scheduled_lighting


def main() -> None:
    v4 = _load_v4()
    v4._install_runtime_validation_adapter()
    implementation = _load_implementation()
    v4._install_reachable_exposure_contract(implementation)
    _install_lighting_function(implementation)
    implementation.__file__ = __file__
    implementation.main()


if __name__ == "__main__":
    main()
