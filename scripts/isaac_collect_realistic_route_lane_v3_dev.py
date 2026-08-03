#!/usr/bin/env python3
"""Development bridge from the frozen forest collector to EmbodiedGen v4 routes.

The G01--G07 collector is content-frozen and must not be edited.  This bridge
verifies that exact source, applies three narrow cross-domain substitutions in
memory, and records both source hashes in the resulting manifest.  It is only
for adapter development; the formal corpus collector must contain native v3
code and pass a new freeze audit before any episode is evaluation-eligible.
"""

from __future__ import annotations

import hashlib
import json
import os
import sys
import traceback
from pathlib import Path
from types import ModuleType


ROOT = Path(__file__).resolve().parents[1]
BASE_COLLECTOR = ROOT / "scripts/isaac_collect_visual_shell_o2_lane.py"
EXPECTED_BASE_SHA256 = "418e7e13e23bbeb074916503bf40482a22d760c1e892d215558ff0ea2e3e60f0"


def _sha256_bytes(value: bytes) -> str:
    return hashlib.sha256(value).hexdigest()


def _replace_once(source: str, old: str, new: str) -> str:
    count = source.count(old)
    if count != 1:
        raise RuntimeError(f"expected exactly one adapter replacement, found {count}: {old[:80]!r}")
    return source.replace(old, new, 1)


def _adapted_source() -> tuple[str, str]:
    raw = BASE_COLLECTOR.read_bytes()
    actual = _sha256_bytes(raw)
    if actual != EXPECTED_BASE_SHA256:
        raise RuntimeError(
            "frozen base collector hash changed; refusing dynamic adapter: "
            f"expected {EXPECTED_BASE_SHA256}, got {actual}"
        )
    source = raw.decode("utf-8")
    source = source.partition('\nif __name__ == "__main__":')[0]
    source = _replace_once(
        source,
        "from kino_vla.sim.realistic_route_protocol import StraightRouteFrame",
        (
            "from kino_vla.sim.realistic_route_protocol_v2 import "
            "RouteFrameV2 as StraightRouteFrame, scene_route_binding"
        ),
    )
    source = _replace_once(
        source,
        '''if compiled.get("passed") is not True or not str(compiled.get("schema_version", "")).startswith(
            "kinofail.forest-hybrid-compiled-scene."
        ):
            raise RuntimeError("input is not the passed forest visual-shell compilation")''',
        '''if compiled.get("passed") is not True:
            raise RuntimeError("input compiled scene did not pass")
        scene_binding = scene_route_binding(compiled)''',
    )
    source = _replace_once(
        source,
        "route_frame = StraightRouteFrame.from_compiled_audit(compiled)",
        "route_frame = scene_binding.frame",
    )
    source = _replace_once(
        source,
        '''custom_floor_path = f"{scene_prim}/Collision/Floor"
        route_surface_path = f"{scene_prim}/Appearance/RouteSurface"''',
        '''custom_floor_path = scene_binding.floor_prim_path(scene_prim)
        route_surface_path = scene_binding.route_surface_prim_path(scene_prim)''',
    )
    source = _replace_once(
        source,
        '''"schema_version": (
                f"kinofail.visual-shell-{args.operator}-fresh-app-lane.v1-development"
            ),''',
        '''"schema_version": (
                f"kinofail.realistic-route-{args.operator}-fresh-app-lane.v3-development"
            ),
            "scene_route_binding": {
                "source_kind": scene_binding.source_kind,
                "floor_prim_suffix": scene_binding.floor_prim_suffix,
                "route_surface_prim_suffix": scene_binding.route_surface_prim_suffix,
                "route_surface_collision_authored": (
                    scene_binding.route_surface_collision_authored
                ),
            },''',
    )
    return source, _sha256_bytes(source.encode("utf-8"))


def _output_from_argv() -> Path | None:
    try:
        index = sys.argv.index("--out")
        return Path(sys.argv[index + 1]).resolve()
    except (ValueError, IndexError):
        return None


def main() -> int:
    if str(ROOT) not in sys.path:
        sys.path.insert(0, str(ROOT))
    os.environ.setdefault("OMNI_KIT_ACCEPT_EULA", "YES")
    os.environ.setdefault("HF_HUB_OFFLINE", "1")
    source, transformed_sha256 = _adapted_source()
    module = ModuleType("kinofail_realistic_route_lane_v3_runtime")
    module.__dict__.update(
        {
            "__file__": str(Path(__file__).resolve()),
            "__name__": module.__name__,
            "__package__": None,
        }
    )
    exec(compile(source, str(BASE_COLLECTOR), "exec"), module.__dict__)
    code = int(module.__dict__["main"]())
    output = _output_from_argv()
    manifest_path = output / "lane_manifest.json" if output is not None else None
    if manifest_path is not None and manifest_path.is_file():
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        manifest["collector_adapter_provenance"] = {
            "formal_corpus_eligible": False,
            "bridge": str(Path(__file__).resolve()),
            "bridge_sha256": _sha256_bytes(Path(__file__).read_bytes()),
            "frozen_base_collector": str(BASE_COLLECTOR),
            "frozen_base_collector_sha256": EXPECTED_BASE_SHA256,
            "adapted_runtime_source_sha256": transformed_sha256,
        }
        manifest["counts_as_a0_a7_evidence"] = False
        manifest_path.write_text(
            json.dumps(manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8"
        )
    return code


if __name__ == "__main__":
    try:
        exit_code = main()
    except Exception:
        traceback.print_exc()
        sys.stdout.flush()
        sys.stderr.flush()
        os._exit(1)
    sys.stdout.flush()
    sys.stderr.flush()
    os._exit(exit_code)
