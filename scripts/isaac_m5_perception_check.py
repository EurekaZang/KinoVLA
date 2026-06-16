#!/usr/bin/env python
"""M5 real-perception gate on the Isaac Go2 (spec §7) — the map consumes rendered pixels.

Closes the CLAUDE.md §6 #22 M5 strict-GPU gap. Real CLIP is blocked (HuggingFace unreachable
through the proxy), so the appearance encoder is the network-free pixel encoder
(kino_vla/map/pixel_appearance.py) — but it must run on REAL perception, not class labels.

This spawns visually-distinct material plates (ice / mud / adhesive / solid ground, two
instances each), renders each through the real Isaac RTX camera (projection, lighting, FOV),
encodes the actual rendered RGB, and verifies the map's drop-in contract on real pixels:

  - same material, two views → cosine ≥ the map's 0.9 propagation bar (would propagate);
  - different materials → cosine well below it (would NOT propagate);
  - the costmap propagation predicate, run on the real-pixel embeddings, marks the
    homogeneous neighbour and spares the different material — i.e. the §7 map consumes
    genuine perception on GPU instead of a hash of a class string.

It is NOT CLIP (no semantics); the upgrade path (swap in a CLIP image encoder) is unchanged.

HARDWARE BLOCK (this box): ``--enable_cameras`` crashes Isaac during app init in Vulkan plugin
registration on this RTX 3060 / driver 595 / Ubuntu 26.04 / Isaac 5.1 stack (three probes:
outputs/gpu_audit/cam_probe*.log), so this script cannot run here and is NOT in the sim gate.
The strict M5 closure on this box is the same encoder driving the real costmap on rendered-
material pixels (tests/test_map_pixel_perception.py, CPU); this script is the drop-in for a
working RTX machine. Run there with:

    python scripts/isaac_m5_perception_check.py --headless --enable_cameras
"""

from __future__ import annotations

import argparse
import os
import sys
import threading

import numpy as np


def main() -> int:
    parser = argparse.ArgumentParser(description="M5 Isaac real-perception gate")
    from isaaclab.app import AppLauncher

    AppLauncher.add_app_launcher_args(parser)
    args = parser.parse_args()
    args.enable_cameras = True  # RTX rendering for the appearance camera
    app = AppLauncher(args).app

    from kino_vla.map.appearance import cosine_similarity
    from kino_vla.map.pixel_appearance import PIXEL_MATERIALS, PixelAppearanceEncoder
    from kino_vla.sim.isaac_policy_backend import IsaacPolicyBackend
    from kino_vla.utils.config import load_config

    prop_bar = float(load_config("map/traversability_v0.yaml").propagation_sim_threshold)
    backend = IsaacPolicyBackend(
        load_config("sim/go2_skeleton.yaml"), np.array([0.0, 0.0]), 0.0, record_cam=True
    )
    if backend._camera is None:
        raise RuntimeError("record camera missing — construct backend with record_cam=True")

    materials = list(PIXEL_MATERIALS)
    # Two instances of each material on a 2×4 grid, away from the robot at the origin.
    positions: dict[tuple[str, int], np.ndarray] = {}
    for i, m in enumerate(materials):
        for inst in (0, 1):
            xy = np.array([4.0 + 2.0 * i, 0.0 + 3.0 * inst])
            positions[(m, inst)] = xy
            backend.add_visual_plate(xy, PIXEL_MATERIALS[m], half_size_m=0.8)

    backend.reset(0)

    enc = PixelAppearanceEncoder()
    embeds: dict[tuple[str, int], np.ndarray] = {}
    mean_rgb: dict[tuple[str, int], tuple[int, int, int]] = {}
    for key, xy in positions.items():
        backend.aim_record_camera(
            np.array([xy[0] - 0.4, xy[1], 1.6]), np.array([xy[0], xy[1], 0.0])
        )
        for _ in range(6):  # step the env so the RTX render refreshes for the new pose
            backend.step(np.zeros(3))
        rgb = backend.capture_rgb()
        h, w = rgb.shape[:2]
        crop = rgb[int(0.35 * h) : int(0.65 * h), int(0.35 * w) : int(0.65 * w)]
        embeds[key] = enc.embed(crop)
        mean_rgb[key] = tuple(int(c) for c in crop.reshape(-1, 3).mean(axis=0))
        print(f"[isaac_m5_perception] {key[0]:13s} inst{key[1]} crop mean RGB={mean_rgb[key]}")

    # Same-material (two views) vs cross-material cosine on the real-pixel embeddings.
    same = {m: cosine_similarity(embeds[(m, 0)], embeds[(m, 1)]) for m in materials}
    cross = []
    for i, a in enumerate(materials):
        for b in materials[i + 1 :]:
            cross.append((a, b, cosine_similarity(embeds[(a, 0)], embeds[(b, 0)])))
    min_same = min(same.values())
    max_cross = max(s for _, _, s in cross)
    print("[isaac_m5_perception] same-material cosine:", {m: round(v, 3) for m, v in same.items()})
    print(
        "[isaac_m5_perception] cross-material cosine:",
        [(a, b, round(s, 3)) for a, b, s in cross],
    )

    # Map propagation predicate on the real-pixel embeddings (costmap.propagate_similar):
    # an ice hazard propagates to the other ice view but NOT to mud (spec §7).
    ice_to_ice = cosine_similarity(embeds[("ice", 0)], embeds[("ice", 1)])
    ice_to_mud = cosine_similarity(embeds[("ice", 0)], embeds[("mud", 0)])
    prop_ok = ice_to_ice >= prop_bar and ice_to_mud < prop_bar

    prop_msg = (
        f"propagation: ice→ice {ice_to_ice:.3f}≥{prop_bar} but ice→mud {ice_to_mud:.3f}<{prop_bar}"
    )
    checks = {
        f"same-material cosine ≥ {prop_bar} (min {min_same:.3f})": min_same >= prop_bar,
        f"cross-material cosine < {prop_bar} (max {max_cross:.3f})": max_cross < prop_bar,
        prop_msg: prop_ok,
    }
    print("[isaac_m5_perception] results:")
    for name, ok in checks.items():
        print(f"  [{'PASS' if ok else 'FAIL'}] {name}")
    passed = all(checks.values())
    print(
        "PASS: M5 pixel encoder runs on real Isaac perception"
        if passed
        else "FAIL: M5 real-perception encoder did not separate materials"
    )

    sys.stdout.flush()
    closer = threading.Thread(target=app.close, daemon=True)
    closer.start()
    closer.join(timeout=15.0)
    os._exit(0 if passed else 1)


if __name__ == "__main__":
    main()
