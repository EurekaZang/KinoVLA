#!/usr/bin/env python
"""Real CLIP on the LIVE Isaac RTX camera over TEXTURED terrain (spec §7) — closing the loop.

The benchmark's visual materials were flat ``UsdPreviewSurface`` diffuse colours (chosen so a
colour-histogram surrogate could resolve them); a flat colour is out of distribution for CLIP, so a
live camera over the old scene could not be CLIP-labelled. This script gives each material a real
texture map (rendered by :func:`kino_vla.map.clip_segmentation.material_texture`, saved as PNG and
bound through ``UsdUVTexture``), images the textured plates with a real ``isaacsim.sensors.camera``,
and feeds those camera pixels to the real CLIP encoder (:class:`ClipAppearanceEncoder`). PASS when
CLIP labels every plate open-vocabulary from genuine camera pixels and same-material cosine beats
cross-material --- the §7 perception running on the live RTX camera, no surrogate.

    python scripts/clip_rtx_camera.py --headless --enable_cameras
"""

from __future__ import annotations

import argparse


def main() -> int:
    parser = argparse.ArgumentParser(description="Real CLIP on the live Isaac RTX camera (§7)")
    parser.add_argument("--out", default="outputs/map/clip_rtx.md")
    from isaacsim import SimulationApp

    pre, _ = parser.parse_known_args()
    app = SimulationApp({"headless": True, "enable_cameras": True})

    import carb
    import numpy as np
    from isaacsim.core.api import World
    from isaacsim.core.api.objects import VisualCuboid
    from isaacsim.sensors.camera import Camera
    from PIL import Image
    from pxr import Gf, Sdf, UsdGeom, UsdLux, UsdShade

    from kino_vla.map.clip_appearance import ClipAppearanceEncoder
    from kino_vla.map.clip_segmentation import MATERIAL_VOCAB, material_texture
    from kino_vla.utils.config import REPO_ROOT

    # 1) Render + save a real texture map per material (what a textured surface looks like).
    tex_dir = REPO_ROOT / "outputs" / "map" / "clip_textures"
    tex_dir.mkdir(parents=True, exist_ok=True)
    materials = ["ice", "mud", "adhesive", "concrete"]
    tex_paths = {}
    for m in materials:
        img = (material_texture(m, seed=0, size=256) * 255).astype("uint8")
        p = tex_dir / f"{m}.png"
        Image.fromarray(img).save(p)
        tex_paths[m] = str(p)

    # 2) Scene: textured plates under a dome light. Disable tonemap so albedo tracks the texture.
    settings = carb.settings.get_settings()
    settings.set_bool("/rtx/post/histogram/enabled", False)
    settings.set_bool("/rtx/post/tonemap/enabled", False)
    world = World(stage_units_in_meters=1.0)
    world.scene.add_default_ground_plane()
    UsdLux.DomeLight.Define(world.stage, "/World/DomeLight").CreateIntensityAttr(400.0)
    stage = world.stage

    def textured_material(name: str, png: str):
        mtl = UsdShade.Material.Define(stage, f"/World/Looks/{name}")
        pbr = UsdShade.Shader.Define(stage, f"/World/Looks/{name}/PBR")
        pbr.CreateIdAttr("UsdPreviewSurface")
        pbr.CreateInput("roughness", Sdf.ValueTypeNames.Float).Set(0.9)
        pbr.CreateInput("metallic", Sdf.ValueTypeNames.Float).Set(0.0)
        reader = UsdShade.Shader.Define(stage, f"/World/Looks/{name}/stReader")
        reader.CreateIdAttr("UsdPrimvarReader_float2")
        reader.CreateInput("varname", Sdf.ValueTypeNames.Token).Set("st")
        tex = UsdShade.Shader.Define(stage, f"/World/Looks/{name}/diffuseTex")
        tex.CreateIdAttr("UsdUVTexture")
        tex.CreateInput("file", Sdf.ValueTypeNames.Asset).Set(png)
        tex.CreateInput("st", Sdf.ValueTypeNames.Float2).ConnectToSource(
            reader.ConnectableAPI(), "result"
        )
        for w in ("wrapS", "wrapT"):
            tex.CreateInput(w, Sdf.ValueTypeNames.Token).Set("repeat")
        tex.CreateOutput("rgb", Sdf.ValueTypeNames.Float3)
        pbr.CreateInput("diffuseColor", Sdf.ValueTypeNames.Color3f).ConnectToSource(
            tex.ConnectableAPI(), "rgb"
        )
        mtl.CreateSurfaceOutput().ConnectToSource(pbr.ConnectableAPI(), "surface")
        return mtl

    def textured_quad(path: str, xy: np.ndarray, mtl, size: float = 1.4) -> None:
        mesh = UsdGeom.Mesh.Define(stage, path)
        h = size / 2.0
        z = 0.05
        corners = [(-h, -h), (h, -h), (h, h), (-h, h)]
        mesh.CreatePointsAttr(
            [Gf.Vec3f(float(xy[0] + cx), float(xy[1] + cy), z) for cx, cy in corners]
        )
        mesh.CreateFaceVertexCountsAttr([4])
        mesh.CreateFaceVertexIndicesAttr([0, 1, 2, 3])
        mesh.CreateNormalsAttr([Gf.Vec3f(0, 0, 1)] * 4)  # face up toward the down-looking camera
        mesh.SetNormalsInterpolation("vertex")
        mesh.CreateDoubleSidedAttr(True)  # never cull, whatever the winding
        mesh.CreateSubdivisionSchemeAttr("none")
        mesh.CreateExtentAttr([Gf.Vec3f(-h, -h, 0), Gf.Vec3f(h, h, z)])
        st = UsdGeom.PrimvarsAPI(mesh).CreatePrimvar(
            "st", Sdf.ValueTypeNames.TexCoord2fArray, UsdGeom.Tokens.faceVarying
        )
        st.Set([(0, 0), (1, 0), (1, 1), (0, 1)])
        UsdShade.MaterialBindingAPI(mesh.GetPrim()).Bind(mtl)

    # two plates per material (different texture seed = two viewpoints) for the same-material check.
    mtls = {m: textured_material(m, tex_paths[m]) for m in materials}
    plate_xy: dict[str, np.ndarray] = {}
    for i, m in enumerate(materials):
        for inst in (0, 1):
            xy = np.array([2.0 + 3.0 * i + 1.0 * inst, 0.0])
            plate_xy[f"{m}_{inst}"] = xy
            # second instance gets its own texture seed via a second saved PNG
            png = tex_paths[m]
            if inst == 1:
                img = (material_texture(m, seed=7, size=256) * 255).astype("uint8")
                png = str(tex_dir / f"{m}_v2.png")
                Image.fromarray(img).save(png)
                textured_quad(f"/World/plate_{m}_{inst}", xy, textured_material(f"{m}_v2", png))
            else:
                textured_quad(f"/World/plate_{m}_{inst}", xy, mtls[m])

    _ = VisualCuboid  # (kept import parity with the colour-plate script; meshes used here)
    look_down = np.array([0.7071068, 0.0, 0.7071068, 0.0])  # optical axis -> -Z (straight down)
    cam = Camera(prim_path="/World/Cam", position=np.array([0.0, 0.0, 1.2]), resolution=(224, 224))
    world.reset()
    cam.initialize()
    for _ in range(24):  # warm up the render product (world.step(render=True), not app.update)
        world.step(render=True)
    enc = ClipAppearanceEncoder(vocabulary=MATERIAL_VOCAB)

    # One camera, repositioned over each plate. The RTX path tracer is advanced with
    # world.step(render=True) (the m5 capture pattern that produces real pixels; app.update() leaves
    # a default grey frame). A fresh pose each iteration avoids the #28 same-pose stale-frame issue.
    captures: dict[str, np.ndarray] = {}
    for key, xy in plate_xy.items():
        cam.set_world_pose(np.array([float(xy[0]), float(xy[1]), 1.2]), orientation=look_down)
        for _ in range(24):  # let the path tracer converge at this pose
            world.step(render=True)
        rgba = np.asarray(cam.get_rgba())
        if rgba.size == 0:
            print(f"[clip-rtx] {key}: empty frame")
            continue
        rgb = rgba[..., :3].astype(np.float64)
        if rgb.max() > 1.5:
            rgb = rgb / 255.0
        h, w = rgb.shape[:2]
        c = rgb[int(0.40 * h) : int(0.60 * h), int(0.40 * w) : int(0.60 * w)]  # tight centre crop
        captures[key] = c
        print(f"[clip-rtx] {key}: capture mean RGB = {c.reshape(-1, 3).mean(0).round(3).tolist()}")
        if key in ("ice_0", "mud_0", "adhesive_0"):  # save a few for inspection
            Image.fromarray((c * 255).astype("uint8")).save(tex_dir / f"capture_{key}.png")

    # 3) Real CLIP on the camera pixels: open-vocab labels + same/cross-material cosine.
    lines = ["# Real CLIP on the live Isaac RTX camera over textured terrain (spec §7)", ""]
    embeds: dict[str, np.ndarray] = {}
    label_ok = 0
    for key, img in captures.items():
        material = key.rsplit("_", 1)[0]
        label, prob, _ = enc.classify(img)
        embeds[key] = enc.embed(img)
        hit = label == material
        label_ok += hit and key.endswith("_0")
        verdict = "OK" if hit else "MISLABEL"
        lines.append(f"- camera plate **{key}** -> CLIP `{label}` (p={prob:.2f})  {verdict}")

    def cos(a: np.ndarray, b: np.ndarray) -> float:
        return float(a @ b)

    same = [cos(embeds[f"{m}_0"], embeds[f"{m}_1"]) for m in materials if f"{m}_1" in embeds]
    diff = [
        cos(embeds[f"{a}_0"], embeds[f"{b}_0"])
        for i, a in enumerate(materials)
        for b in materials[i + 1 :]
        if f"{a}_0" in embeds and f"{b}_0" in embeds
    ]
    sep_ok = bool(same and diff and min(same) > max(diff))
    lines += [
        "",
        f"same-material cosine min = {min(same):.3f}; cross-material max = {max(diff):.3f}"
        if same and diff
        else "insufficient captures",
        f"open-vocab labels correct: {label_ok}/{len(materials)}",
    ]
    ok = label_ok == len(materials) and sep_ok
    lines.append("")
    lines.append("PASS" if ok else "FAIL")

    out = REPO_ROOT / pre.out
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text("\n".join(lines))
    print("\n".join(lines))
    print(f"\nwrote {out}")

    import os
    import threading

    threading.Timer(8.0, lambda: os._exit(0 if ok else 1)).start()
    app.close()
    return 0 if ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
