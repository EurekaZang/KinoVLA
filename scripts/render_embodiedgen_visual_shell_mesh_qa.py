#!/usr/bin/env python3
"""Render auditable cardinal viewpoints from a development visual-shell mesh."""

from __future__ import annotations

import argparse
import json
import sys
from datetime import UTC, datetime
from pathlib import Path

import numpy as np
import torch
import trimesh
from PIL import Image, ImageDraw

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from kino_vla.eval.visual_shell_novel_view import (  # noqa: E402
    evaluate_rendered_view,
    mean_pairwise_rgb_l1,
)
from kino_vla.eval.visual_shell_preflight import sha256_file  # noqa: E402


def _load_trimesh(path: Path) -> trimesh.Trimesh:
    mesh = trimesh.load(path, process=False)
    if not isinstance(mesh, trimesh.Trimesh):
        raise TypeError("novel-view renderer requires one Trimesh")
    return mesh


def _pano_camera_metric_eye(raw_mesh: trimesh.Trimesh, target_height_m: float) -> list[float]:
    vertices = np.asarray(raw_mesh.vertices, dtype=np.float64)
    low, high = np.percentile(vertices[:, 1], [1.0, 99.0])
    scale = float(target_height_m) / float(high - low)
    return [0.0, float(-low * scale), 0.0]


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--scene-id", required=True)
    parser.add_argument(
        "--config",
        type=Path,
        default=ROOT
        / "configs/data/kinofail_embodiedgen_visual_shell_novel_view_qa_v1.json",
    )
    parser.add_argument(
        "--output-root",
        type=Path,
        default=ROOT
        / "outputs/kinofail_realistic/scene_sources/embodiedgen_visual_shell_v1",
    )
    args = parser.parse_args()

    config_path = args.config.resolve()
    config = json.loads(config_path.read_text(encoding="utf-8"))
    mesh_contract = ROOT / config["source_mesh_contract"]["path"]
    if sha256_file(mesh_contract) != config["source_mesh_contract"]["sha256"]:
        raise SystemExit("mesh contract differs from the frozen novel-view contract")

    scene_root = args.output_root.resolve() / args.scene_id
    mesh_root = scene_root / "mesh_development"
    stage_audit_path = mesh_root / "mesh_stage_audit.json"
    stage_audit = json.loads(stage_audit_path.read_text(encoding="utf-8"))
    if not stage_audit.get("passed"):
        raise SystemExit("mesh stage did not pass its development preflight")
    raw_path = mesh_root / "mesh_model_raw.ply"
    metric_path = mesh_root / "mesh_model_metric_visual.ply"
    if stage_audit["raw"]["mesh"]["sha256"] != sha256_file(raw_path):
        raise SystemExit("raw mesh differs from its stage audit")
    if stage_audit["metric_visual"]["mesh"]["sha256"] != sha256_file(metric_path):
        raise SystemExit("metric mesh differs from its stage audit")

    output = mesh_root / config.get("output_subdir", "novel_view_qa")
    if output.exists():
        raise SystemExit("refusing to overwrite an existing novel-view QA attempt")
    output.mkdir(parents=True)
    request_path = output / "render_request.json"
    request = {
        "schema_version": "kinofail.embodiedgen-visual-shell-render-request.v1",
        "created_utc": datetime.now(UTC).isoformat(),
        "development_only": True,
        "scene_id": args.scene_id,
        "config": {"path": str(config_path), "sha256": sha256_file(config_path)},
        "mesh_stage_audit_sha256": sha256_file(stage_audit_path),
        "raw_mesh_sha256": sha256_file(raw_path),
        "metric_mesh_sha256": sha256_file(metric_path),
        "renderer": {"path": str(Path(__file__).resolve()), "sha256": sha256_file(Path(__file__).resolve())},
        "render": config["render"],
        "admission_policy": config["admission_policy"],
        "runtime": {
            "torch": torch.__version__,
            "torch_cuda": torch.version.cuda,
            "gpu": torch.cuda.get_device_name(0) if torch.cuda.is_available() else None,
        },
    }
    request_path.write_text(
        json.dumps(request, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )

    try:
        from pytorch3d.renderer import (
            AmbientLights,
            BlendParams,
            FoVPerspectiveCameras,
            HardFlatShader,
            MeshRasterizer,
            RasterizationSettings,
            look_at_view_transform,
        )
        from pytorch3d.renderer.mesh.textures import TexturesVertex
        from pytorch3d.structures import Meshes

        raw = _load_trimesh(raw_path)
        metric = _load_trimesh(metric_path)
        target_height_m = float(stage_audit["metric_visual"]["metrics"]["target_height_m"])
        eye = _pano_camera_metric_eye(raw, target_height_m)
        vertices = torch.as_tensor(
            np.asarray(metric.vertices, dtype=np.float32), device="cuda"
        )
        faces = torch.as_tensor(np.asarray(metric.faces, dtype=np.int64), device="cuda")
        colors_np = np.asarray(metric.visual.vertex_colors, dtype=np.float32)[:, :3] / 255.0
        colors = torch.as_tensor(colors_np, device="cuda")
        meshes = Meshes(
            verts=[vertices],
            faces=[faces],
            textures=TexturesVertex(verts_features=[colors]),
        )
        render_cfg = config["render"]
        translations = [
            np.asarray(view.get("eye_offset_xz_m", [0.0, 0.0]), dtype=np.float64)
            for view in render_cfg["views"]
        ]
        if any(offset.shape != (2,) for offset in translations):
            raise ValueError("every eye_offset_xz_m must contain exactly two values")
        if any(
            float(np.linalg.norm(offset)) > float(render_cfg.get("maximum_translation_m", 1e9))
            for offset in translations
        ):
            raise ValueError("translated-view offset exceeds the frozen maximum")
        if render_cfg.get("requires_nonzero_translation") and not any(
            float(np.linalg.norm(offset)) > 0.0 for offset in translations
        ):
            raise ValueError("translated-view contract contains no nonzero camera translation")
        raster_settings = RasterizationSettings(
            image_size=(int(render_cfg["image_height"]), int(render_cfg["image_width"])),
            blur_radius=0.0,
            faces_per_pixel=1,
            cull_backfaces=False,
        )
        blend = BlendParams(background_color=tuple(value / 255.0 for value in render_cfg["background_rgb"]))
        lights = AmbientLights(device="cuda", ambient_color=((1.0, 1.0, 1.0),))
        rendered_images: list[np.ndarray] = []
        view_audits = []
        for view, offset in zip(render_cfg["views"], translations):
            direction = view["direction_xz"]
            view_eye = [eye[0] + float(offset[0]), eye[1], eye[2] + float(offset[1])]
            target = [
                view_eye[0]
                + float(direction[0]) * float(render_cfg["target_distance_m"]),
                view_eye[1],
                view_eye[2]
                + float(direction[1]) * float(render_cfg["target_distance_m"]),
            ]
            rotation, translation = look_at_view_transform(
                eye=[view_eye], at=[target], up=[[0.0, 1.0, 0.0]], device="cuda"
            )
            cameras = FoVPerspectiveCameras(
                device="cuda",
                R=rotation,
                T=translation,
                znear=float(render_cfg["znear_m"]),
                zfar=float(render_cfg["zfar_m"]),
                fov=float(render_cfg["fov_degrees"]),
            )
            rasterizer = MeshRasterizer(cameras=cameras, raster_settings=raster_settings)
            fragments = rasterizer(meshes)
            shader = HardFlatShader(
                device="cuda", cameras=cameras, lights=lights, blend_params=blend
            )
            image = shader(fragments, meshes)[0, ..., :3].clamp(0, 1)
            rgb = (image.detach().cpu().numpy() * 255.0).round().astype(np.uint8)
            mask = (fragments.pix_to_face[0, ..., 0] >= 0).detach().cpu().numpy()
            rendered_images.append(rgb)
            image_path = output / f"{view['view_id']}.png"
            Image.fromarray(rgb).save(image_path)
            view_audit = evaluate_rendered_view(
                rgb,
                mask,
                expected_width=int(render_cfg["image_width"]),
                expected_height=int(render_cfg["image_height"]),
                thresholds=config["machine_preflight"],
            )
            view_audit.update(
                {
                    "view_id": view["view_id"],
                    "image": {"path": str(image_path), "sha256": sha256_file(image_path)},
                    "eye_metric": view_eye,
                    "eye_offset_xz_m": offset.tolist(),
                    "target_metric": target,
                }
            )
            view_audits.append(view_audit)
            del fragments, shader, rasterizer, cameras, image
            torch.cuda.empty_cache()

        pairwise_l1 = mean_pairwise_rgb_l1(rendered_images)
        distinct_views = pairwise_l1 >= float(
            config["machine_preflight"]["minimum_mean_pairwise_rgb_l1"]
        )
        tile_width = int(render_cfg["image_width"])
        tile_height = int(render_cfg["image_height"])
        sheet = Image.new("RGB", (tile_width * 2, tile_height * 2), "white")
        draw = ImageDraw.Draw(sheet)
        for index, (view, rgb) in enumerate(zip(render_cfg["views"], rendered_images)):
            left = (index % 2) * tile_width
            top = (index // 2) * tile_height
            sheet.paste(Image.fromarray(rgb), (left, top))
            draw.rectangle((left, top, left + 92, top + 22), fill="white")
            draw.text((left + 5, top + 4), view["view_id"], fill="black")
        sheet_path = output / "novel_view_contact_sheet.png"
        sheet.save(sheet_path)
        audit = {
            "schema_version": "kinofail.embodiedgen-visual-shell-novel-view-stage-audit.v1",
            "scene_id": args.scene_id,
            "development_only": True,
            "views": view_audits,
            "mean_pairwise_rgb_l1": pairwise_l1,
            "distinct_views_passed": distinct_views,
            "contact_sheet": {"path": str(sheet_path), "sha256": sha256_file(sheet_path)},
            "passed": all(view["passed"] for view in view_audits) and distinct_views,
            "counts_as_scene_registry_admission": False,
            "counts_as_a0_a7_evidence": False,
            "independent_human_review_required": True,
        }
        audit_path = output / "novel_view_stage_audit.json"
        audit_path.write_text(
            json.dumps(audit, indent=2, sort_keys=True) + "\n", encoding="utf-8"
        )
        print(json.dumps({"contact_sheet": str(sheet_path), "audit": str(audit_path), "passed": audit["passed"]}))
        if not audit["passed"]:
            raise SystemExit("novel-view machine preflight failed")
    except Exception as exc:
        failure = {
            "schema_version": "kinofail.embodiedgen-visual-shell-render-failure.v1",
            "failed_utc": datetime.now(UTC).isoformat(),
            "failure_class": type(exc).__name__,
            "failure_message": str(exc),
            "request_sha256": sha256_file(request_path),
        }
        (output / "render_failure.json").write_text(
            json.dumps(failure, indent=2, sort_keys=True) + "\n", encoding="utf-8"
        )
        raise


if __name__ == "__main__":
    main()
