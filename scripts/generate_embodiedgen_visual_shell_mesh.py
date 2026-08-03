#!/usr/bin/env python3
"""Reconstruct and preflight a development-only EmbodiedGen visual shell."""

from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
import types
from datetime import UTC, datetime
from importlib.metadata import version
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from kino_vla.eval.visual_shell_mesh import (  # noqa: E402
    create_metric_visual_mesh,
    evaluate_visual_shell_mesh,
)
from kino_vla.eval.visual_shell_preflight import sha256_file  # noqa: E402


def _git(repo: Path, *args: str) -> str:
    return subprocess.check_output(["git", "-C", str(repo), *args], text=True).strip()


def _load_scene(panorama_config: dict, scene_id: str) -> dict:
    rows = [row for row in panorama_config["requests"] if row["scene_id"] == scene_id]
    if len(rows) != 1:
        raise ValueError(f"expected exactly one request for scene_id={scene_id!r}")
    return dict(rows[0])


def _install_torchvision_compatibility_shim() -> None:
    import torchvision.transforms.functional as functional

    shim = types.ModuleType("torchvision.transforms.functional_tensor")
    shim.rgb_to_grayscale = functional.rgb_to_grayscale
    sys.modules["torchvision.transforms.functional_tensor"] = shim


def _install_mesh_only_sr_shim() -> None:
    """Disable an unreachable 3DGS-only SR dependency and fail if it is ever called."""

    class DisabledImageRealESRGAN:
        def __init__(self, outscale: int, model_path: str | None = None) -> None:
            self.outscale = outscale
            self.model_path = model_path

        def __call__(self, image: object) -> object:
            raise RuntimeError(
                "mesh-only compatibility contract violated: RealESRGAN was called"
            )

    shim = types.ModuleType("embodied_gen.models.sr_model")
    shim.ImageRealESRGAN = DisabledImageRealESRGAN
    sys.modules["embodied_gen.models.sr_model"] = shim


def _archive_failed_attempt(output: Path, *, diagnosed_failure_class: str) -> dict[str, str]:
    request = output / "mesh_generation_request.json"
    failure = output / "mesh_generation_failure.json"
    raw_mesh = output / "mesh_model_raw.ply"
    metric_mesh = output / "mesh_model_metric_visual.ply"
    if raw_mesh.exists() or metric_mesh.exists():
        raise SystemExit("refusing to archive an attempt that produced a visual mesh")
    if not request.is_file() or not failure.is_file():
        raise SystemExit("both prior request and failure records are required for archival")
    recorded_failure = json.loads(failure.read_text(encoding="utf-8"))
    if recorded_failure.get("failure_class") != diagnosed_failure_class:
        raise SystemExit(
            "diagnosed failure class does not match the recorded failure: "
            f"{recorded_failure.get('failure_class')!r}"
        )
    attempts = output / "failed_attempts"
    attempts.mkdir(parents=True, exist_ok=True)
    index = len([path for path in attempts.iterdir() if path.is_dir()]) + 1
    attempt = attempts / f"attempt_{index:03d}"
    attempt.mkdir()
    archived_request = attempt / request.name
    archived_failure = attempt / failure.name
    request.replace(archived_request)
    failure.replace(archived_failure)
    return {"request": str(archived_request), "failure": str(archived_failure)}


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--scene-id", required=True)
    parser.add_argument(
        "--config",
        type=Path,
        default=ROOT
        / "configs/data/kinofail_embodiedgen_visual_shell_mesh_development_v1.json",
    )
    parser.add_argument(
        "--embodiedgen-root",
        type=Path,
        default=Path("/home/eureka/dependencies/EmbodiedGen-v2.0.0"),
    )
    parser.add_argument(
        "--output-root",
        type=Path,
        default=ROOT
        / "outputs/kinofail_realistic/scene_sources/embodiedgen_visual_shell_v1",
    )
    parser.add_argument("--skip-generation", action="store_true")
    parser.add_argument(
        "--archive-prior-failure",
        metavar="RECORDED_FAILURE_CLASS",
        help="Archive a matching failed attempt before an explicit retry.",
    )
    args = parser.parse_args()

    config_path = args.config.resolve()
    config = json.loads(config_path.read_text(encoding="utf-8"))
    source_contract_path = ROOT / config["source_panorama_contract"]["path"]
    if sha256_file(source_contract_path) != config["source_panorama_contract"]["sha256"]:
        raise SystemExit("source panorama contract differs from the frozen mesh contract")
    panorama_config = json.loads(source_contract_path.read_text(encoding="utf-8"))
    scene = _load_scene(panorama_config, args.scene_id)

    repo = args.embodiedgen_root.resolve()
    software = config["software"]
    if _git(repo, "rev-parse", "HEAD") != software["embodiedgen_commit"]:
        raise SystemExit("EmbodiedGen checkout differs from the frozen mesh contract")
    if _git(repo, "describe", "--tags", "--exact-match") != software["embodiedgen_release"]:
        raise SystemExit("EmbodiedGen checkout is not at the frozen release tag")
    pano2room = repo / "thirdparty/pano2room"
    if _git(pano2room, "rev-parse", "HEAD") != software["pano2room_commit"]:
        raise SystemExit("Pano2Room checkout differs from the frozen mesh contract")

    scene_root = args.output_root.resolve() / scene["scene_id"]
    pano_path = scene_root / "pano_image.png"
    pano_preflight_path = scene_root / "panorama_preflight.json"
    if not pano_path.is_file() or not pano_preflight_path.is_file():
        raise SystemExit("source panorama or its preflight is missing")
    pano_preflight = json.loads(pano_preflight_path.read_text(encoding="utf-8"))
    if not pano_preflight.get("passed"):
        raise SystemExit("source panorama did not pass development preflight")
    if pano_preflight["image"]["sha256"] != sha256_file(pano_path):
        raise SystemExit("source panorama hash differs from its preflight record")

    output = scene_root / "mesh_development"
    output.mkdir(parents=True, exist_ok=True)
    request_path = output / "mesh_generation_request.json"
    raw_mesh = output / "mesh_model_raw.ply"
    metric_mesh = output / "mesh_model_metric_visual.ply"
    archived_prior_attempt = None
    if args.archive_prior_failure is not None:
        if args.skip_generation:
            raise SystemExit("failure archival cannot be combined with --skip-generation")
        archived_prior_attempt = _archive_failed_attempt(
            output, diagnosed_failure_class=args.archive_prior_failure
        )
    elif request_path.exists() and not args.skip_generation:
        raise SystemExit("refusing to overwrite an existing 3D reconstruction request")
    if (raw_mesh.exists() or metric_mesh.exists()) and not args.skip_generation:
        raise SystemExit("refusing to overwrite an existing 3D visual shell")

    import torch

    runtime = {
        "python": sys.version,
        "torch": torch.__version__,
        "torch_cuda": torch.version.cuda,
        "cuda_available": torch.cuda.is_available(),
        "gpu": torch.cuda.get_device_name(0) if torch.cuda.is_available() else None,
        "capability": list(torch.cuda.get_device_capability(0))
        if torch.cuda.is_available()
        else None,
        "gsplat": version("gsplat"),
        "tinycudann": version("tinycudann"),
        "pytorch3d": version("pytorch3d"),
    }
    expected_runtime = {
        "torch": software["torch_version"],
        "torch_cuda": software["cuda_version"],
        "gsplat": software["gsplat_version"],
        "tinycudann": software["tinycudann_version"],
    }
    if any(runtime[key] != value for key, value in expected_runtime.items()):
        raise SystemExit(f"runtime differs from frozen mesh contract: {runtime}")
    if not runtime["cuda_available"] or runtime["capability"] != [12, 0]:
        raise SystemExit(f"unexpected CUDA runtime/device: {runtime}")

    record = {
        "schema_version": "kinofail.embodiedgen-visual-shell-mesh-request.v1",
        "created_utc": datetime.now(UTC).isoformat(),
        "status": "requested" if not args.skip_generation else "audit_existing",
        "development_only": True,
        "scene": scene,
        "source_panorama": {"path": str(pano_path), "sha256": sha256_file(pano_path)},
        "source_panorama_preflight_sha256": sha256_file(pano_preflight_path),
        "config": {"path": str(config_path), "sha256": sha256_file(config_path)},
        "generator": {"path": str(Path(__file__).resolve()), "sha256": sha256_file(Path(__file__).resolve())},
        "software": software,
        "runtime": runtime,
        "admission_policy": config["admission_policy"],
        "compatibility_policy": config["compatibility_policy"],
        "archived_prior_attempt": archived_prior_attempt,
    }
    if args.skip_generation and request_path.exists():
        record = json.loads(request_path.read_text(encoding="utf-8"))
    else:
        request_path.write_text(
            json.dumps(record, indent=2, sort_keys=True) + "\n", encoding="utf-8"
        )

    if not args.skip_generation:
        try:
            os.environ.setdefault("_CHECK_PEFT", "0")
            _install_torchvision_compatibility_shim()
            pipeline_cfg = config["pipeline"]
            if not pipeline_cfg["emit_3dgs_training_data"]:
                if not config["compatibility_policy"][
                    "disable_realesrgan_in_mesh_only_mode"
                ]:
                    raise RuntimeError("mesh-only SR compatibility policy is not enabled")
                _install_mesh_only_sr_shim()
            sys.path.insert(0, str(repo))
            from embodied_gen.trainer.pono2mesh_trainer import Pano2MeshSRPipeline
            from embodied_gen.utils.config import Pano2MeshSRConfig

            reconstruction_dir = output / "reconstruction"
            reconstruction_dir.mkdir()
            trajectory_dir = repo / "apps/assets/example_scene/camera_trajectory"
            mesh_cfg = Pano2MeshSRConfig(
                mesh_file="mesh_model.ply",
                gs_data_file=(
                    "gs_data.pt" if pipeline_cfg["emit_3dgs_training_data"] else None
                ),
                device=pipeline_cfg["device"],
                fov=int(pipeline_cfg["fov_degrees"]),
                pano_w=int(pipeline_cfg["pano_width"]),
                pano_h=int(pipeline_cfg["pano_height"]),
                cubemap_w=int(pipeline_cfg["cubemap_width"]),
                cubemap_h=int(pipeline_cfg["cubemap_height"]),
                pose_scale=float(pipeline_cfg["pose_scale"]),
                pano_center_offset=tuple(pipeline_cfg["pano_center_offset"]),
                inpaint_frame_stride=int(pipeline_cfg["inpaint_frame_stride"]),
                trajectory_dir=str(trajectory_dir),
                depth_scale_factor=float(pipeline_cfg["depth_scale_factor"]),
                upscale_factor=int(pipeline_cfg["upscale_factor"]),
                visualize=False,
            )
            pipeline = Pano2MeshSRPipeline(mesh_cfg)
            pipeline(str(pano_path), str(reconstruction_dir))
            generated_mesh = reconstruction_dir / "mesh_model.ply"
            if not generated_mesh.is_file():
                raise RuntimeError("Pano2Mesh completed without producing mesh_model.ply")
            generated_mesh.replace(raw_mesh)
        except Exception as exc:
            failure = {
                "schema_version": "kinofail.embodiedgen-visual-shell-mesh-failure.v1",
                "failed_utc": datetime.now(UTC).isoformat(),
                "failure_class": type(exc).__name__,
                "failure_message": str(exc),
                "request_sha256": sha256_file(request_path),
                "raw_mesh_created": raw_mesh.is_file(),
                "metric_mesh_created": metric_mesh.is_file(),
            }
            (output / "mesh_generation_failure.json").write_text(
                json.dumps(failure, indent=2, sort_keys=True) + "\n", encoding="utf-8"
            )
            raise

    if not raw_mesh.is_file():
        raise SystemExit(f"raw visual mesh missing: {raw_mesh}")
    raw_audit = evaluate_visual_shell_mesh(
        raw_mesh, thresholds=config["mesh_preflight"]
    )
    if not metric_mesh.exists():
        metric_transform = create_metric_visual_mesh(
            raw_mesh, metric_mesh, target_height_m=float(scene["real_height_m"])
        )
    else:
        metric_transform = None
    metric_audit = evaluate_visual_shell_mesh(
        metric_mesh,
        thresholds=config["mesh_preflight"],
        target_height_m=float(scene["real_height_m"]),
    )
    audit = {
        "schema_version": "kinofail.embodiedgen-visual-shell-mesh-stage-audit.v1",
        "scene_id": scene["scene_id"],
        "development_only": True,
        "raw": raw_audit,
        "metric_visual": metric_audit,
        "metric_transform": metric_transform,
        "passed": raw_audit["passed"] and metric_audit["passed"],
        "counts_as_scene_registry_admission": False,
        "counts_as_a0_a7_evidence": False,
        "remaining_required_stages": config["admission_policy"]["required_after_mesh"],
    }
    audit_path = output / "mesh_stage_audit.json"
    audit_path.write_text(
        json.dumps(audit, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    print(json.dumps({"mesh": str(metric_mesh), "audit": str(audit_path), "passed": audit["passed"]}))
    if not audit["passed"]:
        raise SystemExit("development visual-shell mesh preflight failed")


if __name__ == "__main__":
    main()
