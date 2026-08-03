#!/usr/bin/env python3
"""Run EmbodiedGen's frozen Real-ESRGAN path on one panorama and audit it."""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
from datetime import UTC, datetime
from pathlib import Path

import numpy as np
from PIL import Image


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from kino_vla.eval.visual_shell_preflight import (  # noqa: E402
    evaluate_visual_shell_panorama,
    sha256_file,
)


def _resolve_frozen(record: dict, *, label: str) -> Path:
    path = (ROOT / record["path"]).resolve()
    if not path.is_file():
        raise FileNotFoundError(f"missing {label}: {path}")
    actual = sha256_file(path)
    if actual != record["sha256"]:
        raise ValueError(f"stale {label}: expected {record['sha256']}, found {actual}")
    return path


def _git(repo: Path, *args: str) -> str:
    return subprocess.check_output(["git", "-C", str(repo), *args], text=True).strip()


def _luminance(rgb: np.ndarray) -> np.ndarray:
    rgb64 = rgb.astype(np.float64)
    return 0.2126 * rgb64[..., 0] + 0.7152 * rgb64[..., 1] + 0.0722 * rgb64[..., 2]


def _detail_audit(source: Image.Image, result: Image.Image) -> dict:
    bicubic = source.resize(result.size, resample=Image.Resampling.BICUBIC)
    result_rgb = np.asarray(result.convert("RGB"), dtype=np.uint8)
    bicubic_rgb = np.asarray(bicubic.convert("RGB"), dtype=np.uint8)
    residual = np.abs(result_rgb.astype(np.float64) - bicubic_rgb.astype(np.float64))
    result_lum = _luminance(result_rgb)
    bicubic_lum = _luminance(bicubic_rgb)

    def gradient_rms(luminance: np.ndarray) -> float:
        dx = np.diff(luminance, axis=1)
        dy = np.diff(luminance, axis=0)
        return float(np.sqrt((np.mean(dx * dx) + np.mean(dy * dy)) / 2.0))

    result_gradient = gradient_rms(result_lum)
    bicubic_gradient = gradient_rms(bicubic_lum)
    return {
        "reference": "PIL bicubic resize of the same source panorama",
        "residual_mae_vs_bicubic": float(np.mean(residual)),
        "result_luminance_gradient_rms": result_gradient,
        "bicubic_luminance_gradient_rms": bicubic_gradient,
        "gradient_rms_ratio_vs_bicubic": (
            result_gradient / bicubic_gradient if bicubic_gradient > 0.0 else None
        ),
        "interpretation": (
            "A nonzero residual confirms learned super-resolution rather than byte-equivalent "
            "interpolation. It does not prove that generated fine detail is physically real."
        ),
    }


def _run_frozen_embodiedgen_realesrgan(
    source: Image.Image, *, weight_path: Path, outscale: int
) -> Image.Image:
    """Faithful extraction of EmbodiedGen v2.0.0 ImageRealESRGAN._lazy_init/call.

    Importing the public wrapper also imports ``embodied_gen.data.utils`` and therefore
    optional Kaolin, although Kaolin is not used by the super-resolution path. Keeping this
    minimal extraction avoids changing the model, weight, architecture, precision, or API
    parameters solely to satisfy that unrelated transitive dependency.
    """
    import types

    import torch
    import torchvision.transforms.functional as functional

    compatibility = types.ModuleType("torchvision.transforms.functional_tensor")
    compatibility.rgb_to_grayscale = functional.rgb_to_grayscale
    sys.modules["torchvision.transforms.functional_tensor"] = compatibility

    from basicsr.archs.rrdbnet_arch import RRDBNet
    from realesrgan import RealESRGANer

    model = RRDBNet(
        num_in_ch=3,
        num_out_ch=3,
        num_feat=64,
        num_block=23,
        num_grow_ch=32,
        scale=4,
    )
    upsampler = RealESRGANer(
        scale=4,
        model_path=str(weight_path),
        model=model,
        pre_pad=0,
        half=True,
    )
    with torch.no_grad():
        output, _ = upsampler.enhance(np.asarray(source), outscale=outscale)
    return Image.fromarray(output)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--config",
        type=Path,
        default=ROOT
        / "configs/data/kinofail_embodiedgen_visual_shell_super_resolution_dev_v1.json",
    )
    parser.add_argument(
        "--embodiedgen-root",
        type=Path,
        default=Path("/home/eureka/dependencies/EmbodiedGen-v2.0.0"),
    )
    args = parser.parse_args()

    config_path = args.config.resolve()
    config = json.loads(config_path.read_text(encoding="utf-8"))
    if config.get("schema_version") not in {
        "kinofail.embodiedgen-visual-shell-super-resolution.v1-development",
        "kinofail.embodiedgen-visual-shell-super-resolution.v2-development",
    }:
        raise ValueError("unsupported super-resolution config schema")

    repo = args.embodiedgen_root.resolve()
    frozen_repo = config["embodiedgen"]
    if _git(repo, "rev-parse", "HEAD") != frozen_repo["commit"]:
        raise SystemExit("EmbodiedGen checkout differs from the frozen contract")
    if _git(repo, "describe", "--tags", "--exact-match") != frozen_repo["release"]:
        raise SystemExit("EmbodiedGen checkout is not at the frozen release")

    source_path = _resolve_frozen(config["source"]["panorama"], label="source panorama")
    request_path = _resolve_frozen(
        config["source"]["generation_request"], label="source generation request"
    )
    preflight_path = _resolve_frozen(config["source"]["preflight"], label="source preflight")
    source_preflight = json.loads(preflight_path.read_text(encoding="utf-8"))
    if source_preflight.get("passed") is not True:
        raise ValueError("source panorama preflight did not pass")

    from huggingface_hub import snapshot_download

    model_root = Path(
        snapshot_download(
            repo_id=frozen_repo["model_repository"],
            allow_patterns=[frozen_repo["model_allow_pattern"]],
        )
    ).resolve()
    weight_path = model_root / frozen_repo["weight_relative_path"]
    if not weight_path.is_file():
        raise FileNotFoundError(f"Real-ESRGAN weight missing: {weight_path}")

    output_dir = (ROOT / config["output"]["directory"]).resolve()
    if output_dir.exists():
        raise FileExistsError(f"refusing to overwrite super-resolution attempt: {output_dir}")
    output_dir.mkdir(parents=True)
    output_path = output_dir / config["output"]["panorama_filename"]
    audit_path = output_dir / config["output"]["audit_filename"]

    source = Image.open(source_path).convert("RGB")
    expected_source = tuple(int(value) for value in config["source"]["expected_resolution"])
    if source.size != expected_source:
        raise ValueError(f"unexpected source dimensions: {source.size} != {expected_source}")

    result = _run_frozen_embodiedgen_realesrgan(
        source,
        weight_path=weight_path,
        outscale=int(config["super_resolution"]["requested_output_scale"]),
    )
    expected_result = tuple(
        int(value) for value in config["super_resolution"]["expected_resolution"]
    )
    if result.size != expected_result:
        raise ValueError(f"unexpected output dimensions: {result.size} != {expected_result}")
    result.save(output_path)

    thresholds = dict(config["visual_preflight"])
    minimum_residual = float(thresholds.pop("minimum_residual_mae_vs_bicubic"))
    image_audit = evaluate_visual_shell_panorama(
        output_path,
        expected_width=expected_result[0],
        expected_height=expected_result[1],
        thresholds=thresholds,
    )
    detail_audit = _detail_audit(source, result)
    checks = {
        "source_preflight_passed": True,
        "output_image_preflight_passed": image_audit["passed"],
        "learned_result_differs_from_bicubic": detail_audit[
            "residual_mae_vs_bicubic"
        ]
        >= minimum_residual,
        "same_scene_not_new_diversity": config["interpretation_policy"][
            "counts_as_additional_scene_diversity"
        ]
        is False,
        "not_a0_a7_evidence": config["interpretation_policy"]["counts_as_a0_a7_evidence"]
        is False,
    }
    audit = {
        "schema_version": "kinofail.embodiedgen-visual-shell-super-resolution-audit.v1",
        "created_utc": datetime.now(UTC).isoformat(),
        "development_only": True,
        "config": {"path": str(config_path), "sha256": sha256_file(config_path)},
        "source": {
            "panorama": str(source_path),
            "panorama_sha256": sha256_file(source_path),
            "generation_request": str(request_path),
            "preflight": str(preflight_path),
        },
        "model": {
            "repository": frozen_repo["model_repository"],
            "resolved_revision": model_root.name,
            "weight": str(weight_path),
            "weight_sha256": sha256_file(weight_path),
            "embodiedgen_commit": frozen_repo["commit"],
            "implementation": config["embodiedgen"].get(
                "implementation",
                "embodied_gen.models.sr_model.ImageRealESRGAN",
            ),
        },
        "output": {
            "panorama": str(output_path),
            "panorama_sha256": sha256_file(output_path),
            "resolution": list(result.size),
        },
        "image_preflight": image_audit,
        "detail_audit": detail_audit,
        "checks": checks,
        "passed": all(checks.values()),
        "scene_registry_eligible": False,
        "counts_as_a0_a7_evidence": False,
        "remaining_gates": [
            "body_fixed_go2_rtx_review",
            "operator_pair_visual_review",
            "three_appearance_randomizations",
            "physical_camera_calibration",
            "independent_human_review",
        ],
    }
    audit_path.write_text(json.dumps(audit, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(
        json.dumps(
            {
                "passed": audit["passed"],
                "panorama": str(output_path),
                "audit": str(audit_path),
                "counts_as_a0_a7_evidence": False,
            },
            indent=2,
        )
    )
    return 0 if audit["passed"] else 2


if __name__ == "__main__":
    raise SystemExit(main())
