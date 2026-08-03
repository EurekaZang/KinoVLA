#!/usr/bin/env python3
"""Generate and preflight one development-only EmbodiedGen visual-shell panorama."""

from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
import types
from datetime import UTC, datetime
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from kino_vla.eval.visual_shell_preflight import (  # noqa: E402
    evaluate_visual_shell_panorama,
    sha256_file,
)


def _git(repo: Path, *args: str) -> str:
    return subprocess.check_output(["git", "-C", str(repo), *args], text=True).strip()


def _load_request(config: dict, scene_id: str) -> dict:
    rows = [row for row in config["requests"] if row["scene_id"] == scene_id]
    if len(rows) != 1:
        raise ValueError(f"expected exactly one request for scene_id={scene_id!r}")
    return dict(rows[0])


def _install_torchvision_compatibility_shim() -> None:
    import torchvision.transforms.functional as functional

    shim = types.ModuleType("torchvision.transforms.functional_tensor")
    shim.rgb_to_grayscale = functional.rgb_to_grayscale
    sys.modules["torchvision.transforms.functional_tensor"] = shim


def _module_device(module: object) -> str | None:
    parameters = getattr(module, "parameters", None)
    if not callable(parameters):
        return None
    try:
        return str(next(parameters()).device)
    except StopIteration:
        return None


def _audit_pipeline_device(pipeline: object, *, expected_prefix: str) -> dict:
    component_devices = {
        name: _module_device(getattr(pipeline, name, None))
        for name in ("text_encoder", "unet", "vae")
    }
    pipeline_device = str(getattr(pipeline, "device", "<missing>"))
    checks = {
        "pipeline_on_expected_device": pipeline_device.startswith(expected_prefix),
        "all_required_components_present": all(
            device is not None for device in component_devices.values()
        ),
        "all_required_components_on_expected_device": all(
            device is not None and device.startswith(expected_prefix)
            for device in component_devices.values()
        ),
        "single_device_policy": len(set(component_devices.values()) | {pipeline_device}) == 1,
    }
    return {
        "policy": "all_cuda_no_offload",
        "pipeline_device": pipeline_device,
        "component_devices": component_devices,
        "checks": checks,
        "passed": all(checks.values()),
    }


def _audit_prompt_tokens(tokenizer: object, prompt: str) -> dict:
    """Reject silent CLIP truncation before spending a frozen generation seed."""
    encoded = tokenizer(
        prompt,
        add_special_tokens=True,
        truncation=False,
        return_attention_mask=False,
    )
    input_ids = encoded["input_ids"]
    if input_ids and isinstance(input_ids[0], list):
        input_ids = input_ids[0]
    token_count = len(input_ids)
    model_max_length = int(getattr(tokenizer, "model_max_length"))
    checks = {
        "nonempty_prompt": bool(prompt.strip()),
        "within_model_context": token_count <= model_max_length,
    }
    return {
        "prompt": prompt,
        "token_count_including_special_tokens": token_count,
        "model_max_length": model_max_length,
        "checks": checks,
        "passed": all(checks.values()),
    }


def _archive_prior_failed_attempt(
    output: Path, *, failure_class: str, generator_sha256: str
) -> dict[str, str]:
    request_path = output / "generation_request.json"
    pano_path = output / "pano_image.png"
    if pano_path.exists():
        raise SystemExit("refusing to archive an attempt that produced a panorama")
    if not request_path.exists():
        raise SystemExit("no prior generation request exists to archive")
    attempts = output / "failed_attempts"
    attempts.mkdir(parents=True, exist_ok=True)
    index = len([path for path in attempts.iterdir() if path.is_dir()]) + 1
    attempt = attempts / f"attempt_{index:03d}"
    attempt.mkdir()
    request = json.loads(request_path.read_text(encoding="utf-8"))
    archived_request = attempt / "generation_request.json"
    archived_request.write_text(
        json.dumps(request, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    existing_failure = output / "generation_failure.json"
    failure = (
        json.loads(existing_failure.read_text(encoding="utf-8"))
        if existing_failure.is_file()
        else {
            "schema_version": "kinofail.embodiedgen-visual-shell-generation-failure.v1",
            "status": "failed",
            "failure_class": failure_class,
            "failure_message": (
                "Prior invocation terminated before panorama creation; the terminal traceback "
                "identified a CPU/CUDA text-encoder input mismatch."
            ),
            "recorded_on_resume_utc": datetime.now(UTC).isoformat(),
            "generator_sha256_recording_failure": generator_sha256,
        }
    )
    archived_failure = attempt / "generation_failure.json"
    archived_failure.write_text(
        json.dumps(failure, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    request_path.unlink()
    if existing_failure.is_file():
        existing_failure.unlink()
    return {
        "request": str(archived_request),
        "failure": str(archived_failure),
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--scene-id", required=True)
    parser.add_argument(
        "--config",
        type=Path,
        default=ROOT / "configs/data/kinofail_embodiedgen_visual_shell_development_v1.json",
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
        choices=("text_encoder_input_device_mismatch",),
        help="Archive a failed request before retrying; required instead of overwriting it.",
    )
    args = parser.parse_args()

    config_path = args.config.resolve()
    config = json.loads(config_path.read_text(encoding="utf-8"))
    request = _load_request(config, args.scene_id)
    embodiedgen = config["embodiedgen"]
    repo = args.embodiedgen_root.resolve()
    if _git(repo, "rev-parse", "HEAD") != embodiedgen["commit"]:
        raise SystemExit("EmbodiedGen checkout differs from the frozen development contract")
    if _git(repo, "describe", "--tags", "--exact-match") != embodiedgen["release"]:
        raise SystemExit("EmbodiedGen checkout is not at the frozen release tag")

    output = args.output_root.resolve() / request["scene_id"]
    output.mkdir(parents=True, exist_ok=True)
    pano_path = output / "pano_image.png"
    request_path = output / "generation_request.json"
    generator_sha256 = sha256_file(Path(__file__).resolve())
    archived_prior_attempt = None
    if request_path.exists() and not args.skip_generation:
        if args.archive_prior_failure is None:
            raise SystemExit(
                "an earlier request exists without a reusable panorama; pass "
                "--archive-prior-failure with the diagnosed failure class"
            )
        archived_prior_attempt = _archive_prior_failed_attempt(
            output,
            failure_class=args.archive_prior_failure,
            generator_sha256=generator_sha256,
        )
    elif args.archive_prior_failure is not None:
        raise SystemExit("--archive-prior-failure was supplied but no prior request exists")
    if pano_path.exists() and not args.skip_generation:
        raise SystemExit("refusing to overwrite an existing frozen-seed panorama")
    request_record = {
        "schema_version": "kinofail.embodiedgen-visual-shell-generation-request.v1",
        "created_utc": datetime.now(UTC).isoformat(),
        "development_only": True,
        "config": {"path": str(config_path), "sha256": sha256_file(config_path)},
        "request": request,
        "embodiedgen": embodiedgen,
        "generation": config["generation"],
        "physics_ownership": config["physics_ownership"],
        "generator": {
            "path": str(Path(__file__).resolve()),
            "sha256": generator_sha256,
            "device_policy": "all_cuda_no_offload",
        },
        "archived_prior_attempt": archived_prior_attempt,
    }
    request_path.write_text(
        json.dumps(request_record, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )

    resolved_model_revision = None
    if not args.skip_generation:
        try:
            os.environ.setdefault("_CHECK_PEFT", "0")
            _install_torchvision_compatibility_shim()
            import torch
            from diffusers import EulerAncestralDiscreteScheduler
            from huggingface_hub import snapshot_download
            from txt2panoimg.pipeline_base import StableDiffusionBlendExtendPipeline

            model_path = Path(
                snapshot_download(
                    embodiedgen["scene_model_repository"], allow_patterns=["sd-base/*"]
                )
            )
            resolved_model_revision = model_path.name
            pipeline = StableDiffusionBlendExtendPipeline.from_pretrained(
                str(model_path / "sd-base"), torch_dtype=torch.float16
            ).to("cuda")
            pipeline.vae.enable_tiling()
            pipeline.scheduler = EulerAncestralDiscreteScheduler.from_config(
                pipeline.scheduler.config
            )
            # txt2panoimg constructs weighted-token tensors on ``pipeline.device``.  With modern
            # diffusers, model CPU offload changes that property to CPU while the text-encoder
            # hook executes on CUDA.  A 32 GB card fits this base pipeline, so the development
            # protocol uses one audited CUDA device and no offload hook.
            device_audit = _audit_pipeline_device(pipeline, expected_prefix="cuda")
            if not device_audit["passed"]:
                raise RuntimeError(f"pipeline device preflight failed: {device_audit}")
            request_record["runtime_device_audit"] = device_audit
            request_record["resolved_model_revision"] = resolved_model_revision
            request_path.write_text(
                json.dumps(request_record, indent=2, sort_keys=True) + "\n",
                encoding="utf-8",
            )
            generation = config["generation"]
            prompt = (
                "<360panorama>, "
                + request["prompt"]
                + ", spacious, open route, coherent scene, photorealistic, trend on artstation, "
                "((best quality)), ((ultra high res))"
            )
            prompt_audit = _audit_prompt_tokens(pipeline.tokenizer, prompt)
            request_record["prompt_audit"] = prompt_audit
            request_path.write_text(
                json.dumps(request_record, indent=2, sort_keys=True) + "\n",
                encoding="utf-8",
            )
            if not prompt_audit["passed"]:
                raise RuntimeError(
                    "generation prompt exceeds the frozen text-encoder context; shorten the "
                    f"request instead of accepting silent truncation: {prompt_audit}"
                )
            image = pipeline(
                prompt,
                negative_prompt=generation["negative_prompt"],
                num_inference_steps=int(generation["num_inference_steps"]),
                height=int(generation["output_height"]),
                width=int(generation["output_width"]),
                guidance_scale=7.5,
                generator=torch.manual_seed(int(request["seed"])),
            ).images[0]
            image.save(pano_path)
        except Exception as exc:
            failure = {
                "schema_version": "kinofail.embodiedgen-visual-shell-generation-failure.v1",
                "status": "failed",
                "failed_utc": datetime.now(UTC).isoformat(),
                "failure_class": type(exc).__name__,
                "failure_message": str(exc),
                "request_sha256": sha256_file(request_path),
                "generator_sha256": generator_sha256,
                "panorama_created": pano_path.is_file(),
            }
            (output / "generation_failure.json").write_text(
                json.dumps(failure, indent=2, sort_keys=True) + "\n", encoding="utf-8"
            )
            raise

    if not pano_path.is_file():
        raise SystemExit(f"panorama missing: {pano_path}")
    audit = evaluate_visual_shell_panorama(
        pano_path,
        expected_width=int(config["generation"]["output_width"]),
        expected_height=int(config["generation"]["output_height"]),
        thresholds=config["visual_preflight"],
    )
    audit["scene_id"] = request["scene_id"]
    audit["domain"] = request["domain"]
    audit["seed"] = request["seed"]
    audit["development_only"] = True
    audit["resolved_model_revision"] = resolved_model_revision
    audit["request_sha256"] = sha256_file(request_path)
    audit_path = output / "panorama_preflight.json"
    audit_path.write_text(
        json.dumps(audit, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    print(json.dumps({"pano": str(pano_path), "audit": str(audit_path), "passed": audit["passed"]}))
    if not audit["passed"]:
        raise SystemExit("development panorama visual preflight failed")


if __name__ == "__main__":
    main()
