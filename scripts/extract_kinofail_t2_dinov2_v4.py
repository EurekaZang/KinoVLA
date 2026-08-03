#!/usr/bin/env python3
"""Extract fine-grained DINOv2 visual descriptors for Kino-Fail T2.

The extractor preserves the robot-front image and derives fixed token pools
from the frozen DINOv2 patch grid.  No semantic mask, simulator ID, operator
metadata, or privileged geometry is used at deployment.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import sys
from datetime import UTC, datetime
from pathlib import Path

import numpy as np
import torch
from PIL import Image
from transformers import AutoImageProcessor, AutoModel


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from scripts.develop_kinofail_conflict_invariant_kino_v4 import (  # noqa: E402
    F42,
    _jsonl,
)


MODEL_ID = "facebook/dinov2-base"
T3_UNION = Path("/data/eureka/KinoVLA/outputs/kinofail_t3_runin_f38_s2/union")


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _temporal_context(frames: np.ndarray) -> np.ndarray:
    return np.concatenate([frames.mean(axis=1), frames[:, -1]], axis=1).astype(
        np.float32
    )


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--output",
        type=Path,
        default=ROOT / "outputs/eval/kino_t2_dinov2_v4_development",
    )
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument(
        "--image-size",
        type=int,
        default=224,
        help=(
            "Square DINOv2 crop resolution. 224 reproduces the original "
            "development cache; larger multiples of 14 retain finer cues."
        ),
    )
    parser.add_argument(
        "--input-region",
        choices=("full", "terrain"),
        default="full",
        help=(
            "Use the full robot-front frame or the registered lower-image "
            "terrain crop. Both are observable RGB-only inputs."
        ),
    )
    parser.add_argument(
        "--all-conflict",
        action="store_true",
        help="Extract both T2 and T3 for fair generic-fusion baselines.",
    )
    parser.add_argument("--model-id", default=MODEL_ID)
    parser.add_argument(
        "--allow-download",
        action="store_true",
        help="Allow Hugging Face to download a model that is not cached.",
    )
    args = parser.parse_args()
    if args.batch_size <= 0:
        raise ValueError("batch size must be positive")
    if args.image_size <= 0 or args.image_size % 14 != 0:
        raise ValueError("image size must be a positive multiple of 14")
    output = args.output.resolve()
    if output.exists():
        raise FileExistsError(output)

    records_path = F42 / "conflict/features/records.jsonl"
    all_rows = _jsonl(records_path)
    rows = (
        all_rows
        if args.all_conflict
        else [
            row
            for row in all_rows
            if str(row["cell"]) == "T2_vision_decisive"
        ]
    )
    frame_count = len(rows[0]["rgb_paths"])
    if any(len(row["rgb_paths"]) != frame_count for row in rows):
        raise RuntimeError("T2 sequences do not share the same frame count")

    if not args.allow_download:
        os.environ.setdefault("HF_HUB_OFFLINE", "1")
    processor = AutoImageProcessor.from_pretrained(args.model_id)
    # Preserve the encoder's original resize/crop ratio (256 -> 224) while
    # increasing spatial resolution.  This keeps the same central field of
    # view and changes only the amount of retained visual detail.
    processor.size = {
        "shortest_edge": int(round(float(args.image_size) * 256.0 / 224.0))
    }
    processor.crop_size = {
        "height": int(args.image_size),
        "width": int(args.image_size),
    }
    model = AutoModel.from_pretrained(args.model_id).to("cuda").eval()
    torch.backends.cuda.matmul.allow_tf32 = True
    hidden = int(model.config.hidden_size)
    names = (
        "cls",
        "patch_mean",
        "ground_mean",
        "ground_max",
        "route_mean",
        "route_max",
    )
    features = {
        name: np.empty((len(rows), frame_count, hidden), dtype=np.float32)
        for name in names
    }
    pending: list[Image.Image] = []
    locations: list[tuple[int, int]] = []

    def flush() -> None:
        nonlocal pending, locations
        if not pending:
            return
        inputs = processor(images=pending, return_tensors="pt")
        pixels = inputs["pixel_values"].to("cuda", non_blocking=True)
        with torch.inference_mode(), torch.autocast(
            device_type="cuda", dtype=torch.bfloat16
        ):
            tokens = model(pixel_values=pixels).last_hidden_state
        tokens = tokens.float()
        cls = tokens[:, 0]
        patches = tokens[:, 1:]
        grid = int(round(float(patches.shape[1]) ** 0.5))
        if grid * grid != int(patches.shape[1]):
            raise RuntimeError("DINO patch tokens do not form a square grid")
        patch_grid = patches.reshape(len(pending), grid, grid, hidden)
        ground_start = int(round(0.40 * grid))
        route_y = int(round(0.35 * grid))
        route_x0 = int(round(0.12 * grid))
        route_x1 = int(round(0.88 * grid))
        ground = patch_grid[:, ground_start:, :, :]
        route = patch_grid[:, route_y:, route_x0:route_x1, :]
        encoded = {
            "cls": cls,
            "patch_mean": patches.mean(dim=1),
            "ground_mean": ground.mean(dim=(1, 2)),
            "ground_max": ground.amax(dim=(1, 2)),
            "route_mean": route.mean(dim=(1, 2)),
            "route_max": route.amax(dim=(1, 2)),
        }
        for name, tensor in encoded.items():
            values = tensor.cpu().numpy().astype(np.float32)
            for (sample_index, frame_index), value in zip(
                locations, values, strict=True
            ):
                features[name][sample_index, frame_index] = value
        pending = []
        locations = []

    for sample_index, row in enumerate(rows):
        source = (
            Path(str(row["source_case_dir"]))
            if "source_case_dir" in row
            else T3_UNION / str(row["scene_cluster"])
        )
        for frame_index, relative in enumerate(row["rgb_paths"]):
            with Image.open(source / str(relative)) as image:
                frame = image.convert("RGB")
                if args.input_region == "terrain":
                    crop = dict(row.get("rgb_crop", {}))
                    y_start = int(crop.get("pixel_y_start", round(0.45 * frame.height)))
                    x_fraction = crop.get("x_fraction", [0.0, 1.0])
                    x0 = int(round(float(x_fraction[0]) * frame.width))
                    x1 = int(round(float(x_fraction[1]) * frame.width))
                    if not (0 <= x0 < x1 <= frame.width and 0 <= y_start < frame.height):
                        raise RuntimeError(
                            f"invalid registered RGB crop for {row['sample_id']}"
                        )
                    frame = frame.crop((x0, y_start, x1, frame.height))
                pending.append(frame.copy())
            locations.append((sample_index, frame_index))
            if len(pending) >= args.batch_size:
                flush()
        if (sample_index + 1) % 500 == 0:
            print(
                json.dumps(
                    {"processed_samples": sample_index + 1, "total": len(rows)}
                ),
                flush=True,
            )
    flush()
    if not all(np.isfinite(values).all() for values in features.values()):
        raise RuntimeError("DINOv2 features contain non-finite values")

    output.mkdir(parents=True, exist_ok=False)
    features_path = output / "features.npz"
    summaries = {name: _temporal_context(values) for name, values in features.items()}
    np.savez_compressed(
        features_path,
        sample_ids=np.asarray([str(row["sample_id"]) for row in rows]),
        **{f"visual_{name}": value for name, value in summaries.items()},
    )
    manifest = {
        "schema_version": "kinofail.conflict-dinov2-v4-development.v1",
        "created_utc": datetime.now(UTC).isoformat(),
        "status": "development_only_complete",
        "confirmatory_evidence": False,
        "encoder": str(args.model_id),
        "encoder_frozen": True,
        "deployment_input": "single five-frame robot-front RGB sequence",
        "image_size": int(args.image_size),
        "input_region": str(args.input_region),
        "processor_shortest_edge": int(processor.size["shortest_edge"]),
        "token_pools": list(names),
        "counts": {"samples": len(rows), "frames_per_sample": frame_count},
        "cells": sorted({str(row["cell"]) for row in rows}),
        "dimensions": {
            name: int(value.shape[1]) for name, value in summaries.items()
        },
        "source_sha256": {"records": _sha256(records_path)},
        "output_sha256": {"features": _sha256(features_path)},
    }
    manifest_path = output / "manifest.json"
    manifest_path.write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n")
    print(json.dumps(manifest, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
