#!/usr/bin/env python3
"""Extract one aligned DINOv2 terrain descriptor from snapshot bundles.

The target order comes from an existing feature archive.  Target records may
alias replacement samples; the extractor resolves those aliases, reads the
corresponding five robot-front RGB frames from the registered snapshot bundle,
and emits one common descriptor schema for every diagnostic battery.
"""

from __future__ import annotations

import argparse
import json
import os
from collections import defaultdict
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image
from transformers import AutoImageProcessor, AutoModel


MODEL_ID = "facebook/dinov2-large"
MODEL_REVISION = "47b73eefe95e8d44ec3623f8890bd894b6ea2d6c"
IMAGE_SIZE = 448


def rows(path: Path) -> list[dict[str, Any]]:
    return [json.loads(line) for line in path.read_text().splitlines() if line]


def record_paths(roots: list[Path]) -> list[Path]:
    result: set[Path] = set()
    for root in roots:
        if root.is_file():
            result.add(root.resolve())
        else:
            result.update(path.resolve() for path in root.rglob("snapshot_records.jsonl"))
    return sorted(result)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--target-features", type=Path, required=True)
    parser.add_argument("--target-record-root", type=Path, action="append", required=True)
    parser.add_argument("--snapshot-root", type=Path, action="append", required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--batch-size", type=int, default=96)
    parser.add_argument(
        "--native-gpu-preprocess",
        action="store_true",
        help=(
            "Perform the fixed resize/crop/normalization on the GPU. This preserves "
            "the DINOv2 input contract while avoiding the per-frame PIL bottleneck."
        ),
    )
    parser.add_argument(
        "--frame-mode", choices=("final", "mean-final"), default="final"
    )
    args = parser.parse_args()
    output = args.output.resolve()
    if output.exists():
        raise FileExistsError(output)
    with np.load(args.target_features.resolve(), allow_pickle=False) as archive:
        target_ids = archive["sample_ids"].astype(str)
    target_set = set(target_ids.tolist())
    aliases: dict[str, str] = {}
    for path in record_paths(args.target_record_root):
        for row in rows(path):
            sample_id = str(row["sample_id"])
            if sample_id in target_set:
                source_id = str(
                    row.get(
                        "source_replacement_sample_id",
                        row.get("source_f33_sample_id", sample_id),
                    )
                )
                if sample_id in aliases and aliases[sample_id] != source_id:
                    raise RuntimeError(f"conflicting target alias: {sample_id}")
                aliases[sample_id] = source_id
    missing_target_records = target_set - set(aliases)
    if missing_target_records:
        raise RuntimeError(
            f"missing {len(missing_target_records)} target records; "
            f"first={sorted(missing_target_records)[:3]}"
        )
    needed_sources = set(aliases.values())
    source_locations: dict[str, tuple[Path, str]] = {}
    for path in record_paths(args.snapshot_root):
        archive_path = path.parent / "snapshots.npz"
        if not archive_path.is_file():
            continue
        for row in rows(path):
            source_id = str(row["sample_id"])
            if source_id not in needed_sources:
                continue
            location = (archive_path, source_id + "__rgb")
            if source_id in source_locations and source_locations[source_id] != location:
                raise RuntimeError(f"duplicate source snapshot: {source_id}")
            source_locations[source_id] = location
    missing_sources = needed_sources - set(source_locations)
    if missing_sources:
        raise RuntimeError(
            f"missing {len(missing_sources)} source snapshots; "
            f"first={sorted(missing_sources)[:3]}"
        )

    os.environ.setdefault("HF_HUB_OFFLINE", "1")
    os.environ.setdefault("TRANSFORMERS_OFFLINE", "1")
    processor = AutoImageProcessor.from_pretrained(MODEL_ID, revision=MODEL_REVISION)
    processor.size = {"shortest_edge": int(round(IMAGE_SIZE * 256.0 / 224.0))}
    processor.crop_size = {"height": IMAGE_SIZE, "width": IMAGE_SIZE}
    device = torch.device("cuda")
    model = AutoModel.from_pretrained(MODEL_ID, revision=MODEL_REVISION).to(device).eval()
    torch.backends.cuda.matmul.allow_tf32 = True
    hidden = int(model.config.hidden_size)
    unique_sources = sorted(needed_sources)
    source_index = {sample_id: index for index, sample_id in enumerate(unique_sources)}
    frame_indices = (4,) if args.frame_mode == "final" else tuple(range(5))
    frame_features = np.empty(
        (len(unique_sources), len(frame_indices), hidden), dtype=np.float32
    )
    pending: list[Image.Image | np.ndarray] = []
    locations: list[tuple[int, int]] = []

    def flush() -> None:
        nonlocal pending, locations
        if not pending:
            return
        if args.native_gpu_preprocess:
            raw = torch.from_numpy(np.stack(pending)).to(device, non_blocking=True)
            pixels = raw.permute(0, 3, 1, 2).to(dtype=torch.float32).div_(255.0)
            source_height, source_width = int(pixels.shape[2]), int(pixels.shape[3])
            resized_height = int(processor.size["shortest_edge"])
            resized_width = int(round(source_width * resized_height / source_height))
            pixels = F.interpolate(
                pixels,
                size=(resized_height, resized_width),
                mode="bicubic",
                align_corners=False,
                antialias=True,
            )
            crop_height = int(processor.crop_size["height"])
            crop_width = int(processor.crop_size["width"])
            top = (resized_height - crop_height) // 2
            left = (resized_width - crop_width) // 2
            pixels = pixels[:, :, top : top + crop_height, left : left + crop_width]
            mean = torch.tensor(processor.image_mean, device=device).view(1, 3, 1, 1)
            std = torch.tensor(processor.image_std, device=device).view(1, 3, 1, 1)
            pixels = pixels.sub_(mean).div_(std)
        else:
            pixels = processor(images=pending, return_tensors="pt")["pixel_values"].to(
                device, non_blocking=True
            )
        with torch.inference_mode(), torch.autocast(
            device_type="cuda", dtype=torch.bfloat16
        ):
            tokens = model(pixel_values=pixels).last_hidden_state[:, 1:].float()
        grid = int(round(float(tokens.shape[1]) ** 0.5))
        patches = tokens.reshape(len(pending), grid, grid, hidden)
        ground = patches[:, int(round(0.40 * grid)) :, :, :].mean(dim=(1, 2))
        values = ground.cpu().numpy().astype(np.float32)
        for (sample_index, frame_index), value in zip(locations, values, strict=True):
            frame_features[sample_index, frame_index] = value
        pending = []
        locations = []

    by_archive: dict[Path, list[tuple[str, str]]] = defaultdict(list)
    for sample_id in unique_sources:
        archive_path, key = source_locations[sample_id]
        by_archive[archive_path].append((sample_id, key))
    processed = 0
    for archive_path, entries in sorted(by_archive.items()):
        with np.load(archive_path, allow_pickle=False) as archive:
            for sample_id, key in entries:
                frames = np.asarray(archive[key], dtype=np.uint8)
                if frames.shape != (5, 360, 640, 3):
                    raise RuntimeError(f"unexpected RGB shape for {sample_id}: {frames.shape}")
                sample_index = source_index[sample_id]
                for local_index, frame_index in enumerate(frame_indices):
                    frame = frames[frame_index]
                    y_start = int(round(0.45 * frame.shape[0]))
                    cropped = frame[y_start:].copy()
                    pending.append(
                        cropped
                        if args.native_gpu_preprocess
                        else Image.fromarray(cropped)
                    )
                    locations.append((sample_index, local_index))
                    if len(pending) >= args.batch_size:
                        flush()
                processed += 1
        if processed % 1000 < len(entries):
            print(
                json.dumps({"processed_sources": processed, "total": len(unique_sources)}),
                flush=True,
            )
    flush()
    if args.frame_mode == "final":
        temporal = frame_features[:, 0].astype(np.float32)
    else:
        temporal = np.concatenate(
            [frame_features.mean(axis=1), frame_features[:, -1]], axis=1
        ).astype(np.float32)
    target_visual = np.stack(
        [temporal[source_index[aliases[sample_id]]] for sample_id in target_ids]
    )
    expected_dimension = hidden if args.frame_mode == "final" else 2 * hidden
    if target_visual.shape != (len(target_ids), expected_dimension):
        raise RuntimeError(f"unexpected target feature shape: {target_visual.shape}")
    output.mkdir(parents=True)
    np.savez_compressed(
        output / "features.npz",
        sample_ids=target_ids,
        visual_ground_mean=target_visual,
    )
    manifest = {
        "schema_version": "kinofail.snapshot-dinov2.v1",
        "created_utc": datetime.now(UTC).isoformat(),
        "model_id": MODEL_ID,
        "model_revision": MODEL_REVISION,
        "image_size": IMAGE_SIZE,
        "input": "five registered robot-front RGB frames; lower-image terrain crop",
        "temporal_summary": args.frame_mode,
        "target_samples": len(target_ids),
        "unique_source_samples": len(unique_sources),
        "dimension": int(target_visual.shape[1]),
        "passed": bool(np.isfinite(target_visual).all()),
    }
    (output / "feature_manifest.json").write_text(
        json.dumps(manifest, indent=2) + "\n"
    )
    print(json.dumps(manifest, indent=2), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
