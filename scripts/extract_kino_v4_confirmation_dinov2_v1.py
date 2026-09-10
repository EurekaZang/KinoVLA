#!/usr/bin/env python3
"""Extract the frozen DINOv2-L/14 terrain descriptor for confirmation rows."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
from datetime import UTC, datetime
from pathlib import Path

import numpy as np
import torch
from PIL import Image
from transformers import AutoImageProcessor, AutoModel


MODEL_ID = "facebook/dinov2-large"
MODEL_REVISION = "47b73eefe95e8d44ec3623f8890bd894b6ea2d6c"
IMAGE_SIZE = 448


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _rows(path: Path) -> list[dict]:
    return [
        json.loads(line)
        for line in path.read_text(encoding="utf-8").splitlines()
        if line
    ]


def _temporal(frames: np.ndarray) -> np.ndarray:
    return np.concatenate([frames.mean(axis=1), frames[:, -1]], axis=1).astype(
        np.float32
    )


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--records", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--batch-size", type=int, default=64)
    args = parser.parse_args()
    records_path = args.records.resolve()
    output = args.output.resolve()
    if output.exists():
        raise FileExistsError(output)
    if args.batch_size <= 0:
        raise ValueError("batch size must be positive")
    rows = _rows(records_path)
    if not rows or any("source_case_dir" not in row for row in rows):
        raise RuntimeError("confirmation records require source_case_dir")
    frame_count = len(rows[0]["rgb_paths"])
    if frame_count != 5 or any(len(row["rgb_paths"]) != frame_count for row in rows):
        raise RuntimeError("frozen confirmation input is five RGB frames")

    os.environ.setdefault("HF_HUB_OFFLINE", "1")
    processor = AutoImageProcessor.from_pretrained(
        MODEL_ID, revision=MODEL_REVISION
    )
    processor.size = {"shortest_edge": int(round(IMAGE_SIZE * 256.0 / 224.0))}
    processor.crop_size = {"height": IMAGE_SIZE, "width": IMAGE_SIZE}
    model = AutoModel.from_pretrained(MODEL_ID, revision=MODEL_REVISION).to("cuda").eval()
    torch.backends.cuda.matmul.allow_tf32 = True
    hidden = int(model.config.hidden_size)
    frame_features = np.empty((len(rows), frame_count, hidden), dtype=np.float32)
    pending: list[Image.Image] = []
    locations: list[tuple[int, int]] = []

    def flush() -> None:
        nonlocal pending, locations
        if not pending:
            return
        pixels = processor(images=pending, return_tensors="pt")["pixel_values"].to(
            "cuda", non_blocking=True
        )
        with torch.inference_mode(), torch.autocast(
            device_type="cuda", dtype=torch.bfloat16
        ):
            tokens = model(pixel_values=pixels).last_hidden_state[:, 1:].float()
        grid = int(round(float(tokens.shape[1]) ** 0.5))
        if grid * grid != int(tokens.shape[1]):
            raise RuntimeError("DINO patch tokens do not form a square grid")
        patches = tokens.reshape(len(pending), grid, grid, hidden)
        ground = patches[:, int(round(0.40 * grid)) :, :, :].mean(dim=(1, 2))
        for (sample_index, frame_index), value in zip(
            locations, ground.cpu().numpy().astype(np.float32), strict=True
        ):
            frame_features[sample_index, frame_index] = value
        pending = []
        locations = []

    for sample_index, row in enumerate(rows):
        source = Path(str(row["source_case_dir"]))
        for frame_index, relative in enumerate(row["rgb_paths"]):
            with Image.open(source / str(relative)) as image:
                frame = image.convert("RGB")
                crop = dict(row.get("rgb_crop", {}))
                y_start = int(crop.get("pixel_y_start", round(0.45 * frame.height)))
                x_fraction = crop.get("x_fraction", [0.0, 1.0])
                x0 = int(round(float(x_fraction[0]) * frame.width))
                x1 = int(round(float(x_fraction[1]) * frame.width))
                if not (0 <= x0 < x1 <= frame.width and 0 <= y_start < frame.height):
                    raise RuntimeError(f"invalid terrain crop: {row['sample_id']}")
                pending.append(frame.crop((x0, y_start, x1, frame.height)).copy())
            locations.append((sample_index, frame_index))
            if len(pending) >= args.batch_size:
                flush()
        if (sample_index + 1) % 500 == 0:
            print(
                json.dumps({"processed_samples": sample_index + 1, "total": len(rows)}),
                flush=True,
            )
    flush()
    visual = _temporal(frame_features)
    if visual.shape != (len(rows), 2 * hidden) or not np.isfinite(visual).all():
        raise RuntimeError("invalid DINO confirmation feature array")

    output.mkdir(parents=True, exist_ok=False)
    features_path = output / "features.npz"
    np.savez_compressed(
        features_path,
        sample_ids=np.asarray([str(row["sample_id"]) for row in rows]),
        visual_ground_mean=visual,
    )
    manifest = {
        "schema_version": "kinofail.kino-v4-confirmation-dinov2.v1",
        "created_utc": datetime.now(UTC).isoformat(),
        "passed": True,
        "model_id": MODEL_ID,
        "model_revision": MODEL_REVISION,
        "image_size": IMAGE_SIZE,
        "input_region": "registered_lower-image terrain crop",
        "pool": "ground_mean",
        "temporal_summary": "mean + final",
        "counts": {"samples": len(rows), "frames_per_sample": frame_count},
        "dimensions": {"frame": hidden, "temporal": int(visual.shape[1])},
        "source_sha256": {"records": _sha256(records_path)},
        "output_sha256": {"features": _sha256(features_path)},
        "selection_uses_feature_or_prediction_values": False,
    }
    (output / "feature_manifest.json").write_text(
        json.dumps(manifest, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    print(json.dumps(manifest, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
