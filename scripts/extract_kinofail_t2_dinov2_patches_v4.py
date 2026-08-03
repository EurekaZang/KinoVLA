#!/usr/bin/env python3
"""Cache final-frame DINOv2 patch tokens for T2 attention development."""

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
DEFAULT_OUTPUT = Path(
    "/data/eureka/KinoVLA/outputs/eval/kino_t2_dinov2_patches_v4_development"
)


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(8 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--batch-size", type=int, default=192)
    args = parser.parse_args()
    output = args.output.resolve()
    if output.exists():
        raise FileExistsError(output)
    output.mkdir(parents=True, exist_ok=False)

    records_path = F42 / "conflict/features/records.jsonl"
    rows = [
        row
        for row in _jsonl(records_path)
        if str(row["cell"]) == "T2_vision_decisive"
    ]
    os.environ.setdefault("HF_HUB_OFFLINE", "1")
    processor = AutoImageProcessor.from_pretrained(MODEL_ID)
    model = AutoModel.from_pretrained(MODEL_ID).to("cuda").eval()
    torch.backends.cuda.matmul.allow_tf32 = True
    hidden = int(model.config.hidden_size)
    sample_ids_path = output / "sample_ids.npy"
    np.save(sample_ids_path, np.asarray([str(row["sample_id"]) for row in rows]))
    patches_path = output / "final_patches.npy"
    cls_path = output / "final_cls.npy"
    patches_memmap: np.memmap | None = None
    cls_memmap = np.lib.format.open_memmap(
        cls_path, mode="w+", dtype=np.float16, shape=(len(rows), hidden)
    )
    pending: list[Image.Image] = []
    indices: list[int] = []

    def flush() -> None:
        nonlocal pending, indices, patches_memmap
        if not pending:
            return
        pixels = processor(images=pending, return_tensors="pt")["pixel_values"].to(
            "cuda", non_blocking=True
        )
        with torch.inference_mode(), torch.autocast(
            device_type="cuda", dtype=torch.bfloat16
        ):
            tokens = model(pixel_values=pixels).last_hidden_state
        values = tokens.detach().cpu().numpy().astype(np.float16)
        if patches_memmap is None:
            patches_memmap = np.lib.format.open_memmap(
                patches_path,
                mode="w+",
                dtype=np.float16,
                shape=(len(rows), values.shape[1] - 1, hidden),
            )
        patches_memmap[indices] = values[:, 1:]
        cls_memmap[indices] = values[:, 0]
        pending = []
        indices = []

    for index, row in enumerate(rows):
        source = Path(str(row["source_case_dir"]))
        final_path = source / str(row["rgb_paths"][-1])
        with Image.open(final_path) as image:
            pending.append(image.convert("RGB").copy())
        indices.append(index)
        if len(pending) >= args.batch_size:
            flush()
        if (index + 1) % 1000 == 0:
            print(json.dumps({"processed": index + 1, "total": len(rows)}), flush=True)
    flush()
    assert patches_memmap is not None
    patches_memmap.flush()
    cls_memmap.flush()
    del patches_memmap, cls_memmap
    patches = np.load(patches_path, mmap_mode="r")
    if patches.shape[0] != len(rows) or not np.isfinite(patches).all():
        raise RuntimeError("patch cache failed validation")

    manifest = {
        "schema_version": "kinofail.t2-dinov2-patches-v4-development.v1",
        "created_utc": datetime.now(UTC).isoformat(),
        "status": "development_only_complete",
        "confirmatory_evidence": False,
        "encoder": MODEL_ID,
        "encoder_frozen": True,
        "deployment_input": "final robot-front RGB frame",
        "counts": {"samples": len(rows)},
        "dimensions": {
            "patch_tokens": list(patches.shape[1:]),
            "cls": hidden,
        },
        "source_sha256": {"records": _sha256(records_path)},
        "output_sha256": {
            "sample_ids": _sha256(sample_ids_path),
            "final_patches": _sha256(patches_path),
            "final_cls": _sha256(cls_path),
        },
    }
    (output / "manifest.json").write_text(
        json.dumps(manifest, indent=2, sort_keys=True) + "\n"
    )
    print(json.dumps(manifest, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
