"""Timestamped, region-local depth faults for realistic O7 camera corruption.

The raw frame must come from the renderer.  This module only transforms the measured depth
channel after acquisition, preserves RGB/segmentation/calibration, and attaches enough provenance
to prove which pixels were changed at which simulation time.
"""

from __future__ import annotations

import hashlib
from collections.abc import Mapping
from dataclasses import dataclass
from typing import Any

import numpy as np


@dataclass(frozen=True)
class DepthFaultRegion:
    """One semantically addressed depth fault in a rendered camera frame."""

    semantic_class: str
    bias_m: float

    def __post_init__(self) -> None:
        if not self.semantic_class:
            raise ValueError("semantic_class must be non-empty")
        if not np.isfinite(self.bias_m):
            raise ValueError("bias_m must be finite")


def _sha256_array(value: np.ndarray) -> str:
    contiguous = np.ascontiguousarray(value)
    return hashlib.sha256(contiguous.tobytes()).hexdigest()


def _semantic_ids(
    id_to_labels: Mapping[object, object], semantic_class: str
) -> set[int]:
    matches: set[int] = set()
    for raw_id, raw_label in id_to_labels.items():
        label = raw_label.get("class") if isinstance(raw_label, Mapping) else str(raw_label)
        if str(label or "").casefold() == semantic_class.casefold():
            matches.add(int(raw_id))
    return matches


def apply_timestamped_depth_faults(
    capture: Mapping[str, Any],
    faults: list[DepthFaultRegion] | tuple[DepthFaultRegion, ...],
    *,
    capture_time_s: float,
    frame_id: int,
    read_time_s: float | None = None,
) -> dict[str, Any]:
    """Copy one real camera capture and bias only finite pixels in named semantic regions."""

    raw_depth = np.asarray(capture["depth"], dtype=np.float64)
    segmentation = np.asarray(capture["seg"])
    if raw_depth.shape != segmentation.shape:
        raise ValueError("depth and segmentation must have identical image shape")

    depth = raw_depth.copy()
    union_mask = np.zeros(raw_depth.shape, dtype=bool)
    rows: list[dict[str, Any]] = []
    labels = capture.get("id_to_labels", {})
    for fault in faults:
        semantic_ids = _semantic_ids(labels, fault.semantic_class)
        mask = np.isfinite(raw_depth) & np.isin(segmentation, list(semantic_ids))
        depth[mask] += float(fault.bias_m)
        union_mask |= mask
        rows.append(
            {
                "semantic_class": fault.semantic_class,
                "semantic_ids": sorted(semantic_ids),
                "requested_bias_m": float(fault.bias_m),
                "affected_pixels": int(mask.sum()),
                "mean_applied_bias_m": (
                    float(np.mean(depth[mask] - raw_depth[mask])) if mask.any() else 0.0
                ),
            }
        )

    read_time = float(capture_time_s if read_time_s is None else read_time_s)
    result = dict(capture)
    result["depth"] = depth
    result["camera_timestamp_s"] = float(capture_time_s)
    result["depth_source_timestamp_s"] = float(capture_time_s)
    result["sensor_read_timestamp_s"] = read_time
    result["camera_frame_id"] = int(frame_id)
    result["depth_fault_telemetry"] = {
        "pipeline": "post_rtx_region_depth_v1",
        "camera_timestamp_s": float(capture_time_s),
        "depth_source_timestamp_s": float(capture_time_s),
        "sensor_read_timestamp_s": read_time,
        "effective_depth_age_s": max(0.0, read_time - float(capture_time_s)),
        "frame_id": int(frame_id),
        "raw_depth_sha256": _sha256_array(raw_depth),
        "output_depth_sha256": _sha256_array(depth),
        "input_rgb_sha256": _sha256_array(np.asarray(capture["rgb"])),
        "output_rgb_sha256": _sha256_array(np.asarray(result["rgb"])),
        "affected_pixels": int(union_mask.sum()),
        "faults": rows,
    }
    return result
