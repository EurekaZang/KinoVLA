"""Runtime provenance and hard quality gates for Kino-Fail realistic episodes.

The design compiler intentionally emits only ``artifact_state=planned`` rows.  This module is the
only supported promotion path from a schedule row to an evaluation-eligible episode:

``planned -> collected -> validated``

Promotion is evidence based.  File hashes are recomputed, RGB/proprioception clocks are checked,
the camera must report a body-fixed Go2 mount, and the privileged operator readback must agree
with the scheduled intervention.  A collector cannot make an episode evaluation eligible merely
by writing ``artifact_state=validated`` into JSON.  Passing artifact QA is also intentionally
separate from publication promotion: engineering smoke collections are retained as
``validated_smoke`` but only the frozen ``formal_pilot`` protocol can become evaluation eligible.
"""

from __future__ import annotations

import copy
import hashlib
import json
import math
import struct
from collections import Counter, defaultdict
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np

RUNTIME_SCHEMA_VERSION = "kinofail.realistic-runtime.v5"
FORMAL_COLLECTION_STATUS = "formal_pilot"
FORMAL_PROTOCOL_SCHEMA_VERSION = "kinofail.formal-collection-protocol.v1"

DEFAULT_RUNTIME_GATES: dict[str, float | int] = {
    "min_rgb_frames": 5,
    "min_distinct_rgb_frames": 3,
    "min_distinct_appearance_view_sequences": 3,
    "min_mean_appearance_pair_rgb_l1": 0.015,
    "min_rgb_width": 320,
    "min_rgb_height": 180,
    "min_proprio_samples": 25,
    "min_proprio_features": 11,
    "max_rgb_proprio_skew_s": 0.04,
    "max_mean_highlight_fraction": 0.45,
    "max_frame_highlight_fraction": 0.90,
    "min_mean_luminance_std": 12.0,
    "min_scheduled_appearance_pixel_fraction": 0.10,
    "min_scene_context_pixel_fraction": 0.08,
    "parameter_abs_tolerance": 1.0e-5,
    "parameter_rel_tolerance": 0.01,
}


@dataclass(frozen=True)
class RuntimeValidation:
    """Result of independently validating one collected episode."""

    passed: bool
    issues: tuple[str, ...]
    gates: dict[str, Any]
    validated_record: dict[str, Any]
    manifest: dict[str, Any]


@dataclass(frozen=True)
class RuntimeCorpusAudit:
    """Corpus-level state counts plus the validated records eligible for evaluation."""

    passed: bool
    summary: dict[str, Any]
    records: tuple[dict[str, Any], ...]


def sha256_file(path: str | Path) -> str:
    """Return a streaming SHA-256 digest for an artifact."""
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def schedule_record_sha256(record: Mapping[str, Any]) -> str:
    """Hash the immutable canonical JSON form of one planned schedule record."""
    payload = json.dumps(record, sort_keys=True, separators=(",", ":"), ensure_ascii=False)
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


def _safe_artifact_path(episode_dir: Path, relative_path: object) -> Path:
    if not isinstance(relative_path, str) or not relative_path:
        raise ValueError("artifact path must be a non-empty relative string")
    relative = Path(relative_path)
    if relative.is_absolute():
        raise ValueError(f"artifact path must be relative: {relative_path!r}")
    root = episode_dir.resolve()
    resolved = (root / relative).resolve()
    try:
        resolved.relative_to(root)
    except ValueError as exc:
        raise ValueError(f"artifact path escapes episode directory: {relative_path!r}") from exc
    return resolved


def _png_dimensions(path: Path) -> tuple[int, int]:
    """Read a PNG IHDR without adding an image-library dependency to the core package."""
    with path.open("rb") as stream:
        header = stream.read(24)
    if len(header) != 24 or header[:8] != b"\x89PNG\r\n\x1a\n" or header[12:16] != b"IHDR":
        raise ValueError(f"not a valid PNG header: {path}")
    width, height = struct.unpack(">II", header[16:24])
    if width <= 0 or height <= 0:
        raise ValueError(f"invalid PNG dimensions: {width}x{height}")
    return width, height


def _strictly_increasing(values: Sequence[float]) -> bool:
    return (
        bool(values)
        and all(math.isfinite(value) for value in values)
        and all(current > previous for previous, current in zip(values, values[1:], strict=False))
    )


def _numbers_close(actual: object, expected: object, *, abs_tol: float, rel_tol: float) -> bool:
    if isinstance(expected, bool) or isinstance(actual, bool):
        return actual == expected
    if isinstance(expected, (int, float)) and isinstance(actual, (int, float)):
        return math.isclose(float(actual), float(expected), abs_tol=abs_tol, rel_tol=rel_tol)
    return actual == expected


def _append_issue(issues: list[str], condition: bool, code: str) -> None:
    if not condition:
        issues.append(code)


def _validate_hashed_readback_artifacts(
    issues: list[str],
    *,
    episode_dir: Path,
    section: Mapping[str, Any],
    issue_prefix: str,
) -> None:
    """Re-hash provenance files copied into an episode; metadata-only claims are insufficient."""
    artifacts = section.get("artifacts")
    if not isinstance(artifacts, list) or not artifacts:
        issues.append(f"{issue_prefix}_artifacts_missing")
        return
    seen_paths: set[str] = set()
    for index, item in enumerate(artifacts):
        if not isinstance(item, Mapping):
            issues.append(f"{issue_prefix}_artifact_{index}_invalid")
            continue
        try:
            relative_path = str(item["path"])
            artifact = _safe_artifact_path(episode_dir, relative_path)
            actual = sha256_file(artifact)
            if actual != item.get("sha256"):
                issues.append(f"{issue_prefix}_artifact_{index}_hash_mismatch")
            if relative_path in seen_paths:
                issues.append(f"{issue_prefix}_artifact_{index}_duplicate")
            seen_paths.add(relative_path)
        except (KeyError, OSError, TypeError, ValueError) as exc:
            issues.append(f"{issue_prefix}_artifact_{index}_unreadable:{type(exc).__name__}")


def _validate_formal_collection_protocol(
    issues: list[str],
    *,
    episode_dir: Path,
    collection: Mapping[str, Any],
    schedule_record: Mapping[str, Any],
) -> None:
    """Require a hash-locked preregistered protocol before granting formal eligibility.

    A free-form ``status=formal_pilot`` string is not evidence that a collection was frozen.  The
    collector must copy the exact protocol into the episode and lock the full schedule plus the
    collector/runtime-validator implementations used for that batch.  The protocol also names the
    only scene/operator/realization/groups that it authorizes, preventing a narrow vertical slice
    from silently certifying unrelated schedule rows.
    """
    reference = collection.get("formal_protocol")
    if not isinstance(reference, Mapping):
        issues.append("formal_protocol_reference_missing")
        return
    try:
        protocol_path = _safe_artifact_path(episode_dir, reference.get("path"))
        protocol_sha256 = sha256_file(protocol_path)
        if protocol_sha256 != reference.get("sha256"):
            issues.append("formal_protocol_hash_mismatch")
        protocol = json.loads(protocol_path.read_text(encoding="utf-8"))
        if not isinstance(protocol, Mapping):
            raise TypeError("formal protocol must be an object")
    except (OSError, TypeError, ValueError, json.JSONDecodeError) as exc:
        issues.append(f"formal_protocol_unreadable:{type(exc).__name__}")
        return

    _append_issue(
        issues,
        protocol.get("schema_version") == FORMAL_PROTOCOL_SCHEMA_VERSION,
        "formal_protocol_schema_mismatch",
    )
    _append_issue(issues, protocol.get("status") == "frozen", "formal_protocol_not_frozen")
    for key in (
        "protocol_id",
        "schedule_sha256",
        "collector_sha256",
        "runtime_manifest_sha256",
    ):
        _append_issue(
            issues,
            isinstance(reference.get(key), str)
            and len(str(reference.get(key))) >= 16
            and reference.get(key) == protocol.get(key),
            f"formal_protocol_{key}_mismatch",
        )
    _append_issue(
        issues,
        collection.get("schedule_sha256") == protocol.get("schedule_sha256"),
        "formal_protocol_collection_schedule_hash_mismatch",
    )
    _append_issue(
        issues,
        collection.get("collector_sha256") == protocol.get("collector_sha256"),
        "formal_protocol_collection_collector_hash_mismatch",
    )
    _append_issue(
        issues,
        collection.get("runtime_manifest_sha256") == protocol.get("runtime_manifest_sha256"),
        "formal_protocol_collection_validator_hash_mismatch",
    )
    _append_issue(
        issues,
        protocol.get("benchmark_id") == schedule_record.get("benchmark_id"),
        "formal_protocol_benchmark_mismatch",
    )

    allowed = protocol.get("allowed")
    if not isinstance(allowed, Mapping):
        issues.append("formal_protocol_allowed_scope_missing")
        return
    scope_fields = {
        "counterfactual_group_ids": "counterfactual_group_id",
        "target_operators": "target_operator",
        "scene_families": "scene_family",
        "physical_realizations": "physical_realization",
        "geometry_profiles": "geometry_profile",
    }
    for protocol_key, schedule_key in scope_fields.items():
        values = allowed.get(protocol_key)
        _append_issue(
            issues,
            isinstance(values, list)
            and bool(values)
            and schedule_record.get(schedule_key) in values,
            f"formal_protocol_{schedule_key}_not_authorized",
        )


def build_collected_manifest(
    schedule_record: Mapping[str, Any],
    *,
    episode_dir: str | Path,
    rgb_frames: Sequence[Mapping[str, Any]],
    rgb_views: Mapping[str, Sequence[Mapping[str, Any]]] | None = None,
    proprio_path: str = "proprio.npz",
    telemetry_path: str = "privileged.jsonl",
    camera: Mapping[str, Any],
    operator_readback: Mapping[str, Any],
    scene_readback: Mapping[str, Any],
    appearance_readback: Mapping[str, Any],
    geometry_readback: Mapping[str, Any],
    collection: Mapping[str, Any],
) -> dict[str, Any]:
    """Build a hash-locked ``collected`` manifest from files already written by a collector.

    RGB entries require ``path`` and ``timestamp_s``.  All artifact paths are relative to the
    episode directory so manifests remain relocatable and path traversal can be rejected.
    """
    root = Path(episode_dir)

    def lock_frames(items: Sequence[Mapping[str, Any]]) -> list[dict[str, Any]]:
        locked: list[dict[str, Any]] = []
        for item in items:
            path = _safe_artifact_path(root, item.get("path"))
            locked.append(
                {
                    "path": str(item["path"]),
                    "timestamp_s": float(item["timestamp_s"]),
                    "sha256": sha256_file(path),
                }
            )
        return locked

    frame_records = lock_frames(rgb_frames)
    scheduled_views = schedule_record.get("appearance_views", [])
    primary_view_id = next(
        (
            str(view["appearance_view_id"])
            for view in scheduled_views
            if isinstance(view, Mapping) and view.get("is_primary") is True
        ),
        "primary",
    )
    supplied_views = dict(rgb_views or {primary_view_id: rgb_frames})
    locked_views = {str(view_id): lock_frames(items) for view_id, items in supplied_views.items()}
    proprio = _safe_artifact_path(root, proprio_path)
    telemetry = _safe_artifact_path(root, telemetry_path)
    return {
        "schema_version": RUNTIME_SCHEMA_VERSION,
        "episode_id": str(schedule_record["episode_id"]),
        "counterfactual_group_id": str(schedule_record["counterfactual_group_id"]),
        "schedule_record_sha256": schedule_record_sha256(schedule_record),
        "artifact_state": "collected",
        "evaluation_eligible": False,
        "artifacts": {
            "rgb_frames": frame_records,
            "primary_rgb_view_id": primary_view_id,
            "rgb_views": locked_views,
            "proprio": {
                "path": proprio_path,
                "sha256": sha256_file(proprio),
                "features_key": "features",
                "timestamps_key": "timestamp_s",
            },
            "telemetry": {
                "path": telemetry_path,
                "sha256": sha256_file(telemetry),
            },
        },
        "camera": copy.deepcopy(dict(camera)),
        "operator_readback": copy.deepcopy(dict(operator_readback)),
        "scene_readback": copy.deepcopy(dict(scene_readback)),
        "appearance_readback": copy.deepcopy(dict(appearance_readback)),
        "geometry_readback": copy.deepcopy(dict(geometry_readback)),
        "collection": copy.deepcopy(dict(collection)),
    }


def write_collected_manifest(
    manifest: Mapping[str, Any], episode_dir: str | Path, *, filename: str = "manifest.json"
) -> Path:
    """Persist a collected manifest after rejecting an invalid state claim."""
    if manifest.get("artifact_state") != "collected" or manifest.get("evaluation_eligible"):
        raise ValueError("a collector may write only collected, evaluation-ineligible manifests")
    destination = _safe_artifact_path(Path(episode_dir), filename)
    destination.parent.mkdir(parents=True, exist_ok=True)
    destination.write_text(
        json.dumps(manifest, indent=2, sort_keys=True, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )
    return destination


def _load_telemetry(path: Path) -> tuple[list[dict[str, Any]], str | None]:
    rows: list[dict[str, Any]] = []
    try:
        with path.open(encoding="utf-8") as stream:
            for line_number, line in enumerate(stream, 1):
                if not line.strip():
                    continue
                value = json.loads(line)
                if not isinstance(value, dict):
                    return [], f"telemetry_non_object_line_{line_number}"
                rows.append(value)
    except (OSError, json.JSONDecodeError) as exc:
        return [], f"telemetry_unreadable:{type(exc).__name__}"
    return rows, None


def _inspect_rgb_sequence(
    entries: object,
    *,
    root: Path,
    gates: Mapping[str, float | int],
    issue_prefix: str = "",
) -> tuple[list[str], dict[str, Any]]:
    """Re-hash and quality-check one timestamped RGB view sequence."""

    def code(value: str) -> str:
        return f"{issue_prefix}{value}"

    issues: list[str] = []
    if not isinstance(entries, list):
        entries = []
        issues.append(code("rgb_manifest_invalid"))
    rgb_times: list[float] = []
    rgb_hashes: list[str] = []
    rgb_dimensions: list[tuple[int, int]] = []
    highlight_fractions: list[float] = []
    luminance_stds: list[float] = []
    for index, entry in enumerate(entries):
        if not isinstance(entry, Mapping):
            issues.append(code(f"rgb_entry_{index}_invalid"))
            continue
        try:
            frame_path = _safe_artifact_path(root, entry.get("path"))
            actual_hash = sha256_file(frame_path)
            expected_hash = str(entry.get("sha256", ""))
            if actual_hash != expected_hash:
                issues.append(code(f"rgb_entry_{index}_hash_mismatch"))
            rgb_times.append(float(entry["timestamp_s"]))
            rgb_hashes.append(actual_hash)
            rgb_dimensions.append(_png_dimensions(frame_path))
            from PIL import Image

            pixels = np.asarray(Image.open(frame_path).convert("RGB"), dtype=np.float32)
            luminance = pixels.mean(axis=2)
            highlight_fractions.append(float(np.mean(luminance >= 235.0)))
            luminance_stds.append(float(np.std(luminance)))
        except (KeyError, OSError, TypeError, ValueError) as exc:
            issues.append(code(f"rgb_entry_{index}_unreadable:{type(exc).__name__}"))
    _append_issue(issues, len(entries) >= int(gates["min_rgb_frames"]), code("too_few_rgb_frames"))
    _append_issue(issues, _strictly_increasing(rgb_times), code("rgb_timestamps_not_monotonic"))
    _append_issue(
        issues,
        len(set(rgb_hashes)) >= int(gates["min_distinct_rgb_frames"]),
        code("insufficient_rgb_temporal_diversity"),
    )
    _append_issue(
        issues,
        bool(rgb_dimensions) and len(set(rgb_dimensions)) == 1,
        code("rgb_dimensions_inconsistent"),
    )
    if rgb_dimensions:
        width, height = rgb_dimensions[0]
        _append_issue(issues, width >= int(gates["min_rgb_width"]), code("rgb_width_too_small"))
        _append_issue(issues, height >= int(gates["min_rgb_height"]), code("rgb_height_too_small"))
    mean_highlight_fraction = float(np.mean(highlight_fractions)) if highlight_fractions else 1.0
    max_highlight_fraction = max(highlight_fractions, default=1.0)
    mean_luminance_std = float(np.mean(luminance_stds)) if luminance_stds else 0.0
    _append_issue(
        issues,
        mean_highlight_fraction <= float(gates["max_mean_highlight_fraction"]),
        code("rgb_mean_highlight_fraction_too_high"),
    )
    _append_issue(
        issues,
        max_highlight_fraction <= float(gates["max_frame_highlight_fraction"]),
        code("rgb_frame_highlight_fraction_too_high"),
    )
    _append_issue(
        issues,
        mean_luminance_std >= float(gates["min_mean_luminance_std"]),
        code("rgb_spatial_contrast_too_low"),
    )
    sequence_sha256 = hashlib.sha256("\n".join(rgb_hashes).encode("ascii")).hexdigest()
    return issues, {
        "entries": entries,
        "times": rgb_times,
        "hashes": rgb_hashes,
        "dimensions": rgb_dimensions,
        "mean_highlight_fraction": mean_highlight_fraction,
        "max_highlight_fraction": max_highlight_fraction,
        "mean_luminance_std": mean_luminance_std,
        "sequence_sha256": sequence_sha256,
    }


def _mean_aligned_rgb_l1(
    primary_entries: Sequence[Mapping[str, Any]],
    alternate_entries: Sequence[Mapping[str, Any]],
    *,
    root: Path,
) -> float:
    """Measure the visible effect of an appearance-only intervention at matched states."""
    if not primary_entries or len(primary_entries) != len(alternate_entries):
        return math.nan
    frame_differences: list[float] = []
    from PIL import Image

    for primary_entry, alternate_entry in zip(primary_entries, alternate_entries, strict=True):
        primary_path = _safe_artifact_path(root, primary_entry.get("path"))
        alternate_path = _safe_artifact_path(root, alternate_entry.get("path"))
        primary = np.asarray(Image.open(primary_path).convert("RGB"), dtype=np.float32)
        alternate = np.asarray(Image.open(alternate_path).convert("RGB"), dtype=np.float32)
        if primary.shape != alternate.shape:
            return math.nan
        frame_differences.append(float(np.mean(np.abs(primary - alternate)) / 255.0))
    return float(np.mean(frame_differences))


def validate_runtime_episode(
    schedule_record: Mapping[str, Any],
    *,
    episode_dir: str | Path,
    manifest: Mapping[str, Any] | None = None,
    manifest_path: str = "manifest.json",
    gate_overrides: Mapping[str, float | int] | None = None,
    write_validated: bool = False,
) -> RuntimeValidation:
    """Independently validate artifacts and return a promoted or rejected schedule record."""
    root = Path(episode_dir)
    gates = dict(DEFAULT_RUNTIME_GATES)
    if gate_overrides:
        gates.update(gate_overrides)
    abs_tol = float(gates["parameter_abs_tolerance"])
    rel_tol = float(gates["parameter_rel_tolerance"])
    if manifest is None:
        manifest_file = _safe_artifact_path(root, manifest_path)
        manifest = json.loads(manifest_file.read_text(encoding="utf-8"))
    runtime = dict(manifest)
    issues: list[str] = []

    _append_issue(
        issues, runtime.get("schema_version") == RUNTIME_SCHEMA_VERSION, "runtime_schema_mismatch"
    )
    _append_issue(
        issues,
        runtime.get("artifact_state") in {"collected", "validated", "validated_smoke"},
        "not_collected",
    )
    _append_issue(
        issues,
        runtime.get("episode_id") == schedule_record.get("episode_id"),
        "episode_id_mismatch",
    )
    _append_issue(
        issues,
        runtime.get("counterfactual_group_id") == schedule_record.get("counterfactual_group_id"),
        "counterfactual_group_mismatch",
    )
    _append_issue(
        issues,
        runtime.get("schedule_record_sha256") == schedule_record_sha256(schedule_record),
        "schedule_hash_mismatch",
    )

    artifacts = runtime.get("artifacts")
    if not isinstance(artifacts, Mapping):
        artifacts = {}
        issues.append("missing_artifacts")
    scheduled_views = schedule_record.get("appearance_views", [])
    expected_view_ids = [
        str(view["appearance_view_id"])
        for view in scheduled_views
        if isinstance(view, Mapping) and "appearance_view_id" in view
    ]
    expected_primary = next(
        (
            str(view["appearance_view_id"])
            for view in scheduled_views
            if isinstance(view, Mapping) and view.get("is_primary") is True
        ),
        "primary",
    )
    rgb_views = artifacts.get("rgb_views")
    if not isinstance(rgb_views, Mapping):
        rgb_views = {}
        issues.append("rgb_views_manifest_invalid")
    _append_issue(
        issues,
        set(rgb_views) == set(expected_view_ids),
        "rgb_view_ids_mismatch",
    )
    _append_issue(
        issues,
        artifacts.get("primary_rgb_view_id") == expected_primary,
        "primary_rgb_view_id_mismatch",
    )
    _append_issue(
        issues,
        artifacts.get("rgb_frames") == rgb_views.get(expected_primary),
        "primary_rgb_alias_mismatch",
    )
    rgb_view_metrics: dict[str, dict[str, Any]] = {}
    for view_id in expected_view_ids:
        prefix = "" if view_id == expected_primary else f"rgb_view_{view_id}_"
        view_issues, metrics = _inspect_rgb_sequence(
            rgb_views.get(view_id), root=root, gates=gates, issue_prefix=prefix
        )
        issues.extend(view_issues)
        rgb_view_metrics[view_id] = metrics
    primary_rgb = rgb_view_metrics.get(expected_primary, {})
    rgb_entries = list(primary_rgb.get("entries", []))
    rgb_times = list(primary_rgb.get("times", []))
    rgb_hashes = list(primary_rgb.get("hashes", []))
    rgb_dimensions = list(primary_rgb.get("dimensions", []))
    mean_highlight_fraction = float(primary_rgb.get("mean_highlight_fraction", 1.0))
    max_highlight_fraction = float(primary_rgb.get("max_highlight_fraction", 1.0))
    mean_luminance_std = float(primary_rgb.get("mean_luminance_std", 0.0))
    sequence_hashes = {str(metrics.get("sequence_sha256")) for metrics in rgb_view_metrics.values()}
    _append_issue(
        issues,
        len(sequence_hashes) >= int(gates["min_distinct_appearance_view_sequences"]),
        "insufficient_rgb_appearance_diversity",
    )
    appearance_pair_rgb_l1: dict[str, float | None] = {}
    for view_id, metrics in rgb_view_metrics.items():
        if view_id == expected_primary:
            continue
        _append_issue(
            issues,
            len(metrics.get("times", [])) == len(rgb_times)
            and all(
                math.isclose(float(actual), float(expected), abs_tol=1.0e-9, rel_tol=0.0)
                for actual, expected in zip(metrics.get("times", []), rgb_times, strict=True)
            ),
            f"rgb_view_{view_id}_timestamps_not_synchronized",
        )
        _append_issue(
            issues,
            metrics.get("dimensions") == rgb_dimensions,
            f"rgb_view_{view_id}_dimensions_not_synchronized",
        )
        try:
            pair_l1 = _mean_aligned_rgb_l1(
                rgb_entries,
                list(metrics.get("entries", [])),
                root=root,
            )
        except (OSError, TypeError, ValueError):
            pair_l1 = math.nan
        appearance_pair_rgb_l1[view_id] = pair_l1 if math.isfinite(pair_l1) else None
        _append_issue(
            issues,
            math.isfinite(pair_l1) and pair_l1 >= float(gates["min_mean_appearance_pair_rgb_l1"]),
            f"rgb_view_{view_id}_appearance_effect_too_small",
        )

    proprio_meta = artifacts.get("proprio", {})
    proprio_times: list[float] = []
    proprio_shape: tuple[int, ...] = ()
    if not isinstance(proprio_meta, Mapping):
        proprio_meta = {}
        issues.append("proprio_manifest_invalid")
    try:
        proprio_path = _safe_artifact_path(root, proprio_meta.get("path"))
        _append_issue(
            issues,
            sha256_file(proprio_path) == proprio_meta.get("sha256"),
            "proprio_hash_mismatch",
        )
        with np.load(proprio_path, allow_pickle=False) as archive:
            features = np.asarray(archive[str(proprio_meta.get("features_key", "features"))])
            timestamps = np.asarray(
                archive[str(proprio_meta.get("timestamps_key", "timestamp_s"))],
                dtype=np.float64,
            )
        proprio_shape = tuple(features.shape)
        proprio_times = [float(value) for value in timestamps.reshape(-1)]
        _append_issue(issues, features.ndim == 2, "proprio_not_matrix")
        if features.ndim == 2:
            _append_issue(
                issues,
                features.shape[0] >= int(gates["min_proprio_samples"]),
                "too_few_proprio_samples",
            )
            _append_issue(
                issues,
                features.shape[1] >= int(gates["min_proprio_features"]),
                "too_few_proprio_features",
            )
            _append_issue(
                issues, len(proprio_times) == features.shape[0], "proprio_timestamp_count_mismatch"
            )
        _append_issue(issues, bool(np.isfinite(features).all()), "proprio_nonfinite")
        _append_issue(
            issues, _strictly_increasing(proprio_times), "proprio_timestamps_not_monotonic"
        )
    except (KeyError, OSError, TypeError, ValueError) as exc:
        issues.append(f"proprio_unreadable:{type(exc).__name__}")

    telemetry_meta = artifacts.get("telemetry", {})
    telemetry_rows: list[dict[str, Any]] = []
    if not isinstance(telemetry_meta, Mapping):
        telemetry_meta = {}
        issues.append("telemetry_manifest_invalid")
    try:
        telemetry_path = _safe_artifact_path(root, telemetry_meta.get("path"))
        _append_issue(
            issues,
            sha256_file(telemetry_path) == telemetry_meta.get("sha256"),
            "telemetry_hash_mismatch",
        )
        telemetry_rows, telemetry_error = _load_telemetry(telemetry_path)
        if telemetry_error:
            issues.append(telemetry_error)
        _append_issue(issues, bool(telemetry_rows), "empty_telemetry")
        telemetry_times = [float(row["timestamp_s"]) for row in telemetry_rows]
        _append_issue(
            issues, _strictly_increasing(telemetry_times), "telemetry_timestamps_not_monotonic"
        )
    except (KeyError, OSError, TypeError, ValueError) as exc:
        issues.append(f"telemetry_unreadable:{type(exc).__name__}")

    max_skew = math.inf
    if rgb_times and proprio_times:
        max_skew = max(min(abs(rgb_t - prop_t) for prop_t in proprio_times) for rgb_t in rgb_times)
    _append_issue(
        issues,
        max_skew <= float(gates["max_rgb_proprio_skew_s"]),
        "rgb_proprio_clock_skew",
    )

    camera = runtime.get("camera")
    if not isinstance(camera, Mapping):
        camera = {}
        issues.append("camera_manifest_invalid")
    _append_issue(
        issues,
        camera.get("profile") == schedule_record.get("camera_profile"),
        "camera_profile_mismatch",
    )
    _append_issue(issues, camera.get("body_fixed") is True, "camera_not_body_fixed")
    _append_issue(
        issues,
        camera.get("pose_sync_method") in {"rigid_base_transform_each_step", "usd_rigid_mount"},
        "camera_pose_sync_unverified",
    )
    extrinsics = camera.get("base_to_camera_xyz_quat_xyzw")
    _append_issue(
        issues,
        isinstance(extrinsics, list)
        and len(extrinsics) == 7
        and all(isinstance(value, (int, float)) and math.isfinite(value) for value in extrinsics),
        "camera_extrinsics_invalid",
    )

    scene = runtime.get("scene_readback")
    if not isinstance(scene, Mapping):
        scene = {}
        issues.append("scene_readback_invalid")
    _append_issue(issues, scene.get("qa_passed") is True, "scene_local_qa_failed")
    for key in ("scene_family", "scene_source", "scene_seed"):
        _append_issue(
            issues,
            scene.get(key) == schedule_record.get(key),
            f"scene_{key}_mismatch",
        )
    _validate_hashed_readback_artifacts(
        issues,
        episode_dir=root,
        section=scene,
        issue_prefix="scene",
    )

    appearance = runtime.get("appearance_readback")
    if not isinstance(appearance, Mapping):
        appearance = {}
        issues.append("appearance_readback_invalid")
    _append_issue(issues, appearance.get("qa_passed") is True, "appearance_local_qa_failed")
    for key in ("appearance_id", "material_family", "surface_state"):
        _append_issue(
            issues,
            appearance.get(key) == schedule_record.get(key),
            f"appearance_{key}_mismatch",
        )
    for key in (
        "uv_scale",
        "uv_rotation_deg",
        "albedo_brightness_multiplier",
        "normal_strength",
        "roughness_multiplier",
    ):
        _append_issue(
            issues,
            _numbers_close(
                appearance.get(key),
                schedule_record.get(key),
                abs_tol=abs_tol,
                rel_tol=rel_tol,
            ),
            f"appearance_{key}_mismatch",
        )
    expected_offset = schedule_record.get("uv_offset")
    actual_offset = appearance.get("uv_offset")
    _append_issue(
        issues,
        isinstance(expected_offset, list)
        and isinstance(actual_offset, list)
        and len(expected_offset) == len(actual_offset) == 2
        and all(
            _numbers_close(actual, expected, abs_tol=abs_tol, rel_tol=rel_tol)
            for actual, expected in zip(actual_offset, expected_offset, strict=True)
        ),
        "appearance_uv_offset_mismatch",
    )
    _validate_hashed_readback_artifacts(
        issues,
        episode_dir=root,
        section=appearance,
        issue_prefix="appearance",
    )
    semantic_summary: dict[str, Any] = {}
    try:
        scene_summary_path = scene.get("semantic_summary_path")
        appearance_summary_path = appearance.get("semantic_summary_path")
        _append_issue(
            issues,
            isinstance(scene_summary_path, str) and scene_summary_path == appearance_summary_path,
            "semantic_summary_path_mismatch",
        )
        semantic_path = _safe_artifact_path(root, scene_summary_path)
        loaded_summary = json.loads(semantic_path.read_text(encoding="utf-8"))
        if not isinstance(loaded_summary, dict):
            raise TypeError("semantic summary is not an object")
        semantic_summary = loaded_summary
        _append_issue(
            issues,
            int(semantic_summary.get("n_frames", -1)) == len(rgb_entries),
            "semantic_summary_frame_count_mismatch",
        )
        _append_issue(
            issues,
            float(semantic_summary.get("mean_scheduled_appearance_pixel_fraction", -1.0))
            >= float(gates["min_scheduled_appearance_pixel_fraction"]),
            "scheduled_appearance_not_visible",
        )
        _append_issue(
            issues,
            float(semantic_summary.get("mean_scene_context_pixel_fraction", -1.0))
            >= float(gates["min_scene_context_pixel_fraction"]),
            "scene_context_not_visible",
        )
    except (OSError, TypeError, ValueError, json.JSONDecodeError) as exc:
        issues.append(f"semantic_summary_unreadable:{type(exc).__name__}")

    scheduled_view_by_id = {
        str(view["appearance_view_id"]): view
        for view in scheduled_views
        if isinstance(view, Mapping) and "appearance_view_id" in view
    }
    runtime_views = appearance.get("views")
    if not isinstance(runtime_views, Mapping):
        runtime_views = {}
        issues.append("appearance_views_readback_invalid")
    _append_issue(
        issues,
        set(runtime_views) == set(scheduled_view_by_id),
        "appearance_view_ids_mismatch",
    )
    appearance_view_summaries: dict[str, dict[str, Any]] = {}
    for view_id, expected_view in scheduled_view_by_id.items():
        actual_view = runtime_views.get(view_id)
        if not isinstance(actual_view, Mapping):
            issues.append(f"appearance_view_{view_id}_readback_invalid")
            continue
        _append_issue(
            issues,
            actual_view.get("qa_passed") is True,
            f"appearance_view_{view_id}_local_qa_failed",
        )
        for key in ("appearance_id", "material_family", "surface_state"):
            _append_issue(
                issues,
                actual_view.get(key) == expected_view.get(key),
                f"appearance_view_{view_id}_{key}_mismatch",
            )
        for key in (
            "uv_scale",
            "uv_rotation_deg",
            "albedo_brightness_multiplier",
            "normal_strength",
            "roughness_multiplier",
        ):
            _append_issue(
                issues,
                _numbers_close(
                    actual_view.get(key),
                    expected_view.get(key),
                    abs_tol=abs_tol,
                    rel_tol=rel_tol,
                ),
                f"appearance_view_{view_id}_{key}_mismatch",
            )
        actual_view_offset = actual_view.get("uv_offset")
        expected_view_offset = expected_view.get("uv_offset")
        _append_issue(
            issues,
            isinstance(actual_view_offset, list)
            and isinstance(expected_view_offset, list)
            and len(actual_view_offset) == len(expected_view_offset) == 2
            and all(
                _numbers_close(actual, expected, abs_tol=abs_tol, rel_tol=rel_tol)
                for actual, expected in zip(actual_view_offset, expected_view_offset, strict=True)
            ),
            f"appearance_view_{view_id}_uv_offset_mismatch",
        )
        _validate_hashed_readback_artifacts(
            issues,
            episode_dir=root,
            section=actual_view,
            issue_prefix=f"appearance_view_{view_id}",
        )
        try:
            summary_path = _safe_artifact_path(root, actual_view.get("semantic_summary_path"))
            loaded = json.loads(summary_path.read_text(encoding="utf-8"))
            if not isinstance(loaded, dict):
                raise TypeError("appearance-view semantic summary is not an object")
            appearance_view_summaries[view_id] = loaded
            _append_issue(
                issues,
                int(loaded.get("n_frames", -1))
                == len(rgb_view_metrics.get(view_id, {}).get("entries", [])),
                f"appearance_view_{view_id}_semantic_frame_count_mismatch",
            )
            _append_issue(
                issues,
                float(loaded.get("mean_scheduled_appearance_pixel_fraction", -1.0))
                >= float(gates["min_scheduled_appearance_pixel_fraction"]),
                f"appearance_view_{view_id}_not_visible",
            )
            _append_issue(
                issues,
                float(loaded.get("mean_scene_context_pixel_fraction", -1.0))
                >= float(gates["min_scene_context_pixel_fraction"]),
                f"appearance_view_{view_id}_scene_context_not_visible",
            )
        except (OSError, TypeError, ValueError, json.JSONDecodeError) as exc:
            issues.append(
                f"appearance_view_{view_id}_semantic_summary_unreadable:{type(exc).__name__}"
            )

    geometry = runtime.get("geometry_readback")
    if not isinstance(geometry, Mapping):
        geometry = {}
        issues.append("geometry_readback_invalid")
    _append_issue(issues, geometry.get("qa_passed") is True, "geometry_local_qa_failed")
    for key in ("geometry_id", "geometry_profile", "physical_realization"):
        _append_issue(
            issues,
            geometry.get(key) == schedule_record.get(key),
            f"geometry_{key}_mismatch",
        )

    readback = runtime.get("operator_readback")
    if not isinstance(readback, Mapping):
        readback = {}
        issues.append("operator_readback_invalid")
    expected_active = schedule_record.get("condition") == "anomaly"
    _append_issue(
        issues,
        readback.get("operator_id") == schedule_record.get("target_operator"),
        "operator_id_mismatch",
    )
    _append_issue(issues, readback.get("active") is expected_active, "operator_state_mismatch")
    _append_issue(issues, readback.get("qa_passed") is True, "operator_local_qa_failed")
    applied = readback.get("applied_parameters")
    expected_parameters = schedule_record.get("physics_parameters", {})
    if not isinstance(applied, Mapping):
        applied = {}
        issues.append("operator_parameters_missing")
    for key, expected in expected_parameters.items():
        if key not in applied:
            issues.append(f"operator_parameter_{key}_missing")
        elif not _numbers_close(applied[key], expected, abs_tol=abs_tol, rel_tol=rel_tol):
            issues.append(f"operator_parameter_{key}_mismatch")

    collection = runtime.get("collection")
    if not isinstance(collection, Mapping):
        collection = {}
        issues.append("collection_manifest_invalid")
    collection_status = collection.get("status")
    if collection_status == FORMAL_COLLECTION_STATUS:
        _validate_formal_collection_protocol(
            issues,
            episode_dir=root,
            collection=collection,
            schedule_record=schedule_record,
        )

    issues = list(dict.fromkeys(issues))
    passed = not issues
    evaluation_eligible = passed and collection_status == FORMAL_COLLECTION_STATUS
    promotion_blockers = [] if evaluation_eligible else ["collection_protocol_not_formal"]
    if not passed:
        promotion_blockers.insert(0, "runtime_quality_gates_failed")
    validation = {
        "passed": passed,
        "issues": issues,
        "evaluation_eligible": evaluation_eligible,
        "collection_status": collection_status,
        "required_collection_status": FORMAL_COLLECTION_STATUS,
        "promotion_blockers": promotion_blockers,
        "measured": {
            "rgb_frames": len(rgb_entries),
            "distinct_rgb_frames": len(set(rgb_hashes)),
            "appearance_views": len(rgb_view_metrics),
            "distinct_appearance_view_sequences": len(sequence_hashes),
            "appearance_view_sequence_sha256": {
                key: value["sequence_sha256"] for key, value in sorted(rgb_view_metrics.items())
            },
            "mean_appearance_pair_rgb_l1": appearance_pair_rgb_l1,
            "rgb_dimensions": list(rgb_dimensions[0]) if rgb_dimensions else None,
            "proprio_shape": list(proprio_shape),
            "telemetry_rows": len(telemetry_rows),
            "max_rgb_proprio_skew_s": max_skew if math.isfinite(max_skew) else None,
            "mean_rgb_highlight_fraction": mean_highlight_fraction,
            "max_rgb_frame_highlight_fraction": max_highlight_fraction,
            "mean_rgb_luminance_std": mean_luminance_std,
            "semantic_summary": semantic_summary,
            "appearance_view_semantic_summaries": appearance_view_summaries,
        },
        "thresholds": gates,
    }
    promoted_manifest = dict(runtime)
    if evaluation_eligible:
        artifact_state = "validated"
    elif passed:
        artifact_state = "validated_smoke"
    else:
        artifact_state = "collected"
    promoted_manifest["artifact_state"] = artifact_state
    promoted_manifest["evaluation_eligible"] = evaluation_eligible
    promoted_manifest["runtime_validation"] = validation
    promoted_record = dict(schedule_record)
    promoted_record["artifact_state"] = artifact_state
    promoted_record["evaluation_eligible"] = evaluation_eligible
    promoted_record["runtime_manifest_sha256"] = hashlib.sha256(
        json.dumps(promoted_manifest, sort_keys=True, separators=(",", ":")).encode("utf-8")
    ).hexdigest()
    promoted_record["runtime_quality_gates"] = validation
    if write_validated:
        destination = _safe_artifact_path(root, manifest_path)
        destination.write_text(
            json.dumps(promoted_manifest, indent=2, sort_keys=True, ensure_ascii=False) + "\n",
            encoding="utf-8",
        )
    return RuntimeValidation(
        passed=passed,
        issues=tuple(issues),
        gates=validation,
        validated_record=promoted_record,
        manifest=promoted_manifest,
    )


def audit_runtime_corpus(
    schedule_records: Sequence[Mapping[str, Any]],
    *,
    corpus_root: str | Path,
    gate_overrides: Mapping[str, float | int] | None = None,
    write_validated: bool = False,
    require_complete: bool = False,
) -> RuntimeCorpusAudit:
    """Audit runtime completeness without conflating missing episodes with failed episodes.

    A partially collected corpus may pass this audit when every artifact that *does* exist is
    valid.  ``require_complete=True`` is the publication-freeze gate: every schedule row and both
    members of every counterfactual pair must then be validated.
    """
    root = Path(corpus_root)
    state_counts: Counter[str] = Counter()
    issue_counts: Counter[str] = Counter()
    invalid_episode_ids: list[str] = []
    promoted_records: list[dict[str, Any]] = []
    pair_states: dict[str, list[bool]] = defaultdict(list)
    for raw_record in schedule_records:
        schedule = dict(raw_record)
        required_outputs = schedule.get("required_outputs", {})
        relative_manifest = (
            required_outputs.get("episode_manifest")
            if isinstance(required_outputs, Mapping)
            else None
        )
        if not isinstance(relative_manifest, str):
            state_counts["invalid_schedule"] += 1
            issue_counts["missing_episode_manifest_path"] += 1
            invalid_episode_ids.append(str(schedule.get("episode_id", "<unknown>")))
            pair_states[str(schedule.get("counterfactual_group_id", "<unknown>"))].append(False)
            continue
        try:
            manifest_file = _safe_artifact_path(root, relative_manifest)
        except ValueError:
            state_counts["invalid_schedule"] += 1
            issue_counts["episode_manifest_path_escape"] += 1
            invalid_episode_ids.append(str(schedule.get("episode_id", "<unknown>")))
            pair_states[str(schedule.get("counterfactual_group_id", "<unknown>"))].append(False)
            continue
        if not manifest_file.exists():
            state_counts["planned"] += 1
            pair_states[str(schedule["counterfactual_group_id"])].append(False)
            continue
        result = validate_runtime_episode(
            schedule,
            episode_dir=manifest_file.parent,
            manifest_path=manifest_file.name,
            gate_overrides=gate_overrides,
            write_validated=write_validated,
        )
        evaluation_eligible = bool(result.validated_record.get("evaluation_eligible"))
        if evaluation_eligible:
            state = "validated"
        elif result.passed:
            state = "validated_smoke"
        else:
            state = "collected_invalid"
        state_counts[state] += 1
        pair_states[str(schedule["counterfactual_group_id"])].append(evaluation_eligible)
        if evaluation_eligible:
            promoted_records.append(result.validated_record)
        elif not result.passed:
            invalid_episode_ids.append(str(schedule["episode_id"]))
            issue_counts.update(result.issues)

    complete_pairs = sum(len(states) == 2 and all(states) for states in pair_states.values())
    incomplete_pairs = len(pair_states) - complete_pairs
    total = len(schedule_records)
    validated = state_counts["validated"]
    validated_smoke = state_counts["validated_smoke"]
    complete = validated == total and complete_pairs * 2 == total
    issues: list[str] = []
    if state_counts["invalid_schedule"]:
        issues.append("invalid_schedule_records")
    if state_counts["collected_invalid"]:
        issues.append("invalid_collected_episodes")
    if require_complete and not complete:
        issues.append("runtime_corpus_incomplete")
    summary = {
        "schema_version": RUNTIME_SCHEMA_VERSION,
        "passed": not issues,
        "publication_freeze_ready": complete and not issues,
        "require_complete": require_complete,
        "issues": issues,
        "scheduled_records": total,
        "state_counts": dict(sorted(state_counts.items())),
        "artifact_qa_passed_records": validated + validated_smoke,
        "validated_smoke_records": validated_smoke,
        "evaluation_eligible_records": validated,
        "evaluation_eligible_fraction": validated / total if total else 0.0,
        "counterfactual_pairs": len(pair_states),
        "complete_counterfactual_pairs": complete_pairs,
        "incomplete_counterfactual_pairs": incomplete_pairs,
        "runtime_issue_counts": dict(sorted(issue_counts.items())),
        "invalid_episode_ids": invalid_episode_ids[:100],
    }
    return RuntimeCorpusAudit(
        passed=not issues,
        summary=summary,
        records=tuple(promoted_records),
    )
