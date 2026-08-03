"""Event-aligned model snapshots from evaluation-eligible realistic episodes.

The key experimental rule is that the anomalous episode supplies the privileged *alignment
event*, while its nominal counterfactual reuses exactly the same decision time.  Appearance
views then reuse both timestamps and proprioception.  Therefore neither the counterfactual nor
the texture-swap comparison is confounded by choosing a different temporal slice.

Every O1--O11 adapter below uses a predeclared privileged mechanism event.  Unknown operators
fail closed; a heuristic such as "largest proprio change" would leak the outcome into sample
selection and is not an acceptable publication protocol.
"""

from __future__ import annotations

import hashlib
import json
import math
from collections import defaultdict
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

import numpy as np
from PIL import Image

from kino_vla.data.runtime_manifest import schedule_record_sha256, sha256_file

SNAPSHOT_SCHEMA_VERSION = "kinofail.realistic-snapshot.v1"
AUDIT_SCHEMA_VERSION = "kinofail.realistic-snapshot-audit.v1"
TEMPORAL_INFEASIBILITY_PREFIXES = (
    "requested RGB timestamps did not resolve to distinct frames",
    "RGB target skew ",
    "insufficient proprio samples before the paired decision time",
    "proprio window does not end at the paired decision time",
    "proprio window exceeds the frozen duration",
)


def _is_temporal_infeasibility(error: ValueError) -> bool:
    message = str(error)
    return any(message.startswith(prefix) for prefix in TEMPORAL_INFEASIBILITY_PREFIXES)


def _read_json(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise TypeError(f"{path} must contain a JSON object")
    return value


def _read_jsonl(path: Path) -> list[dict[str, Any]]:
    return [
        json.loads(line)
        for line in path.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]


def _resolve(repo_root: Path, value: str | Path) -> Path:
    path = Path(value)
    return path if path.is_absolute() else repo_root / path


def _canonical_sha256(value: Mapping[str, Any]) -> str:
    payload = json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=False)
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


def _operator_payload(row: Mapping[str, Any]) -> Mapping[str, Any]:
    value = row.get("operator")
    if isinstance(value, Mapping):
        return value
    legacy_adhesion = row.get("adhesion")
    return legacy_adhesion if isinstance(legacy_adhesion, Mapping) else row


def _independently_certified_operator(schedule: Mapping[str, Any], manifest: Mapping[str, Any]) -> bool:
    """Recover known validator false negatives from immutable physical readback only.

    O4 is owned by the Isaac backend rather than returned as a Python operator object.  The scale-v4
    exposure wrapper consequently marked it false even when backend telemetry proves that forces
    were applied.  This exception is deliberately narrow; it cannot rescue an inactive operator.
    """

    if str(schedule.get("condition")) != "anomaly":
        return manifest.get("operator_readback", {}).get("qa_passed") is True
    if str(schedule.get("target_operator")) != "O4_tether":
        return manifest.get("operator_readback", {}).get("qa_passed") is True
    telemetry = manifest.get("operator_readback", {}).get("telemetry", {})
    feet = [
        value
        for value in telemetry.get("feet", [])
        if isinstance(value, Mapping)
    ]
    peak_force_n = max(
        [float(value.get("max_force_n", 0.0)) for value in feet] + [0.0]
    )
    return (
        int(telemetry.get("total_attachment_cycles", 0)) > 0
        and (
            float(telemetry.get("total_applied_force_n", 0.0)) > 0.0
            or peak_force_n > 0.0
        )
        and float(telemetry.get("total_tangential_work_j", 0.0)) > 0.0
    )


def _runtime_is_accepted(
    schedule: Mapping[str, Any],
    manifest: Mapping[str, Any],
    *,
    allowed_suffixes: Sequence[str],
) -> bool:
    validation = manifest.get("runtime_validation", {})
    if validation.get("passed") is True:
        return True
    issues = [str(value) for value in validation.get("issues", [])]
    if not issues:
        return False
    certified = _independently_certified_operator(schedule, manifest)
    for issue in issues:
        if allowed_suffixes and issue.endswith(tuple(allowed_suffixes)):
            continue
        if issue == "operator_local_qa_failed" and certified:
            continue
        return False
    return True


def _first_matching(rows: Sequence[Mapping[str, Any]], predicate) -> float:
    for row in rows:
        if predicate(_operator_payload(row)):
            timestamp = float(row["timestamp_s"])
            if math.isfinite(timestamp):
                return timestamp
    raise ValueError("telemetry has no measured operator event")


def _first_region_exposure(rows: Sequence[Mapping[str, Any]]) -> float:
    return _first_matching(rows, lambda value: int(value.get("scale_region_exposure_steps", 0)) > 0)


def _first_o2_sinkage(rows: Sequence[Mapping[str, Any]]) -> float:
    return _first_matching(
        rows,
        lambda value: any(
            int(foot.get("contact_steps", 0)) > 0 or float(foot.get("max_sinkage_m", 0.0)) > 0.0
            for foot in value.get("feet", []) if isinstance(foot, Mapping)
        ),
    )


def _first_o3_collapse(rows: Sequence[Mapping[str, Any]]) -> float:
    return _first_matching(
        rows,
        lambda value: any(region.get("collapsed") is True for region in value.get("regions", []) if isinstance(region, Mapping)),
    )


def _first_o4_attachment(rows: Sequence[Mapping[str, Any]]) -> float:
    return _first_matching(
        rows,
        lambda value: int(value.get("total_attachment_cycles", 0)) > 0
        or any(
            foot.get("attached") is True
            or foot.get("event") == "attached"
            or int(foot.get("attachment_count", 0)) > 0
            for foot in value.get("feet", []) if isinstance(foot, Mapping)
        ),
    )


def _first_o5_payload(rows: Sequence[Mapping[str, Any]]) -> float:
    return _first_matching(rows, lambda value: float(value.get("payload_kg", 0.0)) > 0.0)


def _first_o6_push(rows: Sequence[Mapping[str, Any]]) -> float:
    return _first_matching(rows, lambda value: int(value.get("applied_steps", 0)) > 0)


def _first_o10_decay(rows: Sequence[Mapping[str, Any]]) -> float:
    return _first_matching(rows, lambda value: float(value.get("effort_scale", 1.0)) < 0.999)


def _first_o11_bias(rows: Sequence[Mapping[str, Any]]) -> float:
    return _first_matching(
        rows,
        lambda value: value.get("raw_pipeline_active") is True
        and abs(float(value.get("total_raw_tilt_error_rad", 0.0))) > 1.0e-6,
    )


_EVENT_ADAPTERS = {
    "O1_mu_field": _first_region_exposure,
    "O2_compliance": _first_o2_sinkage,
    "O3_collapse": _first_o3_collapse,
    "O4_tether": _first_o4_attachment,
    "O5_payload": _first_o5_payload,
    "O6_push": _first_o6_push,
    "O7_visual_remap": _first_region_exposure,
    "O8_invisible_collider": _first_region_exposure,
    "O9_high_centering": _first_region_exposure,
    "O10_effort_decay": _first_o10_decay,
    "O11_obs_bias": _first_o11_bias,
}


def _nearest_indices(
    available: np.ndarray, targets: np.ndarray, *, max_skew_s: float
) -> tuple[np.ndarray, np.ndarray]:
    if available.ndim != 1 or not len(available):
        raise ValueError("timestamp sequence must be a non-empty vector")
    indices = np.asarray([int(np.argmin(np.abs(available - target))) for target in targets])
    selected = available[indices]
    skews = np.abs(selected - targets)
    if len(set(indices.tolist())) != len(indices):
        raise ValueError("requested RGB timestamps did not resolve to distinct frames")
    if np.any(skews > max_skew_s):
        raise ValueError(f"RGB target skew {float(skews.max()):.6f}s exceeds gate")
    return indices, skews


def _load_rgb(episode_dir: Path, entries: Sequence[Mapping[str, Any]], indices: np.ndarray) -> np.ndarray:
    frames: list[np.ndarray] = []
    for index in indices.tolist():
        entry = entries[index]
        path = episode_dir / str(entry["path"])
        if sha256_file(path) != entry.get("sha256"):
            raise ValueError(f"selected RGB artifact hash mismatch: {path}")
        with Image.open(path) as image:
            frames.append(np.asarray(image.convert("RGB"), dtype=np.uint8))
    if len({frame.shape for frame in frames}) != 1:
        raise ValueError("selected RGB frames have inconsistent dimensions")
    return np.stack(frames, axis=0)


def _proprio_window(
    episode_dir: Path,
    manifest: Mapping[str, Any],
    *,
    decision_time_s: float,
    window_s: float,
    sample_count: int,
    max_end_skew_s: float,
) -> tuple[np.ndarray, np.ndarray, list[str]]:
    artifact = manifest["artifacts"]["proprio"]
    path = episode_dir / str(artifact["path"])
    if sha256_file(path) != artifact.get("sha256"):
        raise ValueError(f"proprio artifact hash mismatch: {path}")
    with np.load(path, allow_pickle=False) as archive:
        features = np.asarray(archive[str(artifact["features_key"])], dtype=np.float32)
        timestamps = np.asarray(archive[str(artifact["timestamps_key"])], dtype=np.float64)
        names = (
            [str(value) for value in archive["feature_names"].tolist()]
            if "feature_names" in archive.files
            else [f"feature_{index}" for index in range(features.shape[1])]
        )
    if features.ndim != 2 or timestamps.shape != (features.shape[0],):
        raise ValueError("invalid proprio array shapes")
    candidates = np.flatnonzero(
        (timestamps > decision_time_s - window_s - 1.0e-9)
        & (timestamps <= decision_time_s + max_end_skew_s)
    )
    if len(candidates) < sample_count:
        raise ValueError("insufficient proprio samples before the paired decision time")
    chosen = candidates[-sample_count:]
    selected_timestamps = timestamps[chosen]
    if abs(float(selected_timestamps[-1]) - decision_time_s) > max_end_skew_s:
        raise ValueError("proprio window does not end at the paired decision time")
    if float(selected_timestamps[0]) < decision_time_s - window_s - 1.0e-9:
        raise ValueError("proprio window exceeds the frozen duration")
    return features[chosen], selected_timestamps, names


def _episode_dir(corpus_root: Path, schedule: Mapping[str, Any]) -> Path:
    return (corpus_root / str(schedule["required_outputs"]["episode_manifest"])).parent


def build_event_aligned_snapshots(
    protocol: Mapping[str, Any], *, repo_root: str | Path
) -> tuple[list[dict[str, Any]], dict[str, np.ndarray], dict[str, Any]]:
    """Extract frozen snapshots and return records, arrays, and a fail-closed audit."""

    root = Path(repo_root)
    schedule_path = _resolve(root, str(protocol["source_schedule"]))
    corpus_root = _resolve(root, str(protocol["source_corpus_root"]))
    overlay_by_pair: dict[str, tuple[Path, str]] = {}
    for overlay in protocol.get("source_corpus_overlays", []):
        overlay_root = _resolve(root, str(overlay["corpus_root"]))
        overlay_protocol_id = str(overlay["required_collection_protocol_id"])
        for pair_id in overlay["counterfactual_group_ids"]:
            key = str(pair_id)
            if key in overlay_by_pair:
                raise ValueError(f"duplicate corpus overlay for {key}")
            overlay_by_pair[key] = (overlay_root, overlay_protocol_id)
    schedules = _read_jsonl(schedule_path)
    expected_schedule_sha256 = protocol.get("source_schedule_sha256")
    if expected_schedule_sha256 is not None and sha256_file(schedule_path) != expected_schedule_sha256:
        raise ValueError("formal snapshot protocol schedule hash mismatch")
    allowed = set(protocol["selection"]["operators_with_admitted_event_adapter"])
    schedules = [row for row in schedules if str(row["target_operator"]) in allowed]
    by_pair: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in schedules:
        pair_id = str(row["counterfactual_group_id"])
        selected_root, selected_protocol_id = overlay_by_pair.get(
            pair_id,
            (
                corpus_root,
                str(protocol["selection"].get("required_collection_protocol_id", "")),
            ),
        )
        manifest_path = selected_root / str(row["required_outputs"]["episode_manifest"])
        if not manifest_path.exists():
            continue
        manifest = _read_json(manifest_path)
        if protocol["selection"]["require_evaluation_eligible"] and not (
            manifest.get("evaluation_eligible") is True
            and manifest.get("runtime_validation", {}).get("passed") is True
        ):
            continue
        if protocol["selection"].get("require_runtime_validation_passed", False):
            allowed_suffixes = [
                str(value)
                for value in protocol["selection"].get(
                    "allow_nonblocking_runtime_issue_suffixes", []
                )
            ]
            if not _runtime_is_accepted(row, manifest, allowed_suffixes=allowed_suffixes):
                continue
        required_protocol = selected_protocol_id or None
        if required_protocol is not None and (
            manifest.get("collection", {}).get("formal_protocol", {}).get("protocol_id")
            != required_protocol
        ):
            continue
        if schedule_record_sha256(row) != manifest.get("schedule_record_sha256"):
            raise ValueError(f"schedule/runtime hash mismatch for {row['episode_id']}")
        row = dict(row)
        row["_manifest"] = manifest
        row["_corpus_root"] = str(selected_root)
        by_pair[pair_id].append(row)

    temporal = protocol["temporal_alignment"]
    records: list[dict[str, Any]] = []
    arrays: dict[str, np.ndarray] = {}
    pair_audits: list[dict[str, Any]] = []
    skipped_incomplete = 0
    temporal_exclusions: list[dict[str, Any]] = []
    for pair_id, pair in sorted(by_pair.items()):
        conditions = {str(row["condition"]): row for row in pair}
        if set(conditions) != {"anomaly", "nominal_counterfactual"} or len(pair) != 2:
            skipped_incomplete += 1
            continue
        anomaly = conditions["anomaly"]
        nominal = conditions["nominal_counterfactual"]
        operator = str(anomaly["target_operator"])
        if operator not in _EVENT_ADAPTERS:
            raise ValueError(f"no admitted event adapter for {operator}")
        anomaly_dir = _episode_dir(Path(str(anomaly["_corpus_root"])), anomaly)
        telemetry_path = anomaly_dir / str(anomaly["_manifest"]["artifacts"]["telemetry"]["path"])
        if sha256_file(telemetry_path) != anomaly["_manifest"]["artifacts"]["telemetry"].get(
            "sha256"
        ):
            raise ValueError(f"telemetry artifact hash mismatch: {telemetry_path}")
        event_time_s = _EVENT_ADAPTERS[operator](_read_jsonl(telemetry_path))
        delay_by_operator = temporal.get("decision_delay_s_by_operator", {})
        decision_time_s = max(
            event_time_s + float(delay_by_operator.get(operator, temporal["decision_delay_s"])),
            float(temporal.get("minimum_decision_time_s", 0.0)),
        )
        target_rgb_times = decision_time_s + np.asarray(
            temporal["rgb_offsets_from_decision_s"], dtype=np.float64
        )

        # Preflight the complete pair before appending any records or arrays.  A short
        # condition may make the anomaly-anchored decision window physically unavailable
        # even when both episodes and all artifacts are structurally valid.  Repeating the
        # terminal frame or moving the frozen decision time would bias the model input, so
        # the only admitted recovery is a pair-level exclusion recorded in the audit.
        temporal_failure: dict[str, Any] | None = None
        for schedule in (anomaly, nominal):
            manifest = schedule["_manifest"]
            episode_dir = _episode_dir(Path(str(schedule["_corpus_root"])), schedule)
            views = manifest["artifacts"]["rgb_views"]
            primary_id = str(manifest["artifacts"]["primary_rgb_view_id"])
            available_times = np.asarray(
                [
                    float(entry["timestamp_s"])
                    for entry in views[primary_id]
                ],
                dtype=np.float64,
            )
            try:
                _nearest_indices(
                    available_times,
                    target_rgb_times,
                    max_skew_s=float(temporal["max_rgb_target_skew_s"]),
                )
                _proprio_window(
                    episode_dir,
                    manifest,
                    decision_time_s=decision_time_s,
                    window_s=float(temporal["proprio_window_s"]),
                    sample_count=int(temporal["proprio_samples"]),
                    max_end_skew_s=float(temporal["max_proprio_end_skew_s"]),
                )
            except ValueError as error:
                if not _is_temporal_infeasibility(error):
                    raise
                temporal_failure = {
                    "counterfactual_group_id": pair_id,
                    "operator": operator,
                    "severity_id": str(anomaly["severity_id"]),
                    "scene_family": str(anomaly["scene_family"]),
                    "failing_condition": str(schedule["condition"]),
                    "event_time_s": event_time_s,
                    "decision_time_s": decision_time_s,
                    "target_rgb_times_s": target_rgb_times.tolist(),
                    "available_rgb_frame_count": int(len(available_times)),
                    "available_rgb_start_s": (
                        float(available_times[0]) if len(available_times) else None
                    ),
                    "available_rgb_end_s": (
                        float(available_times[-1]) if len(available_times) else None
                    ),
                    "reason": str(error),
                }
                break
        if temporal_failure is not None:
            if (
                protocol["selection"].get("on_temporal_alignment_failure")
                != "exclude_complete_pair_and_audit"
            ):
                raise ValueError(
                    "paired temporal alignment is infeasible and the snapshot protocol "
                    f"does not admit pair-level exclusion: {pair_id}"
                )
            temporal_exclusions.append(temporal_failure)
            continue

        pair_selected_times: dict[str, list[float]] = {}
        pair_proprio_times: dict[str, list[float]] = {}
        n_views = None
        for schedule in (anomaly, nominal):
            manifest = schedule["_manifest"]
            episode_dir = _episode_dir(Path(str(schedule["_corpus_root"])), schedule)
            views = manifest["artifacts"]["rgb_views"]
            primary_id = str(manifest["artifacts"]["primary_rgb_view_id"])
            primary_entries = views[primary_id]
            available_times = np.asarray(
                [float(entry["timestamp_s"]) for entry in primary_entries], dtype=np.float64
            )
            indices, skews = _nearest_indices(
                available_times,
                target_rgb_times,
                max_skew_s=float(temporal["max_rgb_target_skew_s"]),
            )
            selected_times = available_times[indices]
            pair_selected_times[str(schedule["condition"])] = selected_times.tolist()
            proprio, proprio_times, feature_names = _proprio_window(
                episode_dir,
                manifest,
                decision_time_s=decision_time_s,
                window_s=float(temporal["proprio_window_s"]),
                sample_count=int(temporal["proprio_samples"]),
                max_end_skew_s=float(temporal["max_proprio_end_skew_s"]),
            )
            pair_proprio_times[str(schedule["condition"])] = proprio_times.tolist()
            if n_views is None:
                n_views = len(views)
            elif n_views != len(views):
                raise ValueError(f"appearance-view count differs within pair {pair_id}")

            reference_timestamps = None
            for view_id, entries in sorted(views.items()):
                view_timestamps = np.asarray(
                    [float(entry["timestamp_s"]) for entry in entries], dtype=np.float64
                )
                if reference_timestamps is None:
                    reference_timestamps = view_timestamps
                elif not np.array_equal(view_timestamps, reference_timestamps):
                    raise ValueError(
                        f"appearance views do not share exact timestamps in {schedule['episode_id']}"
                    )
                rgb = _load_rgb(episode_dir, entries, indices)
                sample_id = f"{schedule['episode_id']}__{view_id}"
                arrays[f"{sample_id}__rgb"] = rgb
                arrays[f"{sample_id}__proprio"] = proprio
                arrays[f"{sample_id}__rgb_timestamp_s"] = selected_times
                arrays[f"{sample_id}__proprio_timestamp_s"] = proprio_times
                appearance = manifest["appearance_readback"]["views"][view_id]
                records.append(
                    {
                        "schema_version": SNAPSHOT_SCHEMA_VERSION,
                        "sample_id": sample_id,
                        "physical_episode_id": schedule["episode_id"],
                        "counterfactual_group_id": pair_id,
                        "statistical_unit_id": pair_id,
                        "appearance_intervention_id": view_id,
                        "appearance_views_are_independent_samples": False,
                        "condition": schedule["condition"],
                        "target_operator": operator,
                        "attribution_category": schedule["attribution_category"],
                        "domain": schedule["domain"],
                        "scene_family": schedule["scene_family"],
                        "camera_profile": schedule["camera_profile"],
                        "severity_id": schedule["severity_id"],
                        "event_time_s_from_anomaly_privileged_telemetry": event_time_s,
                        "decision_time_s": decision_time_s,
                        "rgb_timestamp_s": selected_times.tolist(),
                        "rgb_target_skew_s": skews.tolist(),
                        "proprio_timestamp_s": proprio_times.tolist(),
                        "proprio_feature_names": feature_names,
                        "rgb_shape": list(rgb.shape),
                        "rgb_dtype": str(rgb.dtype),
                        "proprio_shape": list(proprio.shape),
                        "appearance_id": appearance["appearance_id"],
                        "material_family": appearance["material_family"],
                        "source_manifest_sha256": sha256_file(episode_dir / "manifest.json"),
                    }
                )

        timestamps_identical = np.array_equal(
            np.asarray(pair_selected_times["anomaly"]),
            np.asarray(pair_selected_times["nominal_counterfactual"]),
        )
        proprio_timestamps_identical = np.array_equal(
            np.asarray(pair_proprio_times["anomaly"]),
            np.asarray(pair_proprio_times["nominal_counterfactual"]),
        )
        if not timestamps_identical or not proprio_timestamps_identical:
            raise ValueError(f"counterfactual temporal alignment failed for {pair_id}")
        pair_audits.append(
            {
                "counterfactual_group_id": pair_id,
                "operator": operator,
                "event_time_s": event_time_s,
                "decision_time_s": decision_time_s,
                "rgb_timestamps_identical_across_conditions": timestamps_identical,
                "proprio_timestamps_identical_across_conditions": proprio_timestamps_identical,
                "appearance_views_per_physical_episode": n_views,
                "physical_episode_count": 2,
                "independent_counterfactual_pair_count": 1,
            }
        )

    expected_records = sum(2 * int(row["appearance_views_per_physical_episode"]) for row in pair_audits)
    checks = {
        "protocol_mode_matches": protocol.get("development_only")
        is bool(protocol.get("publication_guard", {}).get("may_satisfy_realistic_a0_a7") is False),
        "all_records_from_complete_pairs": len(records) == expected_records,
        "all_pair_rgb_times_identical": all(
            row["rgb_timestamps_identical_across_conditions"] for row in pair_audits
        ),
        "all_pair_proprio_times_identical": all(
            row["proprio_timestamps_identical_across_conditions"] for row in pair_audits
        ),
        "appearance_views_not_counted_as_independent": all(
            row["appearance_views_are_independent_samples"] is False for row in records
        ),
        "all_selected_or_excluded_pairs_accounted_for": (
            len(pair_audits) + len(temporal_exclusions) + skipped_incomplete
            == len(by_pair)
        ),
        "temporal_exclusions_are_pair_level_and_audited": (
            not temporal_exclusions
            or (
                protocol["selection"].get("on_temporal_alignment_failure")
                == "exclude_complete_pair_and_audit"
                and len(
                    {
                        row["counterfactual_group_id"]
                        for row in temporal_exclusions
                    }
                )
                == len(temporal_exclusions)
            )
        ),
        "has_at_least_one_pair": bool(pair_audits),
    }
    audit = {
        "schema_version": AUDIT_SCHEMA_VERSION,
        "protocol_id": protocol["protocol_id"],
        "protocol_sha256": _canonical_sha256(protocol),
        "passed": all(checks.values()),
        "checks": checks,
        "counts": {
            "snapshot_records": len(records),
            "physical_episodes": len({row["physical_episode_id"] for row in records}),
            "independent_counterfactual_pairs": len(pair_audits),
            "appearance_intervention_sequences": len(records),
            "operators": len({row["target_operator"] for row in records}),
            "domains": len({row["domain"] for row in records}),
            "scene_families": len({row["scene_family"] for row in records}),
            "temporally_infeasible_counterfactual_pairs": len(
                temporal_exclusions
            ),
        },
        "skipped_incomplete_pairs": skipped_incomplete,
        "temporal_alignment_exclusions": temporal_exclusions,
        "event_adapter_coverage": {
            "implemented": sorted(_EVENT_ADAPTERS),
            "required_for_publication": 11,
            "ready_for_publication": len(_EVENT_ADAPTERS) == 11,
        },
        "pair_audits": pair_audits,
        "publication_guard": protocol["publication_guard"],
        "interpretation": (
            "Passing validates the hash-bound formal model-input extraction protocol. Experiment "
            "claims still require new training, inference, predictions and A1-A7 statistics."
            if protocol.get("development_only") is False
            else "Passing validates snapshot extraction only and cannot promote a development pilot."
        ),
    }
    return records, arrays, audit


def write_snapshot_bundle(
    records: Sequence[Mapping[str, Any]],
    arrays: Mapping[str, np.ndarray],
    audit: Mapping[str, Any],
    *,
    output_dir: str | Path,
) -> dict[str, str]:
    output = Path(output_dir)
    output.mkdir(parents=True, exist_ok=True)
    records_path = output / "snapshot_records.jsonl"
    arrays_path = output / "snapshots.npz"
    audit_path = output / "extraction_audit.json"
    records_path.write_text(
        "".join(json.dumps(dict(row), sort_keys=True) + "\n" for row in records),
        encoding="utf-8",
    )
    np.savez_compressed(arrays_path, **arrays)
    audit_path.write_text(json.dumps(dict(audit), indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return {
        "snapshot_records": sha256_file(records_path),
        "snapshots": sha256_file(arrays_path),
        "extraction_audit": sha256_file(audit_path),
    }
