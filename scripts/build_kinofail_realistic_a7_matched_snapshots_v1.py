#!/usr/bin/env python3
"""Build frozen observable snapshots for the two realistic A7 matched collections."""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
from collections import defaultdict
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from kino_vla.data.realistic_snapshots import (  # noqa: E402
    _load_rgb,
    _nearest_indices,
    _proprio_window,
    write_snapshot_bundle,
)
from kino_vla.data.runtime_manifest import (  # noqa: E402
    schedule_record_sha256,
    sha256_file,
)


def _json(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise TypeError(path)
    return value


def _jsonl(path: Path) -> list[dict[str, Any]]:
    return [
        json.loads(line)
        for line in path.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _episode_dir(corpus_root: Path, row: Mapping[str, Any]) -> Path:
    return (corpus_root / str(row["required_outputs"]["episode_manifest"])).parent


def _runtime_accepted(manifest: Mapping[str, Any]) -> bool:
    validation = manifest.get("runtime_validation", {})
    if validation.get("passed") is True:
        return True
    issues = [str(value) for value in validation.get("issues", [])]
    allowed = ("appearance_effect_too_small", "rgb_spatial_contrast_too_low")
    return bool(issues) and all(issue.endswith(allowed) for issue in issues)


def _complete_pairs(
    schedule_path: Path,
    corpus_root: Path,
    *,
    required_protocol_id: str,
) -> list[tuple[str, dict[str, dict[str, Any]]]]:
    grouped: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for source in _jsonl(schedule_path):
        row = dict(source)
        episode_dir = _episode_dir(corpus_root, row)
        manifest_path = episode_dir / "manifest.json"
        if not manifest_path.is_file():
            continue
        manifest = _json(manifest_path)
        if schedule_record_sha256(row) != manifest.get("schedule_record_sha256"):
            raise RuntimeError(f"A7 schedule/runtime mismatch: {row['episode_id']}")
        formal = manifest.get("collection", {}).get("formal_protocol", {})
        if formal.get("protocol_id") != required_protocol_id:
            raise RuntimeError(f"A7 formal protocol mismatch: {row['episode_id']}")
        if not _runtime_accepted(manifest):
            raise RuntimeError(
                f"A7 evidence-corrupting runtime failure: {row['episode_id']} "
                f"{manifest.get('runtime_validation', {}).get('issues', [])}"
            )
        row["_manifest"] = manifest
        row["_episode_dir"] = str(episode_dir)
        grouped[str(row["counterfactual_group_id"])].append(row)

    pairs = []
    for pair_id, rows in sorted(grouped.items()):
        by_condition = {str(row["condition"]): row for row in rows}
        if set(by_condition) != {"anomaly", "nominal_counterfactual"} or len(rows) != 2:
            raise RuntimeError(f"incomplete A7 pair: {pair_id}")
        pairs.append((pair_id, by_condition))
    return pairs


def _operator_payload(row: Mapping[str, Any]) -> Mapping[str, Any]:
    value = row.get("operator")
    return value if isinstance(value, Mapping) else row


def _first_event(
    rows: Sequence[Mapping[str, Any]], *, mode: str
) -> float:
    for row in rows:
        value = _operator_payload(row)
        if mode == "legacy_surrogate":
            matched = (
                value.get("implementation") == "legacy_base_wrench"
                and value.get("ever_inside") is True
                and float(value.get("peak_resistance_n", 0.0)) > 0.0
            )
        elif mode == "payload":
            matched = float(value.get("payload_kg", 0.0)) > 0.0
        else:
            raise ValueError(mode)
        if matched:
            return float(row["timestamp_s"])
    raise RuntimeError(f"A7 {mode} telemetry has no mechanism event")


def _proprio_for_pair(
    by_condition: Mapping[str, dict[str, Any]],
    *,
    decision_time_s: float,
    temporal: Mapping[str, Any],
) -> dict[str, tuple[np.ndarray, np.ndarray, list[str]]]:
    output = {}
    for condition, row in by_condition.items():
        output[condition] = _proprio_window(
            Path(row["_episode_dir"]),
            row["_manifest"],
            decision_time_s=decision_time_s,
            window_s=float(temporal["proprio_window_s"]),
            sample_count=int(temporal["proprio_samples"]),
            max_end_skew_s=float(temporal["max_proprio_end_skew_s"]),
        )
    left = output["anomaly"][1]
    right = output["nominal_counterfactual"][1]
    if not np.array_equal(left, right):
        raise RuntimeError("A7 paired proprio timestamps differ")
    return output


def _record(
    row: Mapping[str, Any],
    *,
    sample_id: str,
    appearance_intervention_id: str,
    decision_time_s: float,
    event_time_s: float,
    rgb: np.ndarray,
    rgb_times: np.ndarray,
    rgb_skews: np.ndarray,
    proprio: np.ndarray,
    proprio_times: np.ndarray,
    feature_names: list[str],
    material_family: str,
    source_manifest_sha256: str,
) -> dict[str, Any]:
    return {
        "schema_version": "kinofail.realistic-a7-matched-snapshot.v1",
        "sample_id": sample_id,
        "physical_episode_id": row["episode_id"],
        "counterfactual_group_id": row["counterfactual_group_id"],
        "statistical_unit_id": row["counterfactual_group_id"],
        "appearance_intervention_id": appearance_intervention_id,
        "appearance_views_are_independent_samples": False,
        "condition": row["condition"],
        "target_operator": row["target_operator"],
        "attribution_category": row["attribution_category"],
        "domain": row["domain"],
        "scene_family": row["scene_family"],
        "camera_profile": row["camera_profile"],
        "severity_id": row["severity_id"],
        "event_time_s_from_anomaly_privileged_telemetry": event_time_s,
        "decision_time_s": decision_time_s,
        "rgb_timestamp_s": rgb_times.tolist(),
        "rgb_target_skew_s": rgb_skews.tolist(),
        "proprio_timestamp_s": proprio_times.tolist(),
        "proprio_feature_names": feature_names,
        "rgb_shape": list(rgb.shape),
        "rgb_dtype": str(rgb.dtype),
        "proprio_shape": list(proprio.shape),
        "material_family": material_family,
        "source_manifest_sha256": source_manifest_sha256,
    }


def _build_legacy(
    config: Mapping[str, Any]
) -> tuple[list[dict[str, Any]], dict[str, np.ndarray], dict[str, Any]]:
    section = config["legacy_surrogate"]
    schedule = ROOT / section["schedule"]
    corpus = ROOT / section["corpus_root"]
    pairs = _complete_pairs(
        schedule, corpus, required_protocol_id=section["required_collection_protocol_id"]
    )
    temporal = config["temporal_alignment"]
    records: list[dict[str, Any]] = []
    arrays: dict[str, np.ndarray] = {}
    pair_audits = []
    for pair_id, by_condition in pairs:
        anomaly = by_condition["anomaly"]
        telemetry = _jsonl(
            Path(anomaly["_episode_dir"])
            / str(anomaly["_manifest"]["artifacts"]["telemetry"]["path"])
        )
        event_time = _first_event(telemetry, mode="legacy_surrogate")
        decision_time = max(
            event_time + float(temporal["decision_delay_s"]),
            float(temporal["minimum_decision_time_s"]),
        )
        targets = decision_time + np.asarray(
            temporal["rgb_offsets_from_decision_s"], dtype=np.float64
        )
        proprio_by_condition = _proprio_for_pair(
            by_condition, decision_time_s=decision_time, temporal=temporal
        )
        selected_by_condition = {}
        for condition, row in by_condition.items():
            episode_dir = Path(row["_episode_dir"])
            manifest = row["_manifest"]
            views = manifest["artifacts"]["rgb_views"]
            primary = str(manifest["artifacts"]["primary_rgb_view_id"])
            available = np.asarray(
                [float(entry["timestamp_s"]) for entry in views[primary]],
                dtype=np.float64,
            )
            indices, skews = _nearest_indices(
                available,
                targets,
                max_skew_s=float(temporal["max_rgb_target_skew_s"]),
            )
            selected = available[indices]
            selected_by_condition[condition] = selected
            proprio, proprio_times, names = proprio_by_condition[condition]
            for view_id, entries in sorted(views.items()):
                view_times = np.asarray(
                    [float(entry["timestamp_s"]) for entry in entries],
                    dtype=np.float64,
                )
                if not np.array_equal(view_times, available):
                    raise RuntimeError("A7 surrogate appearance timestamps differ")
                rgb = _load_rgb(episode_dir, entries, indices)
                sample_id = f"{row['episode_id']}__{view_id}"
                arrays[f"{sample_id}__rgb"] = rgb
                arrays[f"{sample_id}__proprio"] = proprio
                arrays[f"{sample_id}__rgb_timestamp_s"] = selected
                arrays[f"{sample_id}__proprio_timestamp_s"] = proprio_times
                appearance = manifest["appearance_readback"]["views"][view_id]
                records.append(
                    _record(
                        row,
                        sample_id=sample_id,
                        appearance_intervention_id=view_id,
                        decision_time_s=decision_time,
                        event_time_s=event_time,
                        rgb=rgb,
                        rgb_times=selected,
                        rgb_skews=skews,
                        proprio=proprio,
                        proprio_times=proprio_times,
                        feature_names=names,
                        material_family=str(appearance["material_family"]),
                        source_manifest_sha256=sha256_file(
                            episode_dir / "manifest.json"
                        ),
                    )
                )
        if not np.array_equal(
            selected_by_condition["anomaly"],
            selected_by_condition["nominal_counterfactual"],
        ):
            raise RuntimeError("A7 surrogate paired RGB timestamps differ")
        pair_audits.append(
            {
                "counterfactual_group_id": pair_id,
                "event_time_s": event_time,
                "decision_time_s": decision_time,
                "appearance_views_per_episode": 3,
            }
        )
    checks = {
        "all_18_pairs_present": len(pairs) == 18,
        "all_36_episodes_present": len({row["physical_episode_id"] for row in records})
        == 36,
        "three_views_per_episode": len(records) == 108,
        "nine_scenes_present": len({row["scene_family"] for row in records}) == 9,
        "both_operators_present": {
            row["target_operator"] for row in records
        } == {"O2_compliance", "O4_tether"},
    }
    return records, arrays, {
        "schema_version": "kinofail.realistic-a7-matched-snapshot-audit.v1",
        "protocol_id": config["protocol_id"],
        "arm": "legacy_surrogate",
        "passed": all(checks.values()),
        "checks": checks,
        "counts": {
            "snapshot_records": len(records),
            "physical_episodes": len({row["physical_episode_id"] for row in records}),
            "independent_counterfactual_pairs": len(pairs),
            "scene_families": len({row["scene_family"] for row in records}),
        },
        "pair_audits": pair_audits,
        "publication_guard": config["publication_guard"],
    }


def _build_visual(
    config: Mapping[str, Any]
) -> tuple[list[dict[str, Any]], dict[str, np.ndarray], dict[str, Any]]:
    section = config["visual_replay"]
    schedule = ROOT / section["schedule"]
    corpus = ROOT / section["corpus_root"]
    pairs = _complete_pairs(
        schedule, corpus, required_protocol_id=section["required_collection_protocol_id"]
    )
    temporal = config["temporal_alignment"]
    expected_arms = list(section["render_arms"])
    records: list[dict[str, Any]] = []
    arrays: dict[str, np.ndarray] = {}
    pair_audits = []
    for pair_id, by_condition in pairs:
        anomaly = by_condition["anomaly"]
        telemetry = _jsonl(
            Path(anomaly["_episode_dir"])
            / str(anomaly["_manifest"]["artifacts"]["telemetry"]["path"])
        )
        event_time = _first_event(telemetry, mode="payload")
        decision_time = max(
            event_time + float(temporal["decision_delay_s_by_operator"]["O5_payload"]),
            float(temporal["minimum_decision_time_s"]),
        )
        targets = decision_time + np.asarray(
            temporal["rgb_offsets_from_decision_s"], dtype=np.float64
        )
        proprio_by_condition = _proprio_for_pair(
            by_condition, decision_time_s=decision_time, temporal=temporal
        )
        selected_by_condition = {}
        for condition, row in by_condition.items():
            episode_dir = Path(row["_episode_dir"])
            manifest = row["_manifest"]
            readback = manifest.get("a7_matched_render_readback", {})
            if readback.get("render_arms") != expected_arms:
                raise RuntimeError("A7 visual render-arm contract mismatch")
            render_manifest = readback.get("render_manifest", {})
            render_manifest_path = episode_dir / str(render_manifest.get("path", ""))
            if (
                not render_manifest_path.is_file()
                or sha256_file(render_manifest_path) != render_manifest.get("sha256")
            ):
                raise RuntimeError("A7 visual render manifest hash mismatch")
            reference = readback["arms"][expected_arms[0]]["frames"]
            available = np.asarray(
                [float(entry["timestamp_s"]) for entry in reference],
                dtype=np.float64,
            )
            indices, skews = _nearest_indices(
                available,
                targets,
                max_skew_s=float(temporal["max_rgb_target_skew_s"]),
            )
            selected = available[indices]
            selected_by_condition[condition] = selected
            proprio, proprio_times, names = proprio_by_condition[condition]
            for arm in expected_arms:
                entries = readback["arms"][arm]["frames"]
                arm_times = np.asarray(
                    [float(entry["timestamp_s"]) for entry in entries],
                    dtype=np.float64,
                )
                if not np.array_equal(arm_times, available):
                    raise RuntimeError("A7 visual arm timestamps differ")
                rgb = _load_rgb(episode_dir, entries, indices)
                sample_id = f"{row['episode_id']}__{arm}"
                arrays[f"{sample_id}__rgb"] = rgb
                arrays[f"{sample_id}__proprio"] = proprio
                arrays[f"{sample_id}__rgb_timestamp_s"] = selected
                arrays[f"{sample_id}__proprio_timestamp_s"] = proprio_times
                records.append(
                    _record(
                        row,
                        sample_id=sample_id,
                        appearance_intervention_id=arm,
                        decision_time_s=decision_time,
                        event_time_s=event_time,
                        rgb=rgb,
                        rgb_times=selected,
                        rgb_skews=skews,
                        proprio=proprio,
                        proprio_times=proprio_times,
                        feature_names=names,
                        material_family=str(row["material_family"]),
                        source_manifest_sha256=sha256_file(
                            episode_dir / "manifest.json"
                        ),
                    )
                )
        if not np.array_equal(
            selected_by_condition["anomaly"],
            selected_by_condition["nominal_counterfactual"],
        ):
            raise RuntimeError("A7 visual paired RGB timestamps differ")
        pair_audits.append(
            {
                "counterfactual_group_id": pair_id,
                "event_time_s": event_time,
                "decision_time_s": decision_time,
                "render_arms_per_episode": len(expected_arms),
            }
        )
    checks = {
        "all_9_pairs_present": len(pairs) == 9,
        "all_18_episodes_present": len({row["physical_episode_id"] for row in records})
        == 18,
        "four_arms_per_episode": len(records) == 72,
        "nine_scenes_present": len({row["scene_family"] for row in records}) == 9,
        "only_o5_present": {row["target_operator"] for row in records}
        == {"O5_payload"},
    }
    return records, arrays, {
        "schema_version": "kinofail.realistic-a7-matched-snapshot-audit.v1",
        "protocol_id": config["protocol_id"],
        "arm": "visual_replay",
        "passed": all(checks.values()),
        "checks": checks,
        "counts": {
            "snapshot_records": len(records),
            "physical_episodes": len({row["physical_episode_id"] for row in records}),
            "independent_counterfactual_pairs": len(pairs),
            "scene_families": len({row["scene_family"] for row in records}),
        },
        "pair_audits": pair_audits,
        "publication_guard": config["publication_guard"],
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--config",
        type=Path,
        default=ROOT
        / "configs/eval/kinofail_realistic_a7_matched_snapshot_v1.json",
    )
    args = parser.parse_args()
    config_path = args.config.resolve()
    config = _json(config_path)
    bound = {
        "design": ROOT / config["design"],
        "legacy_schedule": ROOT / config["legacy_surrogate"]["schedule"],
        "legacy_protocol": ROOT / config["legacy_surrogate"]["collection_protocol"],
        "visual_schedule": ROOT / config["visual_replay"]["schedule"],
        "visual_protocol": ROOT / config["visual_replay"]["collection_protocol"],
        "builder": Path(__file__).resolve(),
    }
    mismatches = {
        key: {"expected": config["input_sha256"][key], "actual": _sha256(path)}
        for key, path in bound.items()
        if not path.is_file() or _sha256(path) != config["input_sha256"][key]
    }
    if mismatches:
        raise RuntimeError(f"A7 matched snapshot frozen-input mismatch: {mismatches}")

    results = {}
    for name, builder in (
        ("legacy_surrogate", _build_legacy),
        ("visual_replay", _build_visual),
    ):
        records, arrays, audit = builder(config)
        output_dir = ROOT / config[name]["snapshot_output_dir"]
        hashes = write_snapshot_bundle(
            records, arrays, audit, output_dir=output_dir
        )
        results[name] = {
            "output_dir": str(output_dir.relative_to(ROOT)),
            "passed": audit["passed"],
            "counts": audit["counts"],
            "sha256": hashes,
        }
    print(json.dumps(results, indent=2, sort_keys=True))
    return 0 if all(row["passed"] for row in results.values()) else 2


if __name__ == "__main__":
    raise SystemExit(main())
