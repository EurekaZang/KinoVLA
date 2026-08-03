#!/usr/bin/env python3
"""Bind 21 untouched F0 scenes and 9 new model-blind admissions to v2 slots."""

from __future__ import annotations

import argparse
import hashlib
import json
from copy import deepcopy
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from scripts import build_kinofail_confirmatory_scene_registry_v1 as legacy


ROOT = Path(__file__).resolve().parents[1]
LEGACY_BUILDER = ROOT / "scripts/build_kinofail_confirmatory_scene_registry_v1.py"
EXPECTED_LEGACY_BUILDER_SHA256 = (
    "1a118387ee63562a4ed719621bfee2ee9e70b6703c21c94fb8620fb0fd7b21f0"
)
EXPECTED_ARCHIVE_REGISTRY_SHA256 = (
    "52c237354621358e1e4ff29a6df925bd35eae949ec6636ced58a5b79eaf2fd44"
)


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _json(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise TypeError(path)
    return value


def _display(path: Path) -> str:
    resolved = path.resolve()
    try:
        return str(resolved.relative_to(ROOT))
    except ValueError:
        return str(resolved)


def _resolve(value: str) -> Path:
    path = Path(value)
    return path.resolve() if path.is_absolute() else (ROOT / path).resolve()


def _resolve_archived(value: str) -> Path:
    path = Path(value)
    try:
        relative = path.resolve().relative_to(
            (ROOT / "outputs/kinofail_confirmatory_v1").resolve()
        )
    except ValueError:
        return _resolve(value)
    return (
        ROOT
        / "outputs/kinofail_confirmatory_invalid_acquisition_pilot_v1_20260725"
        / relative
    ).resolve()


def _verify_file(path: Path, expected: str) -> None:
    if not path.is_file() or _sha256(path) != expected:
        raise RuntimeError(f"evidence hash mismatch: {path}")


def _verify_f0(path: Path) -> dict[str, Any]:
    f0 = _json(path)
    sidecar = path.with_name("freeze_manifest.sha256")
    if (
        not sidecar.is_file()
        or _sha256(path) != sidecar.read_text(encoding="utf-8").split()[0]
        or f0.get("status") != "frozen_before_new_scene_generation"
        or f0.get("confirmatory") is not True
        or f0.get("new_data_available_at_freeze") is not False
    ):
        raise RuntimeError("invalid reconfirmation F0")
    for row in f0.get("frozen_files", []):
        frozen = _resolve(str(row["path"]))
        if (
            not frozen.is_file()
            or frozen.stat().st_size != int(row["bytes"])
            or _sha256(frozen) != str(row["sha256"])
        ):
            raise RuntimeError(f"F0-frozen file changed: {frozen}")
    return f0


def _candidate_terminal(
    config: dict[str, Any], candidate: dict[str, Any]
) -> Path:
    compiled = ROOT / str(config["indoor_pipeline"]["compiled_root"])
    return compiled / str(candidate["scene_id"]) / "terminal_scene_admission.json"


def _new_material_id(
    config: dict[str, Any], candidate: dict[str, Any]
) -> str:
    domain = str(candidate["domain"])
    domain_stream = [
        row
        for row in config["indoor_candidates"]
        if str(row["domain"]) == domain
    ]
    index = next(
        position
        for position, row in enumerate(domain_stream)
        if str(row["scene_id"]) == str(candidate["scene_id"])
    )
    return f"confirm_v2_{domain}_pbr_{index % 10:02d}"


def _attempt_new(
    config: dict[str, Any], candidate: dict[str, Any], index: int
) -> tuple[dict[str, Any], dict[str, Any] | None]:
    terminal_path = _candidate_terminal(config, candidate)
    if not terminal_path.is_file():
        raise FileNotFoundError(
            "new-scene attempts must be a gap-free prefix: "
            f"{terminal_path}"
        )
    terminal = _json(terminal_path)
    expected_material = _new_material_id(config, candidate)
    valid_identity = (
        terminal.get("scene_id") == candidate["scene_id"]
        and terminal.get("model_blind") is True
        and terminal.get("model_or_endpoint_files_read") is False
        and terminal.get("material_id") == expected_material
    )
    if not valid_identity:
        raise RuntimeError(f"invalid model-blind terminal audit: {terminal_path}")
    attempt = {
        "candidate_index": index,
        "candidate": candidate,
        "terminal_audit": _display(terminal_path),
        "terminal_audit_sha256": _sha256(terminal_path),
        "admitted": terminal.get("admitted") is True,
        "failure_stage": terminal.get("failure_stage"),
        "origin": "new_gap_free_candidate",
    }
    return attempt, terminal if attempt["admitted"] else None


def _verify_archive_row(row: dict[str, Any]) -> None:
    for path_key, hash_key in (
        ("episode_usd", "episode_sha256"),
        ("compiled_audit", "compiled_audit_sha256"),
        ("terminal_scene_admission", "terminal_scene_admission_sha256"),
    ):
        _verify_file(_resolve_archived(str(row[path_key])), str(row[hash_key]))
    for source in row.get("geometry_hash_contract", {}).get("sources", []):
        _verify_file(
            _resolve_archived(str(source["path"])), str(source["sha256"])
        )
    geometry_source = row.get("geometry_hash_contract", {}).get("source")
    geometry_source_hash = row.get("geometry_hash_contract", {}).get(
        "source_sha256"
    )
    if geometry_source and geometry_source_hash:
        _verify_file(
            _resolve_archived(str(geometry_source)),
            str(geometry_source_hash),
        )


def _rebased_archive_row(row: dict[str, Any]) -> dict[str, Any]:
    result = deepcopy(row)
    for key in (
        "episode_usd",
        "compiled_audit",
        "terminal_scene_admission",
    ):
        result[key] = _display(_resolve_archived(str(result[key])))
    contract = result.get("geometry_hash_contract", {})
    for source in contract.get("sources", []):
        source["path"] = _display(_resolve_archived(str(source["path"])))
    if contract.get("source"):
        contract["source"] = _display(
            _resolve_archived(str(contract["source"]))
        )
    return result


def _new_scene_row(
    *,
    candidate: dict[str, Any],
    terminal_path: Path,
    terminal: dict[str, Any],
    formal_scene_id: str,
    material_ids: list[str],
) -> dict[str, Any]:
    episode = Path(str(terminal["episode_usd"])).resolve()
    evidence = dict(terminal["evidence"])
    compiled = Path(str(evidence["terrain_audit"]["path"])).resolve()
    if (
        not episode.is_file()
        or not compiled.is_file()
        or evidence["terrain_audit"].get("passed") is not True
        or _sha256(compiled) != str(evidence["terrain_audit"]["sha256"])
    ):
        raise RuntimeError(f"new scene evidence invalid: {candidate['scene_id']}")
    geometry_hash, geometry_contract = legacy._geometry_hash(
        domain=str(candidate["domain"]),
        candidate={
            key: value
            for key, value in candidate.items()
            if key not in {"origin", "archive_registry_index"}
        },
        evidence=evidence,
    )
    return {
        "scene_id": formal_scene_id,
        "candidate_id": str(candidate["scene_id"]),
        "source_scene_id": str(candidate["scene_id"]),
        "domain": str(candidate["domain"]),
        "split": "revised_confirmatory",
        "source": "EmbodiedGen-v2-new-RoomGen-source",
        "episode_usd": _display(episode),
        "episode_sha256": _sha256(episode),
        "compiled_audit": _display(compiled),
        "compiled_audit_sha256": _sha256(compiled),
        "geometry_hash": geometry_hash,
        "geometry_hash_contract": geometry_contract,
        "terminal_scene_admission": _display(terminal_path),
        "terminal_scene_admission_sha256": _sha256(terminal_path),
        "material_ids": material_ids,
        "material_assignment": "contrast-offset-five-two-context",
        "selected_by_model_blind_prefix_rule": True,
        "origin": "new_after_reconfirmation_f0",
        "runtime_seed": int(candidate["runtime_seed"]),
        "source_scene_seed": int(candidate["source_seed"]),
        "room_type": str(candidate["room_type"]),
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--f0-manifest", type=Path, required=True)
    parser.add_argument("--candidate-config", type=Path, required=True)
    parser.add_argument("--material-lock", type=Path, required=True)
    parser.add_argument("--archive-registry", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    out = args.out.resolve()
    if out.exists():
        raise FileExistsError(out)
    if _sha256(LEGACY_BUILDER) != EXPECTED_LEGACY_BUILDER_SHA256:
        raise RuntimeError("legacy scene-registry implementation hash mismatch")

    f0_path = args.f0_manifest.resolve()
    config_path = args.candidate_config.resolve()
    lock_path = args.material_lock.resolve()
    archive_path = args.archive_registry.resolve()
    f0 = _verify_f0(f0_path)
    config = _json(config_path)
    lock = _json(lock_path)
    archive = _json(archive_path)
    if _sha256(archive_path) != EXPECTED_ARCHIVE_REGISTRY_SHA256:
        raise RuntimeError("archive registry hash mismatch")
    frozen = {
        str(row["path"]): str(row["sha256"])
        for row in f0.get("frozen_files", [])
    }
    for path in (config_path, Path(__file__).resolve(), archive_path):
        if frozen.get(_display(path)) != _sha256(path):
            raise RuntimeError(f"F0 does not bind registry input: {path}")

    materials = [dict(row) for row in lock.get("materials", [])]
    expected_material_ids = {
        str(row["material_id"]) for row in f0["design"]["materials"]
    }
    if (
        lock.get("audit", {}).get("passed") is not True
        or len(materials) != 30
        or {str(row["id"]) for row in materials} != expected_material_ids
    ):
        raise RuntimeError("invalid reconfirmation material lock")
    archive_by_source = {
        str(row["source_scene_id"]): dict(row)
        for row in archive.get("scenes", [])
    }

    rows: list[dict[str, Any]] = []
    attrition: dict[str, Any] = {}
    for domain in ("life", "production", "wild"):
        stream = [
            dict(row)
            for row in f0["design"]["scene_candidate_streams"][domain]
        ]
        attempts: list[dict[str, Any]] = []
        selected: list[tuple[dict[str, Any], dict[str, Any] | None]] = []
        for index, candidate in enumerate(stream):
            origin = str(candidate["origin"])
            if origin == "original_f0_unexposed":
                archived = archive_by_source.get(str(candidate["scene_id"]))
                if archived is None:
                    raise RuntimeError(f"missing archived source: {candidate}")
                _verify_archive_row(archived)
                terminal_path = _resolve(
                    str(archived["terminal_scene_admission"])
                )
                attempt = {
                    "candidate_index": index,
                    "candidate": candidate,
                    "terminal_audit": _display(terminal_path),
                    "terminal_audit_sha256": _sha256(terminal_path),
                    "admitted": True,
                    "failure_stage": None,
                    "origin": origin,
                }
                terminal = None
            elif origin == "new_gap_free_candidate":
                attempt, terminal = _attempt_new(config, candidate, index)
            else:
                raise RuntimeError(f"unsupported candidate origin: {origin}")
            attempts.append(attempt)
            if attempt["admitted"]:
                selected.append((candidate, terminal))
            if len(selected) == 10:
                break
        if len(selected) != 10:
            raise RuntimeError(f"{domain}: ten admissions not realized")
        attrition[domain] = {
            "candidate_stream_size": len(stream),
            "attempted_prefix": len(attempts),
            "admitted_in_attempted_prefix": sum(
                bool(row["admitted"]) for row in attempts
            ),
            "selected": 10,
            "attempts": attempts,
        }
        for slot, (candidate, terminal) in enumerate(selected):
            formal = f"confirm_v2_{domain}_scene_{slot:02d}"
            material_ids = [
                f"confirm_v2_{domain}_pbr_{slot:02d}",
                f"confirm_v2_{domain}_pbr_{(slot + 5) % 10:02d}",
            ]
            if candidate["origin"] == "original_f0_unexposed":
                row = _rebased_archive_row(
                    archive_by_source[str(candidate["scene_id"])]
                )
                row.update(
                    {
                        "scene_id": formal,
                        "candidate_id": str(candidate["scene_id"]),
                        "source_scene_id": str(candidate["scene_id"]),
                        "split": "revised_confirmatory",
                        "material_ids": material_ids,
                        "selected_by_model_blind_prefix_rule": True,
                        "origin": "original_f0_unexposed_no_runtime_or_model_outcome",
                        "original_formal_scene_id": candidate[
                            "original_slot_id"
                        ],
                    }
                )
            else:
                if terminal is None:
                    raise RuntimeError("admitted new scene lacks terminal audit")
                row = _new_scene_row(
                    candidate=candidate,
                    terminal_path=_candidate_terminal(config, candidate),
                    terminal=terminal,
                    formal_scene_id=formal,
                    material_ids=material_ids,
                )
            rows.append(row)

    if len(rows) != 30 or len({row["geometry_hash"] for row in rows}) != 30:
        raise RuntimeError("scene count or geometry uniqueness drift")
    excluded = {
        str(row["source_scene_id"])
        for row in _json(
            ROOT
            / "configs/data/kinofail_reconfirmation_exposure_exclusions_v2.json"
        )["records"]
    }
    if {str(row["source_scene_id"]) for row in rows} & excluded:
        raise RuntimeError("invalid-pilot exposed scene entered reconfirmation")
    planned = {str(row["scene_id"]) for row in f0["design"]["scenes"]}
    if {str(row["scene_id"]) for row in rows} != planned:
        raise RuntimeError("realized v2 slots differ from F0")

    registry = {
        "schema_version": "kinofail.unified-reconfirmation-scene-registry.v2",
        "created_utc": datetime.now(UTC).isoformat(),
        "status": "model_blind_scene_admission_complete_pending_f1",
        "selection_uses_model_predictions": False,
        "scene_count": 30,
        "scenes": rows,
        "attrition": attrition,
        "counts": {
            "original_f0_unexposed": 21,
            "new_after_reconfirmation_f0": 9,
            "invalid_pilot_exposed": 0,
        },
        "source_sha256": {
            "f0_manifest": _sha256(f0_path),
            "candidate_config": _sha256(config_path),
            "material_lock": _sha256(lock_path),
            "archive_registry": _sha256(archive_path),
            "builder": _sha256(Path(__file__).resolve()),
        },
    }
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(
        json.dumps(registry, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    print(
        json.dumps(
            {
                "passed": True,
                "scene_count": 30,
                "registry": str(out),
                "sha256": _sha256(out),
            },
            indent=2,
            sort_keys=True,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
