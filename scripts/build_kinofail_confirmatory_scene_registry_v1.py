#!/usr/bin/env python3
"""Select the preregistered model-blind scene prefix and bind 30 F0 slots.

Life and production use the first ten successful candidates in their frozen
ordered streams.  Wild uses all ten fixed candidates.  Selection reads only
terminal scene-admission audits; model predictions and endpoint artifacts are
neither accepted as inputs nor searched on disk.
"""

from __future__ import annotations

import argparse
import hashlib
import json
from datetime import UTC, datetime
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parents[1]


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


def _canonical_sha256(value: Any) -> str:
    return hashlib.sha256(
        json.dumps(
            value,
            sort_keys=True,
            separators=(",", ":"),
            ensure_ascii=False,
        ).encode("utf-8")
    ).hexdigest()


def _geometry_hash(
    *,
    domain: str,
    candidate: dict[str, Any],
    evidence: dict[str, Any],
) -> tuple[str, dict[str, Any]]:
    """Hash geometry-only source/provenance fields, excluding model outcomes."""

    if domain == "wild":
        composition_path = Path(str(evidence["composition_config"]["path"])).resolve()
        composition = _json(composition_path)
        overrides = composition.get("overrides", {})
        geometry_evidence = {
            "candidate": candidate,
            "route": overrides.get("route"),
            "near_field_geometry": overrides.get("near_field"),
            "source_scene_id": overrides.get("source_scene_id"),
        }
        contract = {
            "kind": "canonical_metric_geometry_spec",
            "source": _relative_or_absolute(composition_path),
            "source_sha256": _sha256(composition_path),
        }
    else:
        source_manifest_path = Path(
            str(evidence["source_manifest"]["path"])
        ).resolve()
        source_preflight_path = Path(
            str(evidence["source_preflight"]["path"])
        ).resolve()
        base_audit_path = Path(str(evidence["base_audit"]["path"])).resolve()
        corridor_audit_path = Path(
            str(evidence["corridor_audit"]["path"])
        ).resolve()
        base_audit = _json(base_audit_path)
        corridor_audit = _json(corridor_audit_path)
        geometry_evidence = {
            "candidate": candidate,
            "source_manifest_sha256": _sha256(source_manifest_path),
            "source_geometry_preflight_sha256": _sha256(source_preflight_path),
            "base_geometry": {
                key: base_audit.get(key)
                for key in (
                    "scene_id",
                    "source_kind",
                    "route",
                    "geometry",
                    "collision_contract",
                    "room_bounds",
                )
            },
            "corridor_geometry": {
                key: corridor_audit.get(key)
                for key in (
                    "scene_id",
                    "route",
                    "geometry",
                    "collision_contract",
                    "corridor",
                )
            },
        }
        contract = {
            "kind": "canonical_embodiedgen_source_and_route_geometry_provenance",
            "sources": [
                {
                    "path": _relative_or_absolute(path),
                    "sha256": _sha256(path),
                }
                for path in (
                    source_manifest_path,
                    source_preflight_path,
                    base_audit_path,
                    corridor_audit_path,
                )
            ],
        }
    return _canonical_sha256(geometry_evidence), contract


def _terminal_path(config: dict[str, Any], candidate: dict[str, Any]) -> Path:
    scene_id = str(candidate["scene_id"])
    if candidate["domain"] == "wild":
        return (
            ROOT
            / "outputs/kinofail_confirmatory_v1/scenes/wild"
            / scene_id
            / "terminal_scene_admission.json"
        )
    compiled = ROOT / str(config["indoor_pipeline"]["compiled_root"])
    return compiled / scene_id / "terminal_scene_admission.json"


def _candidate_stream(
    config: dict[str, Any], domain: str
) -> list[dict[str, Any]]:
    source = (
        config["wild_scenes"]
        if domain == "wild"
        else config["indoor_candidates"]
    )
    return [
        dict(row) for row in source if str(row["domain"]) == domain
    ]


def _expected_qa_materials(
    config: dict[str, Any],
    candidate: dict[str, Any],
) -> list[str]:
    domain = str(candidate["domain"])
    stream = _candidate_stream(config, domain)
    index = next(
        position
        for position, row in enumerate(stream)
        if str(row["scene_id"]) == str(candidate["scene_id"])
    )
    primary = f"confirm_v1_{domain}_pbr_{index % 10:02d}"
    if domain == "wild":
        return [primary, f"confirm_v1_wild_pbr_{(index + 5) % 10:02d}"]
    return [primary]


def _select_domain(
    config: dict[str, Any], domain: str
) -> tuple[list[tuple[dict[str, Any], Path, dict[str, Any]]], list[dict[str, Any]]]:
    stream = _candidate_stream(config, domain)
    selected: list[tuple[dict[str, Any], Path, dict[str, Any]]] = []
    attempts: list[dict[str, Any]] = []
    for index, candidate in enumerate(stream):
        terminal_path = _terminal_path(config, candidate)
        if not terminal_path.exists():
            if domain == "wild":
                raise RuntimeError(
                    f"all fixed wild candidates must be attempted: {terminal_path}"
                )
            if len(selected) < 10:
                raise RuntimeError(
                    "indoor candidate attempts must form a gap-free prefix "
                    f"until ten admissions: {terminal_path}"
                )
            break
        terminal = _json(terminal_path)
        expected_qa_materials = _expected_qa_materials(config, candidate)
        terminal_qa_materials = (
            [str(value) for value in terminal.get("material_ids", [])]
            if domain == "wild"
            else [str(terminal.get("material_id", ""))]
        )
        if (
            terminal.get("scene_id") != candidate["scene_id"]
            or terminal.get("model_blind") is not True
            or terminal.get("model_or_endpoint_files_read") is not False
            or terminal_qa_materials != expected_qa_materials
        ):
            raise RuntimeError(f"invalid terminal scene audit: {terminal_path}")
        admitted = terminal.get("admitted") is True
        attempts.append(
            {
                "candidate_index": index,
                "candidate": candidate,
                "terminal_audit": str(terminal_path),
                "terminal_audit_sha256": _sha256(terminal_path),
                "admitted": admitted,
                "failure_stage": terminal.get("failure_stage"),
            }
        )
        if admitted:
            selected.append((candidate, terminal_path, terminal))
        if domain != "wild" and len(selected) == 10:
            break
    if len(selected) != 10:
        raise RuntimeError(
            f"{domain} must produce exactly ten model-blind admissions; "
            f"observed {len(selected)}"
        )
    if domain == "wild" and len(attempts) != 10:
        raise RuntimeError("wild candidate stream must contain ten terminal audits")
    return selected, attempts


def _relative_or_absolute(path: Path) -> str:
    resolved = path.resolve()
    try:
        return str(resolved.relative_to(ROOT))
    except ValueError:
        return str(resolved)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--f0-manifest", type=Path, required=True)
    parser.add_argument(
        "--candidate-config",
        type=Path,
        default=ROOT
        / "configs/data/kinofail_unified_confirmatory_scene_candidates_v1.json",
    )
    parser.add_argument("--material-lock", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    out = args.out.resolve()
    if out.exists():
        raise FileExistsError(f"refusing to overwrite scene registry: {out}")

    f0_path = args.f0_manifest.resolve()
    candidate_path = args.candidate_config.resolve()
    material_lock_path = args.material_lock.resolve()
    f0 = _json(f0_path)
    config = _json(candidate_path)
    material_lock = _json(material_lock_path)
    frozen_files = {
        str(row["path"]): str(row["sha256"])
        for row in f0.get("frozen_files", [])
    }
    candidate_display = _relative_or_absolute(candidate_path)
    if frozen_files.get(candidate_display) != _sha256(candidate_path):
        raise RuntimeError("candidate config is not bound by F0")
    if f0.get("status") != "frozen_before_new_scene_generation":
        raise RuntimeError("invalid F0 state")
    materials = [dict(row) for row in material_lock.get("materials", [])]
    if len(materials) != 30:
        raise RuntimeError("confirmatory material lock must contain 30 assets")
    materials_by_domain = {
        domain: sorted(
            (
                str(row["id"])
                for row in materials
                if domain in row.get("domains", [])
            )
        )
        for domain in ("life", "production", "wild")
    }
    if any(len(values) != 10 for values in materials_by_domain.values()):
        raise RuntimeError("confirmatory material lock must contain ten assets per domain")

    registry_rows: list[dict[str, Any]] = []
    attrition: dict[str, Any] = {}
    for domain in ("life", "production", "wild"):
        selected, attempts = _select_domain(config, domain)
        attrition[domain] = {
            "candidate_stream_size": len(_candidate_stream(config, domain)),
            "attempted_prefix": len(attempts),
            "admitted_in_attempted_prefix": sum(
                bool(row["admitted"]) for row in attempts
            ),
            "selected": 10,
            "attempts": attempts,
        }
        for slot, (candidate, terminal_path, terminal) in enumerate(selected):
            episode_path = Path(str(terminal["episode_usd"])).resolve()
            evidence = dict(terminal["evidence"])
            compiled_key = (
                "compiled_scene"
                if domain == "wild"
                else "terrain_audit"
            )
            compiled_path = Path(
                str(evidence[compiled_key]["path"])
            ).resolve()
            if (
                not episode_path.is_file()
                or not compiled_path.is_file()
                or evidence[compiled_key].get("passed") is not True
                or _sha256(compiled_path)
                != str(evidence[compiled_key]["sha256"])
            ):
                raise RuntimeError(
                    f"selected scene evidence is invalid: {candidate['scene_id']}"
                )
            geometry_hash, geometry_hash_contract = _geometry_hash(
                domain=domain,
                candidate=candidate,
                evidence=evidence,
            )
            material_ids = [
                materials_by_domain[domain][slot],
                materials_by_domain[domain][
                    (slot + 5) % len(materials_by_domain[domain])
                ],
            ]
            row = {
                "scene_id": f"confirm_v1_{domain}_scene_{slot:02d}",
                "candidate_id": str(candidate["scene_id"]),
                "source_scene_id": str(candidate["scene_id"]),
                "domain": domain,
                "split": "confirmatory",
                "source": (
                    "PolyHaven-hybrid-new-metric-geometry"
                    if domain == "wild"
                    else "EmbodiedGen-v2-new-RoomGen-source"
                ),
                "episode_usd": _relative_or_absolute(episode_path),
                "episode_sha256": _sha256(episode_path),
                "compiled_audit": _relative_or_absolute(compiled_path),
                "compiled_audit_sha256": _sha256(compiled_path),
                "geometry_hash": geometry_hash,
                "geometry_hash_contract": geometry_hash_contract,
                "terminal_scene_admission": _relative_or_absolute(
                    terminal_path
                ),
                "terminal_scene_admission_sha256": _sha256(terminal_path),
                "material_ids": material_ids,
                "material_assignment": "contrast-offset-five-two-context",
                "selected_by_model_blind_prefix_rule": True,
                "runtime_seed": int(candidate["runtime_seed"]),
            }
            if domain == "wild":
                row["metric_geometry_seed"] = int(
                    candidate["metric_geometry_seed"]
                )
                row["hdri_id"] = str(candidate["hdri_id"])
            else:
                row["source_scene_seed"] = int(candidate["source_seed"])
                row["room_type"] = str(candidate["room_type"])
            registry_rows.append(row)

    planned_ids = {
        str(row["scene_id"]) for row in f0["design"]["scenes"]
    }
    actual_ids = {str(row["scene_id"]) for row in registry_rows}
    if actual_ids != planned_ids:
        raise RuntimeError("realized scene slots differ from F0")
    out.parent.mkdir(parents=True, exist_ok=True)
    registry = {
        "schema_version": "kinofail.unified-confirmatory-scene-registry.v1",
        "created_utc": datetime.now(UTC).isoformat(),
        "status": "model_blind_scene_admission_complete_pending_f1",
        "selection_uses_model_predictions": False,
        "scene_count": len(registry_rows),
        "scenes": registry_rows,
        "attrition": attrition,
        "source_sha256": {
            "f0_manifest": _sha256(f0_path),
            "candidate_config": _sha256(candidate_path),
            "material_lock": _sha256(material_lock_path),
            "builder": _sha256(Path(__file__).resolve()),
        },
    }
    out.write_text(
        json.dumps(registry, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    print(
        json.dumps(
            {
                "passed": True,
                "scene_count": len(registry_rows),
                "registry": str(out),
                "sha256": _sha256(out),
            },
            indent=2,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
