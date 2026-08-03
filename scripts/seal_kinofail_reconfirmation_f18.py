#!/usr/bin/env python3
"""Seal the byte-identical cache recovery for the pruned F14 audit."""

from __future__ import annotations

import hashlib
import json
import os
import sys
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from scripts import recover_kinofail_reconfirmation_f14_audit_cache_v1 as recovery


OUT = (
    ROOT
    / "outputs/freeze/"
    "unified_moe_v3_reconfirmation_f18_pruned_audit_cache_amendment1"
)
MANIFEST = OUT / "amendment_manifest.json"
F17 = (
    ROOT
    / "outputs/freeze/"
    "unified_moe_v3_reconfirmation_f17_work_conserving_storage_amendment1/"
    "amendment_manifest.json"
)
SCENE04_CORPUS = (
    ROOT
    / "outputs/kinofail_reconfirmation_v2/corpus_ext4/"
    "confirm_v2_production_scene_04"
)
SCENE04_RECEIPT = (
    ROOT
    / "outputs/kinofail_reconfirmation_v2/prune_receipts/"
    "confirm_v2_production_scene_04/completed.json"
)
EVAL_ROOT = ROOT / "outputs/eval/unified_moe_v3_reconfirmation_v2"


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


def _write_json(path: Path, value: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(".json.tmp")
    temporary.write_text(
        json.dumps(value, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    os.replace(temporary, path)


def main() -> int:
    if MANIFEST.exists():
        raise FileExistsError(MANIFEST)
    if not F17.is_file() or _json(F17).get("passed") is not True:
        raise RuntimeError("F17 predecessor is absent or invalid")
    if not SCENE04_CORPUS.is_dir() or SCENE04_RECEIPT.exists():
        raise RuntimeError("F18 must seal during scene04 acquisition")
    if list(EVAL_ROOT.glob("**/*prediction*")):
        raise RuntimeError("prediction artifact exists before F18 seal")
    canonical, expected, transcript_line_sha256 = recovery.recover_canonical()
    if (
        not recovery.CACHE.is_file()
        or recovery.CACHE.read_bytes() != canonical
        or recovery.CACHE.stat().st_size != int(expected["bytes"])
        or _sha256(recovery.CACHE) != expected["sha256"]
    ):
        raise RuntimeError("F14 audit cache is not byte-identical")
    paths = {
        "finalizer_v4": (
            ROOT / "scripts/finalize_kinofail_reconfirmation_v4.py"
        ),
        "finalizer_v3": (
            ROOT / "scripts/finalize_kinofail_reconfirmation_v3.py"
        ),
        "finalizer_v2": (
            ROOT / "scripts/finalize_kinofail_reconfirmation_v2.py"
        ),
        "recovery_script_v1": Path(recovery.__file__).resolve(),
        "f14_audit_cache": recovery.CACHE,
        "f14_raw_inventory": recovery.RAW_INVENTORY,
        "sealer_v1": Path(__file__).resolve(),
    }
    missing = [str(path) for path in paths.values() if not path.is_file()]
    if missing:
        raise FileNotFoundError(missing)
    correction = {
        f"{name}_sha256": _sha256(path) for name, path in paths.items()
    }
    correction.update(
        {
            "f14_audit_cache_bytes": recovery.CACHE.stat().st_size,
            "pruned_original_relative_path": recovery.ORIGINAL_RELATIVE_PATH,
            "pruned_original_inventory_record": expected,
            "transcript_call_id": recovery.CALL_ID,
            "transcript_record_sha256": transcript_line_sha256,
            "byte_identical_to_pruned_original": True,
            "finalizer_preflight_path_only": True,
            "finalizer_scientific_steps_changed": False,
        }
    )
    manifest = {
        "schema_version": (
            "kinofail.reconfirmation-f18-pruned-audit-cache-amendment.v1"
        ),
        "status": "sealed_during_scene04_acquisition_before_any_prediction",
        "created_utc": datetime.now(UTC).isoformat(),
        "passed": True,
        "predecessor_f17": str(F17.relative_to(ROOT)),
        "predecessor_f17_sha256": _sha256(F17),
        "incident": {
            "cause": (
                "the restarted finalizer referenced an F14 launcher audit "
                "that the already-sealed scene01 prune operation removed"
            ),
            "identified_from": "finalizer FileNotFoundError before wait or inference",
            "scene04_collection_affected": False,
            "prediction_or_score_existed": False,
        },
        "correction": correction,
        "scientific_contract": {
            "prediction_or_score_used": False,
            "model_feature_route_threshold_or_analysis_changed": False,
            "collection_schedule_simulation_or_sensor_changed": False,
            "existing_scientific_artifact_modified": False,
        },
        "scope": (
            "redirect only the finalizer preflight read to a byte-identical "
            "authenticated cache of the pruned F14 operational audit"
        ),
    }
    _write_json(MANIFEST, manifest)
    print(json.dumps(manifest, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
