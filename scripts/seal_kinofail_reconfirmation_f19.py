#!/usr/bin/env python3
"""Seal the storage-only scene04 postprocess recovery and successor pipeline."""

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

from scripts import extract_kinofail_confirmatory_t2_features_storage_v1 as storage


SCENE = "confirm_v2_production_scene_04"
OUT = (
    ROOT
    / "outputs/freeze/"
    "unified_moe_v3_reconfirmation_f19_storage_postprocess_amendment1"
)
MANIFEST = OUT / "amendment_manifest.json"
F18 = (
    ROOT
    / "outputs/freeze/"
    "unified_moe_v3_reconfirmation_f18_pruned_audit_cache_amendment1/"
    "amendment_manifest.json"
)
ORCHESTRATION = (
    ROOT / "outputs/kinofail_reconfirmation_v2/orchestration" / SCENE
)
STATE = ORCHESTRATION / "pipeline_state.json"
PRESTATE = ORCHESTRATION / "pipeline_state.f19_pre_recovery.json"
FAILURE_LOG = ORCHESTRATION / "logs/t2_features.log"
LOGICAL_CORPUS = (
    ROOT / "outputs/kinofail_reconfirmation_v2/corpus_ext4" / SCENE
)
DATA_CORPUS = (
    Path("/data/eureka/KinoVLA/outputs/kinofail_reconfirmation_v2/corpus_ext4")
    / SCENE
)
DERIVED = (
    ROOT / "outputs/eval/unified_moe_v3_reconfirmation_v2/shards" / SCENE
)
EVAL_ROOT = ROOT / "outputs/eval/unified_moe_v3_reconfirmation_v2"
CAPSULE = (
    ROOT
    / "outputs/kinofail_reconfirmation_v2/conflict_capsules_ext4/"
    f"{SCENE}/capsule_manifest.json"
)
RECEIPT = (
    ROOT
    / "outputs/kinofail_reconfirmation_v2/prune_receipts/"
    f"{SCENE}/completed.json"
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


def _write_json(path: Path, value: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(".json.tmp")
    temporary.write_text(
        json.dumps(value, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    os.replace(temporary, path)


def _preserve_prestate() -> None:
    value = STATE.read_bytes()
    if PRESTATE.exists():
        if PRESTATE.read_bytes() != value:
            raise RuntimeError("an incompatible F19 prestate already exists")
        return
    temporary = PRESTATE.with_suffix(".json.tmp")
    temporary.write_bytes(value)
    os.replace(temporary, PRESTATE)


def _audit_scene04() -> dict[str, Any]:
    state = _json(STATE)
    if (
        state.get("status") != "terminal_failure"
        or state.get("error")
        != "RuntimeError: model-blind postprocess failed: t2_features"
        or state.get("model_checkpoint_loaded") is not False
        or not FAILURE_LOG.is_file()
        or "is not in the subpath of '/home/eureka/KinoVLA'"
        not in FAILURE_LOG.read_text(encoding="utf-8")
    ):
        raise RuntimeError("scene04 does not match the F19 failure signature")
    if (
        not LOGICAL_CORPUS.is_dir()
        or LOGICAL_CORPUS.resolve() != DATA_CORPUS.resolve()
        or not DATA_CORPUS.is_dir()
        or CAPSULE.exists()
        or RECEIPT.exists()
        or (DERIVED / "c2_t2/features").exists()
    ):
        raise RuntimeError("scene04 recovery boundary is not clean")

    t2_audit_path = (
        DATA_CORPUS
        / "c2_t2/launcher_audits"
        / f"{SCENE}.json"
    )
    t2_audit = _json(t2_audit_path)
    manifests = sorted((DATA_CORPUS / "c2_t2").glob("*/manifest.json"))
    if (
        t2_audit.get("state") != "terminal"
        or t2_audit.get("passed") is not True
        or int(t2_audit.get("case_count", -1)) != 50
        or len(manifests) != 50
    ):
        raise RuntimeError("scene04 T2 acquisition is incomplete")
    t2_bytes = 0
    t2_source_digest = hashlib.sha256()
    for manifest_path in manifests:
        case = _json(manifest_path)
        arrays_path = manifest_path.parent / case["artifacts"]["observables"]["path"]
        if (
            case.get("passed") is not True
            or not arrays_path.is_file()
            or _sha256(arrays_path)
            != case["artifacts"]["observables"]["sha256"]
        ):
            raise RuntimeError(f"scene04 T2 case is invalid: {manifest_path}")
        for source_path in (manifest_path, arrays_path):
            relative = source_path.relative_to(DATA_CORPUS).as_posix()
            size = source_path.stat().st_size
            digest = _sha256(source_path)
            t2_source_digest.update(
                f"{relative}\0{size}\0{digest}\n".encode("utf-8")
            )
            t2_bytes += size

    partition_counts = {}
    for battery, expected_per_partition in (("scale", 88), ("c2_t3", 25)):
        audits = sorted(
            (DATA_CORPUS / "launcher_audits").glob(
                f"{battery}_f12v9_partition_*_of_4.json"
            )
        )
        if len(audits) != 4:
            raise RuntimeError(f"scene04 {battery} partition audit is incomplete")
        rows = []
        for audit_path in audits:
            audit = _json(audit_path)
            attempts = audit.get("attempts", [])
            if (
                len(attempts) != expected_per_partition
                or not all(row.get("state") == "terminal" for row in attempts)
            ):
                raise RuntimeError(
                    f"scene04 {battery} terminal accounting is incomplete"
                )
            rows.append(
                {
                    "path": str(audit_path),
                    "sha256": _sha256(audit_path),
                    "attempts": len(attempts),
                    "passed": sum(row.get("passed") is True for row in attempts),
                }
            )
        partition_counts[battery] = rows

    derived_manifests = {}
    for relative in (
        "scale/snapshots/extraction_audit.json",
        "scale/features/feature_manifest.json",
        "scale/unified_features/feature_manifest.json",
        "c2_t3/snapshots/extraction_audit.json",
        "c2_t3/features/feature_manifest.json",
        "c2_t3/unified_features/feature_manifest.json",
    ):
        artifact = DERIVED / relative
        if not artifact.is_file():
            raise FileNotFoundError(artifact)
        value = _json(artifact)
        if value.get("passed") is not True and value.get("status") != "complete":
            raise RuntimeError(f"scene04 derived artifact is incomplete: {artifact}")
        derived_manifests[relative] = _sha256(artifact)

    logical, physical, relative = storage.mirrored_storage_mapping(
        LOGICAL_CORPUS / "c2_t2"
    )
    return {
        "pipeline_state_sha256": _sha256(STATE),
        "failure_log_sha256": _sha256(FAILURE_LOG),
        "t2_launcher_audit": str(t2_audit_path),
        "t2_launcher_audit_sha256": _sha256(t2_audit_path),
        "t2_case_count": len(manifests),
        "t2_authenticated_source_bytes": t2_bytes,
        "t2_relative_size_content_inventory_sha256": t2_source_digest.hexdigest(),
        "partition_audits": partition_counts,
        "derived_manifests": derived_manifests,
        "logical_t2_corpus": str(logical),
        "physical_t2_corpus": str(physical),
        "repo_relative_t2_corpus": relative.as_posix(),
        "logical_and_physical_relative_paths_equal": True,
        "capsule_existed_at_seal": False,
        "receipt_existed_at_seal": False,
        "t2_features_existed_at_seal": False,
    }


def main() -> int:
    if MANIFEST.exists():
        raise FileExistsError(MANIFEST)
    if not F18.is_file() or _json(F18).get("passed") is not True:
        raise RuntimeError("F18 predecessor is absent or invalid")
    if list(EVAL_ROOT.glob("**/*prediction*")):
        raise RuntimeError("prediction artifact exists before F19 seal")
    evidence = _audit_scene04()
    _preserve_prestate()
    paths = {
        "scene_pipeline_v4": (
            ROOT / "scripts/run_kinofail_reconfirmation_scene_pipeline_v4.py"
        ),
        "scene_pipeline_v3": (
            ROOT / "scripts/run_kinofail_reconfirmation_scene_pipeline_v3.py"
        ),
        "scene_pipeline_v2": (
            ROOT / "scripts/run_kinofail_reconfirmation_scene_pipeline_v2.py"
        ),
        "storage_t2_wrapper_v1": Path(storage.__file__).resolve(),
        "frozen_t2_wrapper_v1": (
            ROOT / "scripts/extract_kinofail_confirmatory_t2_features_v1.py"
        ),
        "frozen_t2_source_v1": (
            ROOT / "scripts/extract_kinofail_realistic_c1_causal_features_v1.py"
        ),
        "all_scenes_v4": (
            ROOT
            / "scripts/run_kinofail_reconfirmation_all_scene_pipelines_v4.py"
        ),
        "scene04_recovery_v1": (
            ROOT / "scripts/recover_kinofail_reconfirmation_scene04_f19.py"
        ),
        "finalizer_v5": (
            ROOT / "scripts/finalize_kinofail_reconfirmation_v5.py"
        ),
        "sealer_v1": Path(__file__).resolve(),
        "scene04_prestate": PRESTATE,
    }
    missing = [str(path) for path in paths.values() if not path.is_file()]
    if missing:
        raise FileNotFoundError(missing)
    correction = {
        f"{name}_sha256": _sha256(path) for name, path in paths.items()
    }
    correction.update(
        {
            "logical_repo_root": str(ROOT),
            "physical_repo_root": str(storage.DATA_REPO_ROOT),
            "t2_feature_entrypoint_only_changed": True,
            "pruner_storage_arguments_only_changed": True,
            "frozen_t2_feature_code_reused": True,
            "frozen_t2_wrapper_reused": True,
            "scene04_physical_acquisition_reexecution_permitted": False,
            "scene04_scale_or_t3_postprocess_reexecution_permitted": False,
        }
    )
    manifest = {
        "schema_version": (
            "kinofail.reconfirmation-f19-storage-postprocess-amendment.v1"
        ),
        "status": "sealed_after_scene04_failure_before_model_or_recovery",
        "created_utc": datetime.now(UTC).isoformat(),
        "passed": True,
        "predecessor_f18": str(F18.relative_to(ROOT)),
        "predecessor_f18_sha256": _sha256(F18),
        "incident": {
            "classification": "storage_symlink_provenance_path_failure",
            "failed_stage": "model_blind_postprocess/t2_features",
            "identified_from": "terminal traceback and artifact existence only",
            "model_feature_value_prediction_score_or_label_used": False,
            "physical_acquisition_complete": True,
            "physical_acquisition_invalidated": False,
        },
        "evidence": evidence,
        "correction": correction,
        "scientific_contract": {
            "feature_values_or_order_changed": False,
            "model_route_threshold_or_analysis_changed": False,
            "collection_simulation_sensor_or_schedule_changed": False,
            "result_dependent_retry_enabled": False,
            "existing_observation_manifest_or_derived_feature_modified": False,
        },
        "recovery_scope": (
            "run only the missing scene04 T2 feature stage through a mirrored "
            "path-provenance adapter, then execute the already-frozen capsule "
            "and prune stages; all future scenes use the same adapter"
        ),
        "pytest_preseal": (
            "3 storage path/command tests passed with manifest validation "
            "excluded until after seal"
        ),
    }
    _write_json(MANIFEST, manifest)
    print(json.dumps(manifest, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
