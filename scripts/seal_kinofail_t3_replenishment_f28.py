#!/usr/bin/env python3
"""Seal an operational relaunch of the frozen F27 T3 replenishment.

F27's scientific design did not reach Python or Isaac Sim because the launcher
omitted the conda environment expected by ``isaaclab.sh``.  F28 changes only
the process environment and writes every new attempt to a separate corpus.
The 91 cases, 182 pair IDs, schedules, seeds, protocols, and collection code
remain byte-identical to the pre-collection F27 freeze.
"""

from __future__ import annotations

import hashlib
import json
from datetime import UTC, datetime
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parents[1]
F27 = ROOT / "outputs/kinofail_t3_replenishment_f27"
OUTPUT = ROOT / "outputs/kinofail_t3_replenishment_f28"
CORPUS = Path("/data/eureka/KinoVLA/outputs/kinofail_t3_replenishment_f28/corpus")
EXPECTED_LAUNCH_ERROR = "Unable to find any Python executable at path"


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def read_json(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise TypeError(path)
    return value


def write_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(value, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )


def main() -> int:
    if OUTPUT.exists() or CORPUS.exists():
        raise FileExistsError("refusing to overwrite F28 seal or corpus")
    freeze_path = F27 / "freeze_manifest.json"
    sidecar = F27 / "freeze_manifest.sha256"
    f27 = read_json(freeze_path)
    if (
        not sidecar.is_file()
        or sidecar.read_text(encoding="utf-8").split()[0] != sha256(freeze_path)
        or f27.get("status") != "sealed_before_collection"
        or f27.get("passed") is not True
        or f27.get("model_prediction_feature_or_score_read") is not False
        or f27.get("counterfactual_pairs") != 182
    ):
        raise RuntimeError("F27 scientific freeze is invalid")
    for section in ("schedules", "protocols", "dependencies"):
        for artifact in f27[section]:
            path = ROOT / str(artifact["path"])
            if not path.is_file() or sha256(path) != artifact["sha256"]:
                raise RuntimeError(f"F27 artifact drift: {path}")

    audit_path = F27 / "final_audit.json"
    audit = read_json(audit_path)
    attempts = [read_json(path) for path in sorted((F27 / "attempts").glob("*.json"))]
    logs = sorted((F27 / "logs").glob("*.log"))
    uniform_prelaunch_failure = (
        len(attempts) == 182
        and len(logs) == 182
        and audit.get("passed") is False
        and audit.get("counts", {}).get("terminal_pairs") == 182
        and audit.get("counts", {}).get("passed_pairs") == 0
        and all(
            row.get("state") == "terminal"
            and row.get("returncode") == 1
            and row.get("summary_exists") is False
            for row in attempts
        )
        and all(EXPECTED_LAUNCH_ERROR in path.read_text(encoding="utf-8", errors="replace") for path in logs)
    )
    if not uniform_prelaunch_failure:
        raise RuntimeError("F27 is not a uniform pre-Isaac launcher failure")

    forbidden = [
        path
        for base in (
            F27,
            ROOT / "outputs/eval/unified_moe_v3_reconfirmation_v2_replenished_f26",
        )
        if base.exists()
        for path in base.rglob("*")
        if path.is_file() and "prediction" in path.name.lower()
    ]
    if forbidden:
        raise RuntimeError(f"prediction artifact exists before F28 seal: {forbidden[:3]}")

    dependencies = [
        ROOT / "scripts/run_kinofail_t3_replenishment_f28.py",
        ROOT / "scripts/isaac_collect_kinofail_confirmatory_pair_v9.py",
        ROOT / "scripts/isaac_collect_kinofail_confirmatory_pair_v8.py",
        ROOT / "kino_vla/sim/isaac_policy_backend.py",
        Path("/home/eureka/IsaacLab-v2.3.0/isaaclab.sh"),
    ]
    seal = {
        "schema_version": "kinofail.t3-replenishment-f28-operational-seal.v1",
        "created_utc": datetime.now(UTC).isoformat(),
        "status": "sealed_before_operational_relaunch",
        "passed": True,
        "scientific_design_changed_from_f27": False,
        "operational_change": "set CONDA_PREFIX, PATH, TERM, and OMNI_KIT_ACCEPT_EULA",
        "selection_changed_from_f27": False,
        "result_dependent_selection_or_retry": False,
        "f27_collector_entered_python_or_isaac": False,
        "model_prediction_feature_or_score_read": False,
        "launcher_attempt_index": 2,
        "scientific_collection_attempt_index": 1,
        "cases": 91,
        "counterfactual_pairs": 182,
        "physical_episodes": 364,
        "corpus_root": str(CORPUS),
        "f27_freeze": str(freeze_path.relative_to(ROOT)),
        "f27_freeze_sha256": sha256(freeze_path),
        "f27_failed_audit": str(audit_path.relative_to(ROOT)),
        "f27_failed_audit_sha256": sha256(audit_path),
        "f27_uniform_launcher_failure": uniform_prelaunch_failure,
        "inherited_schedules": f27["schedules"],
        "inherited_protocols": f27["protocols"],
        "scene_registry": f27["scene_registry"],
        "scene_registry_sha256": f27["scene_registry_sha256"],
        "material_lock": f27["material_lock"],
        "material_lock_sha256": f27["material_lock_sha256"],
        "mapping": f27["mapping"],
        "mapping_sha256": f27["mapping_sha256"],
        "dependencies": [
            {"path": str(path), "sha256": sha256(path)} for path in dependencies
        ],
    }
    seal_path = OUTPUT / "seal_manifest.json"
    write_json(seal_path, seal)
    (OUTPUT / "seal_manifest.sha256").write_text(
        f"{sha256(seal_path)}  {seal_path.name}\n", encoding="utf-8"
    )
    print(json.dumps(seal, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
