#!/usr/bin/env python3
"""Wait for F4e, then admit every F4h extension-scene candidate once."""

from __future__ import annotations

import hashlib
import json
import math
import os
import subprocess
import sys
import time
from datetime import UTC, datetime
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parents[1]
F4H_ROOT = ROOT / "outputs/freeze/kino_v4_confirmation_t3_extension_f4h"
F4H = F4H_ROOT / "seal_manifest.json"
F4I_ROOT = ROOT / "outputs/freeze/kino_v4_confirmation_t3_extension_f4i"
F4I = F4I_ROOT / "amendment_manifest.json"
F4P_ROOT = ROOT / "outputs/freeze/kino_v4_confirmation_t3_reserve_f4p"
F4P = F4P_ROOT / "amendment_manifest.json"
F4Q_ROOT = ROOT / "outputs/freeze/kino_v4_confirmation_t3_extension_f4q"
F4Q = F4Q_ROOT / "handoff_manifest.json"
F4R_ROOT = ROOT / "outputs/freeze/kino_v4_confirmation_t3_extension_f4r"
F4R = F4R_ROOT / "amendment_manifest.json"
F4S_ROOT = ROOT / "outputs/freeze/kino_v4_confirmation_t3_extension_f4s"
F4S = F4S_ROOT / "amendment_manifest.json"
F4T_ROOT = ROOT / "outputs/freeze/kino_v4_confirmation_t3_reserve_f4t"
F4T = F4T_ROOT / "amendment_manifest.json"
F4U_ROOT = ROOT / "outputs/freeze/kino_v4_confirmation_t3_extension_f4u"
F4U = F4U_ROOT / "handoff_manifest.json"
F4V_ROOT = ROOT / "outputs/freeze/kino_v4_confirmation_t3_reserve_order_f4v"
F4V = F4V_ROOT / "amendment_manifest.json"
F4W_ROOT = ROOT / "outputs/freeze/kino_v4_confirmation_t3_extension_f4w"
F4W = F4W_ROOT / "handoff_manifest.json"
F4X_ROOT = ROOT / "outputs/freeze/kino_v4_confirmation_t3_reserve_f4x"
F4X = F4X_ROOT / "amendment_manifest.json"
F4Y_ROOT = ROOT / "outputs/freeze/kino_v4_confirmation_t3_extension_f4y"
F4Y = F4Y_ROOT / "handoff_manifest.json"
F4Z_ROOT = ROOT / "outputs/freeze/kino_v4_confirmation_t3_reserve_f4z"
F4Z = F4Z_ROOT / "amendment_manifest.json"
F4AA_ROOT = ROOT / "outputs/freeze/kino_v4_confirmation_t3_extension_f4aa"
F4AA = F4AA_ROOT / "handoff_manifest.json"
F4AB_ROOT = ROOT / "outputs/freeze/kino_v4_confirmation_t3_extension_f4ab"
F4AB = F4AB_ROOT / "handoff_manifest.json"
F4AC_ROOT = ROOT / "outputs/freeze/kino_v4_confirmation_t3_extension_f4ac"
F4AC = F4AC_ROOT / "amendment_manifest.json"
F4AD_ROOT = ROOT / "outputs/freeze/kino_v4_confirmation_t3_extension_f4ad"
F4AD = F4AD_ROOT / "handoff_manifest.json"
F4AE_ROOT = ROOT / "outputs/freeze/kino_v4_confirmation_t3_extension_f4ae"
F4AE = F4AE_ROOT / "handoff_manifest.json"
F4E_AUDIT = ROOT / "outputs/kinofail_kino_v4_confirmation_t3_runin_f4e/final_audit.json"
CONFIG = F4Z_ROOT / "scene_candidates.json"
MATERIAL_LOCK = Path("/data/eureka/kinofail_kino_v4_confirmation_v1/assets/terrain_pbr/terrain_assets.lock.json")
HDRI_LOCK = ROOT / "outputs/assets/forest_hdri_polyhaven_confirmation_v3/forest_hdri.lock.json"
INDOOR = ROOT / "scripts/run_kinofail_kino_v4_confirmation_indoor_scene.py"
WILD = ROOT / "scripts/run_kinofail_kino_v4_confirmation_wild_scene.py"
FRONT_GATE = ROOT / "scripts/isaac_audit_kino_v4_extension_front_view_v1.py"
MULTIANCHOR_FRONT_GATE = ROOT / "scripts/isaac_audit_kino_v4_extension_front_view_multianchor_f4r.py"
ISAACLAB = Path("/home/eureka/IsaacLab-v2.3.0/isaaclab.sh")
EXPERIENCE = Path("/home/eureka/IsaacLab-v2.3.0/apps/isaaclab.python.headless.rendering.kit")
CONDA_ENV = Path("/home/eureka/miniconda3/envs/kinovla")
OUT = ROOT / "outputs/kinofail_kino_v4_confirmation_t3_extension_f4h"


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def load(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise TypeError(path)
    return value


def jsonl(path: Path) -> list[dict[str, Any]]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line]


def atomic(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    os.replace(temporary, path)


def sidecar_ok(path: Path) -> bool:
    sidecar = path.with_name(path.name.replace(".json", ".sha256"))
    return sidecar.is_file() and sidecar.read_text().split()[0] == sha256(path)


def material_id(domain: str, index: int) -> str:
    return f"kino_v4_confirm_{domain}_pbr_{index % 4:02d}"


def runtime_env() -> dict[str, str]:
    value = dict(os.environ)
    value["CONDA_PREFIX"] = str(CONDA_ENV)
    value["PATH"] = f"{CONDA_ENV / 'bin'}:{value.get('PATH', '')}"
    return value


def available_stage(stage: str, scene_id: str) -> str:
    logs = OUT / "logs"
    candidate = stage
    attempt = 0
    while (logs / f"{scene_id}_{candidate}.log").exists():
        attempt += 1
        candidate = f"{stage}_operational_resume_f4ab_{attempt:02d}"
    return candidate


def run(stage: str, scene_id: str, command: list[str]) -> dict[str, Any]:
    logs = OUT / "logs"
    logs.mkdir(parents=True, exist_ok=True)
    path = logs / f"{scene_id}_{stage}.log"
    started = datetime.now(UTC).isoformat()
    with path.open("x", encoding="utf-8") as stream:
        completed = subprocess.run(
            command,
            cwd=ROOT,
            stdout=stream,
            stderr=subprocess.STDOUT,
            check=False,
            env=runtime_env(),
        )
    return {
        "stage": stage,
        "scene_id": scene_id,
        "started_utc": started,
        "completed_utc": datetime.now(UTC).isoformat(),
        "command": command,
        "log": str(path),
        "returncode": int(completed.returncode),
        "passed": completed.returncode == 0,
    }


def terminal_result(
    stage: str,
    scene_id: str,
    command: list[str],
    audit: Path,
    passed: bool,
) -> dict[str, Any]:
    return {
        "stage": stage,
        "scene_id": scene_id,
        "command": command,
        "audit": str(audit),
        "audit_sha256": sha256(audit),
        "returncode": 0 if passed else 2,
        "passed": passed,
        "reused_terminal_model_blind_audit": True,
    }


def run_gate(
    stage: str,
    scene_id: str,
    command: list[str],
    audit: Path,
) -> dict[str, Any]:
    if audit.is_file():
        return terminal_result(
            stage,
            scene_id,
            command,
            audit,
            load(audit).get("passed") is True,
        )
    result = run(available_stage(stage, scene_id), scene_id, command)
    if not audit.is_file():
        raise RuntimeError(
            f"operational gate failure produced no model-blind audit: {scene_id} {stage}"
        )
    result["audit"] = str(audit)
    result["audit_sha256"] = sha256(audit)
    result["passed"] = load(audit).get("passed") is True
    return result


def no_f4e_collectors() -> bool:
    for proc in Path("/proc").iterdir():
        if not proc.name.isdigit():
            continue
        try:
            command = (proc / "cmdline").read_bytes().replace(b"\0", b" ").decode(errors="ignore")
        except OSError:
            continue
        if "isaac_collect_kinofail_kino_v4_confirmation_t3_runin_v1.py" in command:
            return False
    return True


def main() -> int:
    for path in (F4H, F4I, F4P, F4Q, F4R, F4S, F4T, F4U, F4V, F4W, F4X, F4Y, F4Z, F4AA, F4AB, F4AC, F4AD, F4AE, CONFIG, MATERIAL_LOCK, HDRI_LOCK, INDOOR, WILD, FRONT_GATE, MULTIANCHOR_FRONT_GATE, ISAACLAB, EXPERIENCE, CONDA_ENV / "bin/python"):
        if not path.is_file():
            raise FileNotFoundError(path)
    if not sidecar_ok(F4H) or not sidecar_ok(F4I) or not sidecar_ok(F4P) or not sidecar_ok(F4Q) or not sidecar_ok(F4R) or not sidecar_ok(F4S) or not sidecar_ok(F4T) or not sidecar_ok(F4U) or not sidecar_ok(F4V) or not sidecar_ok(F4W) or not sidecar_ok(F4X) or not sidecar_ok(F4Y) or not sidecar_ok(F4Z) or not sidecar_ok(F4AA) or not sidecar_ok(F4AB) or not sidecar_ok(F4AC) or not sidecar_ok(F4AD) or not sidecar_ok(F4AE):
        raise RuntimeError("F4h--F4ae sidecar mismatch")
    f4h = load(F4H)
    f4i = load(F4I)
    f4p = load(F4P)
    f4q = load(F4Q)
    if f4q.get("passed") is not True:
        raise RuntimeError("F4q handoff is invalid")
    f4r = load(F4R)
    f4s = load(F4S)
    if f4s.get("passed") is not True:
        raise RuntimeError("F4s attrition-margin amendment is invalid")
    f4t = load(F4T)
    f4u = load(F4U)
    if f4u.get("passed") is not True:
        raise RuntimeError("F4u reserve handoff is invalid")
    f4v = load(F4V)
    f4w = load(F4W)
    f4x = load(F4X)
    if f4x.get("passed") is not True or f4x.get("model_prediction_truth_key_or_score_read") is not False:
        raise RuntimeError("F4x reserve amendment is invalid")
    f4y = load(F4Y)
    f4z = load(F4Z)
    if f4z.get("passed") is not True or f4z.get("model_prediction_truth_key_or_score_read") is not False:
        raise RuntimeError("F4z reserve correction is invalid")
    f4aa = load(F4AA)
    f4ab = load(F4AB)
    f4ac = load(F4AC)
    f4ad = load(F4AD)
    f4ae = load(F4AE)
    if (
        f4y.get("passed") is not True
        or f4aa.get("passed") is not True
        or f4ab.get("passed") is not True
        or f4ab.get("parent_f4aa_sha256") != sha256(F4AA)
        or f4ac.get("passed") is not True
        or f4ac.get("parent_f4ab_sha256") != sha256(F4AB)
        or f4ac.get("front_view_gate_sha256") != f4ae.get("predecessor_front_view_gate_sha256")
        or f4ac.get("multianchor_front_gate_sha256") != f4ae.get("predecessor_multianchor_front_gate_sha256")
        or f4ad.get("passed") is not True
        or f4ad.get("parent_f4ac_sha256") != sha256(F4AC)
        or f4ad.get("predecessor_scene_admission_runner_sha256") != f4ac.get("scene_admission_runner_sha256")
        or f4ad.get("replaced_multianchor_front_gate_sha256") != f4r.get("multianchor_front_gate_sha256")
        or f4ad.get("multianchor_front_gate_sha256") != f4ae.get("predecessor_multianchor_front_gate_sha256")
        or f4ae.get("passed") is not True
        or f4ae.get("parent_f4ad_sha256") != sha256(F4AD)
        or f4ae.get("predecessor_scene_admission_runner_sha256") != f4ad.get("scene_admission_runner_sha256")
        or f4ae.get("front_view_gate_sha256") != sha256(FRONT_GATE)
        or f4ae.get("multianchor_front_gate_sha256") != sha256(MULTIANCHOR_FRONT_GATE)
        or f4ae.get("scene_admission_runner_sha256") != sha256(Path(__file__).resolve())
    ):
        raise RuntimeError("F4ae deterministic-process-exit handoff drift")
    for relative, expected in f4i["frozen_files"].items():
        path = ROOT / relative
        if path == FRONT_GATE:
            if (
                expected != f4ac.get("replaced_front_view_gate_sha256")
                or sha256(path) != f4ae.get("front_view_gate_sha256")
            ):
                raise RuntimeError(f"F4ac front-view gate drift: {path}")
            continue
        if not path.is_file() or sha256(path) != expected:
            raise RuntimeError(f"F4i frozen file drift: {path}")
    selected_path = ROOT / str(f4z["selected_scene_candidates"])
    if (
        sha256(selected_path) != f4z["selected_scene_candidates_sha256"]
        or sha256(CONFIG) != f4z["candidate_config_sha256"]
    ):
        raise RuntimeError("F4z candidate bank or ordering drift")
    selected = jsonl(selected_path)
    source = load(CONFIG)
    streams = {
        domain: [row for key in ("indoor_candidates", "wild_scenes") for row in source[key] if row["domain"] == domain]
        for domain in ("life", "production", "wild")
    }
    OUT.mkdir(parents=True, exist_ok=True)
    state = OUT / "scene_admission_state.json"
    atomic(
        state,
        {
            "status": "waiting_for_f4e_terminal_audit",
            "updated_utc": datetime.now(UTC).isoformat(),
            "model_prediction_truth_key_or_score_read": False,
        },
    )
    while not F4E_AUDIT.is_file():
        time.sleep(30)
        atomic(
            state,
            {
                "status": "waiting_for_f4e_terminal_audit",
                "updated_utc": datetime.now(UTC).isoformat(),
                "model_prediction_truth_key_or_score_read": False,
            },
        )
    while not no_f4e_collectors():
        time.sleep(10)
    f4e = load(F4E_AUDIT)
    original_rejected = int(f4e["counts"]["rejected_or_excluded_cases"])
    target_admitted = None
    attrition_margin = None
    for candidate_target in range(9, len(selected) + 1):
        candidate_margin = max(
            10,
            math.ceil(0.20 * original_rejected),
            math.ceil(0.02 * 24 * candidate_target),
        )
        projected = (original_rejected + candidate_margin) / (
            288 + 24 * candidate_target
        )
        if projected < 0.05:
            target_admitted = candidate_target
            attrition_margin = candidate_margin
            break
    if target_admitted is None or attrition_margin is None:
        atomic(
            state,
            {
                "status": "scientific_gate_failed",
                "reason": "frozen_candidate_bank_smaller_than_attrition_sizing_rule",
                "original_rejected_cases": original_rejected,
                "attrition_margin_cases": attrition_margin,
                "target_admitted_scenes": target_admitted,
                "candidate_bank": len(selected),
                "updated_utc": datetime.now(UTC).isoformat(),
            },
        )
        return 2
    atomic(
        state,
        {
            "status": "running_scene_admission",
            "updated_utc": datetime.now(UTC).isoformat(),
            "f4e_passed": f4e.get("passed") is True,
            "original_rejected_cases": original_rejected,
            "attrition_margin_cases": attrition_margin,
            "target_admitted_scenes": target_admitted,
            "candidate_bank": len(selected),
            "model_prediction_truth_key_or_score_read": False,
        },
    )

    results = []
    for position, row in enumerate(selected):
        scene_id = str(row["scene_id"])
        domain = str(row["domain"])
        index = next(i for i, item in enumerate(streams[domain]) if item["scene_id"] == scene_id)
        primary = material_id(domain, index)
        if domain == "wild":
            command = [
                sys.executable,
                str(WILD),
                "--scene-config", str(CONFIG),
                "--scene-id", scene_id,
                "--hdri-lock", str(HDRI_LOCK),
                "--material-lock", str(MATERIAL_LOCK),
                "--route-material-id", primary,
                "--surrounding-material-id", material_id(domain, index + 2),
            ]
            root = Path("/data/eureka/kinofail_kino_v4_confirmation_v1/scenes/wild") / scene_id
            episode = root / "composition/episode.usda"
            compiled = root / "composition/compiled_scene_audit.json"
            seed = int(row["runtime_seed"])
        else:
            command = [
                sys.executable,
                str(INDOOR),
                "--config", str(CONFIG),
                "--scene-id", scene_id,
                "--material-lock", str(MATERIAL_LOCK),
                "--material-id", primary,
            ]
            root = Path("/data/eureka/kinofail_kino_v4_confirmation_v1/scenes/embodiedgen_compiled") / scene_id
            episode = root / "route_surface_v4/terrain_route_v2/episode_terrain_v2.usda"
            compiled = root / "route_surface_v4/terrain_route_v2/compiled_scene_audit.json"
            seed = int(row["runtime_seed"])
        terminal = root / "terminal_scene_admission.json"
        if terminal.is_file():
            generation = terminal_result(
                "f1_admission",
                scene_id,
                command,
                terminal,
                load(terminal).get("admitted") is True,
            )
        else:
            generation = run(available_stage("f1_admission", scene_id), scene_id, command)
            if not terminal.is_file():
                raise RuntimeError(
                    f"operational scene-generation failure produced no terminal audit: {scene_id}"
                )
            generation["audit"] = str(terminal)
            generation["audit_sha256"] = sha256(terminal)
            generation["passed"] = load(terminal).get("admitted") is True
        front = None
        front_anchors = []
        if generation["passed"]:
            gate_out = root / "f4h_registered_front_view_gate"
            gate_command = [
                str(ISAACLAB), "-p", str(FRONT_GATE),
                "--scene-id", scene_id,
                "--episode-usd", str(episode),
                "--compiled-audit", str(compiled),
                "--material-lock", str(MATERIAL_LOCK),
                "--material-id", primary,
                "--material-id", material_id(domain, index + 1),
                "--material-id", material_id(domain, index + 2),
                "--camera-profile", "go2_front_calib_b",
                "--seed", str(seed),
                "--output", str(gate_out),
                "--minimum-swap-l1", "0.015",
                "--headless", "--enable_cameras", "--experience", str(EXPERIENCE),
            ]
            front = run_gate(
                "registered_front_view_gate",
                scene_id,
                gate_command,
                gate_out / "audit.json",
            )
        if front is not None and front["passed"]:
            for anchor_name, route_progress_m in (("p020", 0.20), ("p050", 0.50)):
                gate_out = root / f"f4r_registered_front_view_gate_{anchor_name}"
                gate_command = [
                    str(ISAACLAB), "-p", str(MULTIANCHOR_FRONT_GATE),
                    "--scene-id", scene_id,
                    "--episode-usd", str(episode),
                    "--compiled-audit", str(compiled),
                    "--material-lock", str(MATERIAL_LOCK),
                    "--material-id", primary,
                    "--material-id", material_id(domain, index + 1),
                    "--material-id", material_id(domain, index + 2),
                    "--camera-profile", "go2_front_calib_b",
                    "--route-progress-m", str(route_progress_m),
                    "--seed", str(seed),
                    "--output", str(gate_out),
                    "--minimum-swap-l1", "0.015",
                    "--headless", "--enable_cameras", "--experience", str(EXPERIENCE),
                ]
                result = run_gate(
                    f"registered_front_view_gate_{anchor_name}",
                    scene_id,
                    gate_command,
                    gate_out / "audit.json",
                )
                front_anchors.append(result)
                if not result["passed"]:
                    break
        passed = (
            generation["passed"]
            and front is not None
            and front["passed"]
            and len(front_anchors) == 2
            and all(value["passed"] for value in front_anchors)
        )
        result = {
            "scene_id": scene_id,
            "domain": domain,
            "candidate_position": position,
            "passed": passed,
            "f1_admission": generation,
            "registered_front_view_gate": front,
            "registered_front_view_multianchor_gates": front_anchors,
        }
        results.append(result)
        atomic(
            state,
            {
                "status": "running_scene_admission",
                "updated_utc": datetime.now(UTC).isoformat(),
                "completed_candidates": len(results),
                "passed_candidates": sum(item["passed"] for item in results),
                "target_admitted_scenes": target_admitted,
                "candidate_bank": len(selected),
                "active_or_next_scene": selected[len(results)]["scene_id"] if len(results) < len(selected) else None,
                "model_prediction_truth_key_or_score_read": False,
            },
        )
        if len(results) >= 12 and sum(item["passed"] for item in results) >= target_admitted:
            break
    audit = {
        "schema_version": "kinofail.kino-v4-confirmation-t3-extension-scene-admission-f4j.v2",
        "created_utc": datetime.now(UTC).isoformat(),
        "passed": len(results) >= 12 and sum(item["passed"] for item in results) >= target_admitted,
        "minimum_admitted_scenes_for_combined_attrition_recovery": 9,
        "adaptive_target_admitted_scenes": target_admitted,
        "original_rejected_cases": original_rejected,
        "attrition_margin_cases": attrition_margin,
        "candidate_selection_rule": "gap_free_frozen_prefix_until_target_after_first_twelve_attempts",
        "model_prediction_truth_key_or_score_read": False,
        "f4e_failures_preserved": True,
        "counts": {
            "candidate_bank": len(selected),
            "candidate_attempts": len(results),
            "admitted_scenes": sum(item["passed"] for item in results),
            "failed_scene_admissions": sum(not item["passed"] for item in results),
        },
        "results": results,
        "source_sha256": {
            "f4h": sha256(F4H),
            "f4i": sha256(F4I),
            "f4p": sha256(F4P),
            "f4q": sha256(F4Q),
            "f4r": sha256(F4R),
            "f4s": sha256(F4S),
            "f4t": sha256(F4T),
            "f4u": sha256(F4U),
            "f4v": sha256(F4V),
            "f4w": sha256(F4W),
            "f4x": sha256(F4X),
            "f4y": sha256(F4Y),
            "f4z": sha256(F4Z),
            "f4aa": sha256(F4AA),
            "f4ab": sha256(F4AB),
            "f4ac": sha256(F4AC),
            "f4ad": sha256(F4AD),
            "f4ae": sha256(F4AE),
            "f4e_audit": sha256(F4E_AUDIT),
            "selected_candidates": sha256(selected_path),
            "runner": sha256(Path(__file__).resolve()),
        },
    }
    atomic(OUT / "scene_admission_audit.json", audit)
    atomic(
        state,
        {
            "status": "complete" if audit["passed"] else "scientific_gate_failed",
            "passed": audit["passed"],
            "updated_utc": datetime.now(UTC).isoformat(),
            "counts": audit["counts"],
        },
    )
    print(json.dumps(audit["counts"], indent=2, sort_keys=True), flush=True)
    return 0 if audit["passed"] else 2


if __name__ == "__main__":
    raise SystemExit(main())
