#!/usr/bin/env python3
"""Freeze an O4 protocol only after held-out route-surface v7 admission."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import subprocess
import sys
import tempfile
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parents[1]
ADMISSION_SCHEMA = "kinofail.embodiedgen-route-surface-v7-admission-audit.v1"


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _frozen_file(path: Path) -> dict[str, str]:
    path = path.resolve()
    return {"path": str(path), "sha256": _sha256(path)}


def _json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--route-surface-admission", type=Path, required=True)
    parser.add_argument("--episode-usd", type=Path, required=True)
    parser.add_argument("--compiled-audit", type=Path, required=True)
    parser.add_argument("--calibration-manifest", type=Path, nargs="+", required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--scene-seed", type=int, required=True)
    parser.add_argument("--formal-seeds", type=int, nargs="+", required=True)
    parser.add_argument("--forward-s", type=float, required=True)
    parser.add_argument("--peel-pulse-s", type=float, default=0.10)
    parser.add_argument("--recovery-s", type=float, default=1.00)
    parser.add_argument("--reverse-s", type=float, default=3.0)
    parser.add_argument("--stop-s", type=float, default=1.0)
    parser.add_argument("--rgb-fps", type=float, default=5.0)
    args = parser.parse_args()

    admission_path = args.route_surface_admission.resolve()
    episode = args.episode_usd.resolve()
    compiled_path = args.compiled_audit.resolve()
    out = args.out.resolve()
    if out.exists():
        raise FileExistsError(f"refusing to overwrite frozen protocol: {out}")
    admission = _json(admission_path)
    evidence = admission.get("evidence", {})
    expected = {
        "compiled_audit": (compiled_path, evidence.get("compiled_audit", {})),
        "episode": (episode, evidence.get("episode", {})),
        "rtx_audit": (
            episode.parent / "rtx_qa/rtx_scene_audit.json",
            evidence.get("rtx_audit", {}),
        ),
        "go2_audit": (
            episode.parent / "go2_qa/go2_scene_audit.json",
            evidence.get("go2_audit", {}),
        ),
    }
    checks = {
        "admission_schema": admission.get("schema_version") == ADMISSION_SCHEMA,
        "admission_passed": admission.get("passed") is True,
        "heldout_material": admission.get("material", {}).get("split") in ("val", "test"),
    }
    for name, (path, record) in expected.items():
        checks[f"{name}_path"] = Path(record.get("path", "")).resolve() == path
        checks[f"{name}_hash"] = path.is_file() and record.get("sha256") == _sha256(path)
    if not all(checks.values()):
        raise RuntimeError(f"route-surface admission is stale or ineligible: {checks}")

    old_freezer = ROOT / "scripts/freeze_embodiedgen_o4_protocol.py"
    with tempfile.TemporaryDirectory(prefix="kinofail-v7-protocol-") as temp_dir:
        temporary_protocol = Path(temp_dir) / "protocol.json"
        command = [
            sys.executable,
            str(old_freezer),
            "--episode-usd",
            str(episode),
            "--compiled-audit",
            str(compiled_path),
            "--calibration-manifest",
            *[str(path.resolve()) for path in args.calibration_manifest],
            "--out",
            str(temporary_protocol),
            "--scene-seed",
            str(args.scene_seed),
            "--formal-seeds",
            *[str(seed) for seed in args.formal_seeds],
            "--forward-s",
            str(args.forward_s),
            "--peel-pulse-s",
            str(args.peel_pulse_s),
            "--recovery-s",
            str(args.recovery_s),
            "--reverse-s",
            str(args.reverse_s),
            "--stop-s",
            str(args.stop_s),
            "--rgb-fps",
            str(args.rgb_fps),
        ]
        environment = os.environ.copy()
        environment["PYTHONPATH"] = str(ROOT) + (
            os.pathsep + environment["PYTHONPATH"]
            if environment.get("PYTHONPATH")
            else ""
        )
        subprocess.run(command, cwd=ROOT, env=environment, check=True)
        protocol = _json(temporary_protocol)

    protocol["protocol_id"] = protocol["protocol_id"] + "-route-surface-v7"
    protocol["frozen_files"]["route_surface_v7_admission"] = _frozen_file(admission_path)
    protocol["frozen_files"]["route_surface_v7_protocol_wrapper"] = _frozen_file(
        Path(__file__)
    )
    protocol["heldout_scene_contract"].update(
        {
            "route_surface_v7_admission_verified": True,
            "heldout_material_split_verified": True,
            "material_split": admission["material"]["split"],
            "material_id": admission["material"]["id"],
        }
    )
    protocol["route_surface_v7_contract"] = {
        "admission_schema": ADMISSION_SCHEMA,
        "visual_intervention_only": True,
        "terrain_material_split": admission["material"]["split"],
        "terrain_material_id": admission["material"]["id"],
        "collector_compatibility": (
            "Protocol retains v3 schema; the collector validates its required frozen files and "
            "the batch auditor additionally validates these route-surface v7 bindings."
        ),
    }
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(protocol, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    print(out)
    print(_sha256(out))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
