#!/usr/bin/env python
"""Download Guardian/FailCoT HF assets and REFLECT multi-sensory archives for A8."""

from __future__ import annotations

import argparse
import hashlib
import json
import shutil
import subprocess
import sys
import tarfile
import zipfile
from pathlib import Path
from typing import Any
from urllib.request import urlretrieve

from kino_vla.eval.a7_ablation import artifact_meta, load_yaml, repo_path, write_json
from kino_vla.eval.a8_state_audit import audit_reflect_state

REPO_ROOT = Path(__file__).resolve().parents[1]


def _sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def _hf_download(repo_id: str, dest: Path) -> Path:
    dest.mkdir(parents=True, exist_ok=True)
    # Prefer huggingface_hub snapshot_download when available.
    try:
        from huggingface_hub import snapshot_download

        path = snapshot_download(
            repo_id=repo_id,
            repo_type="dataset",
            local_dir=str(dest),
            local_dir_use_symlinks=False,
        )
        return Path(path)
    except Exception as e:  # pragma: no cover - network path
        print(f"[a8_download] snapshot_download failed for {repo_id}: {e}", file=sys.stderr)
        # Fallback CLI
        cmd = [
            str(Path.home() / "miniconda3/envs/kinovla/bin/hf"),
            "download",
            repo_id,
            "--repo-type",
            "dataset",
            "--local-dir",
            str(dest),
        ]
        subprocess.run(cmd, check=True)
        return dest


def _download_url(url: str, dest_file: Path) -> Path:
    dest_file.parent.mkdir(parents=True, exist_ok=True)
    if dest_file.exists() and dest_file.stat().st_size > 0:
        print(f"[a8_download] skip existing {dest_file}")
        return dest_file
    print(f"[a8_download] downloading {url} -> {dest_file}")
    urlretrieve(url, dest_file)
    return dest_file


def _extract_archive(archive: Path, dest: Path) -> Path:
    dest.mkdir(parents=True, exist_ok=True)
    marker = dest / ".extracted_ok"
    if marker.exists():
        return dest
    print(f"[a8_download] extracting {archive} -> {dest}")
    if archive.suffix == ".zip" or archive.name.endswith(".zip"):
        with zipfile.ZipFile(archive, "r") as zf:
            zf.extractall(dest)
    elif archive.name.endswith(".tar.gz") or archive.suffixes[-2:] == [".tar", ".gz"]:
        with tarfile.open(archive, "r:gz") as tf:
            tf.extractall(dest)
    elif archive.suffix == ".tar":
        with tarfile.open(archive, "r:") as tf:
            tf.extractall(dest)
    else:
        # try tar auto
        try:
            with tarfile.open(archive, "r:*") as tf:
                tf.extractall(dest)
        except Exception as e:
            raise RuntimeError(f"unsupported archive {archive}: {e}") from e
    marker.write_text("ok\n")
    return dest


def download_guardian(cfg: dict[str, Any], *, which: str = "all") -> dict[str, str]:
    g = cfg["guardian"]
    root = repo_path(g["root"])
    root.mkdir(parents=True, exist_ok=True)
    items = []
    if which in ("all", "train"):
        items.extend(g.get("train", []))
    if which in ("all", "test"):
        items.extend(g.get("test", []))
    out: dict[str, str] = {}
    for item in items:
        name = item["name"]
        dest = root / name
        try:
            path = _hf_download(item["repo"], dest)
            # Extract nested records.tar.gz if present
            for tar in dest.rglob("records.tar.gz"):
                _extract_archive(tar, tar.parent / "records")
            out[name] = str(path)
        except Exception as e:
            if item.get("optional"):
                print(f"[a8_download] optional skip {name}: {e}")
                continue
            raise
    return out


def download_reflect(cfg: dict[str, Any], *, include_sim: bool = False) -> dict[str, str]:
    r = cfg["reflect"]
    root = repo_path(r["root"])
    root.mkdir(parents=True, exist_ok=True)
    out: dict[str, str] = {}
    tasks = _download_url(r["tasks_url"], root / "tasks_real_world.json")
    out["tasks"] = str(tasks)
    real_zip = _download_url(r["real_url"], root / "real_data.zip")
    out["real_zip"] = str(real_zip)
    real_dir = _extract_archive(real_zip, root / "real_data")
    out["real_data"] = str(real_dir)
    if include_sim:
        sim_zip = _download_url(r["sim_url"], root / "sim_data.zip")
        out["sim_zip"] = str(sim_zip)
        sim_dir = _extract_archive(sim_zip, root / "sim_data")
        out["sim_data"] = str(sim_dir)
    return out


def main() -> None:
    ap = argparse.ArgumentParser(description="A8 data download")
    ap.add_argument("--config", default="configs/eval/a8.yaml")
    ap.add_argument("--guardian-all", action="store_true")
    ap.add_argument("--guardian-train", action="store_true")
    ap.add_argument("--guardian-test", action="store_true")
    ap.add_argument("--reflect-real", action="store_true")
    ap.add_argument("--reflect-sim", action="store_true")
    ap.add_argument("--skip-audit", action="store_true")
    args = ap.parse_args()

    cfg = load_yaml(args.config)
    out_dir = repo_path(cfg["output_dir"])
    out_dir.mkdir(parents=True, exist_ok=True)

    paths: dict[str, Any] = {}
    if args.guardian_all or (not args.guardian_train and not args.guardian_test and not args.reflect_real):
        # default: everything if no flags? require explicit flags for safety on huge downloads
        pass
    if args.guardian_all:
        paths["guardian"] = download_guardian(cfg, which="all")
    else:
        gpaths: dict[str, str] = {}
        if args.guardian_train:
            gpaths.update(download_guardian(cfg, which="train"))
        if args.guardian_test:
            gpaths.update(download_guardian(cfg, which="test"))
        if gpaths:
            paths["guardian"] = gpaths

    if args.reflect_real or args.reflect_sim:
        paths["reflect"] = download_reflect(cfg, include_sim=args.reflect_sim)

    # hashes for archives that exist
    hashes: dict[str, str] = {}
    for key, val in list(paths.get("reflect", {}).items()):
        p = Path(val)
        if p.is_file():
            hashes[key] = _sha256(p)

    audit = None
    if not args.skip_audit and "reflect" in paths and "real_data" in paths["reflect"]:
        audit = audit_reflect_state(paths["reflect"]["real_data"])
        # Conditional sim download if real insufficient and config allows
        if (
            not audit["pass"]
            and cfg["reflect"].get("download_sim_if_real_insufficient", True)
            and not args.reflect_sim
            and "sim_data" not in paths.get("reflect", {})
        ):
            print("[a8_download] real audit failed; downloading sim_data as fallback")
            more = download_reflect(cfg, include_sim=True)
            paths["reflect"].update(more)
            # re-audit real+sim union
            union = repo_path(cfg["reflect"]["root"]) / "union_probe"
            # audit sim separately and merge counts
            sim_audit = audit_reflect_state(paths["reflect"].get("sim_data", paths["reflect"]["real_data"]))
            audit = {
                "pass": audit["pass"] or sim_audit["pass"],
                "real": audit,
                "sim": sim_audit,
                "reason": "real_ok" if audit["pass"] else ("sim_ok" if sim_audit["pass"] else "both_failed"),
            }
        write_json(out_dir / "a8b_state_audit.json", audit)

    manifest = {
        **artifact_meta(args.config, sources={}),
        "paths": paths,
        "archive_sha256": hashes,
        "a8b_audit_pass": None if audit is None else bool(audit.get("pass")),
    }
    write_json(out_dir / "download_manifest.json", manifest)
    print(json.dumps({"manifest": str(out_dir / "download_manifest.json"), "a8b_pass": manifest["a8b_audit_pass"]}, indent=2))


if __name__ == "__main__":
    main()
