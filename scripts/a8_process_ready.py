#!/usr/bin/env python
"""Process any completed A8 archives: extract records, rebuild cards, optional infer."""

from __future__ import annotations

import argparse
import json
import subprocess
import tarfile
from pathlib import Path

from kino_vla.eval.a7_ablation import artifact_meta, load_yaml, repo_path, write_json
from kino_vla.eval.a8_datasets import build_a8a_cards_from_split, write_a8a_dataset

# Expected approximate sizes (bytes) for completion checks
EXPECTED = {
    "ur5fail_test": 52_300_000,
    "bdv2fail_test": 179_000_000,
    "bdv2fail_train": 780_000_000,
    "rlbenchfail_train": 8_700_000_000,
    "rlbenchfail_test": 45_000_000_000,
    "ood_robofail": 900_000_000,
    "ood_robovqa": 100_000_000,
    "ood_ur5": 50_000_000,
    "reflect_real": 28_000_000_000,
}


def _complete(path: Path, expected: int, tol: float = 0.98) -> bool:
    if not path.exists():
        return False
    sz = path.stat().st_size
    return sz >= int(expected * tol) and sz > 1_000_000


def extract_tar(archive: Path, dest: Path) -> bool:
    marker = dest / ".extracted_ok"
    if marker.exists():
        return True
    if not archive.exists() or archive.stat().st_size < 1_000_000:
        return False
    dest.mkdir(parents=True, exist_ok=True)
    print(f"[process] extract {archive} -> {dest}", flush=True)
    with tarfile.open(archive, "r:*") as tf:
        tf.extractall(dest)
    marker.write_text("ok\n")
    return True


def rebuild_a8a_split(split_dir: Path, name: str, out_root: Path, config_path: str) -> Path | None:
    cards = build_a8a_cards_from_split(
        split_dir, split_name=name, stage="execution", prompt_policy="vanilla"
    )
    if not cards:
        cards = build_a8a_cards_from_split(
            split_dir, split_name=name, stage="planning", prompt_policy="vanilla"
        )
    if not cards:
        print(f"[process] no cards for {name}", flush=True)
        return None
    n_with_img = sum(1 for c in cards if c.get("images") and Path(c["images"][0]).exists())
    print(f"[process] {name}: n={len(cards)} with_images={n_with_img}", flush=True)
    dest = out_root / name.replace("/", "__")
    write_a8a_dataset(
        dest,
        cards,
        {**artifact_meta(config_path, sources={"split": str(split_dir)}), "split": name},
    )
    return dest


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="configs/eval/a8.yaml")
    ap.add_argument("--infer-ready", action="store_true", help="launch ZS infer for rebuilt splits with images")
    ap.add_argument("--limit", type=int, default=None)
    args = ap.parse_args()
    cfg = load_yaml(args.config)
    g_root = repo_path(cfg["guardian"]["root"])
    out_a8a = repo_path(cfg["output_dir"]) / "datasets" / "a8a"
    out_a8a.mkdir(parents=True, exist_ok=True)

    jobs = [
        ("ur5fail_test", g_root / "ur5fail_test" / "records.tar.gz", g_root / "ur5fail_test" / "records", EXPECTED["ur5fail_test"]),
        ("bdv2fail_test", g_root / "bdv2fail_test" / "records.tar.gz", g_root / "bdv2fail_test" / "records", EXPECTED["bdv2fail_test"]),
        ("bdv2fail_train", g_root / "bdv2fail_train" / "records.tar.gz", g_root / "bdv2fail_train" / "records", EXPECTED["bdv2fail_train"]),
        ("ood_bundle/robofail", g_root / "ood_bundle" / "robofail" / "records.tar.gz", g_root / "ood_bundle" / "robofail" / "records", EXPECTED["ood_robofail"]),
        ("ood_bundle/robovqa", g_root / "ood_bundle" / "robovqa" / "records.tar.gz", g_root / "ood_bundle" / "robovqa" / "records", EXPECTED["ood_robovqa"]),
        ("ood_bundle/ur5fail_test", g_root / "ood_bundle" / "ur5fail_test" / "records.tar.gz", g_root / "ood_bundle" / "ur5fail_test" / "records", EXPECTED["ood_ur5"]),
        ("rlbenchfail_train", g_root / "rlbenchfail_train" / "records.tar.gz", g_root / "rlbenchfail_train" / "records", EXPECTED["rlbenchfail_train"]),
        ("rlbenchfail_test", g_root / "rlbenchfail_test" / "records.tar.gz", g_root / "rlbenchfail_test" / "records", EXPECTED["rlbenchfail_test"]),
    ]

    ready = []
    for name, archive, dest, expected in jobs:
        if not _complete(archive, expected):
            sz = archive.stat().st_size if archive.exists() else 0
            print(f"[process] skip {name}: size={sz} expected~{expected}", flush=True)
            continue
        if extract_tar(archive, dest):
            split_dir = archive.parent
            path = rebuild_a8a_split(split_dir, name, out_a8a, args.config)
            if path is not None:
                ready.append((name, path))

    write_json(
        repo_path(cfg["output_dir"]) / "process_ready.json",
        {**artifact_meta(args.config), "ready": [{"name": n, "path": str(p)} for n, p in ready]},
    )

    if args.infer_ready:
        for name, path in ready:
            # only infer test splits
            if "train" in name:
                continue
            out = repo_path(cfg["output_dir"]) / "a8a" / "infer" / f"zero_shot_{name.replace('/', '__')}.json"
            if out.exists() and out.stat().st_size > 1000:
                print(f"[process] infer exists {out}", flush=True)
                continue
            cmd = [
                str(Path.home() / "miniconda3/envs/kinovla/bin/python"),
                "scripts/a8_infer.py",
                "--config",
                args.config,
                "--dataset",
                str(path),
                "--out",
                str(out),
            ]
            if args.limit:
                cmd += ["--limit", str(args.limit)]
            print("[process] launch", " ".join(cmd), flush=True)
            subprocess.Popen(cmd, env={**dict(**{k: v for k, v in __import__('os').environ.items()}), "KINOVLA_MODEL_ID": "/home/eureka/models/Qwen3-VL-4B-Instruct"})

    print(json.dumps({"ready": [n for n, _ in ready]}, indent=2))


if __name__ == "__main__":
    main()
