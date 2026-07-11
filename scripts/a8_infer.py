#!/usr/bin/env python
# ruff: noqa: E501
"""Real Qwen3-VL-4B inference for A8a/A8b cards (zero-shot or LoRA adapter)."""

from __future__ import annotations

import argparse
import json
import re
from pathlib import Path
from typing import Any

import numpy as np
from PIL import Image

from kino_vla.eval.a7_ablation import artifact_meta, load_yaml, repo_path, write_json
from kino_vla.eval.a8_datasets import parse_binary_label
from kino_vla.eval.a8_metrics import accuracy_ci, macro_f1, per_category_recall
from kino_vla.utils.config import load_config
from kino_vla.vla.model import KinoVLA


SYSTEM = (
    "You are a robotic failure verification model. "
    "Given task instruction and observation image(s) of a robot manipulation attempt, "
    "decide whether the execution succeeded or failed. "
    "Reply in exactly this format:\n"
    "<answer> True </answer> <category> success </category>\n"
    "or\n"
    "<answer> False </answer> <category> <short failure category> </category>\n"
    "True means success; False means failure."
)


def _load_cards(ds_dir: Path) -> list[dict[str, Any]]:
    cards_path = ds_dir / "cards.jsonl"
    samples = {}
    if (ds_dir / "samples.jsonl").exists():
        for line in (ds_dir / "samples.jsonl").read_text().splitlines():
            if line.strip():
                r = json.loads(line)
                samples[r["sample_id"]] = r
    cards = []
    if cards_path.exists():
        for line in cards_path.read_text().splitlines():
            if not line.strip():
                continue
            c = json.loads(line)
            s = samples.get(c["sample_id"], {})
            a8 = s.get("a8") or {}
            c.setdefault("binary_label", a8.get("binary_label") or (s.get("ground_truth") or {}).get("category"))
            c.setdefault("failure_mode", s.get("failure_mode"))
            c.setdefault("reward", s.get("reward"))
            c.setdefault("stage", a8.get("stage") or "execution")
            c.setdefault("split", ds_dir.name)
            cards.append(c)
    return cards


def _rgb_from_card(card: dict[str, Any], n_images: int = 2) -> list[np.ndarray]:
    imgs = []
    for p in (card.get("images") or [])[:n_images]:
        path = Path(p)
        if not path.exists():
            continue
        arr = np.asarray(Image.open(path).convert("RGB"), dtype=np.float32) / 255.0
        imgs.append(arr)
    if not imgs:
        imgs = [np.zeros((224, 224, 3), dtype=np.float32)]
    return imgs


def _messages(card: dict[str, Any], n_images: int) -> list[dict]:
    """Build chat messages with explicit image content slots (KinoVLA/Qwen3-VL format)."""
    instr = card.get("task_instruction") or "Verify robot execution success or failure."
    n_img = max(1, int(n_images))
    content: list[dict] = [{"type": "image"} for _ in range(n_img)]
    content.append(
        {
            "type": "text",
            "text": (
                f"Task instruction: {instr}\n"
                "Did the robot succeed? Answer with <answer> True/False </answer> "
                "and <category> ... </category>."
            ),
        }
    )
    return [
        {"role": "system", "content": [{"type": "text", "text": SYSTEM}]},
        {"role": "user", "content": content},
    ]


def _norm_pred(text: str) -> str:
    return parse_binary_label(text)


def run_infer(
    *,
    config_path: str,
    dataset_dir: str,
    out_json: str,
    adapter: str | None,
    limit: int | None,
    max_new_tokens: int = 64,
) -> dict[str, Any]:
    import torch

    if not torch.cuda.is_available():
        raise SystemExit("A8 infer requires CUDA")

    cfg_a8 = load_yaml(config_path)
    vcfg = load_config(
        "vla/sft.yaml",
        {
            "route": "text",
            "data.proprio_detail": "none",
            "model.model_id": cfg_a8["model"]["model_id"],
        },
    )
    device = "cuda"
    dtype = torch.bfloat16
    model = KinoVLA.from_pretrained(vcfg, device=device, dtype=dtype, adapter_dir=adapter)
    model.model.eval()

    ds = repo_path(dataset_dir)
    cards = _load_cards(ds)
    if limit is not None:
        cards = cards[: int(limit)]

    rows = []
    for i, card in enumerate(cards):
        # Prefer 2 frames when available (start/end), else 1.
        rgb_list = _rgb_from_card(card, n_images=2)
        n_img = min(2, max(1, len(rgb_list)))
        rgbs = rgb_list[-n_img:]
        messages = _messages(card, n_images=n_img)
        proprio = np.zeros((25, 8), dtype=np.float32)
        with torch.no_grad():
            text = model.generate(
                messages,
                rgbs,
                proprio_window=proprio,
                max_new_tokens=max_new_tokens,
                temperature=0.0,
            )
        pred = _norm_pred(text)
        truth = parse_binary_label(str(card.get("binary_label")), card.get("reward"))
        rows.append(
            {
                "sample_id": card["sample_id"],
                "truth": truth,
                "pred": pred,
                "correct": pred == truth and truth != "unknown",
                "raw": text,
                "stage": card.get("stage"),
                "split": card.get("split") or ds.name,
                "failure_mode": card.get("failure_mode"),
                "stratum": card.get("stratum", "unassigned"),
            }
        )
        if (i + 1) % 20 == 0:
            acc = accuracy_ci(rows, field="correct")
            print(f"[a8_infer] {i+1}/{len(cards)} acc={acc['rate']}", flush=True)

    y_true = [r["truth"] for r in rows]
    y_pred = [r["pred"] for r in rows]
    summary = {
        **artifact_meta(config_path, sources={"dataset": str(ds)}),
        "adapter": adapter,
        "n": len(rows),
        "accuracy": accuracy_ci(rows, field="correct"),
        "macro_f1": round(macro_f1(y_true, y_pred), 3),
        "per_category_recall": per_category_recall(y_true, y_pred),
        "dataset": str(ds),
    }
    outp = repo_path(out_json)
    outp.parent.mkdir(parents=True, exist_ok=True)
    write_json(outp, {"summary": summary, "rows": rows})
    write_json(outp.with_name(outp.stem + "_summary.json"), summary)
    print(json.dumps(summary["accuracy"], indent=2))
    return summary


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="configs/eval/a8.yaml")
    ap.add_argument("--dataset", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--adapter", default=None)
    ap.add_argument("--limit", type=int, default=None)
    ap.add_argument("--max-new-tokens", type=int, default=64)
    args = ap.parse_args()
    run_infer(
        config_path=args.config,
        dataset_dir=args.dataset,
        out_json=args.out,
        adapter=args.adapter,
        limit=args.limit,
        max_new_tokens=args.max_new_tokens,
    )


if __name__ == "__main__":
    main()
