import json
from pathlib import Path

import numpy as np

from kino_vla.eval.a8_datasets import (
    build_a8a_cards_from_split,
    build_a8b_cards_from_reflect,
    parse_binary_label,
    write_a8a_dataset,
    write_a8b_dataset,
)
from kino_vla.eval.a8_leakage import EVAL_ONLY_FIELDS


def test_parse_binary_label():
    assert parse_binary_label("Success", 1) == "success"
    assert parse_binary_label("Failure: slip", 0) == "failure"


def test_a8a_cards_strip_leakage(tmp_path: Path):
    split = tmp_path / "bdv2"
    split.mkdir()
    # create tiny image
    from PIL import Image

    img = split / "records" / "x.png"
    img.parent.mkdir(parents=True)
    Image.fromarray(np.zeros((32, 32, 3), dtype=np.uint8)).save(img)
    row = {
        "id": "s0",
        "image": ["records/x.png"],
        "conversations": [
            {"from": "human", "value": "<image>\nDid the robot succeed?"},
            {"from": "gpt", "value": "Failure"},
        ],
        "failure_mode": "slip",
        "failure_reason": "dropped",
        "reward": 0,
        "task_instruction": "pick cup",
    }
    (split / "internVL_dataset_execution_vanilla.jsonl").write_text(json.dumps(row) + "\n")
    cards = build_a8a_cards_from_split(split, split_name="bdv2", stage="execution", prompt_policy="vanilla")
    assert len(cards) == 1
    # model-facing fields in cards still include labels for eval; ensure builder write separates
    out = write_a8a_dataset(tmp_path / "ds", cards, {"test": True})
    card_lines = (out / "cards.jsonl").read_text().splitlines()
    card = json.loads(card_lines[0])
    for k in ("failure_reason", "start_caption"):
        assert k not in card or k in EVAL_ONLY_FIELDS
    assert card["binary_label"] == "failure"
    assert (out / "frames.npz").exists()


def test_a8b_cards_and_strata(tmp_path: Path):
    from PIL import Image

    for i in range(3):
        ep = tmp_path / f"ep{i}"
        ep.mkdir()
        Image.fromarray(np.zeros((16, 16, 3), dtype=np.uint8)).save(ep / "front_rgb.png")
        np.save(ep / "robot_state.npy", np.arange(8, dtype=np.float32))
        (ep / "meta.json").write_text(json.dumps({"reward": 0, "failure_mode": "no_close"}))
    cards = build_a8b_cards_from_reflect(tmp_path)
    assert len(cards) >= 3
    assert all(c.get("stratum") in {"E1", "E2", "E3", "E4", "unassigned"} for c in cards)
    out = write_a8b_dataset(tmp_path / "a8bds", cards, {"test": True})
    assert (out / "samples.jsonl").exists()
    assert json.loads((out / "dataset_card.json").read_text())["n_records"] >= 3
