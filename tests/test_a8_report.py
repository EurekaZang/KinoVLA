from pathlib import Path

from scripts.a8_report import render
from kino_vla.eval.a7_ablation import load_yaml, write_json


def test_report_handles_blocked_a8b(tmp_path: Path, monkeypatch):
    cfg = load_yaml("configs/eval/a8.yaml")
    # point output to temp
    cfg = dict(cfg)
    cfg["output_dir"] = str(tmp_path)
    write_json(
        tmp_path / "a8b" / "blocked.json",
        {"status": "blocked", "reason": "no robot-state / proprio files found"},
    )
    write_json(
        tmp_path / "a8a" / "summary.json",
        {
            "status": "partial",
            "note": "test",
            "arms": {
                "zero_shot": {"splits": {}},
                "guardian_8b_paper": {
                    "RoboFail": {"exec": 0.86, "plan": 0.70},
                    "UR5-Fail": {"exec": 0.77, "plan": 0.89},
                    "RoboVQA": {"exec": 0.85, "plan": None},
                },
            },
        },
    )
    md = render(cfg, "configs/eval/a8.yaml")
    assert "blocked" in md
    assert "no robot-state" in md
    assert "Guardian-8B" in md or "0.86" in md
    assert "Allowed inputs" in md
