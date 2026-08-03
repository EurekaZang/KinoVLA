#!/usr/bin/env python3
"""Audit whether realistic Kino-Fail has actually reproduced A0--A7."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from kino_vla.eval.realistic_replication_readiness import (
    audit_realistic_replication_readiness,
)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--contract",
        type=Path,
        default=ROOT / "configs/eval/kinofail_realistic_a0_a7_v1.json",
    )
    parser.add_argument(
        "--out",
        type=Path,
        default=ROOT / "outputs/eval/realistic_a0_a7/readiness_audit.json",
    )
    args = parser.parse_args()
    contract_path = args.contract.resolve()
    contract = json.loads(contract_path.read_text(encoding="utf-8"))
    result = audit_realistic_replication_readiness(contract, root=ROOT)
    result["contract"] = {
        "path": str(contract_path),
        "sha256": __import__("hashlib").sha256(contract_path.read_bytes()).hexdigest(),
    }
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(
        json.dumps(result, indent=2, sort_keys=True, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )
    print(json.dumps({
        "out": str(args.out),
        "n_ready": result["n_ready"],
        "n_required": result["n_required"],
        "realistic_a0_a7_complete": result["realistic_a0_a7_complete"],
        "missing": result["missing_experiments"],
    }))


if __name__ == "__main__":
    main()
