"""Train the Kino-Tokens extractor and report the M4 exit-criterion gates (spec §4, §9).

Privileged-distillation training (teacher-student, spec §4): drive the surrogate Go2
across parameterized operators, regress the privileged physics θ from the 500 ms
proprioception window, plus the self-supervised reconstruction head (its residual is
the OOD score, spec §9) and the auxiliary Kino-Text contrastive head.

Train and held-out splits use *disjoint seed ranges* (no window leakage). Reports
per-target held-out MAE vs configs/tolerances.yaml, the OOD-monotonicity check, and
single-window inference latency. The checkpoint goes to outputs/tokens/ (gitignored,
QA 5.4); the run is deterministic from the config seed and reproducible via
``--eval-only`` (QA 5.2 learned-component bar). ``train_and_eval`` is the shared gate
logic that tests/test_extractor.py asserts against.

Usage:
    python scripts/train_extractor.py [--epochs N] [--device cuda|cpu] [--eval-only]
"""

from __future__ import annotations

import argparse
import json

import torch

from kino_vla.tokens.dataset import build_dataset
from kino_vla.tokens.evaluate import ood_monotonicity, regression_mae
from kino_vla.tokens.extractor import Extractor
from kino_vla.utils.config import REPO_ROOT, Config, load_config
from kino_vla.utils.seeding import seed_everything

EVAL_SEED_BASE = 1_000_000  # disjoint from training seeds (0 .. n_train_seeds)


def train_and_eval(
    cfg: Config,
    tol: Config,
    *,
    device: str,
    out: str,
    eval_only: bool = False,
    verbose: bool = False,
) -> dict:
    """Train (unless ``eval_only``) and score the four M4 exit gates; return a metrics
    dict with per-gate pass flags and an overall ``all_pass``. Shared by the CLI and the
    test suite so the printed numbers and the asserted gates are the same computation."""
    seed_everything(int(cfg.train.seed))
    ckpt = REPO_ROOT / out

    train_seeds = list(range(int(cfg.train.n_train_seeds)))
    eval_seeds = list(range(EVAL_SEED_BASE, EVAL_SEED_BASE + int(cfg.train.n_eval_seeds)))

    if verbose:
        print(f"building held-out eval set ({len(eval_seeds)} seeds)...")
    eval_ds = build_dataset(cfg, eval_seeds)

    if eval_only:
        if verbose:
            print(f"reloading checkpoint {ckpt}.pt on cpu...")
        extractor = Extractor.load(cfg, ckpt, device="cpu")
    else:
        if verbose:
            print(f"building train set ({len(train_seeds)} seeds)...")
        train_ds = build_dataset(cfg, train_seeds)
        if verbose:
            print(f"  train windows: {len(train_ds)}   eval windows: {len(eval_ds)}")
        extractor = Extractor(cfg, device=device)
        if verbose:
            print(f"training on {device} for {int(cfg.train.epochs)} epochs...")
        extractor.fit(train_ds, log=verbose)
        extractor.save(ckpt)
        if verbose:
            print(f"saved checkpoint to {ckpt}.pt")
        # Reload on CPU: that is the deployment / latency target and the reproducible path.
        extractor = Extractor.load(cfg, ckpt, device="cpu")

    # ---- Exit criterion 1: held-out θ regression below per-target tolerances ----------
    mae = regression_mae(extractor, eval_ds)
    reg_pass = {
        name: bool(value <= float(tol.regression.get(f"{name}_mae")))
        for name, value in mae.items()
    }

    # ---- Exit criterion 2: OOD residual monotonic off-manifold ------------------------
    ood = ood_monotonicity(extractor, cfg)
    spearman_ok = bool(ood.spearman >= float(tol.ood.min_spearman))
    sep_ok = bool(ood.separation_ratio >= float(tol.ood.min_separation_ratio))

    # ---- Exit criterion 4: inference latency ------------------------------------------
    p99 = extractor.inference_latency_ms()
    lat_ok = bool(p99 <= float(tol.latency.inference_ms_p99))

    all_ok = all(reg_pass.values()) and spearman_ok and sep_ok and lat_ok
    metrics = {
        "regression_mae": mae,
        "regression_pass": reg_pass,
        "ood": {
            "values": ood.values,
            "curve": ood.ood_curve,
            "spearman": ood.spearman,
            "separation_ratio": ood.separation_ratio,
            "spearman_ok": spearman_ok,
            "separation_ok": sep_ok,
        },
        "inference_ms_p99": p99,
        "latency_ok": lat_ok,
        "all_pass": bool(all_ok),
    }

    if verbose:
        _print_report(metrics, tol)
    out_json = ckpt.parent / "eval_metrics.json"
    out_json.parent.mkdir(parents=True, exist_ok=True)
    out_json.write_text(json.dumps(metrics, indent=2))
    if verbose:
        print(f"\nmetrics written to {out_json}")
        print("PASS: train_extractor" if all_ok else "FAIL: train_extractor")
    return metrics


def _print_report(metrics: dict, tol: Config) -> None:
    print("\n=== Held-out θ regression MAE (spec §4) ===")
    for name, value in metrics["regression_mae"].items():
        budget = float(tol.regression.get(f"{name}_mae"))
        ok = metrics["regression_pass"][name]
        print(f"  {name:14s} MAE={value:.4f}   budget={budget:.4f}   {'PASS' if ok else 'FAIL'}")
    ood = metrics["ood"]
    print("\n=== OOD residual monotonicity (spec §9) ===")
    print(f"  sweep μ={ood['values']}")
    print(f"  OOD  ={[round(v, 4) for v in ood['curve']]}")
    print(
        f"  spearman(-μ, OOD)={ood['spearman']:.3f} (>= {float(tol.ood.min_spearman)}) "
        f"{'PASS' if ood['spearman_ok'] else 'FAIL'}"
    )
    print(
        f"  separation OOD/in-dist={ood['separation_ratio']:.2f} "
        f"(>= {float(tol.ood.min_separation_ratio)}) {'PASS' if ood['separation_ok'] else 'FAIL'}"
    )
    p99 = metrics["inference_ms_p99"]
    budget = float(tol.latency.inference_ms_p99)
    verdict = "PASS" if metrics["latency_ok"] else "FAIL"
    print(f"\n=== Inference latency (cpu) ===\n  p99={p99:.3f} ms (<= {budget} ms) {verdict}")


def main() -> int:
    parser = argparse.ArgumentParser(description="Train + evaluate the Kino-Tokens extractor")
    parser.add_argument("--config", default="tokens/extractor_v0.yaml")
    parser.add_argument("--tolerances", default="tolerances.yaml")
    parser.add_argument("--epochs", type=int, default=None, help="override config epochs")
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--out", default="outputs/tokens/extractor")
    parser.add_argument("--eval-only", action="store_true", help="reload checkpoint, re-eval")
    args = parser.parse_args()

    overrides = {"train.epochs": args.epochs} if args.epochs is not None else None
    cfg = load_config(args.config, overrides)
    tol = load_config(args.tolerances)
    metrics = train_and_eval(
        cfg, tol, device=args.device, out=args.out, eval_only=args.eval_only, verbose=True
    )
    return 0 if metrics["all_pass"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
