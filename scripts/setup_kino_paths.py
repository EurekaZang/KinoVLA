"""Create KINO and KINO-Fail path aliases without moving data or checkpoints."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
REGISTRY = ROOT / "benchmarks/KINO-Fail/evaluation.json"


def publication_links(registry: dict, *, project_alias: Path, data_root: Path) -> dict[Path, Path]:
    feature_root = Path(registry["feature_root"])
    links = {project_alias: ROOT, data_root / "features": feature_root}
    for method, spec in registry["methods"].items():
        checkpoint = (
            ROOT / "checkpoints/KINO"
            if method == "gmu"
            else ROOT / "checkpoints/baselines" / method
        )
        result_name = "KINO" if method == "gmu" else method
        links[checkpoint] = ROOT / spec["freeze_dir"]
        links[ROOT / "results/KINO-Fail" / result_name] = ROOT / spec["reference_dir"]
    action = registry["action"]
    links[ROOT / "results/KINO-Fail/recovery"] = ROOT / action["reference_dir"]
    links[ROOT / "results/KINO-Fail/analysis"] = (ROOT / action["case_csv"]).parent
    # Releases are optional on evaluation-only machines.
    storage_root = feature_root.parent.parent
    for name, source in (
        ("benchmark", storage_root / "kinofail_hf_release_v1"),
        ("raw", storage_root / "kinofail_hf_raw_v1"),
    ):
        if source.is_dir():
            links[data_root / "releases" / name] = source
    return links


def validate_links(links: dict[Path, Path]) -> None:
    # Validate the complete plan before creating anything. Existing paths may
    # only be reused if they already identify the registered source directory.
    for alias, target in links.items():
        if not target.is_dir():
            raise FileNotFoundError(f"Registered directory is unavailable: {target}")
        if alias.exists() or alias.is_symlink():
            if not alias.exists() or not alias.samefile(target):
                raise FileExistsError(f"Path already exists with a different target: {alias}")
            continue
        if alias.is_relative_to(target.resolve()):
            raise ValueError(f"Refusing a recursive directory alias: {alias} -> {target}")
        for parent in alias.parents:
            if (parent.exists() or parent.is_symlink()) and not parent.is_dir():
                raise NotADirectoryError(parent)


def apply_links(links: dict[Path, Path]) -> None:
    validate_links(links)
    for alias, target in links.items():
        if alias.exists():
            continue
        alias.parent.mkdir(parents=True, exist_ok=True)
        alias.symlink_to(target.resolve(), target_is_directory=True)


def main() -> int:
    registry = json.loads(REGISTRY.read_text())
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--project-alias", type=Path, default=ROOT.parent / "KINO")
    parser.add_argument(
        "--data-root",
        type=Path,
        default=Path(registry["feature_root"]).parent.parent / "KINO-Fail",
    )
    parser.add_argument("--apply", action="store_true", help="create the validated aliases")
    args = parser.parse_args()
    # absolute() preserves a requested alias name; resolve() would erase it.
    links = publication_links(
        registry, project_alias=args.project_alias.absolute(), data_root=args.data_root.absolute()
    )
    validate_links(links)
    if args.apply:
        apply_links(links)
    print(json.dumps({str(k): str(v) for k, v in links.items()}, indent=2))
    print("Aliases are ready." if args.apply else "Preview only; use --apply to create aliases.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
