"""Report the runtime environment: Python, CUDA, torch, Isaac Sim / Isaac Lab.

Run this first on any new machine (dev laptop or the RTX 5090 box) to see which
capability tier is available. Exit code 0 always — this is a report, not a gate.

Usage: python scripts/check_env.py
"""

from __future__ import annotations

import importlib
import platform
import shutil
import subprocess
import sys


def probe(module: str) -> str:
    try:
        mod = importlib.import_module(module)
    except ImportError:
        return "MISSING"
    return getattr(mod, "__version__", "installed (no __version__)")


def main() -> None:
    print(f"python      : {sys.version.split()[0]} ({sys.executable})")
    print(f"platform    : {platform.platform()}")

    nvidia_smi = shutil.which("nvidia-smi")
    if nvidia_smi:
        out = subprocess.run(
            [nvidia_smi, "--query-gpu=name,driver_version,memory.total", "--format=csv,noheader"],
            capture_output=True,
            text=True,
            check=False,
        )
        print(f"gpu         : {out.stdout.strip() or out.stderr.strip()}")
    else:
        print("gpu         : nvidia-smi not found (CPU-only dev tier)")

    for module in ("numpy", "yaml", "torch", "isaacsim", "isaaclab"):
        print(f"{module:<12}: {probe(module)}")

    try:
        import torch

        print(f"torch cuda  : available={torch.cuda.is_available()}")
        if torch.cuda.is_available():
            cap = torch.cuda.get_device_capability(0)
            print(f"cuda device : {torch.cuda.get_device_name(0)} (sm_{cap[0]}{cap[1]})")
            print(f"torch cuda v: {torch.version.cuda}")
            # RTX 5090 is Blackwell (sm_120): kernels exist only in CUDA >= 12.8 torch builds.
            if cap >= (12, 0) and tuple(map(int, torch.version.cuda.split("."))) < (12, 8):
                print("WARNING     : Blackwell GPU with pre-12.8 torch build — install cu128+")
    except ImportError:
        pass

    import kino_vla

    print(f"kino_vla    : {kino_vla.__version__} (importable)")


if __name__ == "__main__":
    main()
