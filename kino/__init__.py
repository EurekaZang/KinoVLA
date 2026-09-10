"""KINO: recovery-related physical attribution using a GMU."""

from typing import Any

__version__ = "0.0.1"
__all__ = ["KINO", "FusionDimensions"]


def __getattr__(name: str) -> Any:
    # Import PyTorch only when a model is requested. Re-export the exact
    # registered classes so checkpoint keys and Python type identities stay fixed.
    if name in __all__:
        from kino import models

        value = getattr(models, name)
        globals()[name] = value
        return value
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
