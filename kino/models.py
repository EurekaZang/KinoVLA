"""Public model names for KINO; implementations retain checkpoint compatibility."""

from kino_vla.eval.known_multimodal_fusions import (
    FusionDimensions,
    GatedMultimodalUnit,
)

KINO = GatedMultimodalUnit

__all__ = ["KINO", "FusionDimensions"]
