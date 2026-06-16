"""Semantic traversability map (spec §7): physics-corrected topological memory.

M5 subsystem. Open-vocab segmentation + depth back-projection paints a visual
traversability prior into a persistent odometry-frame costmap; on-ground kinodynamic
failures overwrite it (sticky, high-confidence) and propagate to visually-homogeneous
neighbours via CLIP-feature similarity. The map crop is served to the recovery planner.
"""

from kino_vla.map.appearance import appearance_embedding, cosine_similarity
from kino_vla.map.costmap import Costmap
from kino_vla.map.segmentation import SurrogateSegmenter
from kino_vla.map.traversability_map import TraversabilityMap
from kino_vla.map.types import MapCrop, ObservedRegion, SemanticRegion

__all__ = [
    "Costmap",
    "MapCrop",
    "ObservedRegion",
    "SemanticRegion",
    "SurrogateSegmenter",
    "TraversabilityMap",
    "appearance_embedding",
    "cosine_similarity",
]
