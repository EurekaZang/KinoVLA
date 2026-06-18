"""Privileged-Grounded Hindsight CoT data pipeline (spec §10, M6).

PHASE 1 (procedural counterfactual maze) → PHASE 2 (failure interception + multimodal
snapshot) → PHASE 3 (Oracle annotation + the automated truth-consistency filter that drops
any reflection not grounded in the privileged simulation truth) → dataset format + card.
"""

from kino_vla.data.dataset import (
    build_card,
    config_hash,
    load_frames,
    load_snapshots,
    merge_snapshots,
    read_dataset,
    render_card_md,
    save_snapshots,
    write_dataset,
)
from kino_vla.data.filter import TruthConsistencyFilter
from kino_vla.data.maze import MazeCell, generate_maze
from kino_vla.data.oracle import ApiOracle, OracleClient, ScriptedOracle, build_prompt
from kino_vla.data.pipeline import (
    PipelineResult,
    PipelineStats,
    annotate_snapshots,
    compute_stats,
    run_pipeline,
)
from kino_vla.data.schema import (
    DROP_ATTRIBUTION,
    DROP_PRIMITIVE,
    DROP_SAFETY,
    DROP_SCHEMA,
    KEEP,
    CoTAnnotation,
    CoTParseError,
    DataSample,
    GroundTruth,
    RecoveryPrimitive,
    Snapshot,
    Verdict,
)
from kino_vla.data.snapshot import SnapshotRecorder
from kino_vla.data.taxonomy import FailureTaxonomy

__all__ = [
    "DROP_ATTRIBUTION",
    "DROP_PRIMITIVE",
    "DROP_SAFETY",
    "DROP_SCHEMA",
    "KEEP",
    "ApiOracle",
    "CoTAnnotation",
    "CoTParseError",
    "DataSample",
    "FailureTaxonomy",
    "GroundTruth",
    "MazeCell",
    "OracleClient",
    "PipelineResult",
    "PipelineStats",
    "RecoveryPrimitive",
    "ScriptedOracle",
    "Snapshot",
    "SnapshotRecorder",
    "TruthConsistencyFilter",
    "Verdict",
    "annotate_snapshots",
    "build_card",
    "build_prompt",
    "compute_stats",
    "config_hash",
    "generate_maze",
    "load_frames",
    "load_snapshots",
    "merge_snapshots",
    "read_dataset",
    "render_card_md",
    "run_pipeline",
    "save_snapshots",
    "write_dataset",
]
