"""Hindsight-CoT pipeline orchestration: PHASE 1 → 2 → 3 → truth-consistency filter (spec §10).

Drives each maze cell (PHASE 1) through the same surrogate closed loop the demo uses, the
Kino-Monitor intercepts the failure and packages the snapshot (PHASE 2), the Oracle annotates
it (PHASE 3), and the truth-consistency filter keeps it only if the reflection is grounded in
the privileged θ. Returns every produced sample (kept *and* dropped, each with its verdict) and
the report statistics the dataset card needs: reject-rate by reason, per-operator counts, A/B
balance, ambiguity-pair coverage, and throughput (samples/hour).

Surrogate-driven (numpy, torch-free) so the whole pipeline runs in CI; the loop is the M5
``build_walking_skeleton(operator_factory=…)`` path, so a maze cell is exactly the per-operator
closed loop ``scripts/record_operators.py`` already runs.
"""

from __future__ import annotations

import time
from collections import Counter
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass, field
from threading import Lock
from typing import Any

import numpy as np

from kino_vla.data.filter import TruthConsistencyFilter
from kino_vla.data.maze import MazeCell, generate_maze
from kino_vla.data.oracle import OracleClient
from kino_vla.data.schema import DROP_SCHEMA, DataSample, GroundTruth, Snapshot, Verdict
from kino_vla.data.snapshot import SnapshotRecorder
from kino_vla.data.taxonomy import FailureTaxonomy
from kino_vla.monitor.rule_monitor import RuleMonitor
from kino_vla.sim.operators import OperatorStack
from kino_vla.sim.surrogate import SurrogateBackend
from kino_vla.tokens.features import physics_to_target
from kino_vla.utils.config import Config, load_config
from kino_vla.utils.geometry import Rect
from kino_vla.utils.seeding import rng

# Edge-stoppers halt the robot at the region boundary, so their capture-gate keeps a small
# margin (see _collect_snapshot); every other operator gates strictly in-region (margin 0).
_EDGE_OPS: frozenset[str] = frozenset({"O8_invisible_collider"})


@dataclass(frozen=True)
class PipelineStats:
    """Report statistics (exit criterion 2 + the dataset card, exit criterion 3)."""

    n_cells: int
    n_interceptions: int
    n_no_interception: int
    kept: int
    dropped: int
    reject_rate: float
    by_reason: dict[str, int]
    per_operator_kept: dict[str, int]
    per_operator_total: dict[str, int]
    ab_balance_kept: dict[str, int]
    ambiguity_coverage: dict[str, dict[str, Any]]
    wall_time_s: float
    samples_per_hour: float
    kept_per_hour: float

    def to_dict(self) -> dict[str, Any]:
        return {
            "n_cells": self.n_cells,
            "n_interceptions": self.n_interceptions,
            "n_no_interception": self.n_no_interception,
            "kept": self.kept,
            "dropped": self.dropped,
            "reject_rate": round(self.reject_rate, 4),
            "by_reason": dict(self.by_reason),
            "per_operator_kept": dict(self.per_operator_kept),
            "per_operator_total": dict(self.per_operator_total),
            "ab_balance_kept": dict(self.ab_balance_kept),
            "ambiguity_coverage": dict(self.ambiguity_coverage),
            "wall_time_s": round(self.wall_time_s, 3),
            "samples_per_hour": round(self.samples_per_hour, 1),
            "kept_per_hour": round(self.kept_per_hour, 1),
        }


@dataclass
class PipelineResult:
    samples: list[DataSample] = field(default_factory=list)  # kept + dropped, each with verdict
    stats: PipelineStats | None = None


def run_pipeline(
    cfg: Config,
    *,
    seed: int,
    oracle: OracleClient,
    taxonomy: FailureTaxonomy | None = None,
    filt: TruthConsistencyFilter | None = None,
) -> PipelineResult:
    """Generate, annotate, and filter one Hindsight-CoT dataset from a seeded maze (spec §10)."""
    taxonomy = taxonomy or FailureTaxonomy(cfg)
    filt = filt or TruthConsistencyFilter(taxonomy)
    cells = generate_maze(cfg, seed)

    samples: list[DataSample] = []
    n_no_interception = 0
    wall_start = time.perf_counter()
    for cell in cells:
        snap, ground_truth = _collect_snapshot(cell, cfg, taxonomy)
        if snap is None:
            n_no_interception += 1
            continue
        oracle_text = oracle.annotate(snap)
        annotation, verdict = filt.evaluate(
            oracle_text, ground_truth, prior_outputs=snap.prior_outputs
        )
        samples.append(
            DataSample(
                sample_id=f"{cell.index:04d}_{cell.op_name}",
                snapshot=snap,
                ground_truth=ground_truth,
                annotation=annotation,
                verdict=verdict,
                target_theta=list(physics_to_target(snap.privileged_theta)),
                ambiguity_pair=cell.ambiguity_pair,
            )
        )
    wall = time.perf_counter() - wall_start
    stats = compute_stats(
        cfg, samples, n_cells=len(cells), n_no_interception=n_no_interception, wall=wall
    )
    return PipelineResult(samples=samples, stats=stats)


def annotate_snapshots(
    cfg: Config,
    items: list[tuple[Snapshot, dict[str, float]]],
    *,
    oracle: OracleClient,
    taxonomy: FailureTaxonomy | None = None,
    filt: TruthConsistencyFilter | None = None,
    concurrency: int = 1,
    cache: dict[int, tuple] | None = None,
) -> PipelineResult:
    """PHASE 3 over PRE-COLLECTED snapshots (the decoupled GPU→CPU path, CLAUDE.md #23).

    ``items`` are ``(Snapshot, op_theta)`` pairs from ``load_snapshots`` (real-Go2 snapshots
    collected on the GPU by scripts/isaac_hindsight_collect.py). Each is annotated by ``oracle``
    (the real external LLM) and judged by the truth-consistency filter — identical to the inline
    pipeline, only the snapshot source differs (real Isaac proprioception, not the surrogate).
    ``concurrency`` > 1 runs the (I/O-bound) Oracle calls in a thread pool — the LLM API is the
    wall-time bottleneck at scale, and the Oracle/filter are stateless/read-only so it is safe."""
    taxonomy = taxonomy or FailureTaxonomy(cfg)
    filt = filt or TruthConsistencyFilter(taxonomy)
    pairs = _pair_labels(cfg)

    done = [0]
    lock = Lock()
    total = len(items)

    def process(indexed: tuple[int, tuple[Snapshot, dict[str, float]]]) -> DataSample:
        i, (snap, op_theta) = indexed
        ground_truth = taxonomy.ground_truth(snap.operator_name, op_theta)
        if cache is not None and i in cache:
            annotation, verdict = cache[i]  # reuse a prior good result (retry-failed-only)
        else:
            try:
                annotation, verdict = filt.evaluate(
                    oracle.annotate(snap), ground_truth, prior_outputs=snap.prior_outputs
                )
            except Exception as err:  # noqa: BLE001 - one bad API call must not kill a long run
                annotation = None
                verdict = Verdict(False, DROP_SCHEMA, detail=f"oracle call failed: {err}")
        with lock:
            done[0] += 1
            if done[0] % 25 == 0 or done[0] == total:
                print(f"[annotate] {done[0]}/{total} done", flush=True)
        return DataSample(
            sample_id=f"{i:04d}_{snap.operator_name}",
            snapshot=snap,
            ground_truth=ground_truth,
            annotation=annotation,
            verdict=verdict,
            target_theta=list(physics_to_target(snap.privileged_theta)),
            ambiguity_pair=pairs.get(snap.operator_name),
        )

    wall_start = time.perf_counter()
    if concurrency > 1:
        with ThreadPoolExecutor(max_workers=int(concurrency)) as ex:
            samples = list(ex.map(process, enumerate(items)))
    else:
        samples = [process(x) for x in enumerate(items)]
    wall = time.perf_counter() - wall_start
    stats = compute_stats(cfg, samples, n_cells=len(items), n_no_interception=0, wall=wall)
    return PipelineResult(samples=samples, stats=stats)


def _pair_labels(cfg: Config) -> dict[str, str]:
    """Operator → its ambiguity-pair label (for the dataset card coverage stat)."""
    out: dict[str, str] = {}
    for pair in cfg.maze.ambiguity_pairs:
        a, b = str(pair[0]), str(pair[1])
        label = "|".join(sorted((a, b)))
        out[a] = out[b] = label
    return out


def _collect_snapshot(
    cell: MazeCell, cfg: Config, taxonomy: FailureTaxonomy
) -> tuple[Snapshot | None, GroundTruth | None]:
    """PHASE 1+2: drive one maze cell into its failure with bang-bang excitation, intercept it.

    Reuses the M4 distillation drive (straight +x, square-wave speed) — the recovery loop is
    *not* part of data collection (it is what M7 trains); we only need the robot to experience
    the degradation so the Kino-Monitor can intercept it. Returns ``(snapshot, ground_truth)``
    or ``(None, None)`` if the monitor never fired within the drive (recorded as a miss).
    """
    d = cfg.drive
    sim_cfg = load_config("sim/surrogate.yaml")
    backend = SurrogateBackend(sim_cfg, np.zeros(2), 0.0)
    obs = backend.reset(cell.seed)
    rect = Rect(cx=float(d.hazard_cx), cy=0.0, hx=float(d.hazard_hx), hy=float(d.hazard_hy))
    # Capture-gate IN the hazard region (margin 0) so the snapshot shows the REAL signature, not
    # the bang-bang start transient on firm ground: ice ⇒ slip is in the window; payload/decay
    # (global, but the robot reaches the region here — unlike the heavy/weak Isaac Go2) intercept
    # after warmup once the effort has developed. The O8 edge-stopper keeps a small margin (the
    # wall halts the robot just at the boundary). This mirrors scripts/isaac_hindsight_check.py.
    m = float(d.get("edge_gate_margin_m", 0.4)) if cell.op_name in _EDGE_OPS else 0.0
    gate: Rect | None = Rect(cx=rect.cx, cy=rect.cy, hx=rect.hx + m, hy=rect.hy + m)
    operator = cell.build_operator(rect)
    stack = OperatorStack([operator])
    stack.on_reset(backend)
    ground_truth = taxonomy.ground_truth(cell.op_name, operator.get_privileged_state())

    monitor = RuleMonitor(load_config(str(d.monitor)), dt=backend.dt)
    monitor.reset()
    region = cell.scene_region(rect)
    recorder = SnapshotRecorder(
        cfg,
        scene=[region] if region is not None else [],
        operator_name=cell.op_name,
        appearance_class=cell.appearance,
        privileged_fn=backend.privileged_physics,
        gate_rect=gate,  # intercept at the failure locus, not on a bang-bang decel transient
    )

    r = rng(cell.seed + 100_000)
    phase0 = float(r.uniform(0.0, 1.0))
    dt = float(backend.dt)
    period, duty = float(d.speed_period_s), float(d.speed_duty)
    speed_lo, speed_hi = float(d.speed_lo), float(d.speed_hi)
    for k in range(int(d.n_steps)):
        if obs.fallen:
            break
        obs_measured = stack.transform_obs(obs)
        event = monitor.step(obs_measured)
        recorder.observe(obs_measured, event)
        if recorder.snapshot is not None:
            break
        frac = (phase0 + (k * dt) / period) % 1.0
        speed = speed_hi if frac < duty else speed_lo
        cmd = np.array([speed, 0.05 * r.standard_normal(), 0.05 * r.standard_normal()])
        stack.on_step(backend, obs.t)
        obs = backend.step(cmd)
    return recorder.snapshot, ground_truth


def stats_from_records(
    cfg: Config,
    kept_records: list[dict[str, Any]],
    dropped_records: list[dict[str, Any]],
    *,
    wall_time_s: float = 0.0,
) -> PipelineStats:
    """Recompute :class:`PipelineStats` from kept+dropped JSONL records (dataset merge helper).

    Mirrors :func:`compute_stats` but reads the manifest dicts (``to_record``) instead of live
    ``DataSample`` objects — so two datasets can be merged and the card auto-regenerated (QA 5.4),
    without re-running the Oracle. Throughput is left at 0 (a merge has no single wall-clock)."""

    def op(rec: dict[str, Any]) -> str:
        return rec["snapshot"]["operator_name"]

    n = len(kept_records) + len(dropped_records)
    by_reason = dict(Counter(r["verdict"]["reason"] for r in kept_records + dropped_records))
    coverage: dict[str, dict[str, Any]] = {}
    for pair in cfg.maze.ambiguity_pairs:
        a, b = str(pair[0]), str(pair[1])
        label = "|".join(sorted((a, b)))
        kept_a = sum(1 for r in kept_records if op(r) == a)
        kept_b = sum(1 for r in kept_records if op(r) == b)
        coverage[label] = {a: kept_a, b: kept_b, "both_present": kept_a > 0 and kept_b > 0}
    return PipelineStats(
        n_cells=n,
        n_interceptions=n,
        n_no_interception=0,
        kept=len(kept_records),
        dropped=len(dropped_records),
        reject_rate=(len(dropped_records) / n) if n else 0.0,
        by_reason=by_reason,
        per_operator_kept=dict(Counter(op(r) for r in kept_records)),
        per_operator_total=dict(Counter(op(r) for r in kept_records + dropped_records)),
        ab_balance_kept=dict(Counter(r["ground_truth"]["ab_class"] for r in kept_records)),
        ambiguity_coverage=coverage,
        wall_time_s=wall_time_s,
        samples_per_hour=0.0,
        kept_per_hour=0.0,
    )


def compute_stats(
    cfg: Config,
    samples: list[DataSample],
    *,
    n_cells: int,
    n_no_interception: int,
    wall: float,
) -> PipelineStats:
    kept = [s for s in samples if s.verdict.keep]
    dropped = [s for s in samples if not s.verdict.keep]
    by_reason = dict(Counter(s.verdict.reason for s in samples))
    per_op_kept = dict(Counter(s.snapshot.operator_name for s in kept))
    per_op_total = dict(Counter(s.snapshot.operator_name for s in samples))
    ab_balance = dict(Counter(s.ground_truth.ab_class for s in kept))
    coverage: dict[str, dict[str, Any]] = {}
    for pair in cfg.maze.ambiguity_pairs:
        a, b = str(pair[0]), str(pair[1])
        label = "|".join(sorted((a, b)))
        kept_a = sum(1 for s in kept if s.snapshot.operator_name == a)
        kept_b = sum(1 for s in kept if s.snapshot.operator_name == b)
        coverage[label] = {a: kept_a, b: kept_b, "both_present": kept_a > 0 and kept_b > 0}
    n = len(samples)
    hours = wall / 3600.0 if wall > 0 else float("inf")
    return PipelineStats(
        n_cells=n_cells,
        n_interceptions=n,
        n_no_interception=n_no_interception,
        kept=len(kept),
        dropped=len(dropped),
        reject_rate=(len(dropped) / n) if n else 0.0,
        by_reason=by_reason,
        per_operator_kept=per_op_kept,
        per_operator_total=per_op_total,
        ab_balance_kept=ab_balance,
        ambiguity_coverage=coverage,
        wall_time_s=wall,
        samples_per_hour=(n / hours) if hours > 0 else 0.0,
        kept_per_hour=(len(kept) / hours) if hours > 0 else 0.0,
    )
