"""CBF-QP Safety Shield and Primitive Compiler (spec §6, §5)."""

from kino_vla.shield.cbf_shield import AdmissionResult, CbfShield, ShieldStats
from kino_vla.shield.latency import BUDGETS_MS, LatencyBudget
from kino_vla.shield.passthrough import PassThroughShield, ShieldDecision
from kino_vla.shield.primitive_compiler import (
    AdjustPosture,
    Backstep,
    CompiledCommand,
    HoldAndRequest,
    PrimitiveCompiler,
    ReplanWaypoint,
    SetConstraint,
    SwitchGait,
    UpdateTopology,
)
from kino_vla.shield.qp import CbfConstraints, QpResult, build_constraints, solve_cbf_qp

__all__ = [
    "AdjustPosture",
    "AdmissionResult",
    "BUDGETS_MS",
    "Backstep",
    "CbfConstraints",
    "CbfShield",
    "CompiledCommand",
    "HoldAndRequest",
    "LatencyBudget",
    "PassThroughShield",
    "PrimitiveCompiler",
    "QpResult",
    "ReplanWaypoint",
    "SetConstraint",
    "ShieldDecision",
    "ShieldStats",
    "SwitchGait",
    "UpdateTopology",
    "build_constraints",
    "solve_cbf_qp",
]
