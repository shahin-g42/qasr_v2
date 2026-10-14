"""Executable oracle rules: one frozen semantic per symbolic world kind.

Each rule module exposes KIND, RULE_VERSION, TASK, validate, solve,
independent_solve, surface and attributes_for. rule_for() dispatches a world
to its rule by kind and rejects unknown kinds; the oracle facade in
jev_data.oracle performs the generic evidence, hash, target and lineage
checks around these primitives, so every rule inherits the same audit
envelope without re-implementing it.
"""

from __future__ import annotations

from types import ModuleType

from ..io import require
from . import (
    budget,
    comparison,
    policy,
    readback,
    routing,
    safety,
    slots,
    temporal,
    triage,
)

_MODULES = (budget, comparison, policy, readback, routing, safety, slots, temporal, triage)
RULES = {module.KIND: module for module in _MODULES}
require(len(RULES) == len(_MODULES), "duplicate oracle rule kind")


def rule_for(world: dict) -> ModuleType:
    """Dispatch a symbolic world to its rule by kind; unknown kinds are rejected."""
    require(type(world) is dict, "symbolic world must be an object")
    kind = world.get("kind")
    require(type(kind) is str and kind in RULES, "unsupported symbolic world")
    return RULES[kind]
