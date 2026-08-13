"""Spec schema: the semantic objects the deterministic factory produces.

A *spec* is one fully-decided problem world: typed fills for every declared
slot, free variables that must stay unbound, difficulty features, and an
authenticity profile.  Identity lives here, not in prose: ``semantic_hash``
dedups at the level of meaning, so surface variation can never fake
diversity and surface identity can never hide it.
"""
from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass, field
from typing import Any


@dataclass(frozen=True)
class SlotType:
    """A slot's type is a small feature structure.

    kind: "numeric" samples from [lo, hi]; "choice"/"text" sample from a
    registered paradigm named by ``paradigm``.  ``unit`` rides with numeric
    types so the answer function and the gate can reason about units.
    """

    kind: str
    unit: str | None = None
    lo: float | None = None
    hi: float | None = None
    integer: bool = False
    paradigm: str | None = None
    distinct_from: tuple[str, ...] = ()


@dataclass(frozen=True)
class Slot:
    """One declared position in a construction.

    role semantics:
      binding    -- the value feeds the answer function; the prose must
                    carry it and the gate will extract and diff it.
      incidental -- appears as a fact in the prose but does not bind the
                    answer (places, names); gate still diffs it.
      distractor -- optional color; may be omitted entirely, must not bind.
    """

    name: str
    type: SlotType
    role: str = "binding"


@dataclass(frozen=True)
class FillItem:
    """A sampled fill: surface form plus the semantic value behind it."""

    surface: str
    value: Any


@dataclass(frozen=True)
class Answer:
    """Canonical answer computed from fills -- never stored as prose.

    kind "exact": ``value`` is checkable equality.
    kind "rubric": ``rubric`` describes graded/assumption-bearing answers;
    ``free_vars`` names what the prompt must leave unbound.
    """

    kind: str
    value: Any = None
    rubric: str | None = None
    free_vars: tuple[str, ...] = ()


@dataclass
class Spec:
    """One instance of a construction: the contract all stages negotiate over."""

    construction_id: str
    seed: int
    fills: dict[str, FillItem]
    answer: Answer
    difficulty: dict[str, Any] = field(default_factory=dict)
    authenticity: dict[str, Any] = field(default_factory=dict)

    @property
    def semantic_hash(self) -> str:
        canonical = {
            "construction": self.construction_id,
            "fills": {name: item.value for name, item in sorted(self.fills.items())},
        }
        payload = json.dumps(canonical, sort_keys=True, default=str)
        return hashlib.sha256(payload.encode()).hexdigest()

    def binding_fills(self, construction: "Construction | None" = None) -> dict[str, FillItem]:  # noqa: F821
        if construction is None:
            return dict(self.fills)
        binding = {s.name for s in construction.slots if s.role == "binding"}
        return {k: v for k, v in self.fills.items() if k in binding}

    def free_vars(self) -> tuple[str, ...]:
        return self.answer.free_vars

    def to_dict(self) -> dict[str, Any]:
        return {
            "construction_id": self.construction_id,
            "seed": self.seed,
            "semantic_hash": self.semantic_hash,
            "fills": {k: {"surface": v.surface, "value": v.value} for k, v in self.fills.items()},
            "answer": {
                "kind": self.answer.kind,
                "value": self.answer.value,
                "rubric": self.answer.rubric,
                "free_vars": list(self.answer.free_vars),
            },
            "difficulty": self.difficulty,
            "authenticity": self.authenticity,
        }
