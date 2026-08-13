"""Constructions: form-meaning pairings, the syntagmatic axis.

A construction declares typed slots, computes the canonical answer from
fill values, vets fills with a felicity predicate, and supplies
authenticity pools (personas, stakes, registers) that the sampler draws
from.  The answer function is the construction's *meaning side*: answers
are always derived from fills, never stored as prose, and never sampled.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Callable

from .spec import Answer, Slot

__all__ = ["AuthenticityPools", "Construction", "declare_construction", "get_construction", "construction_ids"]


@dataclass(frozen=True)
class AuthenticityPools:
    """Pools the sampler draws from to make instances feel lived-in.

    Personas and stakes are short descriptive phrases; the realization
    stage turns them into voice.  Registers name the prose style.
    """

    personas: tuple[str, ...]
    stakes: tuple[str, ...]
    registers: tuple[str, ...] = ("casual",)


@dataclass(frozen=True)
class Construction:
    """One problem family.

    answer_fn receives {slot_name: value} for binding slots and returns an
    Answer.  felicity_fn receives the same dict and rejects degenerate
    regions (sampler resamples).  difficulty_fn returns declared features
    for stratification.
    """

    id: str
    domain: str
    description: str
    slots: tuple[Slot, ...]
    answer_fn: Callable[[dict[str, Any]], Answer]
    authenticity: AuthenticityPools
    felicity_fn: Callable[[dict[str, Any]], bool] = lambda values: True
    difficulty_fn: Callable[[dict[str, Any]], dict[str, Any]] = lambda values: {}
    max_distractors: int = 1
    metadata: dict[str, Any] = field(default_factory=dict)

    def slot(self, name: str) -> Slot:
        for s in self.slots:
            if s.name == name:
                return s
        raise ValueError(f"construction {self.id!r} has no slot {name!r}")


CONSTRUCTIONS: dict[str, Construction] = {}


def declare_construction(c: Construction) -> Construction:
    if not c.id or not c.id.strip():
        raise ValueError("construction id must be non-empty")
    if c.id in CONSTRUCTIONS:
        raise ValueError(f"construction {c.id!r} is already declared")
    names = [s.name for s in c.slots]
    if len(names) != len(set(names)):
        raise ValueError(f"construction {c.id!r} has duplicate slot names")
    for s in c.slots:
        for other in s.type.distinct_from:
            if other not in names:
                raise ValueError(f"slot {s.name!r} distinct_from unknown slot {other!r}")
    if not c.authenticity.personas or not c.authenticity.stakes:
        raise ValueError(f"construction {c.id!r} needs non-empty persona and stakes pools")
    CONSTRUCTIONS[c.id] = c
    return c


def get_construction(construction_id: str) -> Construction:
    if construction_id not in CONSTRUCTIONS:
        raise ValueError(
            f"unknown construction {construction_id!r} (declared: {', '.join(sorted(CONSTRUCTIONS))})"
        )
    return CONSTRUCTIONS[construction_id]


def construction_ids() -> tuple[str, ...]:
    return tuple(sorted(CONSTRUCTIONS))
