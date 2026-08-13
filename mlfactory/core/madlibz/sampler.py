"""Deterministic sampling: seed -> Spec.

Selection is blind in the Mad Libs sense: fills are drawn against slot
types and authenticity pools, never against a specific prose outcome.
Felicity failures and distinctness violations resample the whole fill
vector (cheap: generation is free).  The same seed and construction
always produce the same Spec.
"""
from __future__ import annotations

import hashlib
import random
from typing import Any

from .construction import get_construction
from .lexicon import sample_item
from .spec import FillItem, Spec

__all__ = ["sample_spec"]


def _attempt_rng(seed: int, construction_id: str, attempt: int) -> random.Random:
    # Python >= 3.11 only accepts None/int/float/str/bytes as seeds, so
    # derive one deterministically.  sha256 (not hash()) stays stable
    # across processes and PYTHONHASHSEED values.
    digest = hashlib.sha256(f"{seed}:{construction_id}:{attempt}".encode()).digest()
    return random.Random(int.from_bytes(digest[:8], "big"))


def _sample_value(slot_type, rng: random.Random) -> FillItem:
    if slot_type.kind == "numeric":
        lo, hi = slot_type.lo, slot_type.hi
        if lo is None or hi is None:
            raise ValueError("numeric slot type requires lo and hi")
        if slot_type.integer:
            value: Any = rng.randint(int(lo), int(hi))
        else:
            value = round(rng.uniform(lo, hi), 2)
        surface = f"{value:g}" + (f" {slot_type.unit}" if slot_type.unit else "")
        return FillItem(surface=surface, value=value)
    if slot_type.kind in ("choice", "text"):
        if not slot_type.paradigm:
            raise ValueError(f"{slot_type.kind} slot type requires a paradigm")
        item = sample_item(slot_type.paradigm, rng)
        return FillItem(surface=item.surface, value=item.resolved_value())
    raise ValueError(f"unknown slot type kind {slot_type.kind!r}")


def _distinct_ok(construction, values: dict[str, Any]) -> bool:
    for s in construction.slots:
        for other in s.type.distinct_from:
            if values.get(s.name) == values.get(other):
                return False
    return True


def sample_spec(seed: int, construction_id: str, *, max_attempts: int = 64) -> Spec:
    """Draw one Spec deterministically from a seed.

    Attempt sub-seeds derive from (seed, attempt) so replay never depends
    on rejection history of other draws.
    """
    construction = get_construction(construction_id)
    for attempt in range(max_attempts):
        rng = _attempt_rng(seed, construction_id, attempt)
        fills: dict[str, FillItem] = {}
        for slot in construction.slots:
            fills[slot.name] = _sample_value(slot.type, rng)
        values = {name: item.value for name, item in fills.items()}
        if not _distinct_ok(construction, values):
            continue
        binding_values = {
            name: item.value
            for name, item in fills.items()
            if construction.slot(name).role == "binding"
        }
        if not construction.felicity_fn(binding_values):
            continue
        persona = rng.choice(construction.authenticity.personas)
        stakes = rng.choice(construction.authenticity.stakes)
        register = rng.choice(construction.authenticity.registers)
        return Spec(
            construction_id=construction_id,
            seed=int(seed),
            fills=fills,
            answer=construction.answer_fn(binding_values),
            difficulty=construction.difficulty_fn(binding_values),
            authenticity={"persona": persona, "stakes": stakes, "register": register},
        )
    raise RuntimeError(
        f"construction {construction_id!r} produced no felicitous fill in {max_attempts} attempts (seed {seed})"
    )
