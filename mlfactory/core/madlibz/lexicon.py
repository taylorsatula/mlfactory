"""Typed word/number packs: the paradigmatic axis.

A paradigm is an ordered pool of items for one semantic category.  Items
are (surface, value) pairs: the surface is what prose carries, the value is
what the answer function and the gate operate on.  For pure-text paradigms
the value defaults to the surface itself.

Paradigms are the primary extension point: registering a new pool (or
adding items to one) instantly diversifies every construction whose slots
reference it, without touching any construction code.
"""
from __future__ import annotations

import random
from dataclasses import dataclass
from typing import Any

__all__ = ["ParadigmItem", "register_paradigm", "get_paradigm", "paradigm_names", "sample_item"]


@dataclass(frozen=True)
class ParadigmItem:
    surface: str
    value: Any = None

    def resolved_value(self) -> Any:
        return self.surface if self.value is None else self.value


PARADIGMS: dict[str, tuple[ParadigmItem, ...]] = {}


def register_paradigm(name: str, items: list[ParadigmItem | tuple[str, Any] | str], *, replace: bool = False) -> None:
    """Register or extend one paradigm. Duplicate names raise unless replace."""
    if not name or not name.strip():
        raise ValueError("paradigm name must be non-empty")
    resolved: list[ParadigmItem] = []
    for item in items:
        if isinstance(item, ParadigmItem):
            resolved.append(item)
        elif isinstance(item, tuple):
            surface, value = item
            resolved.append(ParadigmItem(surface=str(surface), value=value))
        else:
            resolved.append(ParadigmItem(surface=str(item)))
    if not resolved:
        raise ValueError(f"paradigm {name!r} needs at least one item")
    if name in PARADIGMS and not replace:
        existing = PARADIGMS[name]
        seen = {i.surface for i in existing}
        resolved = list(existing) + [i for i in resolved if i.surface not in seen]
    PARADIGMS[name] = tuple(resolved)


def get_paradigm(name: str) -> tuple[ParadigmItem, ...]:
    if name not in PARADIGMS:
        raise ValueError(f"unknown paradigm {name!r} (registered: {', '.join(sorted(PARADIGMS))})")
    return PARADIGMS[name]


def paradigm_names() -> tuple[str, ...]:
    return tuple(sorted(PARADIGMS))


def sample_item(name: str, rng: random.Random) -> ParadigmItem:
    return rng.choice(get_paradigm(name))


# ---------------------------------------------------------------------------
# Starter packs: concise and generalized; extend via register_paradigm().
# ---------------------------------------------------------------------------
register_paradigm("place", [
    "Marlow", "Fenwick", "Dunmore", "Kettleford", "Ashby", "Rivenhall",
    "Port Ellery", "Camden Crossing", "Norvale", "Weston Flats",
])
register_paradigm("person_name", [
    "Dana", "Ilya", "Marisol", "Kenji", "Priya", "Tomas", "Aisha", "Ruben",
])
register_paradigm("ride_detail", [
    "an old road bike", "a borrowed commuter bike", "a new cassette",
    "a friend riding along", "errands on the far end", "a podcast queued up",
])
register_paradigm("liquid", [
    ("saline", "saline"), ("vinegar", "vinegar"), ("antifreeze", "antifreeze"),
    ("fertilizer feed", "fertilizer feed"),
])
