"""The realization contract: spec -> messages for the prose stage.

This module owns what the realization model is told about its place in the
pipeline.  The model is a cog that knows the machine: it learns what
produced its input, what will happen to its output, and why each rule
exists.  Contract discipline beats prompt pleading -- but a model that
understands the gate is a model that respects it.

Hard invariant: the canonical answer never enters these messages.  The
realization stage may know that an answer exists and is pre-committed; it
may never learn what it is.
"""
from __future__ import annotations

import hashlib
from typing import Any

from .construction import get_construction
from .spec import Spec

__all__ = ["REALIZATION_SYSTEM_PROMPT", "realization_messages", "freeze_record"]

REALIZATION_SYSTEM_PROMPT = """\
You are the prose-realization stage of a dataset-generation pipeline.

Your input comes from a deterministic spec factory: the structured problem
below was assembled deliberately, and a canonical answer has already been
computed from its values. You are not told that answer, and you must never
produce it.

Your output goes to an automated gate. The gate extracts every factual
claim from your prose and diffs it against the spec, solves your prose
independently, and discards anything that drifted. Outputs that fail are
thrown away.

Your single job: write the way a real person would bring this problem to
an assistant. Real people do not pose textbook exercises. They arrive with
circumstances: a reason for asking, a situation, loose phrasing, and the
occasional irrelevant detail. Speak as the persona provided, for the
stakes provided. A short natural message -- a few sentences, first person,
no bullet lists, no meta-commentary.

Gate-enforced rules:
1. Every value under BINDING FACTS must appear in your prose. You may
   phrase around it, but the quantity itself must survive extraction
   unchanged: same number, same unit.
2. Facts under CONTEXT also belong in the message; they are true but do
   not drive the answer.
3. Irrelevant color may come only from DECLARED DISTRACTORS, and you may
   use at most the stated number of them, or none. Invent no other fact
   that constrains the problem.
4. Never state, imply, estimate, or ask about the answer. No solution
   attempt, no guessed figures, no "about X" phrasing that leaks a result.
5. The items under MUST REMAIN OPEN are genuinely unknown in this
   situation. A real person would not know them either. Do not supply or
   imply them, not even as an assumption.
6. Output only the person's message. No preamble, no labels, no quotes.
"""


def realization_messages(spec: Spec) -> list[dict[str, str]]:
    """Build the system+user message pair for the realization model."""
    construction = get_construction(spec.construction_id)
    roles = {s.name: s.role for s in construction.slots}

    binding_lines = [
        f"- {name}: {item.surface}"
        for name, item in sorted(spec.fills.items())
        if roles[name] == "binding"
    ]
    context_lines = [
        f"- {name}: {item.surface}"
        for name, item in sorted(spec.fills.items())
        if roles[name] == "incidental"
    ]
    distractor_lines = [
        f"- {item.surface}"
        for name, item in sorted(spec.fills.items())
        if roles[name] == "distractor"
    ]

    parts: list[str] = [
        f"PERSONA: {spec.authenticity.get('persona', 'an everyday person')}",
        f"STAKES: {spec.authenticity.get('stakes', 'they want a straight answer')}",
        f"REGISTER: {spec.authenticity.get('register', 'casual')}",
        "",
        "BINDING FACTS (must appear exactly as quantities):",
        *binding_lines,
    ]
    if context_lines:
        parts += ["", "CONTEXT (true; include naturally):", *context_lines]
    if distractor_lines:
        parts += [
            "",
            f"DECLARED DISTRACTORS (optional color; use at most {construction.max_distractors}, or none):",
            *distractor_lines,
        ]
    if spec.free_vars():
        parts += ["", "MUST REMAIN OPEN (unknown in this situation; never state or imply):"]
        parts += [f"- {free}" for free in spec.free_vars()]
    parts += ["", f"Problem family: {construction.description}"]

    return [
        {"role": "system", "content": REALIZATION_SYSTEM_PROMPT},
        {"role": "user", "content": "\n".join(parts)},
    ]


def freeze_record(spec: Spec, prose: str, *, model: str, gate: dict[str, Any] | None = None, **provenance: Any) -> dict[str, Any]:
    """Immutable record for an accepted realization.

    The corpus entry binds prose to meaning: semantic identity from the
    spec, surface identity from the prose, and full provenance so any
    future audit can replay generation and re-run the gate.
    """
    if not prose.strip():
        raise ValueError("cannot freeze empty prose")
    return {
        "semantic_hash": spec.semantic_hash,
        "surface_hash": hashlib.sha256(prose.encode()).hexdigest(),
        "construction_id": spec.construction_id,
        "spec_seed": spec.seed,
        "prose": prose,
        "spec": spec.to_dict(),
        "realization_model": model,
        "gate": gate,
        "provenance": provenance,
    }
