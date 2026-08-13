"""Envelope tier: seeds + an open task, judged post-generation.

Design (settled):
- Seeds steer; the open task generates.  A few randomized levers (domain,
  persona, stakes, anomaly genus, detectability granularity) aim the model's
  dream without defining it.  Blind draws by default; levers may be
  specified or mixed by the caller.
- The payload is the ANOMALY, not an answer.  The corpus harvests
  over-deliberation, so prompts are mundane situations carrying an
  engineered conflict.  There is no canonical answer; culling happens
  post-generation with a batch judge.
- Every lever is a classification of the thing, never a numeric scale or a
  self-rating: genuses name *what kind of wrong the anomaly is*,
  detectability names *where it lives relative to the surface*.  (LLMs
  cannot calibrate 0.0-1.0 scores and rubber-stamp subjective ratings;
  they classify reliably.)
- The authoring JSON carries anomaly ground truth.  That metadata is free
  evaluation signal for later classifier/stratifier audits: we know where
  the bodies are buried, so we can measure detection, not trust judgment.
"""
from __future__ import annotations

import hashlib
import random
from dataclasses import dataclass

from .catalog import (
    ANOMALY_GENUS_DESCRIPTIONS,
    ANOMALY_GENUSES,
    DETECTABILITY_DESCRIPTIONS,
    DETECTABILITY_GRANULARS,
    DOMAIN_PROFILES,
)

__all__ = [
    "ANOMALY_GENUS_DESCRIPTIONS",
    "ANOMALY_GENUSES",
    "AUTHORING_SYSTEM_PROMPT",
    "DETECTABILITY_DESCRIPTIONS",
    "DETECTABILITY_GRANULARS",
    "DOMAIN_PROFILES",
    "Envelope",
    "authoring_messages",
    "freeze_authored",
    "sample_envelope",
]


@dataclass(frozen=True)
class Envelope:
    seed: int
    domain: str
    persona: str
    stakes: str
    genus: str
    detectability: str

    @property
    def envelope_hash(self) -> str:
        payload = repr((self.seed, self.domain, self.persona, self.stakes,
                        self.genus, self.detectability))
        return hashlib.sha256(payload.encode()).hexdigest()


def sample_envelope(
    seed: int,
    domain: str,
    *,
    genus: str | None = None,
    detectability: str | None = None,
) -> Envelope:
    """Draw one envelope deterministically.

    Blind draws by default: genus and detectability come from the seed.
    Callers build mixtures by specifying one or both levers per draw.
    """
    if domain not in DOMAIN_PROFILES:
        raise ValueError(f"unknown domain {domain!r} (known: {', '.join(sorted(DOMAIN_PROFILES))})")
    if genus is not None and genus not in ANOMALY_GENUSES:
        raise ValueError(f"unknown genus {genus!r} (known: {', '.join(ANOMALY_GENUSES)})")
    if detectability is not None and detectability not in DETECTABILITY_GRANULARS:
        raise ValueError(f"unknown detectability {detectability!r} (known: {', '.join(DETECTABILITY_GRANULARS)})")
    digest = hashlib.sha256(f"envelope:{seed}:{domain}".encode()).digest()
    rng = random.Random(int.from_bytes(digest[:8], "big"))
    profile = DOMAIN_PROFILES[domain]
    return Envelope(
        seed=int(seed),
        domain=domain,
        persona=rng.choice(profile["personas"]),
        stakes=rng.choice(profile["stakes"]),
        genus=genus or rng.choice(ANOMALY_GENUSES),
        detectability=detectability or rng.choice(DETECTABILITY_GRANULARS),
    )


AUTHORING_SYSTEM_PROMPT = """\
You are the problem-authoring stage of a dataset-generation pipeline.

Upstream, a deterministic sampler drew the seeds below. They aim you; they
do not define the problem. Downstream, a batch judge will cull what you
produce, and every problem you author is checked against its declared
anomaly metadata, so declare honestly.

Dream up one problem a real person in this situation would actually bring
to an assistant. The surface is mundane: an everyday errand, task, or
question, in an authentic first-person voice, a short paragraph. The
payload is the anomaly: one engineered conflict, planted according to its
genus and placed according to its detectability.

- Genus is what kind of wrong the anomaly is. Build exactly that kind.
- Detectability is where the anomaly lives in the prose. Follow the
  supplied placement definition exactly; it describes textual placement,
  not a numeric difficulty level.
- The person must not notice or flag the anomaly. They think they are
  asking an ordinary question.
- No outside knowledge needed, no textbook phrasing, no bullet lists, no
  mention that anything is designed.

Return one JSON object:
{
  "prose": "<the person's message>",
  "surface_question": "<the mundane question they think they are asking>",
  "anomaly": {
    "genus": "<the genus you were given>",
    "detectability": "<the detectability you were given>",
    "what_is_wrong": "<plain-language description of the conflict>",
    "where_it_lives": "<which details in the prose carry it>",
    "why_it_trips_reasoning": "<one line: what over-deliberation it provokes>"
  }
}
"""


def authoring_messages(envelope: Envelope) -> list[dict[str, str]]:
    lines = [
        f"DOMAIN: {envelope.domain}",
        f"PERSONA: {envelope.persona}",
        f"STAKES: {envelope.stakes}",
        f"ANOMALY GENUS: {envelope.genus}",
        f"GENUS DEFINITION: {ANOMALY_GENUS_DESCRIPTIONS[envelope.genus]}",
        f"DETECTABILITY: {envelope.detectability}",
        f"PLACEMENT DEFINITION: {DETECTABILITY_DESCRIPTIONS[envelope.detectability]}",
        "",
        f"envelope_hash: {envelope.envelope_hash}",
        "",
        "Author the problem.",
    ]
    return [
        {"role": "system", "content": AUTHORING_SYSTEM_PROMPT},
        {"role": "user", "content": "\n".join(lines)},
    ]


def freeze_authored(envelope: Envelope, authored: dict, *, model: str, **provenance) -> dict:
    """Immutable corpus record for an authored problem.

    Binds prose to its envelope provenance and the author's own anomaly
    ground truth.  Culling is downstream (batch judge); freezing records
    what was produced, not what survived.
    """
    prose = str(authored.get("prose") or "").strip()
    if not prose:
        raise ValueError("cannot freeze empty prose")
    if "anomaly" not in authored:
        raise ValueError("authored record missing anomaly ground truth")
    return {
        "envelope_hash": envelope.envelope_hash,
        "surface_hash": hashlib.sha256(prose.encode()).hexdigest(),
        "seed": envelope.seed,
        "domain": envelope.domain,
        "prose": prose,
        "surface_question": authored.get("surface_question"),
        "anomaly": authored["anomaly"],
        "envelope": {
            "persona": envelope.persona,
            "stakes": envelope.stakes,
            "genus": envelope.genus,
            "detectability": envelope.detectability,
        },
        "authoring_model": model,
        "provenance": provenance,
    }
