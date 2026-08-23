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
- The code arm authors terse, self-contained software-engineering problems.
  The downstream solver reasons over the text alone — no shell, no repo, no
  tools — so the problem statement must carry every fact needed for the
  resolution.
"""
from __future__ import annotations

import hashlib
import random
from dataclasses import dataclass

from .catalog import (
    ANOMALY_GENUS_DESCRIPTIONS,
    ANOMALY_GENUSES,
    CODE_DOMAIN_PROFILES,
    CODE_FRICTION_DESCRIPTIONS,
    CODE_FRICTIONS,
    CODE_TASK_DESCRIPTIONS,
    CODE_TASKS,
    DETECTABILITY_DESCRIPTIONS,
    DETECTABILITY_GRANULARS,
    DOMAIN_PROFILES,
    TEXTURE_DESCRIPTIONS,
    TEXTURES,
    THRASH_AMPLIFIER_DESCRIPTIONS,
    THRASH_AMPLIFIERS,
    THRASH_DOMAIN_PROFILES,
    THRASH_LOAD_DESCRIPTIONS,
    THRASH_LOADS,
    VERIFIABLE_DOMAIN_DESCRIPTIONS,
    VERIFIABLE_DOMAIN_PROFILES,
    VERIFIABLE_SEARCH_TOPOLOGY_DESCRIPTIONS,
    VERIFIABLE_SEARCH_TOPOLOGIES,
    VERIFIABLE_TASK_DESCRIPTIONS,
    VERIFIABLE_TASKS,
    VERIFIABLE_VERIFIER_DESCRIPTIONS,
    VERIFIABLE_VERIFIERS,
)

__all__ = [
    "ANOMALY_GENUS_DESCRIPTIONS",
    "ANOMALY_GENUSES",
    "AUTHORING_SYSTEM_PROMPT",
    "CLEAN_AUTHORING_SYSTEM_PROMPT",
    "CODE_AUTHORING_SYSTEM_PROMPT",
    "VERIFIABLE_AUTHORING_SYSTEM_PROMPT",
    "CODE_DOMAIN_PROFILES",
    "CODE_FRICTION_DESCRIPTIONS",
    "CODE_FRICTIONS",
    "CODE_TASK_DESCRIPTIONS",
    "CODE_TASKS",
    "DETECTABILITY_DESCRIPTIONS",
    "DETECTABILITY_GRANULARS",
    "DOMAIN_PROFILES",
    "Envelope",
    "TEXTURE_DESCRIPTIONS",
    "TEXTURES",
    "THRASH_AMPLIFIER_DESCRIPTIONS",
    "THRASH_AMPLIFIERS",
    "THRASH_AUTHORING_SYSTEM_PROMPT",
    "THRASH_DOMAIN_PROFILES",
    "THRASH_LOAD_DESCRIPTIONS",
    "THRASH_LOADS",
    "VERIFIABLE_DOMAIN_DESCRIPTIONS",
    "VERIFIABLE_DOMAIN_PROFILES",
    "VERIFIABLE_SEARCH_TOPOLOGY_DESCRIPTIONS",
    "VERIFIABLE_SEARCH_TOPOLOGIES",
    "VERIFIABLE_TASK_DESCRIPTIONS",
    "VERIFIABLE_TASKS",
    "VERIFIABLE_VERIFIER_DESCRIPTIONS",
    "VERIFIABLE_VERIFIERS",
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
    genus: str | None = None
    detectability: str | None = None
    texture: str | None = None
    load_type: str | None = None
    amplifier: str | None = None
    task_kind: str | None = None
    friction: str | None = None
    search_topology: str | None = None
    verifier_kind: str | None = None
    objective_task: str | None = None

    @property
    def envelope_hash(self) -> str:
        if self.texture is not None:
            payload = repr((self.seed, self.domain, self.persona, self.stakes,
                            "clean", self.texture))
        elif self.load_type is not None:
            payload = repr((self.seed, self.domain, self.persona, self.stakes,
                            "thrash", self.load_type, self.amplifier))
        elif self.objective_task is not None:
            payload = repr((self.seed, self.domain, self.persona, self.stakes,
                            "verifiable", self.objective_task, self.search_topology,
                            self.verifier_kind))
        elif self.task_kind is not None:
            payload = repr((self.seed, self.domain, self.persona, self.stakes,
                            "code", self.task_kind, self.friction))
        else:
            payload = repr((self.seed, self.domain, self.persona, self.stakes,
                            self.genus, self.detectability))
        return hashlib.sha256(payload.encode()).hexdigest()


def sample_envelope(
    seed: int,
    domain: str,
    *,
    mode: str = "anomaly",
    genus: str | None = None,
    detectability: str | None = None,
    texture: str | None = None,
    load_type: str | None = None,
    amplifier: str | None = None,
    task_kind: str | None = None,
    friction: str | None = None,
    search_topology: str | None = None,
    verifier_kind: str | None = None,
    objective_task: str | None = None,
) -> Envelope:
    """Draw one envelope deterministically.

    Blind draws by default.  In ``mode="anomaly"`` genus and detectability come
    from the seed; in ``mode="clean"`` a reasoning texture comes from the seed;
    in ``mode="thrash"`` a load type and amplifier come from the seed; in
    ``mode="code"`` a task kind and friction come from the seed instead.
    Callers build mixtures by specifying levers per draw.
    """
    if mode not in ("anomaly", "clean", "thrash", "code", "verifiable"):
        raise ValueError(f"unknown mode {mode!r} (expected 'anomaly', 'clean', 'thrash', 'code', or 'verifiable')")

    if mode == "thrash":
        if domain not in THRASH_DOMAIN_PROFILES:
            raise ValueError(f"unknown thrash domain {domain!r} (known: {', '.join(sorted(THRASH_DOMAIN_PROFILES))})")
        if genus is not None or detectability is not None or texture is not None:
            raise ValueError("genus/detectability/texture are not valid in thrash mode")
        if task_kind is not None or friction is not None:
            raise ValueError("task_kind/friction are not valid in thrash mode")
        if load_type is not None and load_type not in THRASH_LOADS:
            raise ValueError(f"unknown load_type {load_type!r} (known: {', '.join(THRASH_LOADS)})")
        if amplifier is not None and amplifier not in THRASH_AMPLIFIERS:
            raise ValueError(f"unknown amplifier {amplifier!r} (known: {', '.join(THRASH_AMPLIFIERS)})")
        profile = THRASH_DOMAIN_PROFILES[domain]
        digest = hashlib.sha256(f"envelope:thrash:{seed}:{domain}".encode()).digest()
        rng = random.Random(int.from_bytes(digest[:8], "big"))
        return Envelope(
            seed=int(seed),
            domain=domain,
            persona=rng.choice(profile["personas"]),
            stakes=rng.choice(profile["stakes"]),
            load_type=load_type or rng.choice(THRASH_LOADS),
            amplifier=amplifier or rng.choice(THRASH_AMPLIFIERS),
        )

    if mode == "verifiable":
        if domain not in VERIFIABLE_DOMAIN_PROFILES:
            raise ValueError(f"unknown verifiable domain {domain!r} (known: {', '.join(sorted(VERIFIABLE_DOMAIN_PROFILES))})")
        if any(value is not None for value in (genus, detectability, texture, load_type, amplifier, task_kind, friction)):
            raise ValueError("anomaly/clean/thrash/code controls are not valid in verifiable mode")
        if objective_task is not None and objective_task not in VERIFIABLE_TASKS:
            raise ValueError(f"unknown objective_task {objective_task!r} (known: {', '.join(VERIFIABLE_TASKS)})")
        if search_topology is not None and search_topology not in VERIFIABLE_SEARCH_TOPOLOGIES:
            raise ValueError(f"unknown search_topology {search_topology!r} (known: {', '.join(VERIFIABLE_SEARCH_TOPOLOGIES)})")
        if verifier_kind is not None and verifier_kind not in VERIFIABLE_VERIFIERS:
            raise ValueError(f"unknown verifier_kind {verifier_kind!r} (known: {', '.join(VERIFIABLE_VERIFIERS)})")
        profile = VERIFIABLE_DOMAIN_PROFILES[domain]
        digest = hashlib.sha256(f"envelope:verifiable:{seed}:{domain}".encode()).digest()
        rng = random.Random(int.from_bytes(digest[:8], "big"))
        return Envelope(
            seed=int(seed),
            domain=domain,
            persona=rng.choice(profile["personas"]),
            stakes=rng.choice(profile["stakes"]),
            objective_task=objective_task or rng.choice(VERIFIABLE_TASKS),
            search_topology=search_topology or rng.choice(VERIFIABLE_SEARCH_TOPOLOGIES),
            verifier_kind=verifier_kind or rng.choice(VERIFIABLE_VERIFIERS),
        )

    if mode == "code":
        if domain not in CODE_DOMAIN_PROFILES:
            raise ValueError(f"unknown code domain {domain!r} (known: {', '.join(sorted(CODE_DOMAIN_PROFILES))})")
        if genus is not None or detectability is not None or texture is not None:
            raise ValueError("genus/detectability/texture are not valid in code mode")
        if load_type is not None or amplifier is not None:
            raise ValueError("load_type/amplifier are not valid in code mode")
        if task_kind is not None and task_kind not in CODE_TASKS:
            raise ValueError(f"unknown task_kind {task_kind!r} (known: {', '.join(CODE_TASKS)})")
        if friction is not None and friction not in CODE_FRICTIONS:
            raise ValueError(f"unknown friction {friction!r} (known: {', '.join(CODE_FRICTIONS)})")
        profile = CODE_DOMAIN_PROFILES[domain]
        digest = hashlib.sha256(f"envelope:code:{seed}:{domain}".encode()).digest()
        rng = random.Random(int.from_bytes(digest[:8], "big"))
        return Envelope(
            seed=int(seed),
            domain=domain,
            persona=rng.choice(profile["personas"]),
            stakes=rng.choice(profile["stakes"]),
            task_kind=task_kind or rng.choice(CODE_TASKS),
            friction=friction or rng.choice(CODE_FRICTIONS),
        )

    if domain not in DOMAIN_PROFILES:
        raise ValueError(f"unknown domain {domain!r} (known: {', '.join(sorted(DOMAIN_PROFILES))})")
    profile = DOMAIN_PROFILES[domain]

    if mode == "clean":
        if genus is not None or detectability is not None:
            raise ValueError("genus/detectability are not valid in clean mode")
        if load_type is not None or amplifier is not None:
            raise ValueError("load_type/amplifier are not valid in clean mode")
        if task_kind is not None or friction is not None:
            raise ValueError("task_kind/friction are not valid in clean mode")
        if texture is not None and texture not in TEXTURES:
            raise ValueError(f"unknown texture {texture!r} (known: {', '.join(TEXTURES)})")
        digest = hashlib.sha256(f"envelope:clean:{seed}:{domain}".encode()).digest()
        rng = random.Random(int.from_bytes(digest[:8], "big"))
        return Envelope(
            seed=int(seed),
            domain=domain,
            persona=rng.choice(profile["personas"]),
            stakes=rng.choice(profile["stakes"]),
            texture=texture or rng.choice(TEXTURES),
        )

    if texture is not None:
        raise ValueError("texture is only valid in clean mode")
    if load_type is not None or amplifier is not None:
        raise ValueError("load_type/amplifier are only valid in thrash mode")
    if task_kind is not None or friction is not None:
        raise ValueError("task_kind/friction are only valid in code mode")
    if genus is not None and genus not in ANOMALY_GENUSES:
        raise ValueError(f"unknown genus {genus!r} (known: {', '.join(ANOMALY_GENUSES)})")
    if detectability is not None and detectability not in DETECTABILITY_GRANULARS:
        raise ValueError(f"unknown detectability {detectability!r} (known: {', '.join(DETECTABILITY_GRANULARS)})")
    digest = hashlib.sha256(f"envelope:{seed}:{domain}".encode()).digest()
    rng = random.Random(int.from_bytes(digest[:8], "big"))
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


CLEAN_AUTHORING_SYSTEM_PROMPT = """\
You are the problem-authoring stage of a dataset-generation pipeline.

Upstream, a deterministic sampler drew the seeds below. They aim you; they
do not define the problem. Downstream, a batch judge will evaluate what you
produce, and every problem you author is checked against its declared
reasoning metadata, so declare honestly.

Dream up one problem a real person in this situation would actually bring
to an assistant. The surface is mundane: an everyday task or question, in
an authentic first-person voice, a short paragraph.

Create a realistic situation where the person needs a concrete response or
decision, but reaching it requires sustained reasoning through multiple
considerations. Do not plant a hidden anomaly or contradiction, and do not
make the problem a puzzle or a calculation with a single derivable answer.
The difficulty should come from interpretation, context, competing concerns,
incomplete disclosure, or practical judgment.

- Texture names the kind of interpretive difficulty. Build exactly that kind.
- The person must need a real commitment: a course of action, a recommended
  approach, a drafted reply, a framing, a plan, or immediate guidance. The
  question must not evaporate into open-ended musing; there is something
  concrete to produce.
- The person may not articulate the full difficulty. They think they are
  asking a normal question; working through it well is harder than it looks.
- No outside knowledge needed, no textbook phrasing, no bullet lists, no
  mention that anything is designed.

Return one JSON object:
{
  "prose": "<the person's message>",
  "surface_question": "<the concrete thing they literally ask for>",
  "reasoning": {
    "texture": "<the texture you were given>",
    "why_sustained_reasoning_needed": "<one line: what must be weighed or interpreted>",
    "decision_target": "<what the assistant is being asked to produce>"
  }
}
"""


THRASH_AUTHORING_SYSTEM_PROMPT = """\
You are the problem-authoring stage of a dataset-generation pipeline.

Upstream, a deterministic sampler drew the seeds below. They aim you; they
do not define the problem. Downstream, a reasoning model will attempt to
solve what you produce, and we want its trace to show sustained, effortful
work with backtracking, re-derivation, and careful checking.

Create a genuinely complex analytical task. Write it in first person with
a brief framing sentence or two (who you are, why you need help), then
present the core task directly with concrete details — names, numbers,
dates, constraints, data points. The task should be bare and direct, not
wrapped in heavy narrative.

The task must NOT have a single clean deterministic answer. It should
require the solver to juggle multiple pieces of information, reconcile
conflicts or gaps, and work through several steps before reaching a
reasonable conclusion.

- Load type names the kind of cognitive work required. Build exactly that.
- Amplifier names what makes the reasoning messy and effortful. Plant
  exactly that kind of difficulty into the material.
- Include enough concrete detail (at least 5-8 distinct facts, names,
  numbers, or constraints) that the solver must track and integrate them.
- The person genuinely needs help — they are not testing or tricking.
- No bullet lists in the prose itself. Write as natural speech/paragraph.

Return one JSON object:
{
  "prose": "<the person's message with the analytical task>",
  "surface_question": "<the direct question or request>",
  "reasoning": {
    "load_type": "<the load type you were given>",
    "amplifier": "<the amplifier you were given>",
    "why_it_thrashes": "<one line: what forces sustained messy reasoning>"
  }
}
"""


VERIFIABLE_AUTHORING_SYSTEM_PROMPT = """\
You are the problem-authoring stage of a dataset-generation pipeline.

Author one self-contained reasoning problem whose answer is objectively
checkable. The problem must be finite and concrete: all relevant facts,
records, rules, code, numbers, and edge cases appear in the prose, and no
obscure outside knowledge is needed. The solver should have to search rather
than apply one obvious local step, but the search pressure must come from the
structure of the material, not from theatrical language or instructions about
how to reason.

The requested task kind and search topology are controls, not a script for the
solver's trace. Build exactly the supplied task and topology. Include an
initially plausible path, interacting constraints or hypotheses, and enough
information for meaningful checking or revision when the path fails. Do not
say "Wait", "Actually", "reconsider", or require any fixed number of revisions;
do not mention chain-of-thought or ask the solver to narrate its reasoning.
Do not make the task subjective, dependent on current events, or solvable only
by recalling domain trivia. Keep the finite instance modest but nontrivial
(roughly 6–12 entities and 8–20 interacting facts or operations), so the
problem is deep enough to search without becoming a brute-force data dump.
Use concise internal planning and spend the output budget on a complete,
checkable problem and metadata rather than an elaborate hidden derivation.

The prose should state a direct request for the answer, exact output, witness,
assignment, trace, proof result, or minimal correction. It must not reveal the
reference answer or verifier to the solver. The verifier kind identifies a
mechanical check, and the metadata must state the canonical answer or a
fully deterministic verification procedure. If multiple outputs are valid,
define a canonical output or an exact acceptance test.

Return exactly one JSON object:
{
  "prose": "<the complete problem statement>",
  "surface_question": "<the direct request>",
  "problem": {
    "task_kind": "<the task kind you were given>",
    "search_topology": "<the topology you were given>",
    "verifier_kind": "<the verifier kind you were given>",
    "what_must_be_produced": "<the concrete deliverable>",
    "where_search_pressure_lives": "<the interacting structure that makes search necessary>",
    "reference_answer": "<the canonical answer or exact expected output>",
    "deterministic_verifier": "<a mechanical, reproducible check of the answer>",
    "key_invariants": "<facts or invariants that must hold during verification>"
  }
}
"""


CODE_AUTHORING_SYSTEM_PROMPT = """\
You are the problem-authoring stage of a dataset-generation pipeline.

Upstream, a deterministic sampler drew the seeds below. They aim you; they
do not define the problem. Downstream, a solver model will attempt what you
produce by reasoning over the text alone — it has no shell, no repository
access, and no tools, and it can never see any file you do not show.

Author one software-engineering problem: a terse, direct problem statement
asking for a working resolution. Write it the way an engineer hands a
problem to a trusted colleague: one or two sentences of framing (who you
are, why it matters now), then the material and the ask, bare and direct.
No chit-chat, no narrative arc, no mention that anything is designed.

The problem must be fully self-contained. Every snippet, error message,
log line, configuration value, and version number needed for a competent
resolution appears in the problem statement itself. If a component is
relevant but not shown, describe its behavior precisely in prose. Nothing
essential may live in a file the solver cannot see, because there are no
files: the problem statement is the entire codebase.

- Task kind names what the solver must produce. Build exactly that ask.
- Friction names where the difficulty lives. Plant exactly that difficulty
  into the material, and make it resolvable from the presented evidence.
- Include at most five code snippets, and make every snippet load-bearing:
  real evidence or material needed for the resolution. Never include a
  snippet merely to distract.
- The asker genuinely needs the problem resolved. They state what they want
  produced; they do not already know the answer.
- The resolution must be reachable by reasoning over the presented material
  alone — no external lookups, no guessing at unseen code.

Return one JSON object:
{
  "prose": "<the problem statement, snippets included>",
  "surface_question": "<the direct ask in one line>",
  "problem": {
    "task_kind": "<the task kind you were given>",
    "friction": "<the friction you were given>",
    "what_must_be_produced": "<the concrete deliverable>",
    "where_the_difficulty_lives": "<which presented details carry it>",
    "why_it_requires_work": "<one line: what forces sustained reasoning>"
  }
}
"""


def authoring_messages(envelope: Envelope) -> list[dict[str, str]]:
    if envelope.objective_task is not None:
        lines = [
            f"DOMAIN: {envelope.domain}",
            f"DOMAIN DEFINITION: {VERIFIABLE_DOMAIN_DESCRIPTIONS[envelope.domain]}",
            f"PERSONA: {envelope.persona}",
            f"STAKES: {envelope.stakes}",
            f"TASK KIND: {envelope.objective_task}",
            f"TASK DEFINITION: {VERIFIABLE_TASK_DESCRIPTIONS[envelope.objective_task]}",
            f"SEARCH TOPOLOGY: {envelope.search_topology}",
            f"TOPOLOGY DEFINITION: {VERIFIABLE_SEARCH_TOPOLOGY_DESCRIPTIONS[envelope.search_topology]}",
            f"VERIFIER KIND: {envelope.verifier_kind}",
            f"VERIFIER DEFINITION: {VERIFIABLE_VERIFIER_DESCRIPTIONS[envelope.verifier_kind]}",
            "",
            f"envelope_hash: {envelope.envelope_hash}",
            "",
            "Author the objectively checkable problem.",
        ]
        return [
            {"role": "system", "content": VERIFIABLE_AUTHORING_SYSTEM_PROMPT},
            {"role": "user", "content": "\n".join(lines)},
        ]
    if envelope.task_kind is not None:
        lines = [
            f"DOMAIN: {envelope.domain}",
            f"PERSONA: {envelope.persona}",
            f"STAKES: {envelope.stakes}",
            f"TASK KIND: {envelope.task_kind}",
            f"TASK DEFINITION: {CODE_TASK_DESCRIPTIONS[envelope.task_kind]}",
            f"FRICTION: {envelope.friction}",
            f"FRICTION DEFINITION: {CODE_FRICTION_DESCRIPTIONS[envelope.friction]}",
            "",
            f"envelope_hash: {envelope.envelope_hash}",
            "",
            "Author the problem.",
        ]
        return [
            {"role": "system", "content": CODE_AUTHORING_SYSTEM_PROMPT},
            {"role": "user", "content": "\n".join(lines)},
        ]
    if envelope.load_type is not None:
        lines = [
            f"DOMAIN: {envelope.domain}",
            f"PERSONA: {envelope.persona}",
            f"STAKES: {envelope.stakes}",
            f"LOAD TYPE: {envelope.load_type}",
            f"LOAD DEFINITION: {THRASH_LOAD_DESCRIPTIONS[envelope.load_type]}",
            f"AMPLIFIER: {envelope.amplifier}",
            f"AMPLIFIER DEFINITION: {THRASH_AMPLIFIER_DESCRIPTIONS[envelope.amplifier]}",
            "",
            f"envelope_hash: {envelope.envelope_hash}",
            "",
            "Author the problem.",
        ]
        return [
            {"role": "system", "content": THRASH_AUTHORING_SYSTEM_PROMPT},
            {"role": "user", "content": "\n".join(lines)},
        ]
    if envelope.texture is not None:
        lines = [
            f"DOMAIN: {envelope.domain}",
            f"PERSONA: {envelope.persona}",
            f"STAKES: {envelope.stakes}",
            f"TEXTURE: {envelope.texture}",
            f"TEXTURE DEFINITION: {TEXTURE_DESCRIPTIONS[envelope.texture]}",
            "",
            f"envelope_hash: {envelope.envelope_hash}",
            "",
            "Author the problem.",
        ]
        return [
            {"role": "system", "content": CLEAN_AUTHORING_SYSTEM_PROMPT},
            {"role": "user", "content": "\n".join(lines)},
        ]
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

    Binds prose to its envelope provenance and the author's own ground truth.
    Culling is downstream (batch judge); freezing records what was produced,
    not what survived.  Anomaly envelopes carry ``anomaly`` ground truth;
    clean and thrash envelopes carry ``reasoning`` ground truth; code
    envelopes carry ``problem`` ground truth.
    """
    prose = str(authored.get("prose") or "").strip()
    if not prose:
        raise ValueError("cannot freeze empty prose")
    if envelope.objective_task is not None:
        problem = authored.get("problem")
        if not isinstance(problem, dict):
            raise ValueError("authored verifiable problem ground truth must be an object")
        for field, expected in (
            ("task_kind", envelope.objective_task),
            ("search_topology", envelope.search_topology),
            ("verifier_kind", envelope.verifier_kind),
        ):
            if problem.get(field) != expected:
                raise ValueError(
                    f"authored verifiable problem {field} does not match envelope "
                    f"({problem.get(field)!r} != {expected!r})")
        if not isinstance(authored.get("surface_question"), str) or not authored["surface_question"].strip():
            raise ValueError("authored verifiable problem missing non-empty surface_question")
        for field in (
            "what_must_be_produced",
            "where_search_pressure_lives",
            "reference_answer",
            "deterministic_verifier",
            "key_invariants",
        ):
            value = problem.get(field)
            if not isinstance(value, str) or not value.strip():
                raise ValueError(f"authored verifiable problem missing non-empty {field}")
        return {
            "envelope_hash": envelope.envelope_hash,
            "surface_hash": hashlib.sha256(prose.encode()).hexdigest(),
            "seed": envelope.seed,
            "domain": envelope.domain,
            "prose": prose,
            "surface_question": authored["surface_question"],
            "problem": problem,
            "envelope": {
                "persona": envelope.persona,
                "stakes": envelope.stakes,
                "objective_task": envelope.objective_task,
                "search_topology": envelope.search_topology,
                "verifier_kind": envelope.verifier_kind,
            },
            "authoring_model": model,
            "provenance": provenance,
        }
    if envelope.load_type is not None:
        if "reasoning" not in authored:
            raise ValueError("authored record missing reasoning ground truth")
        return {
            "envelope_hash": envelope.envelope_hash,
            "surface_hash": hashlib.sha256(prose.encode()).hexdigest(),
            "seed": envelope.seed,
            "domain": envelope.domain,
            "prose": prose,
            "surface_question": authored.get("surface_question"),
            "reasoning": authored["reasoning"],
            "envelope": {
                "persona": envelope.persona,
                "stakes": envelope.stakes,
                "load_type": envelope.load_type,
                "amplifier": envelope.amplifier,
            },
            "authoring_model": model,
            "provenance": provenance,
        }
    if envelope.task_kind is not None:
        if "problem" not in authored:
            raise ValueError("authored record missing problem ground truth")
        problem = authored["problem"]
        if not isinstance(problem, dict):
            raise ValueError("authored problem ground truth must be an object")
        for field, expected in (("task_kind", envelope.task_kind),
                                ("friction", envelope.friction)):
            if problem.get(field) != expected:
                raise ValueError(
                    f"authored problem {field} does not match envelope "
                    f"({problem.get(field)!r} != {expected!r})")
        surface_question = authored.get("surface_question")
        if not isinstance(surface_question, str) or not surface_question.strip():
            raise ValueError("authored code problem missing non-empty surface_question")
        for field in (
            "what_must_be_produced",
            "where_the_difficulty_lives",
            "why_it_requires_work",
        ):
            value = problem.get(field)
            if not isinstance(value, str) or not value.strip():
                raise ValueError(f"authored problem missing non-empty {field}")
        return {
            "envelope_hash": envelope.envelope_hash,
            "surface_hash": hashlib.sha256(prose.encode()).hexdigest(),
            "seed": envelope.seed,
            "domain": envelope.domain,
            "prose": prose,
            "surface_question": authored.get("surface_question"),
            "problem": problem,
            "envelope": {
                "persona": envelope.persona,
                "stakes": envelope.stakes,
                "task_kind": envelope.task_kind,
                "friction": envelope.friction,
            },
            "authoring_model": model,
            "provenance": provenance,
        }
    if envelope.texture is not None:
        if "reasoning" not in authored:
            raise ValueError("authored record missing reasoning ground truth")
        return {
            "envelope_hash": envelope.envelope_hash,
            "surface_hash": hashlib.sha256(prose.encode()).hexdigest(),
            "seed": envelope.seed,
            "domain": envelope.domain,
            "prose": prose,
            "surface_question": authored.get("surface_question"),
            "reasoning": authored["reasoning"],
            "envelope": {
                "persona": envelope.persona,
                "stakes": envelope.stakes,
                "texture": envelope.texture,
            },
            "authoring_model": model,
            "provenance": provenance,
        }
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
