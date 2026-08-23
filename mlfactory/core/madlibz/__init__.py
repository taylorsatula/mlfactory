"""madlibz — anomaly-seeded prompt generation for mlfactory experiments.

Pipeline (settled design):

1. Envelope sampling (deterministic): ``sample_envelope`` draws seeds —
   domain, persona, stakes, anomaly genus, detectability granularity —
   blind by default, optionally specified or mixed by the caller.
2. Authoring (probabilistic, external model): ``authoring_messages`` hands
   the model the seeds and an open task.  The model dreams up a mundane
   situation carrying one engineered anomaly and returns prose plus
   anomaly ground truth.
3. Freeze (deterministic authority): ``freeze_authored`` binds prose to
   envelope provenance and the declared anomaly.  Culling happens
   downstream with a batch judge; freezing records what was produced.

Quick start::

    from mlfactory.core.madlibz import sample_envelope, authoring_messages, freeze_authored

    env = sample_envelope(
        seed=12,
        domain="diner_breakfast_shift",
        genus="temporal_conflict",
        detectability="distant_pair",
    )
    messages = authoring_messages(env)   # -> model authors the problem
    # freeze_authored(env, authored_json, model="...")
"""
from .envelope import (
    ANOMALY_GENUS_DESCRIPTIONS,
    ANOMALY_GENUSES,
    AUTHORING_SYSTEM_PROMPT,
    CLEAN_AUTHORING_SYSTEM_PROMPT,
    CODE_AUTHORING_SYSTEM_PROMPT,
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
    THRASH_AUTHORING_SYSTEM_PROMPT,
    THRASH_DOMAIN_PROFILES,
    THRASH_LOAD_DESCRIPTIONS,
    THRASH_LOADS,
    VERIFIABLE_AUTHORING_SYSTEM_PROMPT,
    VERIFIABLE_DOMAIN_DESCRIPTIONS,
    VERIFIABLE_DOMAIN_PROFILES,
    VERIFIABLE_SEARCH_TOPOLOGY_DESCRIPTIONS,
    VERIFIABLE_SEARCH_TOPOLOGIES,
    VERIFIABLE_TASK_DESCRIPTIONS,
    VERIFIABLE_TASKS,
    VERIFIABLE_VERIFIER_DESCRIPTIONS,
    VERIFIABLE_VERIFIERS,
    Envelope,
    authoring_messages,
    freeze_authored,
    sample_envelope,
)

__all__ = [
    "ANOMALY_GENUS_DESCRIPTIONS",
    "ANOMALY_GENUSES",
    "AUTHORING_SYSTEM_PROMPT",
    "CLEAN_AUTHORING_SYSTEM_PROMPT",
    "CODE_AUTHORING_SYSTEM_PROMPT",
    "CODE_DOMAIN_PROFILES",
    "CODE_FRICTION_DESCRIPTIONS",
    "CODE_FRICTIONS",
    "CODE_TASK_DESCRIPTIONS",
    "CODE_TASKS",
    "DETECTABILITY_DESCRIPTIONS",
    "DETECTABILITY_GRANULARS",
    "DOMAIN_PROFILES",
    "TEXTURE_DESCRIPTIONS",
    "TEXTURES",
    "THRASH_AMPLIFIER_DESCRIPTIONS",
    "THRASH_AMPLIFIERS",
    "THRASH_AUTHORING_SYSTEM_PROMPT",
    "THRASH_DOMAIN_PROFILES",
    "THRASH_LOAD_DESCRIPTIONS",
    "THRASH_LOADS",
    "VERIFIABLE_AUTHORING_SYSTEM_PROMPT",
    "VERIFIABLE_DOMAIN_DESCRIPTIONS",
    "VERIFIABLE_DOMAIN_PROFILES",
    "VERIFIABLE_SEARCH_TOPOLOGY_DESCRIPTIONS",
    "VERIFIABLE_SEARCH_TOPOLOGIES",
    "VERIFIABLE_TASK_DESCRIPTIONS",
    "VERIFIABLE_TASKS",
    "VERIFIABLE_VERIFIER_DESCRIPTIONS",
    "VERIFIABLE_VERIFIERS",
    "Envelope",
    "authoring_messages",
    "freeze_authored",
    "sample_envelope",
]
