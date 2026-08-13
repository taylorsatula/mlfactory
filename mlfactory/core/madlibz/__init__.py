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
    DETECTABILITY_DESCRIPTIONS,
    DETECTABILITY_GRANULARS,
    DOMAIN_PROFILES,
    Envelope,
    authoring_messages,
    freeze_authored,
    sample_envelope,
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
