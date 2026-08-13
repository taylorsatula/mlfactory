"""madlibz — separation-of-concerns prompt-spec generation for mlfactory.

Three stages, strict division of authority:

1. Spec factory (this package, deterministic): constructions declare typed
   slots and compute canonical answers from fills; the sampler draws
   seed-deterministic Specs.  Identity lives at the semantic level
   (``Spec.semantic_hash``); surface text is augmentation, never identity.
2. Prose realization (probabilistic, external model): ``realization_messages``
   hands the model a cog-aware contract -- it knows what produced its input
   and what the downstream gate will check.  The canonical answer never
   enters those messages.
3. Gate and freeze (deterministic authority): accepted prose is frozen via
   ``freeze_record`` with full provenance.

Quick start::

    from mlfactory.core.madlibz import sample_spec, realization_messages
    import mlfactory.core.madlibz.constructions  # registers reference families

    spec = sample_spec(seed=7, construction_id="round_trip_speed")
    messages = realization_messages(spec)   # -> model writes prose
    # gate verifies prose against spec; then:
    # freeze_record(spec, prose, model="...")
"""
from .construction import (
    CONSTRUCTIONS,
    AuthenticityPools,
    Construction,
    construction_ids,
    declare_construction,
    get_construction,
)
from .constructions import round_trip_speed, round_trip_speed_open_distance
from .constructions_garden import raised_bed_soil_mix, seed_starting_countback
from .envelope import (
    ANOMALY_GENUSES,
    AUTHORING_SYSTEM_PROMPT,
    DETECTABILITY_GRANULARS,
    DOMAIN_PROFILES,
    Envelope,
    authoring_messages,
    freeze_authored,
    sample_envelope,
)
from .lexicon import PARADIGMS, ParadigmItem, get_paradigm, paradigm_names, register_paradigm, sample_item
from .realization import REALIZATION_SYSTEM_PROMPT, freeze_record, realization_messages
from .sampler import sample_spec
from .spec import Answer, FillItem, Slot, SlotType, Spec

__all__ = [
    "Answer",
    "AuthenticityPools",
    "ANOMALY_GENUSES",
    "AUTHORING_SYSTEM_PROMPT",
    "CONSTRUCTIONS",
    "Construction",
    "DETECTABILITY_GRANULARS",
    "DOMAIN_PROFILES",
    "Envelope",
    "FillItem",
    "PARADIGMS",
    "ParadigmItem",
    "REALIZATION_SYSTEM_PROMPT",
    "Slot",
    "SlotType",
    "Spec",
    "construction_ids",
    "declare_construction",
    "freeze_record",
    "get_construction",
    "get_paradigm",
    "paradigm_names",
    "raised_bed_soil_mix",
    "realization_messages",
    "register_paradigm",
    "round_trip_speed",
    "round_trip_speed_open_distance",
    "sample_item",
    "sample_spec",
    "sample_envelope",
    "authoring_messages",
    "freeze_authored",
    "seed_starting_countback",
]
