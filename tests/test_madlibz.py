"""Tests for the madlibz anomaly envelope tier."""
from __future__ import annotations

import pytest

from mlfactory.core.madlibz import (
    ANOMALY_GENUSES,
    AUTHORING_SYSTEM_PROMPT,
    DETECTABILITY_GRANULARS,
    authoring_messages,
    freeze_authored,
    sample_envelope,
)


def test_blind_draws_are_deterministic_and_valid():
    a = sample_envelope(seed=12, domain="household")
    b = sample_envelope(seed=12, domain="household")
    assert a == b and a.envelope_hash == b.envelope_hash
    assert a.genus in ANOMALY_GENUSES
    assert a.detectability in DETECTABILITY_GRANULARS
    assert a.persona and a.stakes


def test_lever_overrides_and_mixtures():
    blind = sample_envelope(seed=12, domain="household")
    specified = sample_envelope(seed=12, domain="household",
                                genus="temporal_conflict", detectability="hidden")
    assert specified.genus == "temporal_conflict" and specified.detectability == "hidden"
    assert specified.envelope_hash != blind.envelope_hash
    # Persona/stakes still come from the blind draw when unspecified.
    assert specified.persona == blind.persona and specified.stakes == blind.stakes


def test_invalid_levers_rejected():
    with pytest.raises(ValueError):
        sample_envelope(seed=1, domain="household", genus="bogus")
    with pytest.raises(ValueError):
        sample_envelope(seed=1, domain="household", detectability="0.7")
    with pytest.raises(ValueError):
        sample_envelope(seed=1, domain="no_such_domain")


def test_authoring_contract_is_classification_not_scoring():
    # The authoring prompt must instruct by placement and kind, never by
    # numeric scale or self-rating.
    assert "0.0" not in AUTHORING_SYSTEM_PROMPT
    assert "confidence" not in AUTHORING_SYSTEM_PROMPT.lower()
    for granular in DETECTABILITY_GRANULARS:
        assert granular in AUTHORING_SYSTEM_PROMPT
    for key in ("prose", "surface_question", "anomaly", "what_is_wrong", "where_it_lives"):
        assert key in AUTHORING_SYSTEM_PROMPT


def test_authoring_messages_carry_envelope():
    env = sample_envelope(seed=4, domain="gardening", genus="red_herring", detectability="indirection")
    user = next(m["content"] for m in authoring_messages(env) if m["role"] == "user")
    assert env.genus in user and env.detectability in user
    assert env.persona in user and env.stakes in user
    assert env.envelope_hash in user


def test_freeze_authored():
    env = sample_envelope(seed=4, domain="gardening", genus="red_herring", detectability="indirection")
    rec = freeze_authored(env, {
        "prose": "quick question about my tomatoes...",
        "surface_question": "when to water",
        "anomaly": {"genus": env.genus, "detectability": env.detectability,
                    "what_is_wrong": "x", "where_it_lives": "y",
                    "why_it_trips_reasoning": "z"},
    }, model="test-model", run="r1")
    assert rec["envelope_hash"] == env.envelope_hash
    assert rec["anomaly"]["genus"] == env.genus
    assert rec["envelope"]["detectability"] == env.detectability
    assert rec["provenance"] == {"run": "r1"}
    with pytest.raises(ValueError):
        freeze_authored(env, {"prose": "  ", "anomaly": {}}, model="test-model")
    with pytest.raises(ValueError):
        freeze_authored(env, {"prose": "hi"}, model="test-model")
