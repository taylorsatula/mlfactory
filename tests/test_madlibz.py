"""Tests for the madlibz anomaly envelope tier."""
from __future__ import annotations

import pytest

from mlfactory.core.madlibz import (
    ANOMALY_GENUS_DESCRIPTIONS,
    ANOMALY_GENUSES,
    AUTHORING_SYSTEM_PROMPT,
    DETECTABILITY_DESCRIPTIONS,
    DETECTABILITY_GRANULARS,
    DOMAIN_PROFILES,
    authoring_messages,
    freeze_authored,
    sample_envelope,
)


def test_catalog_has_curated_control_surfaces():
    assert len(DOMAIN_PROFILES) == 35
    assert len(ANOMALY_GENUSES) == 25
    assert len(DETECTABILITY_GRANULARS) == 25
    assert set(ANOMALY_GENUSES) == set(ANOMALY_GENUS_DESCRIPTIONS)
    assert set(DETECTABILITY_GRANULARS) == set(DETECTABILITY_DESCRIPTIONS)
    assert all(set(profile) == {"personas", "stakes"} for profile in DOMAIN_PROFILES.values())
    assert all(len(profile["personas"]) == 4 for profile in DOMAIN_PROFILES.values())
    assert all(len(profile["stakes"]) == 4 for profile in DOMAIN_PROFILES.values())


def test_blind_draws_are_deterministic_and_valid():
    a = sample_envelope(seed=12, domain="diner_breakfast_shift")
    b = sample_envelope(seed=12, domain="diner_breakfast_shift")
    assert a == b and a.envelope_hash == b.envelope_hash
    assert a.genus in ANOMALY_GENUSES
    assert a.detectability in DETECTABILITY_GRANULARS
    assert a.persona and a.stakes


def test_lever_overrides_and_mixtures():
    blind = sample_envelope(seed=12, domain="diner_breakfast_shift")
    specified = sample_envelope(seed=12, domain="diner_breakfast_shift",
                                genus="temporal_conflict", detectability="distant_pair")
    assert specified.genus == "temporal_conflict" and specified.detectability == "distant_pair"
    assert specified.envelope_hash != blind.envelope_hash
    # Persona/stakes still come from the blind draw when unspecified.
    assert specified.persona == blind.persona and specified.stakes == blind.stakes


def test_invalid_levers_rejected():
    with pytest.raises(ValueError):
        sample_envelope(seed=1, domain="diner_breakfast_shift", genus="bogus")
    with pytest.raises(ValueError):
        sample_envelope(seed=1, domain="diner_breakfast_shift", detectability="0.7")
    with pytest.raises(ValueError):
        sample_envelope(seed=1, domain="no_such_domain")


def test_authoring_contract_is_classification_not_scoring():
    # The authoring prompt must instruct by placement and kind, never by
    # numeric scale or self-rating.
    assert "0.0" not in AUTHORING_SYSTEM_PROMPT
    assert "confidence" not in AUTHORING_SYSTEM_PROMPT.lower()
    for key in ("prose", "surface_question", "anomaly", "what_is_wrong", "where_it_lives"):
        assert key in AUTHORING_SYSTEM_PROMPT


def test_authoring_messages_carry_envelope():
    env = sample_envelope(seed=4, domain="field_ecology_survey",
                          genus="red_herring", detectability="distributed_triad")
    user = next(m["content"] for m in authoring_messages(env) if m["role"] == "user")
    assert env.genus in user and ANOMALY_GENUS_DESCRIPTIONS[env.genus] in user
    assert env.detectability in user and DETECTABILITY_DESCRIPTIONS[env.detectability] in user
    assert env.persona in user and env.stakes in user
    assert env.envelope_hash in user


def test_freeze_authored():
    env = sample_envelope(seed=4, domain="field_ecology_survey",
                          genus="red_herring", detectability="distributed_triad")
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
