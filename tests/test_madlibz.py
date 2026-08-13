"""Tests for the redesigned madlibz spec factory + realization contract."""
from __future__ import annotations

import pytest

from mlfactory.core.madlibz import (
    AuthenticityPools,
    Construction,
    REALIZATION_SYSTEM_PROMPT,
    Slot,
    SlotType,
    construction_ids,
    declare_construction,
    freeze_record,
    get_construction,
    realization_messages,
    sample_spec,
)
from mlfactory.core.madlibz.spec import Answer


def test_determinism():
    a = sample_spec(seed=7, construction_id="round_trip_speed")
    b = sample_spec(seed=7, construction_id="round_trip_speed")
    assert a.to_dict() == b.to_dict()
    assert a.semantic_hash == b.semantic_hash


def test_diversity_and_felicity():
    specs = [sample_spec(seed=s, construction_id="round_trip_speed") for s in range(50)]
    hashes = {s.semantic_hash for s in specs}
    assert len(hashes) == 50
    for s in specs:
        v_out = s.fills["v_out"].value
        v_back = s.fills["v_back"].value
        assert v_out != v_back  # felicity
        expected = round(2 * v_out * v_back / (v_out + v_back), 2)
        assert s.answer.kind == "exact" and s.answer.value == expected
        assert s.fills["place_a"].value != s.fills["place_b"].value  # distinct_from
        assert s.authenticity["persona"] and s.authenticity["stakes"]


def test_underspecified_variant():
    closed = sample_spec(seed=3, construction_id="round_trip_speed")
    open_ = sample_spec(seed=3, construction_id="round_trip_speed_open_distance")
    assert closed.answer.kind == "exact" and closed.free_vars() == ()
    assert open_.answer.kind == "rubric"
    assert open_.free_vars() and "distance" in open_.free_vars()[0]
    assert open_.difficulty["underspecified"] is True
    user_open = next(m["content"] for m in realization_messages(open_) if m["role"] == "user")
    user_closed = next(m["content"] for m in realization_messages(closed) if m["role"] == "user")
    assert "MUST REMAIN OPEN" in user_open
    assert "MUST REMAIN OPEN" not in user_closed


def test_realization_contract_is_cog_aware():
    sys_prompt = REALIZATION_SYSTEM_PROMPT
    for concept in ("prose-realization stage", "spec factory", "gate", "canonical answer"):
        assert concept in sys_prompt
    # The model must know an answer exists and is withheld, not see the answer.
    assert "not told that answer" in sys_prompt


def test_realization_messages_carry_spec_but_never_the_answer():
    for seed in range(10):
        spec = sample_spec(seed=seed, construction_id="round_trip_speed")
        text = "\n".join(m["content"] for m in realization_messages(spec))
        for name in ("v_out", "v_back"):
            assert spec.fills[name].surface in text
        assert spec.fills["place_a"].surface in text
        # Answer leakage check (skip seeds where answer coincides with an input).
        inputs = {spec.fills["v_out"].value, spec.fills["v_back"].value}
        if spec.answer.value not in inputs:
            assert f"{spec.answer.value:g}" not in text


def test_freeze_record():
    spec = sample_spec(seed=11, construction_id="round_trip_speed")
    rec = freeze_record(spec, "Hey, quick one about my ride...", model="test-model", gate={"passed": True}, run="x")
    assert rec["semantic_hash"] == spec.semantic_hash
    assert rec["surface_hash"] and rec["construction_id"] == "round_trip_speed"
    assert rec["spec"]["fills"]["v_out"]["surface"] == spec.fills["v_out"].surface
    assert rec["gate"] == {"passed": True} and rec["provenance"] == {"run": "x"}
    with pytest.raises(ValueError):
        freeze_record(spec, "   ", model="test-model")


def test_garden_constructions():
    # Known-value check on the soil-mix answer function: 4ft x 3ft x 12in bed,
    # half compost, 1.5 cu ft bags -> 6 sq ft compost -> ceil(6/1.5) = 4 bags.
    from mlfactory.core.madlibz import get_construction
    c = get_construction("raised_bed_soil_mix")
    ans = c.answer_fn({"bed_length": 4.0, "bed_width": 3.0, "fill_depth": 12,
                       "compost_fraction": 0.5, "bag_size": 1.5})
    assert ans.kind == "exact" and ans.value == 4
    spec = sample_spec(seed=23, construction_id="raised_bed_soil_mix")
    text = next(m["content"] for m in realization_messages(spec) if m["role"] == "user")
    for name in ("bed_length", "bed_width", "fill_depth", "compost_fraction", "bag_size"):
        assert spec.fills[name].surface in text
    seed_spec = sample_spec(seed=23, construction_id="seed_starting_countback")
    expected = seed_spec.fills["germination_days"].value + 7 * seed_spec.fills["weeks_indoors"].value + seed_spec.fills["hardening_days"].value
    assert seed_spec.answer.value == expected


def test_envelope_blind_draws_and_overrides():
    from mlfactory.core.madlibz import (ANOMALY_GENUSES, DETECTABILITY_GRANULARS,
                                        authoring_messages, freeze_authored, sample_envelope)
    a = sample_envelope(seed=12, domain="household")
    b = sample_envelope(seed=12, domain="household")
    assert a == b and a.envelope_hash == b.envelope_hash
    assert a.genus in ANOMALY_GENUSES and a.detectability in DETECTABILITY_GRANULARS
    c = sample_envelope(seed=12, domain="household", genus="temporal_conflict", detectability="hidden")
    assert c.genus == "temporal_conflict" and c.detectability == "hidden"
    assert c.envelope_hash != a.envelope_hash
    with pytest.raises(ValueError):
        sample_envelope(seed=1, domain="household", genus="bogus")
    with pytest.raises(ValueError):
        sample_envelope(seed=1, domain="household", detectability="0.7")


def test_envelope_authoring_contract_and_freeze():
    from mlfactory.core.madlibz import authoring_messages, freeze_authored, sample_envelope
    env = sample_envelope(seed=4, domain="gardening", genus="red_herring", detectability="indirection")
    msgs = authoring_messages(env)
    user = next(m["content"] for m in msgs if m["role"] == "user")
    assert env.genus in user and env.detectability in user and env.envelope_hash in user
    rec = freeze_authored(env, {"prose": "quick question about my tomatoes...",
                                "surface_question": "when to water",
                                "anomaly": {"genus": env.genus, "detectability": env.detectability,
                                            "what_is_wrong": "x", "where_it_lives": "y",
                                            "why_it_trips_reasoning": "z"}}, model="test")
    assert rec["envelope_hash"] == env.envelope_hash
    assert rec["anomaly"]["genus"] == env.genus
    assert rec["envelope"]["detectability"] == env.detectability
    with pytest.raises(ValueError):
        freeze_authored(env, {"prose": "", "anomaly": {}}, model="test")
    with pytest.raises(ValueError):
        freeze_authored(env, {"prose": "hi"}, model="test")


def test_registry_hygiene():
    assert "round_trip_speed" in construction_ids()
    with pytest.raises(ValueError):
        get_construction("no_such_construction")
    with pytest.raises(ValueError):
        declare_construction(get_construction("round_trip_speed"))  # duplicate id
    with pytest.raises(ValueError):
        declare_construction(Construction(
            id="bad_distinct",
            domain="test",
            description="x",
            slots=(Slot("a", SlotType(kind="text", paradigm="place", distinct_from=("ghost",))),),
            answer_fn=lambda v: Answer(kind="exact", value=0),
            authenticity=AuthenticityPools(personas=("p",), stakes=("s",)),
        ))
