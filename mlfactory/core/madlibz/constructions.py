"""Reference constructions: worked examples of the pattern.

Every construction here pairs a text schema with a computation over the
same slots.  These are also the regression anchor for the pipeline: if the
factory changes meaning, these answers change detectably.
"""
from __future__ import annotations

from .construction import AuthenticityPools, Construction, declare_construction
from .spec import Answer, Slot, SlotType

__all__ = ["round_trip_speed", "round_trip_speed_open_distance"]

_SPEED = SlotType(kind="numeric", unit="km/h", lo=5.0, hi=45.0)
_PLACE_A = SlotType(kind="text", paradigm="place", distinct_from=("place_b",))
_PLACE_B = SlotType(kind="text", paradigm="place")
_RIDE_DETAIL = SlotType(kind="choice", paradigm="ride_detail")

_PERSONAS = (
    "a commuter cyclist",
    "a delivery courier between drops",
    "someone planning a weekend ride with friends",
    "a triathlete checking training numbers",
    "a parent shuttling a kid to practice",
)
_STAKES = (
    "settling a bet with a friend",
    "deciding whether the route fits inside a lunch break",
    "figuring out if a faster bike would actually save time",
    "writing up trip notes for a cycling forum",
    "estimating departure time for tomorrow",
)


def _harmonic_mean(values: dict) -> Answer:
    v_out, v_back = values["v_out"], values["v_back"]
    return Answer(kind="exact", value=round(2 * v_out * v_back / (v_out + v_back), 2))


def _round_trip_difficulty(values: dict) -> dict:
    return {"binding_facts": 2, "derived_steps": 1, "band": "easy"}


round_trip_speed = declare_construction(Construction(
    id="round_trip_speed",
    domain="rates",
    description="Average speed over an out-and-back trip with equal legs.",
    slots=(
        Slot("place_a", _PLACE_A, role="incidental"),
        Slot("place_b", _PLACE_B, role="incidental"),
        Slot("v_out", _SPEED, role="binding"),
        Slot("v_back", _SPEED, role="binding"),
        Slot("ride_detail", _RIDE_DETAIL, role="distractor"),
    ),
    answer_fn=_harmonic_mean,
    felicity_fn=lambda v: v["v_out"] != v["v_back"],
    difficulty_fn=_round_trip_difficulty,
    authenticity=AuthenticityPools(personas=_PERSONAS, stakes=_STAKES),
    max_distractors=1,
))


def _harmonic_mean_open(values: dict) -> Answer:
    # The equal-leg assumption is NOT given here: leg distances are a free
    # variable, so only the conditional answer is licensed.
    v_out, v_back = values["v_out"], values["v_back"]
    conditional = round(2 * v_out * v_back / (v_out + v_back), 2)
    return Answer(
        kind="rubric",
        value=None,
        rubric=(
            "Leg distances are unspecified. Credit: flag the missing "
            f"information; under equal legs the average is {conditional} km/h; "
            "otherwise it depends on the distance ratio."
        ),
        free_vars=("the distance of each leg, or whether they are equal",),
    )


round_trip_speed_open_distance = declare_construction(Construction(
    id="round_trip_speed_open_distance",
    domain="rates",
    description="Average speed over an out-and-back trip; leg distances deliberately unspecified.",
    slots=round_trip_speed.slots,
    answer_fn=_harmonic_mean_open,
    felicity_fn=lambda v: v["v_out"] != v["v_back"],
    difficulty_fn=lambda v: {"binding_facts": 2, "derived_steps": 1, "band": "easy", "underspecified": True},
    authenticity=AuthenticityPools(personas=_PERSONAS, stakes=_STAKES),
    max_distractors=1,
))
