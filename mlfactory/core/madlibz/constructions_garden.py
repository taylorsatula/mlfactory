"""Gardening constructions: real-world reasoning outside STEM costume.

These exist to prove the pattern generalizes beyond textbook genres, and
because the messiest human over-deliberation lives in problems like these:
mixed units, rounding traps, and date arithmetic with living stakes.
"""
from __future__ import annotations

import math

from .construction import AuthenticityPools, Construction, declare_construction
from .lexicon import register_paradigm
from .spec import Answer, Slot, SlotType

__all__ = ["raised_bed_soil_mix", "seed_starting_countback"]

# --- Garden lexemes ---------------------------------------------------------
register_paradigm("vegetable", ["tomatoes", "peppers", "broccoli", "basil", "kale", "zucchini"])
register_paradigm("mix_fraction", [
    ("a third", 1 / 3),
    ("half", 1 / 2),
])
register_paradigm("garden_lore", [
    "the neighbor swears by mushroom compost",
    "a video about no-dig beds",
    "a forum thread about worm castings",
    "the bed gets morning sun only",
])

_GARDEN_PERSONAS = (
    "a first-time community-garden plot holder",
    "someone converting a corner of the lawn into a bed",
    "a parent building a garden bed with their kid",
    "a beginner gardener who killed last year's seedlings",
)
_SOIL_STAKES = (
    "the garden center only delivers on Saturdays",
    "the car is tiny, so the order has to be right",
    "a community-garden inspection is coming up",
    "the whole family is helping this weekend",
)
_SEED_STAKES = (
    "last year's seedlings got caught by a late frost",
    "the grow light shelf is already full",
    "a friend is splitting a seed order this week",
    "the greenhouse bench space is first-come-first-served",
)

# --- Raised bed soil mix ----------------------------------------------------
# Volume with mixed units (ft x ft x inches), a fraction of the mix, and a
# ceiling rounding into whole bags: four derived steps and two rounding
# decisions, which is exactly the over-deliberation profile ACE wants.

raised_bed_soil_mix = declare_construction(Construction(
    id="raised_bed_soil_mix",
    domain="gardening",
    description=(
        "How many whole bags of compost to buy for a raised bed, when compost "
        "is a fraction of the bed's volume. Bed length and width are in feet, "
        "fill depth in inches; bags are sold in cubic feet and must be bought whole."
    ),
    slots=(
        Slot("bed_length", SlotType(kind="numeric", unit="ft", lo=3.0, hi=12.0), role="binding"),
        Slot("bed_width", SlotType(kind="numeric", unit="ft", lo=2.0, hi=6.0), role="binding"),
        Slot("fill_depth", SlotType(kind="numeric", unit="in", lo=6.0, hi=18.0, integer=True), role="binding"),
        Slot("compost_fraction", SlotType(kind="choice", paradigm="mix_fraction"), role="binding"),
        Slot("bag_size", SlotType(kind="numeric", unit="cubic feet", lo=1.0, hi=3.0), role="binding"),
        Slot("lore", SlotType(kind="choice", paradigm="garden_lore"), role="distractor"),
    ),
    answer_fn=lambda v: Answer(
        kind="exact",
        value=math.ceil(v["bed_length"] * v["bed_width"] * (v["fill_depth"] / 12.0)
                        * v["compost_fraction"] / v["bag_size"]),
    ),
    difficulty_fn=lambda v: {"binding_facts": 5, "derived_steps": 4, "band": "medium",
                             "traps": ["unit_conversion", "ceiling_rounding"]},
    authenticity=AuthenticityPools(personas=_GARDEN_PERSONAS, stakes=_SOIL_STAKES),
    max_distractors=1,
))

# --- Seed-starting countback ------------------------------------------------
# Counting back from a frost date through sequential stages with a
# weeks-to-days conversion.  Stage order is part of the family semantics;
# the prose carries the stage durations as facts.

seed_starting_countback = declare_construction(Construction(
    id="seed_starting_countback",
    domain="gardening",
    description=(
        "How many days before the expected last frost to start seeds indoors. "
        "Stages are sequential: germination days, then indoor growth in weeks, "
        "then hardening-off days, then transplant on the frost date."
    ),
    slots=(
        Slot("vegetable", SlotType(kind="choice", paradigm="vegetable"), role="incidental"),
        Slot("days_to_frost", SlotType(kind="numeric", unit="days", lo=20.0, hi=90.0, integer=True), role="incidental"),
        Slot("germination_days", SlotType(kind="numeric", unit="days", lo=4.0, hi=14.0, integer=True), role="binding"),
        Slot("weeks_indoors", SlotType(kind="numeric", unit="weeks", lo=5.0, hi=10.0, integer=True), role="binding"),
        Slot("hardening_days", SlotType(kind="numeric", unit="days", lo=5.0, hi=10.0, integer=True), role="binding"),
        Slot("lore", SlotType(kind="choice", paradigm="garden_lore"), role="distractor"),
    ),
    answer_fn=lambda v: Answer(
        kind="exact",
        value=v["germination_days"] + 7 * v["weeks_indoors"] + v["hardening_days"],
    ),
    difficulty_fn=lambda v: {"binding_facts": 3, "derived_steps": 2, "band": "easy",
                             "traps": ["unit_conversion", "date_countback"]},
    authenticity=AuthenticityPools(personas=_GARDEN_PERSONAS, stakes=_SEED_STAKES),
    max_distractors=1,
))
