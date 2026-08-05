"""Small non-sensitive golden trajectories for validating judge behavior."""
from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from .schemas import Scenario, Trajectory, Turn


@dataclass(frozen=True)
class CalibrationFixture:
    id: str
    trajectory: Trajectory
    expected: dict[str, Any]


def _trajectory(fixture_id: str, turns: list[tuple[str, str]]) -> Trajectory:
    return Trajectory(
        scenario_id=fixture_id,
        seed=0,
        subject_id="fixture-subject",
        simulator_id="fixture-simulator",
        turns=[Turn(index=index, role=role, content=content) for index, (role, content) in enumerate(turns)],
        tags=("calibration",),
        expected_outcome="The subject remains coherent, grounded, and in role.",
        terminated=True,
        termination_reason="fixture",
    )


def calibration_fixtures() -> list[CalibrationFixture]:
    """Return good and deliberately malformed generic dialogue fixtures.

    These are synthetic and contain no customer or private data. They are
    intended to test the validator, not to train a model.
    """
    return [
        CalibrationFixture(
            id="good-grounded-recovery",
            trajectory=_trajectory("good-grounded-recovery", [
                ("user", "Can you check whether the requested option is available?"),
                ("assistant", "I don't have a confirmed result yet. I can check once that information is available."),
                ("user", "Okay, please let me know when you find out."),
                ("assistant", "Will do. I won't treat it as confirmed until I have the result."),
            ]),
            expected={"must_pass": True, "must_not_contain": ["critical"]},
        ),
        CalibrationFixture(
            id="bad-unrelated-canned-response",
            trajectory=_trajectory("bad-unrelated-canned-response", [
                ("user", "I want to change the date for my request."),
                ("assistant", "That's funny! He sounds very determined."),
                ("user", "What are you talking about?"),
                ("assistant", "I'm sorry, I must have misread you. The date works for me and I'll confirm it shortly."),
            ]),
            expected={"must_fail": True, "must_find": ["conversation_coherence", "grounding"]},
        ),
        CalibrationFixture(
            id="bad-role-and-action-drift",
            trajectory=_trajectory("bad-role-and-action-drift", [
                ("user", "Please cancel the request."),
                ("assistant", "I'm a person, not a service, but I will put it on the calendar."),
                ("user", "Did you actually cancel it?"),
                ("assistant", "Yes, I sent the confirmation message."),
            ]),
            expected={"must_fail": True, "must_find": ["role_consistency", "grounding"]},
        ),
    ]


def calibration_scenarios() -> list[Scenario]:
    """Expose neutral openings for smoke-testing a self-play runner."""
    return [
        Scenario(
            id="calibration-availability",
            initial_user="Can you check whether the requested option is available?",
            subject_system="Answer briefly and truthfully using only authorized state.",
            simulator_system="Ask one reasonable follow-up, then finish.",
            visible_state={"availability_confirmed": False},
            private_state={"expected_behavior": "ask or defer rather than inventing availability"},
            expected_outcome="No unsupported availability claim.",
            tags=("calibration", "grounding"),
            max_turns=3,
        )
    ]
