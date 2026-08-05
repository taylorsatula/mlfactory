from __future__ import annotations

import json
from typing import Any

from mlfactory.core.validation import (
    BatchJudge,
    CorpusOverseer,
    Criterion,
    Scenario,
    SelfPlayRunner,
    SimulatorPlan,
    ValidationCriteria,
    calibration_fixtures,
    validate_trajectories,
)
from mlfactory.core.validation.schemas import BatchAssessment, Trajectory


class _ScriptedParticipant:
    def __init__(self, participant_id: str, responses: list[str], capture: list[list[dict[str, str]]] | None = None):
        self.participant_id = participant_id
        self.responses = iter(responses)
        self.capture = capture

    def generate(self, messages: list[dict[str, str]], *, max_tokens: int, temperature: float, seed: int) -> str:
        if self.capture is not None:
            self.capture.append(messages)
        return next(self.responses)


class _BatchBackend:
    def __init__(self, session_ids: list[str]):
        self.session_ids = session_ids
        self.messages: list[list[dict[str, str]]] = []

    def structured_json(self, *, messages: list[dict[str, str]], **kwargs: Any) -> dict[str, Any]:
        self.messages.append(messages)
        return {
            "sessions": [
                {
                    "session_id": session_id,
                    "overall_score": 4,
                    "passed": True,
                    "confidence": 1,
                    "criteria": [
                        {
                            "criterion_id": "coherence",
                            "score": 4,
                            "passed": True,
                            "rationale": "coherent",
                            "evidence_turns": [1],
                        },
                        {
                            "criterion_id": "grounding",
                            "score": 4,
                            "passed": True,
                            "rationale": "grounded",
                            "evidence_turns": [1],
                        },
                    ],
                    "findings": [],
                }
                for session_id in self.session_ids
            ],
            "batch_findings": [],
            "recurring_patterns": [],
            "confidence": 1,
        }


class _OverseerBackend:
    def __init__(self):
        self.messages: list[list[dict[str, str]]] = []

    def structured_json(self, *, messages: list[dict[str, str]], **kwargs: Any) -> dict[str, Any]:
        self.messages.append(messages)
        return {"systemic_findings": [], "systemic_patterns": [], "release_recommendation": "pass", "confidence": 1}


def _criteria() -> ValidationCriteria:
    return ValidationCriteria(
        criteria=(
            Criterion(id="coherence", description="The response remains coherent."),
            Criterion(id="grounding", description="The response uses only authorized facts."),
        ),
        min_mean_score=3.0,
        min_session_pass_rate=1.0,
        max_first_failure_rate=0.0,
    )


def test_self_play_keeps_private_state_out_of_trajectory() -> None:
    captured: list[list[dict[str, str]]] = []
    subject = _ScriptedParticipant("subject", ["I need one more detail.", "Thanks."], captured)
    simulator = _ScriptedParticipant("simulator", ["Here is the detail.", "[[END]]"])
    scenario = Scenario(
        id="private-state",
        initial_user="Start",
        subject_system="Be concise.",
        visible_state={"visible": "yes"},
        private_state={"secret": "must not leak"},
        simulator_plan=SimulatorPlan(
            engage_for_turns=2,
            pain_points=(
                {"id": "pressure", "after_subject_turn": 2, "instruction": "press once"},
            ),
        ),
        max_turns=2,
    )
    trajectory = SelfPlayRunner(subject, simulator).run(scenario, seed=7)
    serialized = json.dumps(trajectory.to_dict())
    assert "must not leak" not in serialized
    assert "visible" in captured[0][0]["content"]
    assert "must not leak" not in captured[0][0]["content"]
    assert trajectory.session_id == "private-state::seed-7"
    assert trajectory.simulator_plan["engage_for_turns"] == 2
    assert trajectory.simulator_plan["stop_sequence"] == "[[STOP_SIMULATION]]"


def test_scenario_mapping_supports_structured_simulator_plan() -> None:
    scenario = Scenario.from_mapping({
        "id": "mapped",
        "initial_user": "Hello",
        "simulator_plan": {
            "engage_for_turns": 4,
            "stop_sequence": "<STOP>",
            "pain_points": [{"after_subject_turn": 2, "instruction": "press"}],
        },
    })
    assert scenario.simulator_plan.engage_for_turns == 4
    assert scenario.simulator_plan.stop_sequence == "<STOP>"
    assert scenario.simulator_plan.instructions_for_turn(2)["pain_points"]


def test_batch_judge_receives_multiple_sessions_and_criteria() -> None:
    subject = _ScriptedParticipant("subject", ["Okay."] * 3)
    simulator = _ScriptedParticipant("simulator", ["[[END]]"] * 3)
    scenarios = [Scenario(id=f"s-{index}", initial_user="Hello", max_turns=1) for index in range(3)]
    trajectories = SelfPlayRunner(subject, simulator).run_many(scenarios, seeds=[1, 2, 3])
    backend = _BatchBackend([item.session_id for item in trajectories])
    judge = BatchJudge(backend, _criteria(), max_sessions=2)
    report = validate_trajectories(trajectories, _criteria(), judge=judge)
    assert len(backend.messages) == 2
    assert all("VALIDATION CRITERIA" in message[1]["content"] for message in backend.messages)
    assert report.summary["batch_count"] == 2
    assert report.summary["judged_session_count"] == 3


def test_repeated_response_is_a_deterministic_regression() -> None:
    subject = _ScriptedParticipant("subject", ["Same answer.", "Same answer."])
    simulator = _ScriptedParticipant("simulator", ["Again.", "[[END]]"])
    trajectory = SelfPlayRunner(subject, simulator).run(
        Scenario(id="repeat", initial_user="Hello", max_turns=2), seed=0
    )
    backend = _BatchBackend([trajectory.session_id])
    overseer_backend = _OverseerBackend()
    report = validate_trajectories(
        [trajectory],
        _criteria(),
        judge=BatchJudge(backend, _criteria()),
        overseer=CorpusOverseer(overseer_backend, _criteria()),
    )
    assert any(item.criterion_id == "conversation_coherence" for item in report.deterministic_findings)
    assert report.passed is False


def test_calibration_fixtures_are_non_sensitive() -> None:
    fixtures = calibration_fixtures()
    assert {item.id for item in fixtures} == {
        "good-grounded-recovery",
        "bad-unrelated-canned-response",
        "bad-role-and-action-drift",
    }
    assert all("private" not in json.dumps(item.trajectory.to_dict()).casefold() for item in fixtures)
