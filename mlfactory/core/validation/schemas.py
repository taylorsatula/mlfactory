"""Serializable schemas for reusable trajectory validation.

The schemas are deliberately model- and modality-agnostic. An experiment owns
its scenarios and writes the returned report into its own run artifacts.
"""
from __future__ import annotations

from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Literal
import json


Scope = Literal["turn", "session", "batch", "corpus"]
Severity = Literal["info", "minor", "major", "critical"]


@dataclass(frozen=True)
class Criterion:
    """A judgeable validation criterion supplied by the calling evaluation."""

    id: str
    description: str
    scope: Scope = "session"
    severity: Severity = "major"
    rubric: dict[int, str] = field(default_factory=lambda: {
        4: "fully satisfies the criterion",
        3: "mostly satisfies it with a minor issue",
        2: "partially satisfies it with a noticeable issue",
        1: "barely satisfies it",
        0: "clearly fails it",
    })
    evidence_required: bool = True
    min_score: float = 3.0
    max_failures: int | None = None

    @classmethod
    def from_mapping(cls, value: dict[str, Any]) -> "Criterion":
        rubric = value.get("rubric")
        if rubric is None:
            normalized_rubric = cls.__dataclass_fields__["rubric"].default_factory()  # type: ignore[index]
        else:
            normalized_rubric = {int(k): str(v) for k, v in dict(rubric).items()}
        return cls(
            id=str(value["id"]),
            description=str(value["description"]),
            scope=str(value.get("scope", "session")),  # type: ignore[arg-type]
            severity=str(value.get("severity", "major")),  # type: ignore[arg-type]
            rubric=normalized_rubric,
            evidence_required=bool(value.get("evidence_required", True)),
            min_score=float(value.get("min_score", 3.0)),
            max_failures=(int(value["max_failures"]) if value.get("max_failures") is not None else None),
        )

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(frozen=True)
class ValidationCriteria:
    """Named criteria plus global batch/corpus thresholds."""

    criteria: tuple[Criterion, ...]
    name: str = "default"
    description: str = ""
    max_critical_failures: int = 0
    min_mean_score: float = 3.0
    min_session_pass_rate: float = 1.0
    max_first_failure_rate: float | None = None

    @classmethod
    def from_mapping(cls, value: dict[str, Any] | list[dict[str, Any]]) -> "ValidationCriteria":
        if isinstance(value, list):
            return cls(criteria=tuple(Criterion.from_mapping(item) for item in value))
        raw = value.get("criteria", [])
        return cls(
            criteria=tuple(Criterion.from_mapping(item) for item in raw),
            name=str(value.get("name", "default")),
            description=str(value.get("description", "")),
            max_critical_failures=int(value.get("max_critical_failures", 0)),
            min_mean_score=float(value.get("min_mean_score", 3.0)),
            min_session_pass_rate=float(value.get("min_session_pass_rate", 1.0)),
            max_first_failure_rate=(
                float(value["max_first_failure_rate"])
                if value.get("max_first_failure_rate") is not None
                else None
            ),
        )

    @classmethod
    def from_file(cls, path: str | Path) -> "ValidationCriteria":
        path = Path(path)
        text = path.read_text(encoding="utf-8")
        if path.suffix.lower() in {".yaml", ".yml"}:
            import yaml
            value = yaml.safe_load(text)
        else:
            value = json.loads(text)
        if not isinstance(value, (dict, list)):
            raise ValueError(f"criteria file must contain a mapping or list: {path}")
        return cls.from_mapping(value)

    def to_dict(self) -> dict[str, Any]:
        return {
            "name": self.name,
            "description": self.description,
            "criteria": [criterion.to_dict() for criterion in self.criteria],
            "max_critical_failures": self.max_critical_failures,
            "min_mean_score": self.min_mean_score,
            "min_session_pass_rate": self.min_session_pass_rate,
            "max_first_failure_rate": self.max_first_failure_rate,
        }

    def prompt_text(self) -> str:
        lines = [f"Criteria set: {self.name}"]
        if self.description:
            lines.append(self.description)
        for criterion in self.criteria:
            lines.append(
                json.dumps(
                    {
                        "id": criterion.id,
                        "scope": criterion.scope,
                        "severity": criterion.severity,
                        "description": criterion.description,
                        "rubric": criterion.rubric,
                        "evidence_required": criterion.evidence_required,
                    },
                    ensure_ascii=False,
                    sort_keys=True,
                )
            )
        return "\n".join(lines)


@dataclass(frozen=True)
class SimulatorPlan:
    """Structured control information supplied only to the faux-human model."""

    engage_for_turns: int | None = None
    stop_sequence: str = "[[STOP_SIMULATION]]"
    persona: str = "natural user"
    additional_context: dict[str, Any] = field(default_factory=dict)
    pain_points: tuple[dict[str, Any], ...] = ()
    turn_instructions: tuple[dict[str, Any], ...] = ()

    @classmethod
    def from_mapping(cls, value: dict[str, Any] | None) -> "SimulatorPlan":
        value = value or {}
        return cls(
            engage_for_turns=(int(value["engage_for_turns"]) if value.get("engage_for_turns") is not None else None),
            stop_sequence=str(value.get("stop_sequence", "[[STOP_SIMULATION]]")),
            persona=str(value.get("persona", "natural user")),
            additional_context=dict(value.get("additional_context", {})),
            pain_points=tuple(dict(item) for item in value.get("pain_points", [])),
            turn_instructions=tuple(dict(item) for item in value.get("turn_instructions", [])),
        )

    def instructions_for_turn(self, subject_turn: int) -> dict[str, Any]:
        def active(item: dict[str, Any]) -> bool:
            at = item.get("after_subject_turn", item.get("subject_turn"))
            if at is None:
                return True
            try:
                return int(at) == subject_turn
            except (TypeError, ValueError):
                return False

        return {
            "pain_points": [dict(item) for item in self.pain_points if active(item)],
            "turn_instructions": [dict(item) for item in self.turn_instructions if active(item)],
        }

    def to_dict(self) -> dict[str, Any]:
        return {
            "engage_for_turns": self.engage_for_turns,
            "stop_sequence": self.stop_sequence,
            "persona": self.persona,
            "additional_context": self.additional_context,
            "pain_points": [dict(item) for item in self.pain_points],
            "turn_instructions": [dict(item) for item in self.turn_instructions],
        }


@dataclass(frozen=True)
class Scenario:
    """A public task opening plus private simulator state.

    ``private_state`` is supplied only to the simulator and is never emitted by
    ``Trajectory.to_dict``. The subject receives ``visible_state`` only.
    """

    id: str
    initial_user: str
    subject_system: str = ""
    simulator_system: str = ""
    visible_state: dict[str, Any] = field(default_factory=dict)
    private_state: dict[str, Any] = field(default_factory=dict)
    simulator_plan: SimulatorPlan = field(default_factory=SimulatorPlan)
    expected_outcome: str = ""
    forbidden_outcomes: tuple[str, ...] = ()
    tags: tuple[str, ...] = ()
    max_turns: int = 8

    @classmethod
    def from_mapping(cls, value: dict[str, Any]) -> "Scenario":
        return cls(
            id=str(value["id"]),
            initial_user=str(value["initial_user"]),
            subject_system=str(value.get("subject_system", "")),
            simulator_system=str(value.get("simulator_system", "")),
            visible_state=dict(value.get("visible_state", {})),
            private_state=dict(value.get("private_state", {})),
            simulator_plan=SimulatorPlan.from_mapping(value.get("simulator_plan")),
            expected_outcome=str(value.get("expected_outcome", "")),
            forbidden_outcomes=tuple(str(item) for item in value.get("forbidden_outcomes", [])),
            tags=tuple(str(item) for item in value.get("tags", [])),
            max_turns=int(value.get("max_turns", 8)),
        )

    def public_dict(self) -> dict[str, Any]:
        return {
            "id": self.id,
            "initial_user": self.initial_user,
            "subject_system": self.subject_system,
            "simulator_system": self.simulator_system,
            "visible_state": self.visible_state,
            "simulator_plan": self.simulator_plan.to_dict(),
            "expected_outcome": self.expected_outcome,
            "forbidden_outcomes": list(self.forbidden_outcomes),
            "tags": list(self.tags),
            "max_turns": self.max_turns,
        }


@dataclass
class Turn:
    index: int
    role: str
    content: str
    metadata: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass
class Trajectory:
    scenario_id: str
    seed: int
    subject_id: str
    simulator_id: str
    turns: list[Turn] = field(default_factory=list)
    tags: tuple[str, ...] = ()
    visible_state: dict[str, Any] = field(default_factory=dict)
    simulator_plan: dict[str, Any] = field(default_factory=dict)
    forbidden_outcomes: tuple[str, ...] = ()
    expected_outcome: str = ""
    terminated: bool = False
    termination_reason: str = ""

    @property
    def session_id(self) -> str:
        return f"{self.scenario_id}::seed-{self.seed}"

    @property
    def assistant_turns(self) -> list[Turn]:
        return [turn for turn in self.turns if turn.role == "assistant"]

    @property
    def user_turns(self) -> list[Turn]:
        return [turn for turn in self.turns if turn.role == "user"]

    def messages(self) -> list[dict[str, str]]:
        return [{"role": turn.role, "content": turn.content} for turn in self.turns]

    def to_dict(self) -> dict[str, Any]:
        return {
            "session_id": self.session_id,
            "scenario_id": self.scenario_id,
            "seed": self.seed,
            "subject_id": self.subject_id,
            "simulator_id": self.simulator_id,
            "turns": [turn.to_dict() for turn in self.turns],
            "tags": list(self.tags),
            "visible_state": self.visible_state,
            "simulator_plan": self.simulator_plan,
            "forbidden_outcomes": list(self.forbidden_outcomes),
            "expected_outcome": self.expected_outcome,
            "terminated": self.terminated,
            "termination_reason": self.termination_reason,
        }


@dataclass
class Finding:
    source: str
    criterion_id: str
    message: str
    severity: Severity = "major"
    scope: Scope = "session"
    session_id: str | None = None
    turn_indices: list[int] = field(default_factory=list)
    evidence: list[str] = field(default_factory=list)
    score: float | None = None
    confidence: float | None = None

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass
class CriterionAssessment:
    criterion_id: str
    score: float
    passed: bool
    rationale: str = ""
    evidence_turns: list[int] = field(default_factory=list)
    confidence: float | None = None
    critical: bool = False

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass
class SessionAssessment:
    session_id: str
    scores: list[CriterionAssessment] = field(default_factory=list)
    findings: list[Finding] = field(default_factory=list)
    overall_score: float = 0.0
    passed: bool = False
    confidence: float | None = None
    judge_error: str | None = None

    def to_dict(self) -> dict[str, Any]:
        return {
            "session_id": self.session_id,
            "scores": [score.to_dict() for score in self.scores],
            "findings": [finding.to_dict() for finding in self.findings],
            "overall_score": self.overall_score,
            "passed": self.passed,
            "confidence": self.confidence,
            "judge_error": self.judge_error,
        }


@dataclass
class BatchAssessment:
    batch_id: str
    session_assessments: list[SessionAssessment] = field(default_factory=list)
    batch_findings: list[Finding] = field(default_factory=list)
    recurring_patterns: list[str] = field(default_factory=list)
    confidence: float | None = None
    judge_error: str | None = None

    def to_dict(self) -> dict[str, Any]:
        return {
            "batch_id": self.batch_id,
            "session_assessments": [item.to_dict() for item in self.session_assessments],
            "batch_findings": [item.to_dict() for item in self.batch_findings],
            "recurring_patterns": self.recurring_patterns,
            "confidence": self.confidence,
            "judge_error": self.judge_error,
        }


@dataclass
class OverseerAssessment:
    findings: list[Finding] = field(default_factory=list)
    systemic_patterns: list[str] = field(default_factory=list)
    release_recommendation: str = "unknown"
    confidence: float | None = None
    judge_error: str | None = None

    def to_dict(self) -> dict[str, Any]:
        return {
            "findings": [finding.to_dict() for finding in self.findings],
            "systemic_patterns": self.systemic_patterns,
            "release_recommendation": self.release_recommendation,
            "confidence": self.confidence,
            "judge_error": self.judge_error,
        }


@dataclass
class ValidationReport:
    criteria: ValidationCriteria
    trajectories: list[Trajectory] = field(default_factory=list)
    deterministic_findings: list[Finding] = field(default_factory=list)
    batch_assessments: list[BatchAssessment] = field(default_factory=list)
    overseer: OverseerAssessment | None = None
    summary: dict[str, Any] = field(default_factory=dict)
    passed: bool = False
    baseline_delta: dict[str, Any] | None = None

    def to_dict(self, include_trajectories: bool = True) -> dict[str, Any]:
        output: dict[str, Any] = {
            "criteria": self.criteria.to_dict(),
            "deterministic_findings": [item.to_dict() for item in self.deterministic_findings],
            "batch_assessments": [item.to_dict() for item in self.batch_assessments],
            "overseer": self.overseer.to_dict() if self.overseer else None,
            "summary": self.summary,
            "passed": self.passed,
            "baseline_delta": self.baseline_delta,
        }
        if include_trajectories:
            output["trajectories"] = [item.to_dict() for item in self.trajectories]
        return output
