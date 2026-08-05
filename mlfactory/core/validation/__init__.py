"""Reusable self-play and overseer validation fixtures for mlfactory.

This is library infrastructure, not an experiment stage. Existing experiment
 evaluators call it and own any resulting run artifacts and release decisions.
"""
from .analyzers import PatternRule, analyze_batch, analyze_trajectory, summarize_findings
from .fixtures import CalibrationFixture, calibration_fixtures, calibration_scenarios
from .judge import BatchJudge
from .overseer import CorpusOverseer
from .pipeline import validate_self_play, validate_trajectories, write_report
from .protocols import CallableParticipant, OpenAIChatParticipant, Participant
from .schemas import (
    BatchAssessment,
    Criterion,
    CriterionAssessment,
    Finding,
    OverseerAssessment,
    Scenario,
    SimulatorPlan,
    SessionAssessment,
    Trajectory,
    Turn,
    ValidationCriteria,
    ValidationReport,
)
from .self_play import SelfPlayConfig, SelfPlayRunner

__all__ = [
    "BatchAssessment",
    "BatchJudge",
    "CalibrationFixture",
    "CallableParticipant",
    "CorpusOverseer",
    "Criterion",
    "CriterionAssessment",
    "Finding",
    "OpenAIChatParticipant",
    "OverseerAssessment",
    "Participant",
    "PatternRule",
    "Scenario",
    "SimulatorPlan",
    "SelfPlayConfig",
    "SelfPlayRunner",
    "SessionAssessment",
    "Trajectory",
    "Turn",
    "ValidationCriteria",
    "ValidationReport",
    "analyze_batch",
    "analyze_trajectory",
    "calibration_fixtures",
    "calibration_scenarios",
    "summarize_findings",
    "validate_self_play",
    "validate_trajectories",
    "write_report",
]
