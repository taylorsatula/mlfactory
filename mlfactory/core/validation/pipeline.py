"""Reusable self-play validation fixture.

This module is intentionally not an mlfactory experiment stage. Callers own
runs, manifests, artifact locations, and promotion decisions.
"""
from __future__ import annotations

from collections.abc import Iterable, Sequence
import json
from pathlib import Path
from typing import Any

from .analyzers import PatternRule, analyze_batch, analyze_trajectory, summarize_findings
from .judge import BatchJudge
from .overseer import CorpusOverseer
from .schemas import (
    BatchAssessment,
    Finding,
    OverseerAssessment,
    Scenario,
    Trajectory,
    ValidationCriteria,
    ValidationReport,
)
from .self_play import SelfPlayConfig, SelfPlayRunner


def _criteria(value: ValidationCriteria | dict[str, Any] | list[dict[str, Any]]) -> ValidationCriteria:
    if isinstance(value, ValidationCriteria):
        return value
    return ValidationCriteria.from_mapping(value)


def _numeric_summary(report: ValidationReport) -> dict[str, Any]:
    sessions = [
        session
        for batch in report.batch_assessments
        for session in batch.session_assessments
    ]
    all_findings = list(report.deterministic_findings)
    all_findings.extend(
        finding
        for batch in report.batch_assessments
        for session in batch.session_assessments
        for finding in session.findings
    )
    all_findings.extend(
        finding
        for batch in report.batch_assessments
        for finding in batch.batch_findings
    )
    if report.overseer:
        all_findings.extend(report.overseer.findings)
    critical = sum(item.severity == "critical" for item in all_findings)
    session_ids = {item.session_id for item in report.trajectories}
    sessions_with_findings = {
        item.session_id for item in all_findings if item.session_id in session_ids
    }
    mean_score = (
        sum(item.overall_score for item in sessions) / len(sessions)
        if sessions else None
    )
    pass_rate = (
        sum(item.passed for item in sessions) / len(sessions)
        if sessions else None
    )
    criterion_failures: dict[str, int] = {}
    for session in sessions:
        for score in session.scores:
            if not score.passed:
                criterion_failures[score.criterion_id] = criterion_failures.get(score.criterion_id, 0) + 1
    return {
        "trajectory_count": len(report.trajectories),
        "batch_count": len(report.batch_assessments),
        "judged_session_count": len(sessions),
        "mean_session_score": mean_score,
        "session_pass_rate": pass_rate,
        "criterion_failures": criterion_failures,
        "critical_failure_count": critical,
        "sessions_with_findings": len(sessions_with_findings),
        "first_failure_rate": (
            len(sessions_with_findings) / len(session_ids) if session_ids else 0.0
        ),
        "deterministic": summarize_findings(report.deterministic_findings),
    }


def _baseline_delta(current: dict[str, Any], baseline: dict[str, Any] | None) -> dict[str, Any] | None:
    if not baseline:
        return None
    delta: dict[str, Any] = {}
    for key in ("mean_session_score", "session_pass_rate", "first_failure_rate", "critical_failure_count"):
        current_value = current.get(key)
        baseline_value = baseline.get(key)
        if isinstance(current_value, (int, float)) and isinstance(baseline_value, (int, float)):
            delta[key] = current_value - baseline_value
    return delta


def _gate(report: ValidationReport) -> bool:
    summary = report.summary
    criteria = report.criteria
    if not report.trajectories:
        return False
    if summary.get("critical_failure_count", 0) > criteria.max_critical_failures:
        return False
    mean_score = summary.get("mean_session_score")
    if mean_score is not None and mean_score < criteria.min_mean_score:
        return False
    pass_rate = summary.get("session_pass_rate")
    if pass_rate is not None and pass_rate < criteria.min_session_pass_rate:
        return False
    criterion_failures = summary.get("criterion_failures", {})
    for criterion in criteria.criteria:
        if criterion.max_failures is not None and criterion_failures.get(criterion.id, 0) > criterion.max_failures:
            return False
    if (
        criteria.max_first_failure_rate is not None
        and summary.get("first_failure_rate", 0.0) > criteria.max_first_failure_rate
    ):
        return False
    if report.overseer and report.overseer.release_recommendation in {"guard", "fail", "unknown"}:
        return False
    return True


def validate_trajectories(
    trajectories: Sequence[Trajectory],
    criteria: ValidationCriteria | dict[str, Any] | list[dict[str, Any]],
    *,
    judge: BatchJudge | None = None,
    overseer: CorpusOverseer | None = None,
    pattern_rules: Iterable[PatternRule] = (),
    min_cross_session_reuse: int = 3,
    baseline_summary: dict[str, Any] | None = None,
) -> ValidationReport:
    """Assess replayed trajectories with deterministic and optional LLM layers."""
    normalized_criteria = _criteria(criteria)
    items = list(trajectories)
    report = ValidationReport(criteria=normalized_criteria, trajectories=items)
    for trajectory in items:
        report.deterministic_findings.extend(analyze_trajectory(trajectory, pattern_rules))
    # Retain the corpus-level signal in the report as well as passing the
    # per-batch slice to each judge. This keeps recurring phrases auditable.
    report.deterministic_findings.extend(analyze_batch(
        items,
        min_cross_session_reuse=min_cross_session_reuse,
    ))

    if judge:
        for index, batch in enumerate(judge.pack_batches(items)):
            batch_ids = {item.session_id for item in batch}
            batch_findings = [
                item for item in report.deterministic_findings
                if item.session_id in batch_ids
            ]
            batch_findings.extend(analyze_batch(
                batch,
                min_cross_session_reuse=min_cross_session_reuse,
            ))
            report.batch_assessments.append(
                judge.assess_batch(f"batch-{index:04d}", batch, batch_findings)
            )

        if overseer:
            report.overseer = overseer.assess(
                report.batch_assessments,
                summarize_findings(report.deterministic_findings),
                baseline_summary=baseline_summary,
            )

    report.summary = _numeric_summary(report)
    report.baseline_delta = _baseline_delta(report.summary, baseline_summary)
    report.summary["baseline_delta"] = report.baseline_delta
    report.summary["judge_enabled"] = judge is not None
    report.summary["overseer_enabled"] = overseer is not None
    report.passed = _gate(report)
    return report


def validate_self_play(
    scenarios: Sequence[Scenario],
    subject: Any,
    simulator: Any,
    criteria: ValidationCriteria | dict[str, Any] | list[dict[str, Any]],
    *,
    judge: BatchJudge | None = None,
    overseer: CorpusOverseer | None = None,
    self_play_config: SelfPlayConfig | None = None,
    seeds: list[int] | None = None,
    pattern_rules: Iterable[PatternRule] = (),
    min_cross_session_reuse: int = 3,
    baseline_summary: dict[str, Any] | None = None,
) -> ValidationReport:
    """Generate fresh multi-turn sessions, then validate them in batches."""
    runner = SelfPlayRunner(subject, simulator, config=self_play_config)
    trajectories = runner.run_many(list(scenarios), seeds=seeds)
    return validate_trajectories(
        trajectories,
        criteria,
        judge=judge,
        overseer=overseer,
        pattern_rules=pattern_rules,
        min_cross_session_reuse=min_cross_session_reuse,
        baseline_summary=baseline_summary,
    )


def write_report(
    report: ValidationReport,
    path: str | Path,
    *,
    include_trajectories: bool = False,
) -> Path:
    """Write a caller-owned validation artifact.

    Trajectories are omitted by default to reduce accidental persistence of
    private conversation data. Use a redacted trajectory copy when retaining
    transcripts for drill-down.
    """
    destination = Path(path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    destination.write_text(
        json.dumps(report.to_dict(include_trajectories=include_trajectories), indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )
    return destination
