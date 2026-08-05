"""Batch-level LLM judging for trajectory validation."""
from __future__ import annotations

from collections.abc import Callable
import json
from typing import Any

from mlfactory.core.api import APIClient, extract_json

from .schemas import (
    BatchAssessment,
    Criterion,
    CriterionAssessment,
    Finding,
    SessionAssessment,
    Trajectory,
    ValidationCriteria,
)


Redactor = Callable[[str], str]


def _redact_trajectory(trajectory: Trajectory, redactor: Redactor) -> dict[str, Any]:
    record = trajectory.to_dict()
    for turn in record["turns"]:
        turn["content"] = redactor(str(turn["content"]))
    return record


def _finding_from_payload(
    value: dict[str, Any],
    *,
    source: str,
    session_id: str | None = None,
    default_scope: str = "session",
) -> Finding:
    severity = str(value.get("severity", "major"))
    if severity not in {"info", "minor", "major", "critical"}:
        severity = "major"
    scope = str(value.get("scope", default_scope))
    if scope not in {"turn", "session", "batch", "corpus"}:
        scope = default_scope
    raw_turns = value.get("turn_indices", value.get("evidence_turns", []))
    try:
        turn_indices = [int(item) for item in raw_turns]
    except (TypeError, ValueError):
        turn_indices = []
    raw_evidence = value.get("evidence", [])
    if isinstance(raw_evidence, str):
        raw_evidence = [raw_evidence]
    return Finding(
        source=source,
        criterion_id=str(value.get("criterion_id", "overall")),
        message=str(value.get("message", value.get("rationale", "unspecified finding"))),
        severity=severity,  # type: ignore[arg-type]
        scope=scope,  # type: ignore[arg-type]
        session_id=session_id or value.get("session_id"),
        turn_indices=turn_indices,
        evidence=[str(item) for item in raw_evidence],
        score=float(value["score"]) if value.get("score") is not None else None,
        confidence=float(value["confidence"]) if value.get("confidence") is not None else None,
    )


class BatchJudge:
    """Ask an independent model to assess a batch of complete sessions.

    The criteria are included in every request. A batch, rather than an
    isolated transcript, lets the judge detect recurring phrases and systemic
    mode/role regressions.
    """

    def __init__(
        self,
        backend: APIClient | Any,
        criteria: ValidationCriteria,
        *,
        model: str | None = None,
        max_sessions: int = 8,
        max_chars: int = 36_000,
        max_tokens: int = 2_048,
        redactor: Redactor | None = None,
    ) -> None:
        self.backend = backend
        self.criteria = criteria
        self.model = model
        self.max_sessions = max_sessions
        self.max_chars = max_chars
        self.max_tokens = max_tokens
        self.redactor = redactor or (lambda text: text)

    def pack_batches(self, trajectories: list[Trajectory]) -> list[list[Trajectory]]:
        batches: list[list[Trajectory]] = []
        current: list[Trajectory] = []
        current_chars = 0
        for trajectory in trajectories:
            size = len(json.dumps(_redact_trajectory(trajectory, self.redactor), ensure_ascii=False))
            if current and (len(current) >= self.max_sessions or current_chars + size > self.max_chars):
                batches.append(current)
                current = []
                current_chars = 0
            current.append(trajectory)
            current_chars += size
        if current:
            batches.append(current)
        return batches

    def build_messages(
        self,
        trajectories: list[Trajectory],
        deterministic_findings: list[Finding],
        batch_id: str,
    ) -> list[dict[str, str]]:
        sessions = [_redact_trajectory(item, self.redactor) for item in trajectories]
        findings = [item.to_dict() for item in deterministic_findings]
        schema = {
            "sessions": [
                {
                    "session_id": "string",
                    "overall_score": "number 0-4",
                    "passed": "boolean",
                    "confidence": "number 0-1",
                    "criteria": [
                        {
                            "criterion_id": "string",
                            "score": "number 0-4",
                            "passed": "boolean",
                            "rationale": "string",
                            "evidence_turns": ["integer"],
                            "confidence": "number 0-1",
                        }
                    ],
                    "findings": [
                        {
                            "criterion_id": "string",
                            "severity": "info|minor|major|critical",
                            "message": "string",
                            "turn_indices": ["integer"],
                            "evidence": ["string"],
                        }
                    ],
                }
            ],
            "batch_findings": [
                {
                    "criterion_id": "string",
                    "severity": "info|minor|major|critical",
                    "message": "string",
                    "session_id": "optional string",
                    "evidence": ["string"],
                }
            ],
            "recurring_patterns": ["string"],
            "confidence": "number 0-1",
        }
        system = (
            "You are an independent validation judge. Assess the subject model, not the "
            "simulator. Inspect every complete session and then compare the batch as a "
            "whole. Look for failures that recur across unrelated sessions, especially "
            "canned replies, role drift, mode collapse, local loops, and systemic "
            "grounding errors. Do not infer hidden state for the subject. Return only one "
            "JSON object matching the requested schema. Evidence must quote or identify "
            "turns from the supplied sessions."
        )
        user = (
            f"Batch ID: {batch_id}\n\n"
            f"VALIDATION CRITERIA\n{self.criteria.prompt_text()}\n\n"
            f"DETERMINISTIC FINDINGS\n{json.dumps(findings, ensure_ascii=False)}\n\n"
            f"SESSIONS\n{json.dumps(sessions, ensure_ascii=False)}\n\n"
            f"RETURN JSON SCHEMA\n{json.dumps(schema, ensure_ascii=False)}"
        )
        return [{"role": "system", "content": system}, {"role": "user", "content": user}]

    def _call(self, messages: list[dict[str, str]]) -> dict[str, Any]:
        structured_error: Exception | None = None
        if hasattr(self.backend, "structured_json"):
            try:
                return self.backend.structured_json(
                    messages=messages,
                    model=self.model,
                    temperature=0.0,
                    max_tokens=self.max_tokens,
                )
            except Exception as exc:
                structured_error = exc
        if hasattr(self.backend, "chat_completion"):
            try:
                text = self.backend.chat_completion(
                    messages=messages,
                    model=self.model,
                    temperature=0.0,
                    max_tokens=self.max_tokens,
                )
                return extract_json(text)
            except Exception as exc:
                if structured_error is not None:
                    raise RuntimeError(f"structured judge failed: {structured_error}; plain retry failed: {exc}") from exc
                raise
        if callable(self.backend):
            result = self.backend(messages)
            return result if isinstance(result, dict) else extract_json(str(result))
        raise TypeError("judge backend must provide structured_json/chat_completion or be callable")

    def assess_batch(
        self,
        batch_id: str,
        trajectories: list[Trajectory],
        deterministic_findings: list[Finding],
    ) -> BatchAssessment:
        messages = self.build_messages(trajectories, deterministic_findings, batch_id)
        try:
            payload = self._call(messages)
        except Exception as exc:
            return BatchAssessment(
                batch_id=batch_id,
                session_assessments=[
                    SessionAssessment(
                        session_id=item.session_id,
                        passed=False,
                        judge_error=f"{type(exc).__name__}: {exc}",
                    )
                    for item in trajectories
                ],
                judge_error=f"{type(exc).__name__}: {exc}",
            )

        by_id = {item.session_id: item for item in trajectories}
        criterion_by_id = {item.id: item for item in self.criteria.criteria}
        assessments: list[SessionAssessment] = []
        for raw in payload.get("sessions", []):
            if not isinstance(raw, dict):
                continue
            session_id = str(raw.get("session_id", ""))
            if session_id not in by_id:
                continue
            scores: list[CriterionAssessment] = []
            session_findings: list[Finding] = []
            for score in raw.get("criteria", []):
                if not isinstance(score, dict):
                    continue
                criterion_id = str(score.get("criterion_id", ""))
                criterion = criterion_by_id.get(criterion_id)
                if criterion is None:
                    continue
                numeric = max(0.0, min(4.0, float(score.get("score", 0))))
                passed = bool(score.get("passed", numeric >= criterion.min_score))
                evidence = score.get("evidence_turns", [])
                scores.append(CriterionAssessment(
                    criterion_id=criterion_id,
                    score=numeric,
                    passed=passed,
                    rationale=str(score.get("rationale", "")),
                    evidence_turns=[int(item) for item in evidence if str(item).isdigit()],
                    confidence=(float(score["confidence"]) if score.get("confidence") is not None else None),
                    critical=criterion.severity == "critical",
                ))
            for finding in raw.get("findings", []):
                if isinstance(finding, dict):
                    session_findings.append(_finding_from_payload(finding, source="judge", session_id=session_id))
            missing = [criterion.id for criterion in self.criteria.criteria if criterion.scope in {"turn", "session"} and criterion.id not in {item.criterion_id for item in scores}]
            if missing:
                session_findings.append(Finding(
                    source="judge",
                    criterion_id="judge_schema",
                    message=f"Judge omitted required criteria: {', '.join(missing)}",
                    severity="critical",
                    session_id=session_id,
                ))
            overall = float(raw.get("overall_score", sum(item.score for item in scores) / max(1, len(scores))))
            passed = bool(raw.get("passed", bool(scores) and all(item.passed for item in scores))) and not any(item.severity == "critical" for item in session_findings)
            assessments.append(SessionAssessment(
                session_id=session_id,
                scores=scores,
                findings=session_findings,
                overall_score=max(0.0, min(4.0, overall)),
                passed=passed,
                confidence=(float(raw["confidence"]) if raw.get("confidence") is not None else None),
            ))

        present = {item.session_id for item in assessments}
        for trajectory in trajectories:
            if trajectory.session_id not in present:
                assessments.append(SessionAssessment(
                    session_id=trajectory.session_id,
                    passed=False,
                    judge_error="judge omitted this session",
                ))
        batch_findings = [
            _finding_from_payload(item, source="judge", default_scope="batch")
            for item in payload.get("batch_findings", [])
            if isinstance(item, dict)
        ]
        return BatchAssessment(
            batch_id=batch_id,
            session_assessments=assessments,
            batch_findings=batch_findings,
            recurring_patterns=[str(item) for item in payload.get("recurring_patterns", [])],
            confidence=(float(payload["confidence"]) if payload.get("confidence") is not None else None),
        )
