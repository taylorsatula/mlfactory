"""Corpus-level assessment over batch judge outputs."""
from __future__ import annotations

import json
from typing import Any

from mlfactory.core.api import APIClient, extract_json

from .schemas import (
    BatchAssessment,
    Finding,
    OverseerAssessment,
    ValidationCriteria,
)


class CorpusOverseer:
    """Find systemic regressions after individual batches have been judged."""

    def __init__(
        self,
        backend: APIClient | Any,
        criteria: ValidationCriteria,
        *,
        model: str | None = None,
        max_tokens: int = 2_048,
    ) -> None:
        self.backend = backend
        self.criteria = criteria
        self.model = model
        self.max_tokens = max_tokens

    def build_messages(
        self,
        assessments: list[BatchAssessment],
        deterministic_summary: dict[str, Any],
        *,
        baseline_summary: dict[str, Any] | None = None,
    ) -> list[dict[str, str]]:
        compact = [
            {
                "batch_id": item.batch_id,
                "session_assessments": [
                    {
                        "session_id": session.session_id,
                        "overall_score": session.overall_score,
                        "passed": session.passed,
                        "scores": [score.to_dict() for score in session.scores],
                        "findings": [finding.to_dict() for finding in session.findings],
                        "judge_error": session.judge_error,
                    }
                    for session in item.session_assessments
                ],
                "batch_findings": [finding.to_dict() for finding in item.batch_findings],
                "recurring_patterns": item.recurring_patterns,
                "judge_error": item.judge_error,
            }
            for item in assessments
        ]
        schema = {
            "systemic_findings": [
                {
                    "criterion_id": "string",
                    "severity": "info|minor|major|critical",
                    "message": "string",
                    "evidence": ["batch IDs, session IDs, or recurring pattern examples"],
                }
            ],
            "systemic_patterns": ["string"],
            "release_recommendation": "pass|guard|fail|unknown",
            "confidence": "number 0-1",
        }
        system = (
            "You are the corpus-level overseer for an ML validation run. Review the "
            "batch assessments as a population, not as isolated cases. Look for "
            "regressions visible only in aggregate: repeated canned responses, early "
            "failure concentration, role or mode collapse, systematic hallucinations, "
            "judge disagreement, and baseline degradation. Do not average away a "
            "critical failure. Return only JSON matching the schema."
        )
        user = (
            f"VALIDATION CRITERIA\n{self.criteria.prompt_text()}\n\n"
            f"DETERMINISTIC SUMMARY\n{json.dumps(deterministic_summary, ensure_ascii=False)}\n\n"
            f"BATCH ASSESSMENTS\n{json.dumps(compact, ensure_ascii=False)}\n\n"
            f"BASELINE SUMMARY\n{json.dumps(baseline_summary or {}, ensure_ascii=False)}\n\n"
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
                return extract_json(self.backend.chat_completion(
                    messages=messages,
                    model=self.model,
                    temperature=0.0,
                    max_tokens=self.max_tokens,
                ))
            except Exception as exc:
                if structured_error is not None:
                    raise RuntimeError(f"structured overseer failed: {structured_error}; plain retry failed: {exc}") from exc
                raise
        if callable(self.backend):
            result = self.backend(messages)
            return result if isinstance(result, dict) else extract_json(str(result))
        raise TypeError("overseer backend must provide structured_json/chat_completion or be callable")

    def assess(
        self,
        assessments: list[BatchAssessment],
        deterministic_summary: dict[str, Any],
        *,
        baseline_summary: dict[str, Any] | None = None,
    ) -> OverseerAssessment:
        try:
            payload = self._call(self.build_messages(
                assessments,
                deterministic_summary,
                baseline_summary=baseline_summary,
            ))
        except Exception as exc:
            return OverseerAssessment(
                release_recommendation="unknown",
                judge_error=f"{type(exc).__name__}: {exc}",
            )
        findings: list[Finding] = []
        for item in payload.get("systemic_findings", payload.get("findings", [])):
            if not isinstance(item, dict):
                continue
            severity = str(item.get("severity", "major"))
            if severity not in {"info", "minor", "major", "critical"}:
                severity = "major"
            evidence = item.get("evidence", [])
            if isinstance(evidence, str):
                evidence = [evidence]
            findings.append(Finding(
                source="overseer",
                criterion_id=str(item.get("criterion_id", "systemic_quality")),
                message=str(item.get("message", "systemic finding")),
                severity=severity,  # type: ignore[arg-type]
                scope="corpus",
                evidence=[str(value) for value in evidence],
            ))
        recommendation = str(payload.get("release_recommendation", "unknown")).lower()
        if recommendation not in {"pass", "guard", "fail", "unknown"}:
            recommendation = "unknown"
        return OverseerAssessment(
            findings=findings,
            systemic_patterns=[str(item) for item in payload.get("systemic_patterns", [])],
            release_recommendation=recommendation,
            confidence=(float(payload["confidence"]) if payload.get("confidence") is not None else None),
        )
