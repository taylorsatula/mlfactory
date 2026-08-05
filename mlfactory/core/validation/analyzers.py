"""Cheap, auditable trajectory checks used before LLM judging."""
from __future__ import annotations

from collections import Counter, defaultdict
from dataclasses import dataclass
import re
from typing import Iterable

from .schemas import Finding, Trajectory


_WORDS = re.compile(r"[\w']+", re.UNICODE)


def normalize(text: str) -> str:
    return re.sub(r"\s+", " ", str(text).casefold()).strip()


def _ngrams(text: str, size: int = 4) -> set[str]:
    words = _WORDS.findall(normalize(text))
    return {" ".join(words[i : i + size]) for i in range(max(0, len(words) - size + 1))}


@dataclass(frozen=True)
class PatternRule:
    """Optional domain-specific deterministic rule supplied by an evaluator."""

    id: str
    pattern: str
    message: str
    severity: str = "major"
    criterion_id: str = "grounding"
    flags: int = re.IGNORECASE

    def compiled(self) -> re.Pattern[str]:
        return re.compile(self.pattern, self.flags)


def analyze_trajectory(
    trajectory: Trajectory,
    pattern_rules: Iterable[PatternRule] = (),
) -> list[Finding]:
    findings: list[Finding] = []
    assistant_turns = trajectory.assistant_turns
    if not assistant_turns:
        findings.append(Finding(
            source="deterministic",
            criterion_id="response_validity",
            message="The subject produced no assistant response.",
            severity="critical",
            session_id=trajectory.session_id,
        ))
        return findings

    seen: dict[str, int] = {}
    for turn in assistant_turns:
        text = normalize(turn.content)
        if not text:
            findings.append(Finding(
                source="deterministic",
                criterion_id="response_validity",
                message="The subject produced an empty response.",
                severity="critical",
                session_id=trajectory.session_id,
                turn_indices=[turn.index],
            ))
            continue
        if text in seen:
            findings.append(Finding(
                source="deterministic",
                criterion_id="conversation_coherence",
                message="The subject repeated an earlier assistant response exactly.",
                severity="major",
                session_id=trajectory.session_id,
                turn_indices=[seen[text], turn.index],
                evidence=[turn.content],
            ))
        else:
            seen[text] = turn.index

        for rule in pattern_rules:
            if rule.compiled().search(turn.content):
                findings.append(Finding(
                    source="deterministic",
                    criterion_id=rule.criterion_id,
                    message=rule.message,
                    severity=rule.severity,  # type: ignore[arg-type]
                    session_id=trajectory.session_id,
                    turn_indices=[turn.index],
                    evidence=[turn.content],
                ))
    return findings


def analyze_batch(
    trajectories: list[Trajectory],
    *,
    min_cross_session_reuse: int = 3,
    ngram_size: int = 4,
) -> list[Finding]:
    """Find exact phrase reuse across distinct sessions.

    This is intentionally conservative and only reports phrases shared by at
    least ``min_cross_session_reuse`` sessions. The batch judge can decide
    whether a repeated phrase is a legitimate template or a regression.
    """
    phrase_sessions: dict[str, set[str]] = defaultdict(set)
    phrase_examples: dict[str, str] = {}
    for trajectory in trajectories:
        phrases: set[str] = set()
        for turn in trajectory.assistant_turns:
            phrases.update(_ngrams(turn.content, size=ngram_size))
        for phrase in phrases:
            phrase_sessions[phrase].add(trajectory.session_id)
            phrase_examples.setdefault(phrase, next(
                (turn.content for turn in trajectory.assistant_turns if phrase in _ngrams(turn.content, ngram_size)),
                phrase,
            ))

    findings: list[Finding] = []
    for phrase, sessions in sorted(phrase_sessions.items()):
        if len(sessions) < min_cross_session_reuse:
            continue
        findings.append(Finding(
            source="deterministic",
            criterion_id="cross_session_reuse",
            message=f"A phrase recurs across {len(sessions)} distinct sessions.",
            severity="minor",
            scope="batch",
            session_id=None,
            evidence=[phrase_examples[phrase]],
            confidence=1.0,
        ))
    return findings


def summarize_findings(findings: list[Finding]) -> dict[str, object]:
    by_criterion = Counter(item.criterion_id for item in findings)
    by_severity = Counter(item.severity for item in findings)
    first_failure_sessions = {item.session_id for item in findings if item.session_id}
    return {
        "total": len(findings),
        "by_criterion": dict(sorted(by_criterion.items())),
        "by_severity": dict(sorted(by_severity.items())),
        "sessions_with_findings": len(first_failure_sessions),
    }
