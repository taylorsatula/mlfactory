"""Participant protocols for model-agnostic validation."""
from __future__ import annotations

from collections.abc import Callable
from typing import Any, Protocol

from mlfactory.core.api import APIClient


class Participant(Protocol):
    """A chat participant used by self-play or replay assessment."""

    participant_id: str

    def generate(
        self,
        messages: list[dict[str, str]],
        *,
        max_tokens: int,
        temperature: float,
        seed: int,
    ) -> str:
        ...


class CallableParticipant:
    """Adapt a simple callable into the Participant protocol."""

    def __init__(
        self,
        function: Callable[[list[dict[str, str]]], str],
        participant_id: str = "callable",
    ) -> None:
        self.function = function
        self.participant_id = participant_id

    def generate(
        self,
        messages: list[dict[str, str]],
        *,
        max_tokens: int,
        temperature: float,
        seed: int,
    ) -> str:
        return str(self.function(messages))


class OpenAIChatParticipant:
    """Use any OpenAI-compatible endpoint as a self-play participant."""

    def __init__(
        self,
        client: APIClient,
        participant_id: str,
        model: str | None = None,
        extra_body: dict[str, Any] | None = None,
    ) -> None:
        self.client = client
        self.participant_id = participant_id
        self.model = model
        self.extra_body = dict(extra_body or {})

    def generate(
        self,
        messages: list[dict[str, str]],
        *,
        max_tokens: int,
        temperature: float,
        seed: int,
    ) -> str:
        # Most OpenAI-compatible local servers accept seed, while APIClient
        # keeps it optional for providers that ignore it.
        return self.client.chat_completion(
            messages=messages,
            model=self.model,
            temperature=temperature,
            max_tokens=max_tokens,
            extra_body={**self.extra_body, "seed": seed},
        )


def ensure_participant(value: Participant | Callable[[list[dict[str, str]]], str], participant_id: str) -> Participant:
    if callable(value) and not hasattr(value, "generate"):
        return CallableParticipant(value, participant_id=participant_id)
    return value  # type: ignore[return-value]


def participant_id(value: Any, fallback: str) -> str:
    return str(getattr(value, "participant_id", fallback))
