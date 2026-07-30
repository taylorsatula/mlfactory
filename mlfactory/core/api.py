"""Reusable API client / judge for OpenAI-compatible endpoints.

Provides retries with exponential backoff, structured JSON extraction, and
consistent error handling. Experiments no longer need to write their own retry
loops.
"""
from __future__ import annotations

import json
import re
import time
import traceback
from dataclasses import dataclass
from typing import Any, Callable

import requests


@dataclass
class APIConfig:
    base_url: str
    api_key: str = "none"
    model: str | None = None
    timeout: float = 120.0
    max_retries: int = 3
    backoff_base: float = 2.0
    site_url: str | None = None
    app_name: str | None = None


class APIClient:
    """Thin retrying wrapper around OpenAI-compatible chat completions."""

    def __init__(self, config: APIConfig):
        self.config = config
        self._session = requests.Session()
        self._session.headers.update(
            {
                "Authorization": f"Bearer {config.api_key}",
                "Content-Type": "application/json",
            }
        )
        if config.site_url:
            self._session.headers["HTTP-Referer"] = config.site_url
        if config.app_name:
            self._session.headers["X-Title"] = config.app_name

    def chat_completion(
        self,
        messages: list[dict[str, str]],
        model: str | None = None,
        temperature: float | None = None,
        max_tokens: int | None = None,
        top_p: float | None = None,
        top_k: int | None = None,
        response_format: dict[str, str] | None = None,
        extra_body: dict[str, Any] | None = None,
    ) -> str:
        """Return the raw text content of a chat completion, with retries."""
        payload: dict[str, Any] = {
            "model": model or self.config.model,
            "messages": messages,
        }
        if temperature is not None:
            payload["temperature"] = temperature
        if max_tokens is not None:
            payload["max_tokens"] = max_tokens
        if top_p is not None:
            payload["top_p"] = top_p
        if response_format is not None:
            payload["response_format"] = response_format

        body_extras: dict[str, Any] = {}
        if top_k is not None:
            body_extras["top_k"] = top_k
        if extra_body:
            body_extras.update(extra_body)
        if body_extras:
            payload["extra_body"] = body_extras

        url = f"{self.config.base_url.rstrip('/')}/chat/completions"
        last_exc: Exception | None = None
        for attempt in range(self.config.max_retries):
            try:
                resp = self._session.post(
                    url,
                    json=payload,
                    timeout=self.config.timeout,
                )
                resp.raise_for_status()
                data = resp.json()
                choices = data.get("choices", [])
                if not choices:
                    raise RuntimeError("empty choices in response")
                content = choices[0].get("message", {}).get("content", "")
                return content
            except Exception as e:
                last_exc = e
                wait = self.config.backoff_base * (2 ** attempt)
                time.sleep(wait)
        raise RuntimeError(f"API call failed after {self.config.max_retries} retries: {last_exc}")

    def structured_json(
        self,
        messages: list[dict[str, str]],
        model: str | None = None,
        response_format: dict[str, str] | None = None,
        **kwargs: Any,
    ) -> dict[str, Any]:
        """Return a parsed JSON object from the model, with fallback extraction."""
        if response_format is None:
            response_format = {"type": "json_object"}
        text = self.chat_completion(
            messages=messages,
            model=model,
            response_format=response_format,
            **kwargs,
        )
        return extract_json(text)


def extract_json(text: str | None) -> dict[str, Any]:
    """Extract the first JSON object from model output."""
    if text is None:
        raise ValueError("empty response")
    text = text.strip()
    if not text:
        raise ValueError("empty response")

    # Strip markdown fences.
    if text.startswith("```"):
        text = re.sub(r"^```(?:json)?\s*", "", text, flags=re.IGNORECASE)
        text = re.sub(r"\s*```$", "", text)
        text = text.strip()

    try:
        return json.loads(text)
    except json.JSONDecodeError:
        pass

    match = re.search(r"\{.*\}", text, re.DOTALL)
    if match:
        try:
            return json.loads(match.group(0))
        except json.JSONDecodeError as exc:
            raise ValueError(f"no valid JSON object found in response") from exc
    raise ValueError("no JSON object found in response")


class Judge(APIClient):
    """Opinionated judge client: low-temperature A/B or rubric calls."""

    def compare(
        self,
        prompt: str,
        candidate_a: str,
        candidate_b: str,
        criterion: str = "overall quality",
    ) -> str:
        system = "You compare two candidate responses and return only A or B."
        user = (
            f"Criterion: {criterion}\n\n=== PROMPT ===\n{prompt}\n\n"
            f"=== CANDIDATE A ===\n{candidate_a}\n\n"
            f"=== CANDIDATE B ===\n{candidate_b}\n\n"
            "Which is better? Return only A or B."
        )
        text = self.chat_completion(
            messages=[
                {"role": "system", "content": system},
                {"role": "user", "content": user},
            ],
            temperature=0.0,
            max_tokens=4,
        )
        text = text.strip().upper()
        if "A" in text and "B" not in text:
            return "A"
        if "B" in text and "A" not in text:
            return "B"
        return "TIE"


@dataclass
class JudgeResult:
    wins: int
    ties: int
    total: int

    @property
    def jmq(self) -> float:
        return 2.0 * (self.wins + self.ties * 0.5) / max(self.total, 1)


def run_judge_pairwise(
    judge_client: Judge,
    prompts: list[str],
    hyps: list[str],
    refs: list[str],
    criterion: str = "overall quality",
    seed: int = 42,
) -> JudgeResult:
    """Run pairwise comparisons and return aggregated JMQ-style score."""
    import random

    random.seed(seed)
    wins = ties = 0
    for p, h, r in zip(prompts, hyps, refs):
        order = random.random() > 0.5
        a, b = (h, r) if order else (r, h)
        choice = judge_client.compare(p, a, b, criterion=criterion)
        if choice == "TIE":
            ties += 1
        elif (order and choice == "A") or (not order and choice == "B"):
            wins += 1
    return JudgeResult(wins=wins, ties=ties, total=len(prompts))
