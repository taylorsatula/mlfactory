"""Markdown prompt loading with optional placeholder injection.

Prompts live as ``.md`` files inside experiment directories so they can be
dited without touching code. Use ``str.format`` style placeholders for
injectable content.

Examples:

    from mlfactory.core.prompts import render_markdown

    system_prompt = render_markdown(
        "mlfactory/experiments/ace/prompts/classifier_system_prompt.md",
        extra_instructions="Focus on mathematical reasoning.",
    )
"""
from __future__ import annotations

from pathlib import Path
from typing import Any


class PromptTemplateError(ValueError):
    pass


def load_markdown(path: str | Path) -> str:
    """Return the raw contents of a markdown prompt file."""
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"prompt file not found: {path}")
    return path.read_text(encoding="utf-8")


def render_markdown(path: str | Path, **kwargs: Any) -> str:
    """Load a markdown prompt and inject placeholders via ``str.format``.

    Missing placeholders are left unchanged so the prompt can still contain
    literal braces that are not meant as templates.
    """
    text = load_markdown(path)
    try:
        return text.format(**kwargs)
    except KeyError as exc:
        raise PromptTemplateError(f"missing prompt placeholder: {exc}") from exc
    except ValueError as exc:
        raise PromptTemplateError(f"invalid prompt template syntax: {exc}") from exc
