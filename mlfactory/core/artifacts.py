"""Reusable artifact/checkpoint helpers.

Saves model checkpoints, tokenizers, and arbitrary files into a run's artifact
directory while computing hashes and updating the manifest.
"""
from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from mlfactory.core.manifest import RunManifest


def _write_json_atomic(path: Path, data: Any) -> None:
    tmp = path.with_suffix(path.suffix + ".tmp")
    with tmp.open("w", encoding="utf-8") as f:
        json.dump(data, f, indent=2, ensure_ascii=False, default=str)
        f.flush()
    tmp.replace(path)


def save_checkpoint(
    run_dir: str | Path,
    step: int | None,
    model: Any,
    tokenizer: Any | None = None,
    manifest: RunManifest | None = None,
    label: str | None = None,
    *,
    title: str | None = None,
    description: str | None = None,
    tags: list[str] | None = None,
    caveats: str | None = None,
    sensitivity: str | None = None,
    schema: dict[str, Any] | None = None,
    name: str | None = None,
) -> Path:
    """Save a PEFT adapter + tokenizer to ``artifacts/checkpoint-<step>/``.

    Also updates the manifest artifact list if one is provided. The same
    lab-notebook metadata as :func:`mlfactory.core.datasave.datasave` is attached
    to every file in the checkpoint (title/description are required to make
    data discoverable, but default sensibly so existing callers keep working).
    """
    run_dir = Path(run_dir)
    artifacts_dir = run_dir / "artifacts"
    artifacts_dir.mkdir(parents=True, exist_ok=True)

    name = name or label or (f"checkpoint-{step}" if step is not None else "checkpoint-final")
    title = title or f"Checkpoint {name}"
    description = description or "Model checkpoint saved during training."

    ckpt_dir = artifacts_dir / name
    ckpt_dir.mkdir(parents=True, exist_ok=True)

    model.save_pretrained(ckpt_dir)
    if tokenizer is not None:
        tokenizer.save_pretrained(ckpt_dir)

    if manifest is not None:
        from mlfactory.core.datasave import register_checkpoint_dir

        register_checkpoint_dir(
            manifest, run_dir, ckpt_dir,
            title=title, description=description, name=name,
            tags=tags, caveats=caveats, sensitivity=sensitivity,
            data_schema=schema,
        )

    return ckpt_dir


def save_summary(
    run_dir: str | Path,
    summary: dict[str, Any],
    manifest: RunManifest | None = None,
) -> Path:
    """Write ``artifacts/summary.json`` and optionally update the manifest."""
    run_dir = Path(run_dir)
    out = run_dir / "artifacts" / "summary.json"
    out.parent.mkdir(parents=True, exist_ok=True)
    _write_json_atomic(out, summary)

    if manifest is not None:
        manifest.summary = summary
        manifest.write(run_dir / "manifest.json")
    return out


def save_config(
    run_dir: str | Path,
    config: dict[str, Any],
    name: str = "config.json",
) -> Path:
    """Write a JSON config into the run artifacts."""
    run_dir = Path(run_dir)
    out = run_dir / "artifacts" / name
    out.parent.mkdir(parents=True, exist_ok=True)
    _write_json_atomic(out, config)
    return out
