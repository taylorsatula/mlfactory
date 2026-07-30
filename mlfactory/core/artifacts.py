"""Reusable artifact/checkpoint helpers.

Saves model checkpoints, tokenizers, and arbitrary files into a run's artifact
directory while computing hashes and updating the manifest.
"""
from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any

from mlfactory.core.manifest import FileRecord, RunManifest, sha256_file


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
) -> Path:
    """Save a PEFT adapter + tokenizer to ``artifacts/checkpoint-<step>/``.

    Also updates the manifest artifact list if one is provided.
    """
    run_dir = Path(run_dir)
    artifacts_dir = run_dir / "artifacts"
    artifacts_dir.mkdir(parents=True, exist_ok=True)

    name = label or (f"checkpoint-{step}" if step is not None else "checkpoint-final")
    ckpt_dir = artifacts_dir / name
    ckpt_dir.mkdir(parents=True, exist_ok=True)

    model.save_pretrained(ckpt_dir)
    if tokenizer is not None:
        tokenizer.save_pretrained(ckpt_dir)

    if manifest is not None:
        new_artifacts: list[FileRecord] = []
        for p in ckpt_dir.rglob("*"):
            if p.is_file():
                new_artifacts.append(
                    FileRecord(
                        path=str(p.resolve()),
                        sha256=sha256_file(p),
                        role=f"artifact:{name}/{p.relative_to(ckpt_dir)}",
                        size_bytes=p.stat().st_size,
                    )
                )
        manifest.artifacts.extend(new_artifacts)
        manifest.write(run_dir / "manifest.json")

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
