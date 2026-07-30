"""Unified metrics logging for mlfactory runs.

Writes per-step metrics to:
- ``dashboard.jsonl`` in the run directory (for the live TUI)
- the SQLite registry metrics table (for querying/history)
- stdout (optional)

Experiments should create one MetricsLogger per run and call ``.step()`` or
``.log()`` as work progresses.
"""
from __future__ import annotations

import json
import os
import sys
from collections.abc import Mapping
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from mlfactory.core.registry import Registry


def _jsonify(value: Any) -> Any:
    """Make a value JSON-serializable for the dashboard line."""
    if isinstance(value, (str, int, float, bool, type(None))):
        return value
    if isinstance(value, Mapping):
        return {k: _jsonify(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [_jsonify(v) for v in value]
    return str(value)


class MetricsLogger:
    """Append-only metrics logger for one run."""

    def __init__(
        self,
        run_dir: str | Path,
        run_id: str | None = None,
        registry: Registry | None = None,
        echo: bool = True,
    ):
        self.run_dir = Path(run_dir)
        self.run_dir.mkdir(parents=True, exist_ok=True)
        self.run_id = run_id or self.run_dir.name
        self.registry = registry
        self.echo = echo
        self._dashboard_path = self.run_dir / "dashboard.jsonl"
        self._step: int | None = None

    def _now(self) -> str:
        return datetime.now(timezone.utc).isoformat()

    def _write_dashboard(self, record: dict[str, Any]) -> None:
        line = json.dumps(record, ensure_ascii=False, default=str) + "\n"
        with open(self._dashboard_path, "a", encoding="utf-8") as f:
            f.write(line)
            f.flush()
            os.fsync(f.fileno())

    def log(
        self,
        key: str,
        value: Any,
        step: int | None = None,
        timestamp: str | None = None,
    ) -> None:
        """Log a single scalar metric."""
        ts = timestamp or self._now()
        record = {
            "timestamp": ts,
            "run_id": self.run_id,
            "key": key,
            "value": _jsonify(value),
        }
        if step is not None:
            record["step"] = step
        self._write_dashboard(record)
        if self.registry is not None:
            try:
                scalar = float(value) if isinstance(value, (int, float)) else None
            except (TypeError, ValueError):
                scalar = None
            self.registry.insert_metric(
                run_id=self.run_id,
                key=key,
                value=scalar,
                json_value=_jsonify(value) if scalar is None else None,
                step=step,
                timestamp=ts,
            )
        if self.echo:
            print(json.dumps(record, ensure_ascii=False, default=str), flush=True)

    def step(self, step: int, **metrics: Any) -> None:
        """Log a bundle of metrics for a training/eval step."""
        self._step = step
        ts = self._now()
        flat: dict[str, Any] = {"step": step, "timestamp": ts}
        for key, value in metrics.items():
            flat[key] = _jsonify(value)
            self.log(key, value, step=step, timestamp=ts)
        # Also emit a combined dashboard row for the TUI.
        self._write_dashboard(flat)

    def event(self, event: str, detail: dict[str, Any] | None = None) -> None:
        """Log a discrete event (checkpoint saved, guard triggered, etc.)."""
        record = {
            "timestamp": self._now(),
            "run_id": self.run_id,
            "event": event,
        }
        if detail:
            record["detail"] = _jsonify(detail)
        self._write_dashboard(record)
        if self.echo:
            print(json.dumps(record, ensure_ascii=False, default=str), flush=True)
