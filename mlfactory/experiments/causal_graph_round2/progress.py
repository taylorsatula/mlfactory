"""Append-only progress events shared by the runner and child processes."""
from __future__ import annotations

import json
import os
import time
from pathlib import Path
from typing import Any


def emit(path: str | Path | None, event: str, **values: Any) -> None:
    if not path:
        return
    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    record = {"timestamp": time.time(), "event": event, "pid": os.getpid(), **values}
    with target.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(record, ensure_ascii=False, sort_keys=True) + "\n")
        handle.flush()


def latest(path: str | Path) -> dict[str, Any] | None:
    target = Path(path)
    if not target.exists():
        return None
    last = None
    with target.open(encoding="utf-8", errors="ignore") as handle:
        for line in handle:
            if line.strip():
                try:
                    last = json.loads(line)
                except json.JSONDecodeError:
                    continue
    return last
