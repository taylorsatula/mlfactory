"""Base class for mlfactory stage plugins.

A plugin turns a spec into a run directory of artifacts + a manifest.
"""
from __future__ import annotations

from abc import ABC, abstractmethod
from pathlib import Path
from typing import Any, ClassVar

from mlfactory.core.manifest import RunManifest


class StagePlugin(ABC):
    """Abstract base for collect, classify, train, eval, etc."""

    stage: ClassVar[str]

    def __init__(self, manifest: RunManifest):
        self.manifest = manifest

    @property
    def run_dir(self) -> Path:
        return Path(self.manifest.source.path).parent if self.manifest.source else Path("runs") / self.manifest.run_id

    @abstractmethod
    def prepare(self) -> None:
        """Validate inputs, start services, fetch data."""
        ...

    @abstractmethod
    def execute(self) -> None:
        """Run the actual workload."""
        ...

    @abstractmethod
    def finalize(self) -> None:
        """Write summary, checksum artifacts, stop services."""
        ...

    def run(self) -> RunManifest:
        from datetime import datetime, timezone

        self.manifest.status = "running"
        self.manifest.started_at = datetime.now(timezone.utc).isoformat()
        try:
            self.prepare()
            self.execute()
            if self.manifest.status != "guarded":
                self.manifest.status = "completed"
        except Exception as exc:
            self.manifest.status = "failed"
            self.manifest.summary["error"] = f"{type(exc).__name__}: {exc}"
            raise
        finally:
            self.manifest.completed_at = datetime.now(timezone.utc).isoformat()
            self.finalize()
        return self.manifest


class PluginRegistry:
    """Maps stage names to plugin classes."""

    def __init__(self):
        self._plugins: dict[str, type[StagePlugin]] = {}

    def register(self, plugin_cls: type[StagePlugin]) -> None:
        self._plugins[plugin_cls.stage] = plugin_cls

    def get(self, stage: str) -> type[StagePlugin]:
        if stage not in self._plugins:
            raise KeyError(f"no plugin registered for stage {stage!r}; known: {sorted(self._plugins)}")
        return self._plugins[stage]

    def list_stages(self) -> list[str]:
        return sorted(self._plugins.keys())


PLUGINS = PluginRegistry()
