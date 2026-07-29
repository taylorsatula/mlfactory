"""Factory plugin that wraps the legacy ACE classify.py.

TODO: implement prepare/execute/finalize; for now registered so specs can name
stage ``classify``.
"""
from __future__ import annotations

from mlfactory.plugins.base import PLUGINS, StagePlugin


class ClassifyPlugin(StagePlugin):
    stage = "classify"

    def prepare(self) -> None:
        raise NotImplementedError("classify plugin not yet ported")

    def execute(self) -> None:
        raise NotImplementedError("classify plugin not yet ported")

    def finalize(self) -> None:
        pass


PLUGINS.register(ClassifyPlugin)
