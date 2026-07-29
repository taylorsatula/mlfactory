"""Factory plugin that wraps the legacy DFT eval.py.

TODO: implement prepare/execute/finalize; for now registered so specs can name
stage ``eval``.
"""
from __future__ import annotations

from mlfactory.plugins.base import PLUGINS, StagePlugin


class EvalPlugin(StagePlugin):
    stage = "eval"

    def prepare(self) -> None:
        raise NotImplementedError("eval plugin not yet ported")

    def execute(self) -> None:
        raise NotImplementedError("eval plugin not yet ported")

    def finalize(self) -> None:
        pass


PLUGINS.register(EvalPlugin)
