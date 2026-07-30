"""Factory plugin that wraps the native ACE collect.py.

Starts a disposable llama-server via the reusable ``model()`` resource manager
and runs ``collect.py`` against it.
"""
from __future__ import annotations

import os
import subprocess
import sys
from pathlib import Path

from mlfactory.core.manifest import FileRecord, sha256_file
from mlfactory.core.model_server import model
from mlfactory.plugins.base import PLUGINS, StagePlugin


class CollectPlugin(StagePlugin):
    stage = "collect"

    def __init__(self, manifest):
        super().__init__(manifest)
        self.run_dir = Path(self.manifest.source.path).parent
        self.spec = manifest.spec

    def _script_path(self, name: str) -> Path:
        return Path(__file__).parent / name

    def _env(self) -> dict[str, str]:
        env = dict(os.environ)
        env.update(self.spec.get("env", {}))
        env.setdefault("PYTHONUNBUFFERED", "1")
        return env

    def prepare(self) -> None:
        (self.run_dir / "artifacts").mkdir(parents=True, exist_ok=True)
        (self.run_dir / "logs").mkdir(parents=True, exist_ok=True)

    def execute(self) -> None:
        s = self.spec
        python = s.get("python", sys.executable)
        model_alias = s.get("model", "qwen3.5:4b")
        gpu = s.get("gpu", 0)

        with model(model_alias, gpu=gpu) as srv:
            cmd = [
                python,
                str(self._script_path("collect.py")),
                "--run-dir", str(self.run_dir),
                "--prompts", str(s["prompts"]),
                "--base-url", srv.base_url,
                "--model-name", str(s.get("model_name", "Qwen/Qwen3.5-4B")),
                "--provider", str(s.get("provider", "llama")),
                "--max-model-len", str(s.get("max_model_len", 32768)),
                "--max-output-tokens", str(s.get("max_output_tokens", 16000)),
                "--request-timeout", str(s.get("request_timeout", 1200)),
                "--time-budget-seconds", str(s.get("time_budget_seconds", 900)),
                "--seed-offset", str(s.get("seed_offset", 0)),
                "--stratified-extras", str(s.get("stratified_extras", 3)),
            ]
            if s.get("max_samples"):
                cmd.extend(["--max-samples", str(s["max_samples"])])

            log = self.run_dir / "logs" / "collect.log"
            err = self.run_dir / "logs" / "collect.err"
            with open(log, "w") as lf, open(err, "w") as ef:
                proc = subprocess.Popen(cmd, env=self._env(), stdout=lf, stderr=ef)
                rc = proc.wait()
            if rc != 0:
                raise RuntimeError(f"collect.py exited with code {rc}")

    def finalize(self) -> None:
        artifacts_dir = self.run_dir / "artifacts"
        for name in ["generations.jsonl"]:
            p = artifacts_dir / name
            if p.exists():
                self.manifest.artifacts.append(
                    FileRecord(
                        path=str(p.resolve()),
                        sha256=sha256_file(p),
                        role=f"artifact:{name}",
                        size_bytes=p.stat().st_size,
                    )
                )
        self.manifest.summary = {
            "model_alias": self.spec.get("model"),
            "model_name": self.spec.get("model_name"),
            "prompts": str(Path(self.spec["prompts"]).resolve()),
        }
        self.manifest.write(self.run_dir / "manifest.json")


PLUGINS.register(CollectPlugin)
