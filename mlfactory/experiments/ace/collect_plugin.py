"""Factory plugin that wraps the legacy ACE collect.py / llama-server flow.

The plugin uses the reusable ``model()`` resource manager to start a disposable
llama-server for the duration of the run, then runs ``collect.py`` against it.
"""
from __future__ import annotations

import os
import shutil
import subprocess
import sys
from pathlib import Path
from typing import Any

from mlfactory.core.manifest import FileRecord, sha256_file
from mlfactory.core.model_server import model
from mlfactory.plugins.base import PLUGINS, StagePlugin


class CollectPlugin(StagePlugin):
    stage = "collect"

    def __init__(self, manifest):
        super().__init__(manifest)
        self.run_dir = Path(self.manifest.source.path).parent if self.manifest.source else Path("runs") / self.manifest.run_id
        self.spec = manifest.spec

    def _env(self, base_url: str) -> dict[str, str]:
        env = dict(os.environ)
        env.update(self.spec.get("env", {}))
        env.setdefault("PYTHONUNBUFFERED", "1")
        env["ACE_BASE_URL"] = base_url
        env["ACE_MODEL_NAME"] = self.spec.get("model_name", "Qwen/Qwen3.5-4B")
        env["ACE_PROVIDER"] = self.spec.get("provider", "llama")
        env["ACE_MAX_MODEL_LEN"] = str(self.spec.get("max_model_len", 32768))
        env["ACE_MAX_OUTPUT_TOKENS"] = str(self.spec.get("max_output_tokens", 16000))
        env["ACE_REQUEST_TIMEOUT"] = str(self.spec.get("request_timeout", 1200))
        env["ACE_TIME_BUDGET_SECONDS"] = str(self.spec.get("time_budget_seconds", 900))
        env["ACE_SEED_OFFSET"] = str(self.spec.get("seed_offset", 0))
        env["ACE_STRATIFIED_EXTRAS"] = str(self.spec.get("stratified_extras", 3))
        return env

    def _script_path(self, name: str) -> Path:
        return Path(__file__).parent / name

    def prepare(self) -> None:
        (self.run_dir / "artifacts").mkdir(parents=True, exist_ok=True)
        (self.run_dir / "logs").mkdir(parents=True, exist_ok=True)

    def execute(self) -> None:
        model_alias = self.spec.get("model", "qwen3.5:4b")
        gpu = self.spec.get("gpu", 0)
        python = self.spec.get("python", sys.executable)

        with model(model_alias, gpu=gpu) as srv:
            env = self._env(srv.base_url)
            out_dir = self.run_dir / "artifacts"
            prompts = Path(self.spec["prompts"])

            cmd = [
                python,
                str(self._script_path("collect.py")),
                "--prompts", str(prompts),
                "--out-dir", str(out_dir),
            ]
            if self.spec.get("max_samples"):
                cmd.extend(["--max-samples", str(self.spec["max_samples"])])

            log = self.run_dir / "logs" / "collect.log"
            with open(log, "w") as lf:
                proc = subprocess.Popen(
                    cmd,
                    env=env,
                    stdout=lf,
                    stderr=subprocess.STDOUT,
                )
                rc = proc.wait()
            if rc != 0:
                raise RuntimeError(f"collect.py exited with code {rc}")

    def finalize(self) -> None:
        out_dir = self.run_dir / "artifacts"
        for name in ["generations.jsonl", "manifest.json"]:
            p = out_dir / name
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
            "provider": self.spec.get("provider"),
            "model_name": self.spec.get("model_name"),
            "model_alias": self.spec.get("model"),
            "prompts": str(Path(self.spec["prompts"]).resolve()),
        }
        self.manifest.write(self.run_dir / "manifest.json")


PLUGINS.register(CollectPlugin)
