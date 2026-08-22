"""Factory plugin that wraps the native ACE classify.py classifier harness."""
from __future__ import annotations

import os
import subprocess
import sys
from pathlib import Path

from mlfactory.core.manifest import FileRecord, sha256_file
from mlfactory.plugins.base import PLUGINS, StagePlugin


class ClassifyPlugin(StagePlugin):
    stage = "classify"

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

        cmd = [
            python,
            str(self._script_path("classify.py")),
            "--run-dir", str(self.run_dir),
            "--input", str(Path(s["input"]).resolve()),
            "--base-url", str(s.get("base_url", "https://openrouter.ai/api/v1")),
            "--model", str(s.get("model", "qwen/qwen3.6-27b")),
            "--api-key", str(s.get("api_key", "none")),
            "--temperature", str(s.get("temperature", 0.6)),
            "--max-tokens", str(s.get("max_tokens", 65536)),
            "--site-url", str(s.get("site_url", "https://localhost")),
            "--app-name", str(s.get("app_name", "ace-baseline-classifier")),
        ]

        if s.get("system_prompt"):
            cmd.extend(["--system-prompt", str(s["system_prompt"])])
        if s.get("extra_instructions"):
            cmd.extend(["--extra-instructions", str(s["extra_instructions"])])
        if s.get("extra_body"):
            cmd.extend(["--extra-body", str(s["extra_body"])])
        if s.get("resume", True):
            cmd.append("--resume")
        if s.get("max_records"):
            cmd.extend(["--max-records", str(s["max_records"])])
        if s.get("no_json_mode"):
            cmd.append("--no-json-mode")

        log = self.run_dir / "logs" / "classify.log"
        err = self.run_dir / "logs" / "classify.err"
        with open(log, "w") as lf, open(err, "w") as ef:
            proc = subprocess.Popen(cmd, env=self._env(), stdout=lf, stderr=ef)
            rc = proc.wait()
        if rc != 0:
            raise RuntimeError(f"classify.py exited with code {rc}")

    def finalize(self) -> None:
        bucket_dir = self.run_dir / "artifacts" / "buckets"
        for p in bucket_dir.glob("*.jsonl"):
            if p.is_file():
                self.manifest.artifacts.append(
                    FileRecord(
                        path=str(p.resolve()),
                        sha256=sha256_file(p),
                        role=f"artifact:buckets/{p.name}",
                        size_bytes=p.stat().st_size,
                    )
                )
        self.manifest.summary = {
            "input": str(Path(self.spec["input"]).resolve()),
            "model": self.spec.get("model"),
            "bucket_dir": str(bucket_dir.resolve()),
        }
        self.manifest.write(self.run_dir / "manifest.json")


PLUGINS.register(ClassifyPlugin)
