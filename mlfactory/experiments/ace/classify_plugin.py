"""Factory plugin that wraps the ACE classify.py classifier harness.

Classifies a generations JSONL into KEEP / BORDERLINE / REJECT buckets using a
judge model accessed through an OpenAI-compatible endpoint.
"""
from __future__ import annotations

import os
import subprocess
import sys
from pathlib import Path
from typing import Any

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
        env.setdefault("PYTHONUNBUFFERED", "1")
        return env

    def _arg(self, key: str, default: Any | None = None) -> Any:
        return self.spec.get(key, default)

    def prepare(self) -> None:
        (self.run_dir / "artifacts").mkdir(parents=True, exist_ok=True)
        (self.run_dir / "logs").mkdir(parents=True, exist_ok=True)

    def execute(self) -> None:
        env = self._env()
        s = self.spec
        python = s.get("python", sys.executable)

        bucket_dir = self.run_dir / "artifacts" / "buckets"
        bucket_dir.mkdir(parents=True, exist_ok=True)

        cmd = [
            python,
            str(self._script_path("classify.py")),
            str(Path(s["input"]).resolve()),
            "--bucket-dir", str(bucket_dir),
            "--base-url", str(s.get("base_url", "https://openrouter.ai/api/v1")),
            "--model", str(s.get("model", "qwen/qwen3.6-27b")),
            "--api-key", str(s.get("api_key", os.environ.get("ACE_CLASSIFIER_API_KEY", "none"))),
            "--temperature", str(s.get("temperature", 0.6)),
            "--max-tokens", str(s.get("max_tokens", 65536)),
            "--site-url", str(s.get("site_url", "https://localhost")),
            "--app-name", str(s.get("app_name", "ace-baseline-classifier")),
            "--manifest-name", "classifier_manifest.json",
        ]

        if s.get("system_prompt"):
            cmd.extend(["--system-prompt", str(s["system_prompt"])])
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
            proc = subprocess.Popen(cmd, env=env, stdout=lf, stderr=ef)
            rc = proc.wait()
        if rc != 0:
            raise RuntimeError(f"classify.py exited with code {rc}")

    def finalize(self) -> None:
        bucket_dir = self.run_dir / "artifacts" / "buckets"
        for pattern in ["*.jsonl", "*.json"]:
            for p in bucket_dir.glob(pattern):
                if p.is_file():
                    self.manifest.artifacts.append(
                        FileRecord(
                            path=str(p.resolve()),
                            sha256=sha256_file(p),
                            role=f"artifact:buckets/{p.name}",
                            size_bytes=p.stat().st_size,
                        )
                    )
        manifest_path = bucket_dir / "classifier_manifest.json"
        if manifest_path.exists():
            import json
            summary = json.loads(manifest_path.read_text(encoding="utf-8"))
            self.manifest.summary = summary
        self.manifest.write(self.run_dir / "manifest.json")


PLUGINS.register(ClassifyPlugin)
