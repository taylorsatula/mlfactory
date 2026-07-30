"""Factory plugin that wraps the native DFT build_pilot.py pilot builder."""
from __future__ import annotations

import os
import subprocess
import sys
from pathlib import Path

from mlfactory.core.manifest import FileRecord, sha256_file
from mlfactory.plugins.base import PLUGINS, StagePlugin


class BuildPilotPlugin(StagePlugin):
    stage = "build-pilot"

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
            str(self._script_path("build_pilot.py")),
            "--run-dir", str(self.run_dir),
            "--train-size", str(s.get("train_size", 10000)),
            "--test-size", str(s.get("test_size", 500)),
            "--fineweb-sample", str(s.get("fineweb_sample", "CC-MAIN-2024-10")),
            "--seed", str(s.get("seed", 42)),
            "--tokenizer", str(s.get("tokenizer", "qwen/Qwen3.6-27B")),
        ]
        if s.get("dry_run", 0):
            cmd.extend(["--dry-run", str(s["dry_run"])])
        if s.get("resume"):
            cmd.append("--resume")
        if s.get("extra_instructions"):
            cmd.extend(["--extra-instructions", str(s["extra_instructions"])])
        if s.get("clean_prompt"):
            cmd.extend(["--clean-prompt", str(s["clean_prompt"])])
        if s.get("metadata_prompt"):
            cmd.extend(["--metadata-prompt", str(s["metadata_prompt"])])
        if s.get("outline_prompt"):
            cmd.extend(["--outline-prompt", str(s["outline_prompt"])])

        log = self.run_dir / "logs" / "build_pilot.log"
        err = self.run_dir / "logs" / "build_pilot.err"
        with open(log, "w") as lf, open(err, "w") as ef:
            proc = subprocess.Popen(cmd, env=self._env(), stdout=lf, stderr=ef)
            rc = proc.wait()
        if rc != 0:
            raise RuntimeError(f"build_pilot.py exited with code {rc}")

    def finalize(self) -> None:
        artifacts_dir = self.run_dir / "artifacts"
        for pattern in ["train.jsonl", "test.jsonl", "partial.jsonl", "summary.json", "build_pilot_config.json"]:
            for p in artifacts_dir.glob(pattern):
                if p.is_file():
                    self.manifest.artifacts.append(
                        FileRecord(
                            path=str(p.resolve()),
                            sha256=sha256_file(p),
                            role=f"artifact:{p.name}",
                            size_bytes=p.stat().st_size,
                        )
                    )
        summary_path = artifacts_dir / "summary.json"
        if summary_path.exists():
            import json
            summary = json.loads(summary_path.read_text(encoding="utf-8"))
            self.manifest.summary = summary
        self.manifest.write(self.run_dir / "manifest.json")


PLUGINS.register(BuildPilotPlugin)
