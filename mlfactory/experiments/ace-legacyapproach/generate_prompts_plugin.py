"""Factory plugin that wraps the native ACE generate_prompts.py corpus builder."""
from __future__ import annotations

import os
import subprocess
import sys
from pathlib import Path

from mlfactory.core.manifest import FileRecord, sha256_file
from mlfactory.plugins.base import PLUGINS, StagePlugin


class GeneratePlugin(StagePlugin):
    stage = "generate"

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
            str(self._script_path("generate_prompts.py")),
            "--run-dir", str(self.run_dir),
            "--target", str(s.get("target", 400)),
        ]
        if s.get("extra_instructions"):
            cmd.extend(["--extra-instructions", str(s["extra_instructions"])])

        log = self.run_dir / "logs" / "generate.log"
        err = self.run_dir / "logs" / "generate.err"
        with open(log, "w") as lf, open(err, "w") as ef:
            proc = subprocess.Popen(cmd, env=self._env(), stdout=lf, stderr=ef)
            rc = proc.wait()
        if rc != 0:
            raise RuntimeError(f"generate_prompts.py exited with code {rc}")

    def finalize(self) -> None:
        artifacts_dir = self.run_dir / "artifacts"
        for pattern in ["prompts.jsonl", "verifiers.jsonl", "verifiers/*"]:
            for p in artifacts_dir.glob(pattern):
                if p.is_file():
                    self.manifest.artifacts.append(
                        FileRecord(
                            path=str(p.resolve()),
                            sha256=sha256_file(p),
                            role=f"artifact:{p.relative_to(artifacts_dir)}",
                            size_bytes=p.stat().st_size,
                        )
                    )
        self.manifest.summary = {
            "target": self.spec.get("target", 400),
            "prompts_path": str((artifacts_dir / "prompts.jsonl").resolve()),
        }
        self.manifest.write(self.run_dir / "manifest.json")


PLUGINS.register(GeneratePlugin)
