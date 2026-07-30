"""Factory plugin that wraps the DFT eval.py evaluation harness.

Evaluates a generator against a reference test set using MMD, token L2,
slop diagnostics, and optional LLM-as-judge pairwise comparisons.
"""
from __future__ import annotations

import os
import subprocess
import sys
from pathlib import Path
from typing import Any

from mlfactory.core.manifest import FileRecord, sha256_file
from mlfactory.core.model_server import model
from mlfactory.plugins.base import PLUGINS, StagePlugin


class EvalPlugin(StagePlugin):
    stage = "eval"

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

    def prepare(self) -> None:
        (self.run_dir / "artifacts").mkdir(parents=True, exist_ok=True)
        (self.run_dir / "logs").mkdir(parents=True, exist_ok=True)

    def execute(self) -> None:
        s = self.spec
        python = s.get("python", sys.executable)
        env = self._env()

        out_dir = self.run_dir / "artifacts"
        out_dir.mkdir(parents=True, exist_ok=True)

        cmd = [
            python,
            str(self._script_path("eval.py")),
            "--run-dir", str(self.run_dir),
            "--test-file", str(s["test_file"]),
            "--gen-model", str(s.get("gen_model", "Qwopus3.6-27b")),
            "--tokenizer", str(s.get("tokenizer", "qwen/Qwen3.6-27B")),
            "--embed-model", str(s.get("embed_model", "nvidia/llama-embed-nemotron-8b")),
            "--embed-device", str(s.get("embed_device", "cuda:0")),
            "--temps", str(s.get("temps", "0.7,1.0")),
            "--top-p", str(s.get("top_p", 1.0)),
            "--top-k", str(s.get("top_k", 0)),
            "--repeat-penalty", str(s.get("repeat_penalty", 1.0)),
            "--presence-penalty", str(s.get("presence_penalty", 0.0)),
            "--frequency-penalty", str(s.get("frequency_penalty", 0.0)),
            "--max-tokens", str(s.get("max_tokens", 512)),
            "--judge-criterion", str(s.get("judge_criterion", "overall quality")),
            "--judge-samples", str(s.get("judge_samples", 50)),
            "--seed", str(s.get("seed", 42)),
        ]

        if s.get("n"):
            cmd.extend(["--n", str(s["n"])])
        if s.get("system_prompt"):
            cmd.extend(["--system-prompt", str(s["system_prompt"])])
        if s.get("judge_url"):
            cmd.extend(["--judge-url", str(s["judge_url"])])
        if s.get("judge_model"):
            cmd.extend(["--judge-model", str(s["judge_model"])])
        if s.get("strip_think", True):
            cmd.append("--strip-think")
        else:
            cmd.append("--no-strip-think")

        model_alias = s.get("model")
        if model_alias:
            gpu = s.get("gpu", 0)
            with model(model_alias, gpu=gpu) as srv:
                cmd.extend(["--gen-url", srv.base_url])
                self._run(cmd, env)
        else:
            cmd.extend(["--gen-url", str(s.get("gen_url", "http://localhost:3090/v1"))])
            self._run(cmd, env)

    def _run(self, cmd: list[str], env: dict[str, str]) -> None:
        log = self.run_dir / "logs" / "eval.log"
        err = self.run_dir / "logs" / "eval.err"
        with open(log, "w") as lf, open(err, "w") as ef:
            proc = subprocess.Popen(cmd, env=env, stdout=lf, stderr=ef)
            rc = proc.wait()
        if rc != 0:
            raise RuntimeError(f"eval.py exited with code {rc}")

    def finalize(self) -> None:
        out_dir = self.run_dir / "artifacts"
        for pattern in ["summary.json", "completions_*.jsonl"]:
            for p in out_dir.glob(pattern):
                if p.is_file():
                    self.manifest.artifacts.append(
                        FileRecord(
                            path=str(p.resolve()),
                            sha256=sha256_file(p),
                            role=f"artifact:{p.relative_to(out_dir)}",
                            size_bytes=p.stat().st_size,
                        )
                    )
        summary_path = out_dir / "summary.json"
        if summary_path.exists():
            import json
            summary = json.loads(summary_path.read_text(encoding="utf-8"))
            self.manifest.summary = summary
        self.manifest.write(self.run_dir / "manifest.json")


PLUGINS.register(EvalPlugin)
