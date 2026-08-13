"""Reproducible runner for the human-preference DPO style pass."""
from __future__ import annotations

import json
import os
import shutil
import subprocess
import sys
from pathlib import Path

from mlfactory.core.env import DEFAULT_TRAINING_ENV
from mlfactory.core.manifest import FileRecord, RunManifest, sha256_file
from mlfactory.plugins.base import PLUGINS, StagePlugin


class VoiceDPOTrainPlugin(StagePlugin):
    stage = "voice-dpo-train"

    def __init__(self, manifest: RunManifest):
        super().__init__(manifest)
        self.spec = manifest.spec

    def _env(self) -> dict[str, str]:
        env = dict(os.environ)
        for key, value in DEFAULT_TRAINING_ENV.items():
            env.setdefault(key, str(value))
        env.update({str(key): str(value) for key, value in self.spec.get("env", {}).items()})
        env.setdefault("PYTHONPATH", str(Path(__file__).resolve().parents[3]))
        return env

    def prepare(self) -> None:
        base = Path(self.spec["base_model"]).resolve()
        adapter = Path(self.spec["init_adapter"]).resolve()
        threads = Path(self.spec["threads"]).resolve()
        if not (base / "config.json").is_file(): raise FileNotFoundError(base)
        if not adapter.is_dir(): raise FileNotFoundError(adapter)
        if not threads.is_dir(): raise FileNotFoundError(threads)
        (self.run_dir / "artifacts").mkdir(parents=True, exist_ok=True)
        (self.run_dir / "logs").mkdir(parents=True, exist_ok=True)
        snapshot = self.run_dir / "artifacts" / "source_snapshot"; snapshot.mkdir(parents=True, exist_ok=True)
        for name in ("dpo_train_plugin.py", "train_dpo_voice.py", "train_voice.py", "train_robust_voice.py", "voice_prompt.py", "voice_safety.py"):
            shutil.copy2(Path(__file__).parent / name, snapshot / name)
        (snapshot / "effective_spec.json").write_text(json.dumps(self.spec, indent=2, sort_keys=True) + "\n", encoding="utf-8")

    def execute(self) -> None:
        s = self.spec; python = str(s.get("python", sys.executable))
        cmd = [python, str(Path(__file__).parent / "train_dpo_voice.py"), "--run-dir", str(self.run_dir), "--base-model", str(Path(s["base_model"]).resolve()), "--init-adapter", str(Path(s["init_adapter"]).resolve()), "--threads", str(Path(s["threads"]).resolve()), "--device", str(s.get("device", "cuda:0"))]
        for key in ("seed", "max_real_train", "max_real_eval", "generation_batch_size", "max_new_tokens", "max_length", "max_target_tokens", "batch_size", "gradient_accumulation_steps", "max_steps", "eval_every", "eval_examples", "save_every", "learning_rate", "weight_decay", "max_grad_norm", "warmup_steps", "beta"):
            if key in s: cmd.extend([f"--{key.replace('_', '-')}", str(s[key])])
        log = self.run_dir / "logs" / "train.log"; err = self.run_dir / "logs" / "train.err"
        with log.open("w", encoding="utf-8") as out, err.open("w", encoding="utf-8") as error:
            out.write("COMMAND " + " ".join(cmd) + "\n"); out.flush()
            if subprocess.call(cmd, env=self._env(), stdout=out, stderr=error) != 0: raise RuntimeError("DPO voice training failed")

    def finalize(self) -> None:
        artifacts = self.run_dir / "artifacts"
        self.manifest.artifacts = [FileRecord(path=str(p.resolve()), sha256=sha256_file(p), role=f"artifact:{p.relative_to(artifacts)}", size_bytes=p.stat().st_size) for p in sorted(artifacts.rglob("*")) if p.is_file()]
        summary = artifacts / "summary.json"
        if summary.exists(): self.manifest.summary = json.loads(summary.read_text(encoding="utf-8"))
        self.manifest.write(self.run_dir / "manifest.json")


PLUGINS.register(VoiceDPOTrainPlugin)
