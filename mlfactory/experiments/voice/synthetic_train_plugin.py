"""mlfactory plugin for the second, synthetic-data voice adapter."""
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


class SyntheticVoiceTrainPlugin(StagePlugin):
    stage = "voice-synthetic-train"

    def __init__(self, manifest: RunManifest):
        super().__init__(manifest)
        self.spec = manifest.spec

    def _env(self) -> dict[str, str]:
        env = dict(os.environ)
        for key, value in DEFAULT_TRAINING_ENV.items():
            env.setdefault(key, value)
        env.update({str(key): str(value) for key, value in self.spec.get("env", {}).items()})
        return env

    def _script_path(self) -> Path:
        return Path(__file__).parent / "train_synthetic_voice.py"

    def prepare(self) -> None:
        s = self.spec
        base = Path(s["base_model"]).resolve()
        teacher = Path(s["init_adapter"]).resolve()
        train_file = Path(s["train_file"]).resolve()
        eval_file = Path(s["eval_file"]).resolve()
        if not (base / "config.json").is_file():
            raise FileNotFoundError(f"base model is incomplete: {base}")
        for path in (teacher, train_file, eval_file):
            if not path.exists():
                raise FileNotFoundError(path)
        if self.run_dir.resolve() == teacher or teacher in self.run_dir.resolve().parents:
            raise ValueError("second run cannot write inside the teacher adapter")
        (self.run_dir / "artifacts").mkdir(parents=True, exist_ok=True)
        (self.run_dir / "logs").mkdir(parents=True, exist_ok=True)
        snapshot = self.run_dir / "artifacts" / "source_snapshot"
        snapshot.mkdir(parents=True, exist_ok=True)
        shutil.copy2(self._script_path(), snapshot / "train_synthetic_voice.py")
        shutil.copy2(Path(__file__), snapshot / "synthetic_train_plugin.py")
        (snapshot / "effective_spec.json").write_text(json.dumps(s, indent=2, sort_keys=True) + "\n", encoding="utf-8")

    def execute(self) -> None:
        s = self.spec
        python = str(s.get("python", sys.executable))
        cmd = [
            python, str(self._script_path()),
            "--run-dir", str(self.run_dir),
            "--base-model", str(Path(s["base_model"]).resolve()),
            "--init-adapter", str(Path(s["init_adapter"]).resolve()),
            "--train-file", str(Path(s["train_file"]).resolve()),
            "--eval-file", str(Path(s["eval_file"]).resolve()),
            "--device", str(s.get("device", "cuda:0")),
            "--seed", str(s.get("seed", 20260805)),
            "--max-length", str(s.get("max_length", 512)),
            "--max-target-tokens", str(s.get("max_target_tokens", 192)),
            "--batch-size", str(s.get("batch_size", 1)),
            "--gradient-accumulation-steps", str(s.get("gradient_accumulation_steps", 4)),
            "--max-steps", str(s.get("max_steps", 900)),
            "--eval-every", str(s.get("eval_every", 50)),
            "--eval-examples", str(s.get("eval_examples", 256)),
            "--save-every", str(s.get("save_every", 100)),
            "--learning-rate", str(s.get("learning_rate", 1e-5)),
            "--weight-decay", str(s.get("weight_decay", 0.01)),
            "--max-grad-norm", str(s.get("max_grad_norm", 1.0)),
            "--warmup-steps", str(s.get("warmup_steps", 50)),
        ]
        if s.get("load_in_4bit"):
            cmd.append("--load-in-4bit")
        log_path = self.run_dir / "logs" / "train.log"
        err_path = self.run_dir / "logs" / "train.err"
        with log_path.open("w", encoding="utf-8") as log, err_path.open("w", encoding="utf-8") as err:
            log.write("COMMAND " + " ".join(cmd) + "\n")
            log.flush()
            rc = subprocess.call(cmd, env=self._env(), stdout=log, stderr=err)
        if rc != 0:
            raise RuntimeError(f"train_synthetic_voice.py exited with code {rc}")

    def finalize(self) -> None:
        artifacts_dir = self.run_dir / "artifacts"
        self.manifest.artifacts = []
        for path in sorted(artifacts_dir.rglob("*")):
            if path.is_file():
                self.manifest.artifacts.append(FileRecord(path=str(path.resolve()), sha256=sha256_file(path), role=f"artifact:{path.relative_to(artifacts_dir)}", size_bytes=path.stat().st_size))
        summary_path = artifacts_dir / "summary.json"
        if summary_path.exists():
            self.manifest.summary = json.loads(summary_path.read_text(encoding="utf-8"))
        self.manifest.write(self.run_dir / "manifest.json")


PLUGINS.register(SyntheticVoiceTrainPlugin)
