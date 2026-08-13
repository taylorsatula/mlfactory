"""mlfactory plugin for the local Qwen voice LoRA proof of concept."""
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


class VoiceTrainPlugin(StagePlugin):
    """Run voice/train_voice.py while keeping raw SMS data out of artifacts."""

    stage = "voice-train"

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
        return Path(__file__).parent / "train_voice.py"

    def prepare(self) -> None:
        (self.run_dir / "artifacts").mkdir(parents=True, exist_ok=True)
        (self.run_dir / "logs").mkdir(parents=True, exist_ok=True)
        threads = Path(self.spec["threads"]).resolve()
        model = Path(self.spec["base_model"]).resolve()
        if not threads.is_dir():
            raise FileNotFoundError(f"voice threads directory not found: {threads}")
        if not model.is_dir():
            raise FileNotFoundError(f"base model directory not found: {model}")
        if not (model / "config.json").is_file():
            raise FileNotFoundError(f"base model is not a complete local HF snapshot: {model}")
        # git archive captures only committed files; preserve the exact local
        # executable policy for this run without copying any private corpus.
        snapshot = self.run_dir / "artifacts" / "source_snapshot"
        snapshot.mkdir(parents=True, exist_ok=True)
        shutil.copy2(self._script_path(), snapshot / "train_voice.py")
        shutil.copy2(Path(__file__), snapshot / "train_plugin.py")
        (snapshot / "effective_spec.json").write_text(
            json.dumps(self.spec, indent=2, sort_keys=True) + "\n", encoding="utf-8"
        )

    def execute(self) -> None:
        s = self.spec
        python = str(s.get("python", sys.executable))
        cmd = [
            python,
            str(self._script_path()),
            "--run-dir", str(self.run_dir),
            "--base-model", str(Path(s["base_model"]).resolve()),
            "--threads", str(Path(s["threads"]).resolve()),
            "--device", str(s.get("device", "cuda:0")),
            "--seed", str(s.get("seed", 42)),
            "--test-fraction", str(s.get("test_fraction", 0.2)),
            "--history-messages", str(s.get("history_messages", 8)),
            "--max-train-examples", str(s.get("max_train_examples", 300)),
            "--max-test-examples", str(s.get("max_test_examples", 60)),
            "--max-length", str(s.get("max_length", 384)),
            "--max-target-tokens", str(s.get("max_target_tokens", 128)),
            "--batch-size", str(s.get("batch_size", 1)),
            "--gradient-accumulation-steps", str(s.get("gradient_accumulation_steps", 4)),
            "--max-steps", str(s.get("max_steps", 80)),
            "--eval-every", str(s.get("eval_every", 20)),
            "--eval-examples", str(s.get("eval_examples", 24)),
            "--save-every", str(s.get("save_every", 100)),
            "--learning-rate", str(s.get("learning_rate", 2e-4)),
            "--weight-decay", str(s.get("weight_decay", 0.0)),
            "--max-grad-norm", str(s.get("max_grad_norm", 1.0)),
            "--lora-r", str(s.get("lora_r", 8)),
            "--lora-alpha", str(s.get("lora_alpha", 16)),
            "--lora-dropout", str(s.get("lora_dropout", 0.05)),
            "--target-modules", str(s.get("target_modules", "q_proj,k_proj,v_proj,o_proj")),
            "--generation-tokens", str(s.get("generation_tokens", 96)),
            "--model-class", str(s.get("model_class", "causal_lm")),
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
            raise RuntimeError(f"train_voice.py exited with code {rc}")

    def finalize(self) -> None:
        artifacts_dir = self.run_dir / "artifacts"
        self.manifest.artifacts = []
        for path in sorted(artifacts_dir.rglob("*")):
            if path.is_file():
                self.manifest.artifacts.append(
                    FileRecord(
                        path=str(path.resolve()),
                        sha256=sha256_file(path),
                        role=f"artifact:{path.relative_to(artifacts_dir)}",
                        size_bytes=path.stat().st_size,
                    )
                )
        summary_path = artifacts_dir / "summary.json"
        if summary_path.exists():
            self.manifest.summary = json.loads(summary_path.read_text(encoding="utf-8"))
        self.manifest.write(self.run_dir / "manifest.json")


PLUGINS.register(VoiceTrainPlugin)
