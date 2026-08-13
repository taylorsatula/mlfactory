"""Reproducible runner for the fresh, diverse robust voice adapter."""
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


class RobustVoiceTrainPlugin(StagePlugin):
    stage = "voice-robust-train"

    def __init__(self, manifest: RunManifest):
        super().__init__(manifest)
        self.spec = manifest.spec

    def _env(self) -> dict[str, str]:
        env = dict(os.environ)
        for key, value in DEFAULT_TRAINING_ENV.items():
            env.setdefault(key, value)
        env.update({str(key): str(value) for key, value in self.spec.get("env", {}).items()})
        env.setdefault("PYTHONPATH", str(Path(__file__).resolve().parents[3]))
        return env

    def _snapshot(self) -> list[Path]:
        return [
            Path(__file__).parent / "robust_train_plugin.py",
            Path(__file__).parent / "train_robust_voice.py",
            Path(__file__).parent / "build_robust_voice_data.py",
            Path(__file__).parent / "generate_grounded_voice.py",
            Path(__file__).parent / "filter_grounded_voice.py",
            Path(__file__).parent / "data" / "grounded_scenario_catalog.json",
            Path(__file__).parent / "voice_prompt.py",
            Path(__file__).parent / "voice_safety.py",
            Path(__file__).parent / "train_voice.py",
        ]

    def prepare(self) -> None:
        base = Path(self.spec["base_model"]).resolve()
        threads = Path(self.spec["threads"]).resolve()
        if not (base / "config.json").is_file():
            raise FileNotFoundError(f"base model is incomplete: {base}")
        if not threads.is_dir():
            raise FileNotFoundError(threads)
        (self.run_dir / "artifacts").mkdir(parents=True, exist_ok=True)
        (self.run_dir / "logs").mkdir(parents=True, exist_ok=True)
        snapshot = self.run_dir / "artifacts" / "source_snapshot"
        snapshot.mkdir(parents=True, exist_ok=True)
        for source in self._snapshot():
            shutil.copy2(source, snapshot / source.name)
        (snapshot / "effective_spec.json").write_text(json.dumps(self.spec, indent=2, sort_keys=True) + "\n", encoding="utf-8")

    def execute(self) -> None:
        s = self.spec
        python = str(s.get("python", sys.executable))
        synthetic_dir = self.run_dir / "artifacts" / "synthetic_data"
        builder = str(Path(__file__).parent / "build_robust_voice_data.py")
        train = str(Path(__file__).parent / "train_robust_voice.py")
        log_path = self.run_dir / "logs" / "train.log"
        err_path = self.run_dir / "logs" / "train.err"
        with log_path.open("w", encoding="utf-8") as log, err_path.open("w", encoding="utf-8") as err:
            supplied_data = s.get("synthetic_data_dir")
            if supplied_data:
                source_dir = Path(str(supplied_data)).resolve()
                required = [source_dir / name for name in ("train.jsonl", "eval.jsonl", "replay.jsonl")]
                if not source_dir.is_dir() or not all(path.is_file() for path in required):
                    raise FileNotFoundError(f"synthetic_data_dir must contain train.jsonl, eval.jsonl, replay.jsonl: {source_dir}")
                synthetic_dir.mkdir(parents=True, exist_ok=True)
                for source in required:
                    shutil.copy2(source, synthetic_dir / source.name)
                log.write("SYNTHETIC_DATA " + str(source_dir) + "\n")
            else:
                build_cmd = [python, builder, "--output-dir", str(synthetic_dir), "--seed", str(s.get("data_seed", 20260806))]
                log.write("BUILD " + " ".join(build_cmd) + "\n"); log.flush()
                if subprocess.call(build_cmd, env=self._env(), stdout=log, stderr=err) != 0:
                    raise RuntimeError("robust synthetic data build failed")
            log.flush()
            cmd = [
                python, train, "--run-dir", str(self.run_dir), "--base-model", str(Path(s["base_model"]).resolve()),
                "--threads", str(Path(s["threads"]).resolve()), "--synthetic-train", str(synthetic_dir / "train.jsonl"),
                "--synthetic-eval", str(synthetic_dir / "eval.jsonl"), "--replay-file", str(synthetic_dir / "replay.jsonl"),
                "--device", str(s.get("device", "cuda:0")), "--seed", str(s.get("seed", 20260806)),
            ]
            for key in ("max_real_train", "max_real_eval", "synthetic_repeats", "replay_repeats", "max_length", "max_target_tokens", "batch_size", "gradient_accumulation_steps", "max_steps", "eval_every", "eval_examples", "save_every", "learning_rate", "weight_decay", "max_grad_norm", "warmup_steps", "lora_r", "lora_alpha", "lora_dropout", "target_modules"):
                if key in s:
                    cmd.extend([f"--{key.replace('_', '-')}", str(s[key])])
            if s.get("load_in_4bit"):
                cmd.append("--load-in-4bit")
            log.write("TRAIN " + " ".join(cmd) + "\n"); log.flush()
            if subprocess.call(cmd, env=self._env(), stdout=log, stderr=err) != 0:
                raise RuntimeError("robust voice training failed")

    def finalize(self) -> None:
        artifacts = self.run_dir / "artifacts"
        self.manifest.artifacts = []
        for path in sorted(artifacts.rglob("*")):
            if path.is_file():
                self.manifest.artifacts.append(FileRecord(path=str(path.resolve()), sha256=sha256_file(path), role=f"artifact:{path.relative_to(artifacts)}", size_bytes=path.stat().st_size))
        summary = artifacts / "summary.json"
        if summary.exists():
            self.manifest.summary = json.loads(summary.read_text(encoding="utf-8"))
        self.manifest.write(self.run_dir / "manifest.json")


PLUGINS.register(RobustVoiceTrainPlugin)
