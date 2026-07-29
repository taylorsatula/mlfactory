"""Factory plugin that wraps the legacy DFT train_dft.py.

The plugin translates the mlfactory spec into train_dft.py CLI arguments and
records the resulting checkpoints / summary / logs as artifacts.
"""
from __future__ import annotations

import os
import subprocess
import sys
from pathlib import Path
from typing import Any

from mlfactory.core.manifest import FileRecord, sha256_file
from mlfactory.plugins.base import PLUGINS, StagePlugin


class TrainPlugin(StagePlugin):
    stage = "train"

    def __init__(self, manifest):
        super().__init__(manifest)
        self.run_dir = Path(self.manifest.source.path).parent
        self.spec = manifest.spec

    def _env(self) -> dict[str, str]:
        env = dict(os.environ)
        env.update(self.spec.get("env", {}))
        env.setdefault("PYTHONUNBUFFERED", "1")
        # Standard DFT environment defaults.
        env.setdefault("PYTORCH_ALLOC_CONF", "expandable_segments:True,roundup_power2_divisions:[32:256,64:128,256:64,>:32]")
        env.setdefault("PYTORCH_CUDA_ALLOC_CONF", env["PYTORCH_ALLOC_CONF"])
        env.setdefault("TRITON_DISABLE_AUTOTUNING", "1")
        return env

    def _script_path(self, name: str) -> Path:
        return Path(__file__).parent / name

    def prepare(self) -> None:
        (self.run_dir / "artifacts").mkdir(parents=True, exist_ok=True)
        (self.run_dir / "logs").mkdir(parents=True, exist_ok=True)

    def execute(self) -> None:
        env = self._env()
        s = self.spec
        out_dir = self.run_dir / "artifacts"
        python = s.get("python", sys.executable)

        cmd = [
            python,
            str(self._script_path("train_dft.py")),
            "--model-name", str(s.get("model_name", "Qwen/Qwen3.5-4B")),
            "--train-file", str(s["train_file"]),
            "--test-file", str(s["test_file"]),
            "--out-dir", str(out_dir),
            "--embed-model", str(s.get("embed_model", "nvidia/llama-embed-nemotron-8b")),
            "--embed-device", str(s.get("embed_device", "cuda:1")),
            "--device", str(s.get("device", "cuda:0")),
            "--ref-device", str(s.get("ref_device", "cuda:1")),
            "--mmd-ref-sample-size", str(s.get("mmd_ref_sample_size", 256)),
            "--kernel", str(s.get("kernel", "rq")),
            "--mmd-bandwidth", str(s.get("mmd_bandwidth", "median")),
            "--rq-alpha", str(s.get("rq_alpha", 1.0)),
            "--lora-r", str(s.get("lora_r", 16)),
            "--lora-alpha", str(s.get("lora_alpha", 32)),
            "--lora-dropout", str(s.get("lora_dropout", 0.0)),
            "--lora-target", str(s.get("lora_target", "q_proj,k_proj,v_proj,o_proj")),
            "--max-prompt-length", str(s.get("max_prompt_length", 512)),
            "--max-response-length", str(s.get("max_response_length", 1024)),
            "--temperature", str(s.get("temperature", 0.8)),
            "--top-p", str(s.get("top_p", 0.95)),
            "--top-k", str(s.get("top_k", 50)),
            "--rollout-temperature", str(s.get("rollout_temperature", 1.0)),
            "--rollout-top-p", str(s.get("rollout_top_p", 1.0)),
            "--rollout-top-k", str(s.get("rollout_top_k", 0)),
            "--batch-size", str(s.get("batch_size", 8)),
            "--gradient-accumulation-steps", str(s.get("gradient_accumulation_steps", 1)),
            "--ppo-epochs", str(s.get("ppo_epochs", 4)),
            "--num-train-epochs", str(s.get("num_train_epochs", 3)),
            "--lr", str(s.get("lr", 1e-5)),
            "--kl-coef", str(s.get("kl_coef", 0.2)),
            "--cliprange", str(s.get("cliprange", 0.2)),
            "--vf-coef", str(s.get("vf_coef", 0.5)),
            "--ent-coef", str(s.get("ent_coef", 0.0)),
            "--gamma", str(s.get("gamma", 1.0)),
            "--lam", str(s.get("lam", 1.0)),
            "--eval-every", str(s.get("eval_every", 100)),
            "--num-eval-samples", str(s.get("num_eval_samples", 100)),
            "--save-every", str(s.get("save_every", 100)),
            "--max-steps", str(s.get("max_steps", -1)),
            "--seed", str(s.get("seed", 42)),
        ]
        if s.get("load_in_4bit"):
            cmd.append("--load-in-4bit")
        if s.get("critic_mode"):
            cmd.extend(["--critic-mode", str(s["critic_mode"])])

        log = self.run_dir / "logs" / "train.log"
        err = self.run_dir / "logs" / "train.err"
        with open(log, "w") as lf, open(err, "w") as ef:
            proc = subprocess.Popen(
                cmd,
                env=env,
                stdout=lf,
                stderr=ef,
            )
            rc = proc.wait()
        if rc != 0:
            raise RuntimeError(f"train_dft.py exited with code {rc}")

    def finalize(self) -> None:
        out_dir = self.run_dir / "artifacts"
        # Hash all checkpoint / final / summary files.
        for pattern in ["summary.json", "train_config.json", "final/**/*", "checkpoint-*/**/*", "logs/*"]:
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
            if summary.get("guard") or summary.get("guard_report"):
                self.manifest.status = "guarded"
        self.manifest.write(self.run_dir / "manifest.json")


PLUGINS.register(TrainPlugin)
