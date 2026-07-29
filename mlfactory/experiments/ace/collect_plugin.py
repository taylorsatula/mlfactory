"""Factory plugin that wraps the legacy ACE collect.py / llama-server flow.

This is an adapter: the heavy lifting still lives in ``collect.py``; the plugin
provides the mlfactory harness around it (manifest, source snapshot, telemetry,
artifact hashing).
"""
from __future__ import annotations

import os
import shutil
import signal
import subprocess
import sys
import time
from pathlib import Path
from typing import Any

from mlfactory.core.manifest import FileRecord, sha256_file
from mlfactory.plugins.base import PLUGINS, StagePlugin


class CollectPlugin(StagePlugin):
    stage = "collect"

    def __init__(self, manifest):
        super().__init__(manifest)
        self.run_dir = Path(self.manifest.source.path).parent
        self.server_proc: subprocess.Popen | None = None
        self.spec = manifest.spec

    def _env(self) -> dict[str, str]:
        env = dict(os.environ)
        env.update(self.spec.get("env", {}))
        env.setdefault("PYTHONUNBUFFERED", "1")
        return env

    def _script_path(self, name: str) -> Path:
        return Path(__file__).parent / name

    def prepare(self) -> None:
        (self.run_dir / "artifacts").mkdir(parents=True, exist_ok=True)
        (self.run_dir / "logs").mkdir(parents=True, exist_ok=True)
        provider = self.spec.get("provider", "llama")
        if provider == "llama":
            self._start_llama_server()

    def _start_llama_server(self) -> None:
        server = self.spec.get("server", {})
        port = server.get("port", 3090)
        gpu = server.get("gpu", 0)
        model = server.get("model", "/home/admin/models/Qwen3.5-4B-UD-Q8_K_XL.gguf")
        env = self._env()
        env["ACE_LLAMA_PORT"] = str(port)
        env["ACE_MAIN_GPU"] = str(gpu)
        env["ACE_MODEL_GGUF"] = str(model)

        cmd = [
            "bash",
            str(self._script_path("launch_llama_qwen35_4b.sh")),
        ]
        log = self.run_dir / "logs" / "llama-server.log"
        self.server_proc = subprocess.Popen(
            cmd,
            env=env,
            stdout=open(log, "w"),
            stderr=subprocess.STDOUT,
            start_new_session=True,
        )
        # Wait for health endpoint.
        url = f"http://127.0.0.1:{port}/v1/models"
        for _ in range(120):
            try:
                out = subprocess.check_output(
                    ["curl", "-sf", "-m", "5", url],
                    stderr=subprocess.DEVNULL,
                    text=True,
                )
                if self.spec.get("model_name", "") in out:
                    break
            except Exception:
                pass
            if self.server_proc.poll() is not None:
                raise RuntimeError("llama-server exited early")
            time.sleep(5)
        else:
            self._stop_server()
            raise RuntimeError("llama-server did not become ready")

    def execute(self) -> None:
        env = self._env()
        env["ACE_BASE_URL"] = f"http://127.0.0.1:{self.spec.get('server', {}).get('port', 3090)}/v1"
        env["ACE_MODEL_NAME"] = self.spec.get("model_name", "Qwen/Qwen3.5-4B")
        env["ACE_PROVIDER"] = self.spec.get("provider", "llama")
        env["ACE_MAX_MODEL_LEN"] = str(self.spec.get("max_model_len", 32768))
        env["ACE_MAX_OUTPUT_TOKENS"] = str(self.spec.get("max_output_tokens", 16000))
        env["ACE_REQUEST_TIMEOUT"] = str(self.spec.get("request_timeout", 1200))
        env["ACE_TIME_BUDGET_SECONDS"] = str(self.spec.get("time_budget_seconds", 900))
        env["ACE_SEED_OFFSET"] = str(self.spec.get("seed_offset", 0))
        env["ACE_STRATIFIED_EXTRAS"] = str(self.spec.get("stratified_extras", 3))

        prompts = Path(self.spec["prompts"])
        out_dir = self.run_dir / "artifacts"
        python = self.spec.get("python", sys.executable)
        cmd = [
            python,
            str(self._script_path("collect.py")),
            "--prompts", str(prompts),
            "--out-dir", str(out_dir),
        ]
        if self.spec.get("max_samples"):
            cmd.extend(["--max-samples", str(self.spec["max_samples"])])

        log = self.run_dir / "logs" / "collect.log"
        with open(log, "w") as lf:
            proc = subprocess.Popen(
                cmd,
                env=env,
                stdout=lf,
                stderr=subprocess.STDOUT,
            )
            rc = proc.wait()
        if rc != 0:
            raise RuntimeError(f"collect.py exited with code {rc}")

    def finalize(self) -> None:
        self._stop_server()
        out_dir = self.run_dir / "artifacts"
        for name in ["generations.jsonl", "manifest.json"]:
            p = out_dir / name
            if p.exists():
                self.manifest.artifacts.append(
                    FileRecord(
                        path=str(p.resolve()),
                        sha256=sha256_file(p),
                        role=f"artifact:{name}",
                        size_bytes=p.stat().st_size,
                    )
                )
        self.manifest.summary = {
            "provider": self.spec.get("provider"),
            "model_name": self.spec.get("model_name"),
            "prompts": str(Path(self.spec["prompts"]).resolve()),
        }
        self.manifest.write(self.run_dir / "manifest.json")

    def _stop_server(self) -> None:
        if self.server_proc is None:
            return
        try:
            os.killpg(os.getpgid(self.server_proc.pid), signal.SIGTERM)
        except ProcessLookupError:
            pass
        try:
            self.server_proc.wait(timeout=30)
        except subprocess.TimeoutExpired:
            try:
                os.killpg(os.getpgid(self.server_proc.pid), signal.SIGKILL)
            except ProcessLookupError:
                pass
        self.server_proc = None


PLUGINS.register(CollectPlugin)
