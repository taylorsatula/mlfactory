"""Friendly model-server resource for experiments.

Usage:
    from mlfactory.core.model_server import model

    with model("qwen3.5:4b", gpu=0) as srv:
        client = srv.client()
        response = client.chat.completions.create(...)

The context manager starts a disposable llama-server, waits for it to become
healthy, yields a handle, and stops the server on exit.
"""
from __future__ import annotations

import os
import signal
import socket
import subprocess
import sys
import time
from contextlib import contextmanager
from pathlib import Path
from typing import Any, Iterator

import yaml
from pydantic import BaseModel, Field, field_validator


class ModelSpec(BaseModel):
    """Configuration for one model alias."""

    path: str
    alias: str | None = None
    family: str = "unknown"
    context_size: int = 32768
    max_output_tokens: int | None = None
    gpu_layers: int = 999
    main_gpu: int = 0
    split_mode: str = "none"
    flash_attn: bool = True
    reasoning: bool = False
    reasoning_format: str | None = None
    reasoning_budget: int | None = None
    mtp: bool = False
    mtp_path: str | None = None
    batch_size: int = 2048
    ubatch_size: int = 512
    cache_type_k: str = "q8_0"
    cache_type_v: str = "q8_0"
    extra_args: list[str] = Field(default_factory=list)


class ServerConfig(BaseModel):
    """Global server settings from the model registry."""

    binary: str = "/home/admin/llama.cpp/build/bin/llama-server"
    host: str = "127.0.0.1"
    port_range_start: int = 3090
    port_range_end: int = 3100
    stop_services: list[str] = Field(default_factory=list)


class ModelRegistry(BaseModel):
    models: dict[str, ModelSpec]
    server: ServerConfig = Field(default_factory=ServerConfig)

    @field_validator("models")
    @classmethod
    def _normalize_aliases(cls, v: dict[str, Any]) -> dict[str, Any]:
        # Allow aliases like "qwen3.5:4b" (already normalized by YAML).
        return v

    @classmethod
    def load(cls, path: Path | None = None) -> "ModelRegistry":
        if path is None:
            path = Path("models.yaml")
        data = yaml.safe_load(path.read_text(encoding="utf-8"))
        return cls.model_validate(data)

    def get(self, alias: str) -> ModelSpec:
        alias = alias.lower().strip()
        if alias not in self.models:
            raise KeyError(
                f"unknown model alias {alias!r}; known: {sorted(self.models.keys())}"
            )
        spec = self.models[alias]
        if spec.alias is None:
            spec.alias = alias
        return spec


# ---------------------------------------------------------------------------
# helpers
# ---------------------------------------------------------------------------

def _find_free_port(host: str, start: int, end: int) -> int:
    for port in range(start, end + 1):
        with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
            s.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
            try:
                s.bind((host, port))
                return port
            except OSError:
                continue
    raise RuntimeError(f"no free port in range {start}-{end}")


def _port_in_use(host: str, port: int) -> bool:
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
        s.settimeout(1)
        return s.connect_ex((host, port)) == 0


def _kill_port_occupant(host: str, port: int) -> None:
    try:
        pids = subprocess.check_output(
            ["lsof", "-ti", f"tcp:{port}"], stderr=subprocess.DEVNULL, text=True
        ).split()
        for pid in pids:
            try:
                os.kill(int(pid), signal.SIGTERM)
            except ProcessLookupError:
                pass
        time.sleep(1)
        for pid in pids:
            try:
                os.kill(int(pid), signal.SIGKILL)
            except ProcessLookupError:
                pass
    except subprocess.CalledProcessError:
        pass


def _stop_services(services: list[str]) -> None:
    for svc in services:
        try:
            out = subprocess.run(
                ["systemctl", "is-active", "--quiet", svc],
                capture_output=True,
            )
            if out.returncode != 0:
                continue
            subprocess.run(
                ["sudo", "-n", "systemctl", "stop", svc],
                check=True,
                capture_output=True,
            )
            subprocess.run(
                ["sudo", "-n", "systemctl", "disable", svc],
                capture_output=True,
            )
            for _ in range(24):
                check = subprocess.run(
                    ["systemctl", "is-active", "--quiet", svc],
                    capture_output=True,
                )
                if check.returncode != 0:
                    break
                time.sleep(5)
        except (subprocess.CalledProcessError, FileNotFoundError):
            pass


def _wait_for_server(base_url: str, timeout: float = 120.0) -> None:
    deadline = time.time() + timeout
    models_url = f"{base_url}/models"
    while time.time() < deadline:
        try:
            out = subprocess.check_output(
                ["curl", "-sf", "-m", "5", models_url],
                stderr=subprocess.DEVNULL,
                text=True,
            )
            if out.strip():
                return
        except subprocess.CalledProcessError:
            pass
        time.sleep(2)
    raise RuntimeError(f"server did not become healthy within {timeout}s")


# ---------------------------------------------------------------------------
# ModelServer
# ---------------------------------------------------------------------------

class ModelServer:
    """A disposable llama-server instance."""

    def __init__(
        self,
        alias: str,
        gpu: int | None = None,
        port: int | None = None,
        host: str | None = None,
        overrides: dict[str, Any] | None = None,
        registry: ModelRegistry | None = None,
    ):
        self.registry = registry or ModelRegistry.load()
        self.spec = self.registry.get(alias)
        self.host = host or self.registry.server.host
        self.port = port
        self.gpu = gpu if gpu is not None else self.spec.main_gpu
        self.overrides = overrides or {}
        self.proc: subprocess.Popen | None = None
        self.log_file: Path | None = None
        self._client: Any | None = None

    @property
    def base_url(self) -> str:
        return f"http://{self.host}:{self.port}/v1"

    def _resolve_param(self, key: str, default: Any = None) -> Any:
        if key in self.overrides:
            return self.overrides[key]
        return getattr(self.spec, key, default)

    def _build_command(self) -> list[str]:
        binary = self.registry.server.binary
        model_path = self.spec.path
        if not Path(model_path).exists():
            raise FileNotFoundError(f"model file not found: {model_path}")

        alias = self.spec.alias or Path(model_path).stem
        cmd = [
            binary,
            "--model", model_path,
            "--alias", alias,
            "--host", self.host,
            "--port", str(self.port),
            "--ctx-size", str(self._resolve_param("context_size")),
            "--n-gpu-layers", str(self._resolve_param("gpu_layers")),
            "--main-gpu", str(self.gpu),
            "--split-mode", str(self._resolve_param("split_mode")),
            "--batch-size", str(self._resolve_param("batch_size")),
            "--ubatch-size", str(self._resolve_param("ubatch_size")),
            "--cache-type-k", str(self._resolve_param("cache_type_k")),
            "--cache-type-v", str(self._resolve_param("cache_type_v")),
            "--parallel", "1",
            "--n-predict", "-1",
            "--jinja",
        ]

        if self._resolve_param("flash_attn"):
            cmd.extend(["--flash-attn", "on"])

        if self._resolve_param("reasoning"):
            cmd.extend(["--reasoning", "on"])
            fmt = self._resolve_param("reasoning_format")
            if fmt:
                cmd.extend(["--reasoning-format", fmt])
            budget = self._resolve_param("reasoning_budget")
            if budget is not None:
                cmd.extend(["--reasoning-budget", str(budget)])

        if self._resolve_param("mtp"):
            cmd.extend(["--spec-type", "draft-mtp", "--spec-draft-n-max", "3"])
            mtp_path = self._resolve_param("mtp_path")
            if mtp_path and Path(mtp_path).exists():
                # llama.cpp accepts --model-draft for an external draft model.
                cmd.extend(["--model-draft", mtp_path])

        cmd.extend(self.spec.extra_args)
        return cmd

    def start(self) -> "ModelServer":
        # Stop mutually exclusive systemd services.
        _stop_services(self.registry.server.stop_services)

        # Allocate a port if not provided.
        if self.port is None:
            self.port = _find_free_port(
                self.host,
                self.registry.server.port_range_start,
                self.registry.server.port_range_end,
            )

        # If something is already on this port, kill it.
        if _port_in_use(self.host, self.port):
            _kill_port_occupant(self.host, self.port)

        cmd = self._build_command()
        self.log_file = Path(f"/tmp/mlfactory_modelserver_{self.port}.log")
        self.log_file.parent.mkdir(parents=True, exist_ok=True)
        self.proc = subprocess.Popen(
            cmd,
            stdout=open(self.log_file, "w"),
            stderr=subprocess.STDOUT,
            start_new_session=True,
        )

        try:
            _wait_for_server(f"http://{self.host}:{self.port}")
        except Exception:
            self.stop()
            raise
        return self

    def stop(self) -> None:
        if self.proc is None:
            return
        try:
            os.killpg(os.getpgid(self.proc.pid), signal.SIGTERM)
        except ProcessLookupError:
            pass
        try:
            self.proc.wait(timeout=30)
        except subprocess.TimeoutExpired:
            try:
                os.killpg(os.getpgid(self.proc.pid), signal.SIGKILL)
            except ProcessLookupError:
                pass
        self.proc = None

    def client(self) -> Any:
        """Return an OpenAI client pointing at this server."""
        if self._client is None:
            try:
                from openai import OpenAI
            except ImportError as exc:
                raise ImportError("openai package is required for ModelServer.client()") from exc
            self._client = OpenAI(base_url=self.base_url, api_key="none")
        return self._client

    def __enter__(self) -> "ModelServer":
        return self.start()

    def __exit__(self, exc_type, exc, tb) -> None:
        self.stop()


@contextmanager
def model(alias: str, **kwargs: Any) -> Iterator[ModelServer]:
    """Context manager: start a model server, yield it, then stop it."""
    server = ModelServer(alias, **kwargs)
    try:
        yield server.start()
    finally:
        server.stop()
