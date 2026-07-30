"""Run manifest schema and helpers.

A manifest is the single source of truth for one experiment invocation.
It is written at the start of a run and finalized at the end.
"""
from __future__ import annotations

import hashlib
import json
import os
import platform
import subprocess
import sys
import tarfile
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from pydantic import BaseModel, Field, field_validator


STAGES = {"collect", "classify", "stratify", "generate", "train", "eval", "build-pilot", "transform", "analyze"}
RUN_STATUSES = {"pending", "running", "completed", "failed", "guarded", "aborted"}


class GitInfo(BaseModel):
    commit: str | None = None
    dirty: bool = True
    branch: str | None = None
    tag: str | None = None
    remote_url: str | None = None


class SourceArchive(BaseModel):
    path: str
    sha256: str
    archive_type: str = "tar.gz"


class FileRecord(BaseModel):
    path: str
    sha256: str
    role: str = "input"  # input, artifact, log, etc.
    size_bytes: int | None = None


class EnvironmentInfo(BaseModel):
    python_version: str = sys.version
    platform: str = platform.platform()
    freeze_path: str | None = None
    runtime_path: str | None = None
    env_vars: dict[str, str] = Field(default_factory=dict)


class HardwareInfo(BaseModel):
    gpus: list[dict[str, Any]] = Field(default_factory=list)
    cpu_count: int = Field(default_factory=os.cpu_count)
    total_ram_gib: float | None = None


class RunManifest(BaseModel):
    """Top-level manifest for a single run."""

    run_id: str
    stage: str
    status: str = "pending"
    created_at: str = Field(default_factory=lambda: datetime.now(timezone.utc).isoformat())
    started_at: str | None = None
    completed_at: str | None = None

    spec: dict[str, Any] = Field(default_factory=dict)

    git: GitInfo = Field(default_factory=GitInfo)
    source: SourceArchive | None = None
    inputs: list[FileRecord] = Field(default_factory=list)
    artifacts: list[FileRecord] = Field(default_factory=list)
    logs: list[FileRecord] = Field(default_factory=list)

    env: EnvironmentInfo = Field(default_factory=EnvironmentInfo)
    hardware: HardwareInfo = Field(default_factory=HardwareInfo)

    parent_runs: list[str] = Field(default_factory=list)
    child_runs: list[str] = Field(default_factory=list)

    summary: dict[str, Any] = Field(default_factory=dict)
    guard_report: dict[str, Any] | None = None

    @field_validator("stage")
    @classmethod
    def _valid_stage(cls, v: str) -> str:
        if v not in STAGES:
            raise ValueError(f"stage must be one of {sorted(STAGES)}, got {v!r}")
        return v

    @field_validator("status")
    @classmethod
    def _valid_status(cls, v: str) -> str:
        if v not in RUN_STATUSES:
            raise ValueError(f"status must be one of {sorted(RUN_STATUSES)}, got {v!r}")
        return v

    # ------------------------------------------------------------------
    # persistence
    # ------------------------------------------------------------------
    def write(self, path: Path) -> None:
        tmp = path.with_suffix(".tmp")
        tmp.write_text(self.model_dump_json(indent=2), encoding="utf-8")
        tmp.replace(path)

    @classmethod
    def read(cls, path: Path) -> "RunManifest":
        return cls.model_validate_json(path.read_text(encoding="utf-8"))

    def to_db_row(self) -> tuple[str, str, str, str, str | None, str | None, str]:
        """Return row tuple for the registry: (run_id, stage, status, created_at, started_at, completed_at, json)."""
        return (
            self.run_id,
            self.stage,
            self.status,
            self.created_at,
            self.started_at,
            self.completed_at,
            self.model_dump_json(),
        )


# ---------------------------------------------------------------------------
# helpers
# ---------------------------------------------------------------------------

def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(65536), b""):
            h.update(chunk)
    return h.hexdigest()


def sha256_bytes(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def git_info(repo_dir: Path | None = None) -> GitInfo:
    repo_dir = repo_dir or Path.cwd()

    def run(cmd: list[str], **kw) -> str | None:
        try:
            return subprocess.check_output(
                cmd, cwd=repo_dir, text=True, stderr=subprocess.DEVNULL, **kw
            ).strip()
        except Exception:
            return None

    commit = run(["git", "rev-parse", "HEAD"])
    branch = run(["git", "rev-parse", "--abbrev-ref", "HEAD"])
    tag = run(["git", "describe", "--tags", "--always"])
    dirty = run(["git", "status", "--porcelain"]) not in (None, "")
    remote_url = run(["git", "remote", "get-url", "origin"])
    return GitInfo(
        commit=commit,
        dirty=dirty,
        branch=branch,
        tag=tag,
        remote_url=remote_url,
    )


def archive_source(repo_dir: Path, dest: Path, prefix: str = "source/") -> SourceArchive:
    """Create a reproducible tar.gz of the git tree at the current commit.

    Uses ``git archive`` so ignored files (builds, .venv, outputs) are excluded
    and the archive is byte-stable for a given commit.
    """
    repo_dir = repo_dir or Path.cwd()
    dest.parent.mkdir(parents=True, exist_ok=True)
    try:
        archive_bytes = subprocess.check_output(
            ["git", "archive", "--format=tar.gz", "HEAD"],
            cwd=repo_dir,
        )
    except subprocess.CalledProcessError as exc:
        raise RuntimeError("git archive failed; is this a git repo with a commit?") from exc

    dest.write_bytes(archive_bytes)
    return SourceArchive(
        path=str(dest),
        sha256=sha256_bytes(archive_bytes),
        archive_type="tar.gz",
    )


def freeze_environment(dest_dir: Path) -> EnvironmentInfo:
    """Capture pip freeze and runtime diagnostics into dest_dir."""
    dest_dir.mkdir(parents=True, exist_ok=True)
    freeze_path = dest_dir / "requirements.freeze.txt"
    runtime_path = dest_dir / "environment.runtime.txt"

    try:
        freeze = subprocess.check_output(
            [sys.executable, "-m", "pip", "freeze"], text=True, stderr=subprocess.DEVNULL
        )
    except Exception as e:
        freeze = f"# pip freeze failed: {e}\n"
    freeze_path.write_text(freeze, encoding="utf-8")

    runtime_lines = [
        f"python {platform.python_version()}",
        f"platform {platform.platform()}",
        f"machine {platform.machine()}",
        f"processor {platform.processor()}",
    ]
    runtime_path.write_text("\n".join(runtime_lines) + "\n", encoding="utf-8")

    return EnvironmentInfo(
        python_version=platform.python_version(),
        platform=platform.platform(),
        freeze_path=str(freeze_path),
        runtime_path=str(runtime_path),
        env_vars={k: os.environ.get(k, "") for k in ["CUDA_VISIBLE_DEVICES", "PYTORCH_CUDA_ALLOC_CONF"]},
    )


def hardware_info() -> HardwareInfo:
    gpus: list[dict] = []
    try:
        out = subprocess.check_output(
            ["nvidia-smi", "--query-gpu=index,name,driver_version,memory.total", "--format=csv,noheader"],
            text=True,
            stderr=subprocess.DEVNULL,
        )
        for line in out.strip().splitlines():
            parts = [p.strip() for p in line.split(",")]
            if len(parts) >= 4:
                gpus.append(
                    {"index": parts[0], "name": parts[1], "driver": parts[2], "memory": parts[3]}
                )
    except Exception:
        pass

    total_ram_gib: float | None = None
    try:
        with open("/proc/meminfo", "r") as f:
            for line in f:
                if line.startswith("MemTotal:"):
                    kb = int(line.split()[1])
                    total_ram_gib = kb / (1024 * 1024)
                    break
    except Exception:
        pass

    return HardwareInfo(gpus=gpus, total_ram_gib=total_ram_gib)


def new_run_id(stage: str, suffix: str = "") -> str:
    ts = datetime.now(timezone.utc).strftime("%Y%m%d-%H%M%S")
    base = f"{ts}.{stage}"
    if suffix:
        base = f"{base}.{suffix}"
    return base
