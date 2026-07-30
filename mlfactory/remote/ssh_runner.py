"""Generic SSH remote runner for mlfactory.

This module is backend-agnostic SSH plumbing. Vast.ai-specific behavior lives in
``vast_runner.py``.
"""
from __future__ import annotations

import subprocess
from dataclasses import dataclass
from pathlib import Path
from typing import Any


@dataclass
class SSHConfig:
    host: str
    user: str = "root"
    port: int = 22
    key: str | None = None
    remote_workdir: str = "/workspace/mlfactory"
    python: str = "python3"

    def target(self) -> str:
        return f"{self.user}@{self.host}"

    def ssh_base(self) -> list[str]:
        cmd = ["ssh"]
        if self.port != 22:
            cmd.extend(["-p", str(self.port)])
        if self.key:
            cmd.extend(["-i", self.key])
        cmd.extend(["-o", "StrictHostKeyChecking=accept-new", "-o", "BatchMode=yes"])
        return cmd

    def rsync_ssh_arg(self) -> str:
        parts = ["ssh", "-o", "StrictHostKeyChecking=accept-new", "-o", "BatchMode=yes"]
        if self.port != 22:
            parts.extend(["-p", str(self.port)])
        if self.key:
            parts.extend(["-i", self.key])
        return " ".join(parts)


class SSHRunner:
    """Low-level SSH helper: run commands and rsync to/from a remote host."""

    def __init__(self, config: SSHConfig):
        self.config = config

    def run_remote(
        self,
        command: str,
        check: bool = True,
        capture: bool = False,
        timeout: float | None = None,
    ) -> subprocess.CompletedProcess:
        cmd = self.config.ssh_base() + [self.config.target(), command]
        return subprocess.run(
            cmd, check=check, capture_output=capture, text=True, timeout=timeout
        )

    def run_remote_stream(self, command: str) -> None:
        """Run a remote command streaming stdout/stderr to the local terminal."""
        cmd = self.config.ssh_base() + [self.config.target(), command]
        subprocess.run(cmd, check=True)

    def rsync_to_remote(
        self,
        local_path: Path,
        remote_path: str,
        excludes: list[str] | None = None,
    ) -> None:
        cmd = ["rsync", "-az", "--info=progress2", "--partial"]
        for ex in excludes or []:
            cmd.extend(["--exclude", ex])
        cmd.extend(["-e", self.config.rsync_ssh_arg()])
        cmd.extend([str(local_path.rstrip("/")) + "/", f"{self.config.target()}:{remote_path}/"])
        subprocess.run(cmd, check=True)

    def rsync_from_remote(
        self,
        remote_path: str,
        local_path: Path,
    ) -> None:
        cmd = ["rsync", "-az", "--info=progress2", "--partial"]
        cmd.extend(["-e", self.config.rsync_ssh_arg()])
        local_path.mkdir(parents=True, exist_ok=True)
        cmd.extend([f"{self.config.target()}:{remote_path}/", str(local_path.rstrip("/")) + "/"])
        subprocess.run(cmd, check=True)

    def ensure_dir(self, remote_path: str) -> None:
        self.run_remote(f"mkdir -p {remote_path}")

    def remote_path_exists(self, remote_path: str) -> bool:
        result = self.run_remote(f"test -e {remote_path}", check=False, capture=True)
        return result.returncode == 0
