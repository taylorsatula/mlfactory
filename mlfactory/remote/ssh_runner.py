"""Generic SSH remote runner for mlfactory.

Supports any SSH-accessible host; Vast.ai is just the first target.
The runner is intentionally thin: it copies the repo, runs the driver, and
rsyncs outputs back. Heavy lifting stays in the local driver.
"""
from __future__ import annotations

import os
import subprocess
import sys
import tempfile
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

    def ssh_base(self) -> list[str]:
        cmd = ["ssh"]
        if self.port != 22:
            cmd.extend(["-p", str(self.port)])
        if self.key:
            cmd.extend(["-i", self.key])
        cmd.extend(["-o", "StrictHostKeyChecking=accept-new"])
        return cmd

    def target(self) -> str:
        return f"{self.user}@{self.host}"

    def run_remote(self, command: str, check: bool = True, capture: bool = False) -> subprocess.CompletedProcess:
        cmd = self.ssh_base() + [self.target(), command]
        return subprocess.run(cmd, check=check, capture_output=capture, text=True)

    def rsync_to_remote(self, local_path: Path, remote_path: str) -> None:
        cmd = ["rsync", "-az", "--info=progress2", "--exclude", ".venv", "--exclude", ".venv312",
               "--exclude", "__pycache__", "--exclude", "runs", "--exclude", "data"]
        if self.key:
            cmd.extend(["-e", f"ssh -i {self.key} -p {self.port} -o StrictHostKeyChecking=accept-new"])
        else:
            cmd.extend(["-e", f"ssh -p {self.port} -o StrictHostKeyChecking=accept-new"])
        cmd.extend([str(local_path) + "/", f"{self.target()}:{remote_path}/"])
        subprocess.run(cmd, check=True)

    def rsync_from_remote(self, remote_path: str, local_path: Path) -> None:
        cmd = ["rsync", "-az", "--partial", "--info=progress2"]
        if self.key:
            cmd.extend(["-e", f"ssh -i {self.key} -p {self.port} -o StrictHostKeyChecking=accept-new"])
        else:
            cmd.extend(["-e", f"ssh -p {self.port} -o StrictHostKeyChecking=accept-new"])
        cmd.extend([f"{self.target()}:{remote_path}/", str(local_path) + "/"])
        subprocess.run(cmd, check=True)


class RemoteRunner:
    def __init__(self, config: SSHConfig):
        self.config = config

    def setup(self) -> None:
        """Ensure remote workdir exists and dependencies are installed."""
        self.config.run_remote(f"mkdir -p {self.config.remote_workdir}")
        self.config.run_remote(f"cd {self.config.remote_workdir} && {self.config.python} -m pip install -e .")

    def run(self, spec_name: str, run_id: str | None = None, parent_runs: list[str] | None = None) -> str:
        """Sync code, execute a run remotely, and sync outputs back."""
        self.config.rsync_to_remote(Path.cwd(), self.config.remote_workdir)

        cmd_parts = [
            f"cd {self.config.remote_workdir}",
            f"{self.config.python} -m mlfactory.cli -r data/registry.db init specs/{spec_name}.yaml",
        ]
        if run_id:
            cmd_parts[-1] += f" --run-id {run_id}"
        init_cmd = " && ".join(cmd_parts)
        result = self.config.run_remote(init_cmd, capture=True)
        # Extract run id from stdout.
        remote_run_id = run_id or self._extract_run_id(result.stdout)
        if not remote_run_id:
            raise RuntimeError(f"could not determine run id from remote output:\n{result.stdout}\n{result.stderr}")

        exec_cmd = (
            f"cd {self.config.remote_workdir} && "
            f"{self.config.python} -m mlfactory.cli -r data/registry.db run specs/{spec_name}.yaml "
            f"--run-id {remote_run_id}"
        )
        try:
            self.config.run_remote(exec_cmd)
        finally:
            local_runs = Path("runs")
            local_runs.mkdir(parents=True, exist_ok=True)
            self.config.rsync_from_remote(
                f"{self.config.remote_workdir}/runs/{remote_run_id}",
                local_runs / remote_run_id,
            )
            # Pull registry updates too.
            self.config.rsync_from_remote(
                f"{self.config.remote_workdir}/data/registry.db",
                Path("data") / "registry-remote.db",
            )
        return remote_run_id

    @staticmethod
    def _extract_run_id(stdout: str) -> str | None:
        for line in stdout.splitlines():
            if line.startswith("Created run "):
                return line.split()[2].strip()
        return None


def vast_config(host: str, port: int = 22, key: str | None = None) -> SSHConfig:
    """Convenience factory for a Vast.ai H100 instance."""
    return SSHConfig(
        host=host,
        user="root",
        port=port,
        key=key,
        remote_workdir="/workspace/mlfactory",
        python="python3",
    )
