"""Vast.ai remote runner for mlfactory.

Manages Vast instances and runs experiments on them. Requires the ``vastai``
CLI to be installed and a Vast API key available via env var or config file.

Typical flow:

    runner = VastRunner.from_search(
        query="gpu_name == H100",
        api_key=os.environ.get("VAST_API_KEY"),
    )
    runner.provision()
    runner.setup(experiment="dft")
    runner.run_spec("mlfactory/experiments/dft/specs/dft_train_h100_validation.yaml")
    runner.pull_outputs()
    runner.stop()  # or destroy()
"""
from __future__ import annotations

import json
import os
import subprocess
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from mlfactory.remote.ssh_runner import SSHConfig, SSHRunner


# Image tags drift on Vast hosts — verify the tag before renting
# (docs/VAST_REMOTE.md §image-choice). The previous default
# (nvidia/cuda:12.9.0-devel-ubuntu26.04) 404'd on 2026-08-26; this one is
# verified working that date. Note it starts a preinstalled llama
# supervisor service: stop it before GPU work. Torch training stacks
# install their own venv regardless of this image.
DEFAULT_IMAGE = "vastai/llama-cpp:b10182-cuda-12.9"
DEFAULT_REMOTE_WORKDIR = "/workspace/mlfactory"


@dataclass
class VastConfig:
    api_key: str | None = None
    image: str = DEFAULT_IMAGE
    disk_gb: float = 300.0
    remote_workdir: str = DEFAULT_REMOTE_WORKDIR
    python: str = "python3"


class VastAPIError(RuntimeError):
    pass


def _vast_cli() -> str:
    return "vastai"


def _run_vast(args: list[str], api_key: str | None = None, capture: bool = True) -> subprocess.CompletedProcess:
    cmd = [_vast_cli(), "--raw"]
    if api_key:
        cmd.extend(["--api-key", api_key])
    cmd.extend(args)
    return subprocess.run(cmd, check=True, capture_output=capture, text=True)


def _vast_json(args: list[str], api_key: str | None = None) -> Any:
    result = _run_vast(args, api_key=api_key)
    if not result.stdout.strip():
        return []
    try:
        return json.loads(result.stdout)
    except json.JSONDecodeError as exc:
        raise VastAPIError(f"could not parse vastai output: {result.stdout[:200]}") from exc


def load_api_key() -> str | None:
    """Return API key from env var or Vast config file."""
    key = os.environ.get("VAST_API_KEY")
    if key:
        return key
    config_path = Path.home() / ".config" / "vastai" / "vast_api_key"
    if config_path.exists():
        return config_path.read_text(encoding="utf-8").strip()
    return None


def search_offers(query: str, api_key: str | None = None) -> list[dict]:
    """Search Vast offers matching a query string."""
    return _vast_json(["search", "offers", query], api_key=api_key)


def find_h100_offer(
    api_key: str | None = None,
    min_vram_gb: int = 80,
    num_gpus: int = 2,
    disk_gb: float = 300.0,
) -> dict | None:
    """Find a suitable H100 offer. Returns the first match or None."""
    query = (
        f"gpu_name == H100 num_gpus >= {num_gpus} "
        f"disk_space >= {disk_gb} direct_port_count >= 1"
    )
    offers = search_offers(query, api_key=api_key)
    for offer in offers:
        gpus = offer.get("num_gpus", 0)
        vram = offer.get("gpu_ram", 0)
        if gpus >= num_gpus and vram >= min_vram_gb:
            return offer
    return None


def create_instance(
    offer_id: int,
    image: str = DEFAULT_IMAGE,
    disk_gb: float = 300.0,
    api_key: str | None = None,
    label: str = "mlfactory",
) -> dict:
    """Create a Vast instance from an offer."""
    return _vast_json(
        [
            "create", "instance",
            str(offer_id),
            "--image", image,
            "--disk", str(disk_gb),
            "--label", label,
        ],
        api_key=api_key,
    )


def list_instances(api_key: str | None = None) -> list[dict]:
    return _vast_json(["show", "instances"], api_key=api_key)


def wait_for_instance(
    instance_id: int,
    api_key: str | None = None,
    timeout: float = 600.0,
    poll_interval: float = 10.0,
) -> dict:
    """Poll Vast until the instance is running and has SSH info."""
    deadline = time.time() + timeout
    while time.time() < deadline:
        instances = list_instances(api_key=api_key)
        for inst in instances:
            if inst.get("id") == instance_id:
                status = inst.get("actual_status", "")
                if status == "running" and inst.get("ssh_port"):
                    return inst
                print(f"instance {instance_id} status: {status}")
                break
        time.sleep(poll_interval)
    raise TimeoutError(f"instance {instance_id} did not become ready within {timeout}s")


def ssh_url(instance_id: int, api_key: str | None = None) -> str:
    """Return ssh url like root@host:port."""
    result = _run_vast(["ssh-url", str(instance_id)], api_key=api_key)
    return result.stdout.strip()


def parse_ssh_url(url: str) -> tuple[str, int]:
    """Parse 'root@host:port' into (host, port)."""
    user_host, port_str = url.rsplit(":", 1)
    _, host = user_host.split("@", 1)
    return host, int(port_str)


class VastRunner(SSHRunner):
    """High-level runner for a Vast.ai instance."""

    def __init__(
        self,
        config: VastConfig,
        instance_id: int | None = None,
        ssh_config: SSHConfig | None = None,
    ):
        self.vast_config = config
        self.instance_id = instance_id
        self.instance_info: dict | None = None
        if ssh_config is None and instance_id is not None:
            ssh_config = self._ssh_config_for_instance(instance_id)
        super().__init__(ssh_config or SSHConfig(host="placeholder", remote_workdir=config.remote_workdir))

    def _ssh_config_for_instance(self, instance_id: int) -> SSHConfig:
        url = ssh_url(instance_id, api_key=self.vast_config.api_key)
        host, port = parse_ssh_url(url)
        return SSHConfig(
            host=host,
            port=port,
            remote_workdir=self.vast_config.remote_workdir,
            python=self.vast_config.python,
        )

    @classmethod
    def from_search(
        cls,
        query: str | None = None,
        api_key: str | None = None,
        **kwargs: Any,
    ) -> "VastRunner":
        """Create a runner configured to provision from a search query."""
        key = api_key or load_api_key()
        if key is None:
            raise VastAPIError("Vast API key not found. Set VAST_API_KEY env var or configure vastai CLI.")
        config = VastConfig(api_key=key, **kwargs)
        return cls(config=config)

    @classmethod
    def from_instance_id(cls, instance_id: int, api_key: str | None = None, **kwargs: Any) -> "VastRunner":
        key = api_key or load_api_key()
        config = VastConfig(api_key=key, **kwargs)
        return cls(config=config, instance_id=instance_id)

    def provision(self, query: str | None = None) -> dict:
        """Search for an offer and create an instance."""
        if self.instance_id is not None:
            raise VastAPIError("instance already provisioned")
        if query is None:
            offer = find_h100_offer(api_key=self.vast_config.api_key)
        else:
            offers = search_offers(query, api_key=self.vast_config.api_key)
            offer = offers[0] if offers else None
        if offer is None:
            raise VastAPIError("no matching Vast offer found")
        offer_id = offer["id"]
        print(f"creating instance from offer {offer_id}...")
        result = create_instance(
            offer_id=offer_id,
            image=self.vast_config.image,
            disk_gb=self.vast_config.disk_gb,
            api_key=self.vast_config.api_key,
        )
        self.instance_id = result.get("new_contract")
        if not self.instance_id:
            raise VastAPIError(f"could not determine instance id from: {result}")
        print(f"instance {self.instance_id} created; waiting for readiness...")
        self.instance_info = wait_for_instance(self.instance_id, api_key=self.vast_config.api_key)
        # Refresh SSH config now that host/port are known.
        self.config = self._ssh_config_for_instance(self.instance_id)
        return self.instance_info

    def destroy(self) -> None:
        if self.instance_id is None:
            raise VastAPIError("no instance to destroy")
        _run_vast(["destroy", "instance", str(self.instance_id)], api_key=self.vast_config.api_key)
        self.instance_id = None

    def stop(self) -> None:
        if self.instance_id is None:
            raise VastAPIError("no instance to stop")
        _run_vast(["stop", "instance", str(self.instance_id)], api_key=self.vast_config.api_key)

    def start(self) -> dict:
        if self.instance_id is None:
            raise VastAPIError("no instance to start")
        _run_vast(["start", "instance", str(self.instance_id)], api_key=self.vast_config.api_key)
        self.instance_info = wait_for_instance(self.instance_id, api_key=self.vast_config.api_key)
        self.config = self._ssh_config_for_instance(self.instance_id)
        return self.instance_info

    def sync_code(self, local_path: Path | None = None, excludes: list[str] | None = None) -> None:
        """Sync the local repo to the remote workdir."""
        local_path = local_path or Path.cwd()
        excludes = excludes or [
            ".venv", ".venv312", "venv", "__pycache__", ".git",
            "runs", ".mlfactory", "*.egg-info", "*.safetensors", "*.bin",
        ]
        self.ensure_dir(self.config.remote_workdir)
        self.rsync_to_remote(local_path, self.config.remote_workdir, excludes=excludes)

    def setup(self, experiment: str | None = None, setup_script: str | None = None) -> None:
        """Install mlfactory and optionally run an experiment setup script."""
        self.run_remote(f"cd {self.config.remote_workdir} && {self.config.python} -m pip install -e .")
        if setup_script:
            self.run_remote_stream(f"cd {self.config.remote_workdir} && bash {setup_script}")
        elif experiment == "dft":
            script = "mlfactory/experiments/dft/setup_h100.sh"
            if self.remote_path_exists(f"{self.config.remote_workdir}/{script}"):
                self.run_remote_stream(f"cd {self.config.remote_workdir} && bash {script}")

    def run_spec(self, spec_path: str, run_id: str | None = None) -> str:
        """Run a spec remotely. Returns the remote run id."""
        self.sync_code()
        remote_spec = f"{self.config.remote_workdir}/{spec_path}"
        cmd = f"cd {self.config.remote_workdir} && {self.config.python} -m mlfactory.cli init {remote_spec}"
        if run_id:
            cmd += f" --run-id {run_id}"
        result = self.run_remote(cmd, capture=True)
        # Extract run id from stdout.
        remote_run_id = run_id
        if not remote_run_id:
            for line in result.stdout.splitlines():
                if line.startswith("Created run "):
                    remote_run_id = line.split()[2].strip()
                    break
        if not remote_run_id:
            raise RuntimeError(f"could not determine run id from remote output:\n{result.stdout}\n{result.stderr}")

        exec_cmd = (
            f"cd {self.config.remote_workdir} && "
            f"{self.config.python} -m mlfactory.cli run {remote_spec} --run-id {remote_run_id}"
        )
        try:
            self.run_remote_stream(exec_cmd)
        finally:
            self.pull_outputs(remote_run_id)
        return remote_run_id

    def pull_outputs(self, run_id: str, local_runs_dir: str | Path = "runs") -> None:
        """Pull a remote run directory back to local."""
        local_path = Path(local_runs_dir)
        local_path.mkdir(parents=True, exist_ok=True)
        remote_run_path = f"{self.config.remote_workdir}/runs/{run_id}"
        self.rsync_from_remote(remote_run_path, local_path / run_id)

    def pull_registry(self, local_registry: str | Path = ".mlfactory/registry-remote.db") -> None:
        """Pull the remote registry.db for merging into the local registry."""
        remote_registry = f"{self.config.remote_workdir}/.mlfactory/registry.db"
        self.rsync_from_remote(remote_registry, Path(local_registry))
