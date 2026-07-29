"""Main run driver: spec -> manifest -> plugin execution -> registry."""
from __future__ import annotations

import os
import shutil
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import yaml

from mlfactory.core.manifest import (
    RunManifest,
    archive_source,
    freeze_environment,
    git_info,
    hardware_info,
    new_run_id,
    sha256_file,
)
from mlfactory.core.registry import Registry
from mlfactory.plugins.base import PLUGINS

# Register built-in experiment plugins.
import mlfactory.experiments.ace.classify_plugin  # noqa: E402,F401
import mlfactory.experiments.ace.collect_plugin  # noqa: E402,F401
import mlfactory.experiments.dft.eval_plugin  # noqa: E402,F401
import mlfactory.experiments.dft.train_plugin  # noqa: E402,F401


def _load_spec(path: Path) -> dict[str, Any]:
    text = path.read_text(encoding="utf-8")
    if path.suffix in {".yaml", ".yml"}:
        return yaml.safe_load(text)
    if path.suffix == ".json":
        import json

        return json.loads(text)
    raise ValueError(f"unsupported spec format: {path.suffix}")


def _resolve_inputs(spec: dict[str, Any], run_dir: Path) -> list:
    from mlfactory.core.manifest import FileRecord

    inputs: list[FileRecord] = []
    for key, value in spec.items():
        if not isinstance(value, (str, os.PathLike)):
            continue
        p = Path(value)
        if p.exists() and p.is_file():
            inputs.append(
                FileRecord(
                    path=str(p.resolve()),
                    sha256=sha256_file(p),
                    role=f"input:{key}",
                    size_bytes=p.stat().st_size,
                )
            )
    return inputs


def _link_inputs_to_run_dir(inputs: list, run_dir: Path) -> None:
    inputs_dir = run_dir / "inputs"
    inputs_dir.mkdir(parents=True, exist_ok=True)
    for fr in inputs:
        src = Path(fr.path)
        link = inputs_dir / src.name
        if link.exists() or link.is_symlink():
            continue
        try:
            os.symlink(src.resolve(), link)
        except OSError:
            shutil.copy2(src, link)


def create_run(
    spec_path: Path,
    stage: str | None = None,
    run_id: str | None = None,
    runs_dir: Path = Path("runs"),
    repo_dir: Path | None = None,
    parent_runs: list[str] | None = None,
) -> RunManifest:
    """Create a run directory and manifest from a spec file.

    Does not execute the plugin. Use :func:`run_from_spec` for full execution.
    """
    spec = _load_spec(spec_path)
    stage = stage or spec.get("stage")
    if not stage:
        raise ValueError("spec must contain 'stage' or --stage must be provided")

    run_id = run_id or new_run_id(stage, spec.get("name", ""))
    run_dir = runs_dir / run_id
    if run_dir.exists():
        raise FileExistsError(f"run directory already exists: {run_dir}")
    run_dir.mkdir(parents=True)
    (run_dir / "artifacts").mkdir()
    (run_dir / "logs").mkdir()

    repo_dir = repo_dir or Path.cwd()
    git = git_info(repo_dir)
    source_archive = archive_source(repo_dir, run_dir / "source.tar.gz")
    env = freeze_environment(run_dir / "env")
    hw = hardware_info()
    inputs = _resolve_inputs(spec, run_dir)
    _link_inputs_to_run_dir(inputs, run_dir)

    manifest = RunManifest(
        run_id=run_id,
        stage=stage,
        status="pending",
        spec=spec,
        git=git,
        source=source_archive,
        inputs=inputs,
        env=env,
        hardware=hw,
        parent_runs=parent_runs or [],
    )
    manifest.write(run_dir / "manifest.json")
    return manifest


def run_from_spec(
    spec_path: Path,
    stage: str | None = None,
    run_id: str | None = None,
    runs_dir: Path = Path("runs"),
    registry: Registry | None = None,
    repo_dir: Path | None = None,
    parent_runs: list[str] | None = None,
) -> RunManifest:
    """Create, execute, and register a run."""
    manifest = create_run(
        spec_path=spec_path,
        stage=stage,
        run_id=run_id,
        runs_dir=runs_dir,
        repo_dir=repo_dir,
        parent_runs=parent_runs,
    )

    registry = registry or Registry()
    registry.register(manifest)
    for parent in manifest.parent_runs:
        registry.link_runs(parent, manifest.run_id, "input")

    plugin_cls = PLUGINS.get(manifest.stage)
    plugin = plugin_cls(manifest)

    try:
        plugin.run()
    finally:
        manifest.write(Path(manifest.source.path).parent / "manifest.json")
        registry.register(manifest)

    return manifest
