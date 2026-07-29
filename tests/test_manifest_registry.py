"""Smoke tests for manifest round-trip and registry operations."""
from __future__ import annotations

import json
import tempfile
from pathlib import Path

from mlfactory.core.manifest import (
    EnvironmentInfo,
    FileRecord,
    GitInfo,
    HardwareInfo,
    RunManifest,
    SourceArchive,
)
from mlfactory.core.registry import Registry


def test_manifest_round_trip() -> None:
    m = RunManifest(
        run_id="test.run.001",
        stage="collect",
        spec={"model_name": "test-model"},
        git=GitInfo(commit="abc123", dirty=False, branch="main"),
        source=SourceArchive(path="/tmp/src.tar.gz", sha256="0" * 64),
        inputs=[FileRecord(path="/tmp/in.jsonl", sha256="1" * 64, role="input:data")],
        env=EnvironmentInfo(python_version="3.12", platform="linux"),
        hardware=HardwareInfo(gpus=[]),
    )
    with tempfile.TemporaryDirectory() as td:
        p = Path(td) / "manifest.json"
        m.write(p)
        m2 = RunManifest.read(p)
    assert m2.run_id == "test.run.001"
    assert m2.git.commit == "abc123"


def test_registry_crud() -> None:
    with tempfile.TemporaryDirectory() as td:
        db = Path(td) / "registry.db"
        registry = Registry(db)
        m = RunManifest(
            run_id="test.run.002",
            stage="train",
            status="completed",
            spec={"lr": 1e-5},
        )
        registry.register(m)
        found = registry.get("test.run.002")
        assert found is not None
        assert found.stage == "train"
        assert len(registry.find(stage="train")) == 1
        assert len(registry.find(stage="collect")) == 0


def test_lineage() -> None:
    with tempfile.TemporaryDirectory() as td:
        db = Path(td) / "registry.db"
        registry = Registry(db)
        parent = RunManifest(run_id="parent", stage="collect", status="completed")
        child = RunManifest(run_id="child", stage="train", status="completed", parent_runs=["parent"])
        registry.ingest_manifest(parent)
        registry.ingest_manifest(child)
        assert registry.parents("child") == [("parent", "input")]
        assert registry.children("parent") == [("child", "input")]
