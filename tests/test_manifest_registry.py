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


def test_merge_registry() -> None:
    with tempfile.TemporaryDirectory() as td:
        local_db = Path(td) / "local.db"
        remote_db = Path(td) / "remote.db"

        local = Registry(local_db)
        remote = Registry(remote_db)

        local_parent = RunManifest(run_id="local_parent", stage="collect", status="completed")
        local_child = RunManifest(run_id="local_child", stage="train", status="completed", parent_runs=["local_parent"])
        local.ingest_manifest(local_parent)
        local.ingest_manifest(local_child)
        local.insert_metric("local_child", "loss", value=0.5, step=1)

        remote_parent = RunManifest(run_id="remote_parent", stage="collect", status="completed")
        remote_child = RunManifest(run_id="remote_child", stage="train", status="completed", parent_runs=["remote_parent"])
        remote.ingest_manifest(remote_parent)
        remote.ingest_manifest(remote_child)
        remote.insert_metric("remote_child", "loss", value=0.4, step=1)

        counts = local.merge_from(remote_db)
        assert counts == {"runs_added": 2, "runs_replaced": 0, "runs_skipped": 0, "lineage": 1, "metrics": 1}

        assert local.get("remote_child") is not None
        assert local.parents("remote_child") == [("remote_parent", "input")]
        assert len(local.metrics_for_run("remote_child")) == 1
        # Local runs should be untouched.
        assert local.get("local_child") is not None
        assert len(local.metrics_for_run("local_child")) == 1


def test_merge_registry_replace() -> None:
    with tempfile.TemporaryDirectory() as td:
        local_db = Path(td) / "local.db"
        remote_db = Path(td) / "remote.db"

        local = Registry(local_db)
        remote = Registry(remote_db)

        shared = RunManifest(run_id="shared", stage="collect", status="completed", spec={"version": "local"})
        local.register(shared)
        local.insert_metric("shared", "loss", value=0.9, step=1)

        remote_shared = RunManifest(run_id="shared", stage="train", status="completed", spec={"version": "remote"})
        remote.register(remote_shared)
        remote.insert_metric("shared", "loss", value=0.1, step=1)
        remote.insert_metric("shared", "acc", value=0.99, step=1)

        counts = local.merge_from(remote_db, on_conflict="replace")
        assert counts["runs_added"] == 0
        assert counts["runs_replaced"] == 1
        assert counts["runs_skipped"] == 0
        assert counts["metrics"] == 2

        merged = local.get("shared")
        assert merged.stage == "train"
        assert merged.spec["version"] == "remote"
        metrics = local.metrics_for_run("shared")
        assert len(metrics) == 2
        assert {m["key"] for m in metrics} == {"loss", "acc"}


def test_merge_registry_skip_conflict() -> None:
    with tempfile.TemporaryDirectory() as td:
        local_db = Path(td) / "local.db"
        remote_db = Path(td) / "remote.db"

        local = Registry(local_db)
        remote = Registry(remote_db)

        shared = RunManifest(run_id="shared", stage="collect", status="completed", spec={"version": "local"})
        local.register(shared)

        remote_shared = RunManifest(run_id="shared", stage="train", status="completed", spec={"version": "remote"})
        remote.register(remote_shared)

        counts = local.merge_from(remote_db, on_conflict="skip")
        assert counts == {"runs_added": 0, "runs_replaced": 0, "runs_skipped": 1, "lineage": 0, "metrics": 0}

        merged = local.get("shared")
        assert merged.stage == "collect"
