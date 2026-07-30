#!/usr/bin/env python3
"""Non-destructively ingest existing ace-baseline-trajectories and dft-eval-harness
runs into the mlfactory registry.

Run from the mlfactory repo root:

    python migrations/ingest_existing_runs.py

This script only reads from the legacy directories and writes to
``data/registry.db`` (and a migration log). It does not move or modify the
original runs.
"""
from __future__ import annotations

import hashlib
import json
import shutil
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

# Ensure mlfactory is importable without installation.
sys.path.insert(0, str(Path(__file__).parent.parent))

from mlfactory.core.manifest import (
    EnvironmentInfo,
    FileRecord,
    GitInfo,
    HardwareInfo,
    RunManifest,
    SourceArchive,
)
from mlfactory.core.registry import Registry


ACE_ROOT = Path("/home/admin/ace-baseline-trajectories")
DFT_ROOT = Path("/home/admin/dft-eval-harness")
REGISTRY_PATH = Path(".mlfactory/registry.db")
MIGRATION_LOG = Path("migrations/ingest_existing_runs.log")


def _now() -> str:
    return datetime.now(timezone.utc).isoformat()


def _sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(65536), b""):
            h.update(chunk)
    return h.hexdigest()


def _write_log(msg: str) -> None:
    MIGRATION_LOG.parent.mkdir(parents=True, exist_ok=True)
    with open(MIGRATION_LOG, "a", encoding="utf-8") as f:
        f.write(f"[{_now()}] {msg}\n")


def _input_records_from_dir(run_dir: Path) -> list[FileRecord]:
    """Best-effort input records from known input files."""
    inputs: list[FileRecord] = []
    candidates = ["prompts.jsonl", "train.jsonl", "test.jsonl", "generations.jsonl"]
    for name in candidates:
        p = run_dir / name
        if not p.exists():
            p = run_dir.parent / name
        if p.exists() and p.is_file():
            inputs.append(
                FileRecord(
                    path=str(p.resolve()),
                    sha256=_sha256_file(p),
                    role=f"input:{name}",
                    size_bytes=p.stat().st_size,
                )
            )
    return inputs


def _artifact_records_from_dir(run_dir: Path) -> list[FileRecord]:
    """Hash all files under run_dir/artifacts-like subdirs."""
    artifacts: list[FileRecord] = []
    for sub in ["final", "checkpoint-100", "checkpoint-80", "checkpoint-60", "checkpoint-40", "checkpoint-20"]:
        subdir = run_dir / sub
        if subdir.exists():
            for p in subdir.rglob("*"):
                if p.is_file():
                    artifacts.append(
                        FileRecord(
                            path=str(p.resolve()),
                            sha256=_sha256_file(p),
                            role=f"artifact:{sub}/{p.relative_to(subdir)}",
                            size_bytes=p.stat().st_size,
                        )
                    )
    return artifacts


def _summarize_dft_train(summary_path: Path | None, train_config_path: Path | None) -> dict[str, Any]:
    summary: dict[str, Any] = {}
    if train_config_path and train_config_path.exists():
        summary["config"] = json.loads(train_config_path.read_text(encoding="utf-8"))
    if summary_path and summary_path.exists():
        data = json.loads(summary_path.read_text(encoding="utf-8"))
        summary.update(data)
        # Flatten final eval metrics if present.
        if "eval_steps" in data and data["eval_steps"]:
            last_eval = data["eval_steps"][-1]
            summary["final_eval_step"] = last_eval.get("step")
            for k in ["mmd2", "l2_1gram", "repetition_rate", "non_english_char_rate", "self_bleu"]:
                if k in last_eval:
                    summary[f"final_eval_{k}"] = last_eval[k]
    return summary


def _summarize_dft_eval(summary_path: Path | None) -> dict[str, Any]:
    summary: dict[str, Any] = {}
    if summary_path and summary_path.exists():
        data = json.loads(summary_path.read_text(encoding="utf-8"))
        summary.update(data)
        if "results" in data and data["results"]:
            for k, v in data["results"][0].items():
                summary[f"result_0_{k}"] = v
    return summary


def _make_placeholder_git() -> GitInfo:
    return GitInfo(commit=None, dirty=True, branch=None, tag=None, remote_url=None)


def _make_placeholder_source(run_dir: Path) -> None:
    """Legacy runs keep their original directories; do not create derived archives."""
    return None


def _env_and_hardware_from_ace_manifest(data: dict[str, Any]) -> tuple[EnvironmentInfo, HardwareInfo]:
    env = EnvironmentInfo(
        python_version=data.get("python_version", ""),
        platform=data.get("platform", ""),
        freeze_path=None,
        runtime_path=None,
        env_vars={},
    )
    gpus: list[dict] = []
    for line in data.get("gpu_info", []):
        parts = [p.strip() for p in line.split(",")]
        if len(parts) >= 3:
            gpus.append({"index": parts[0], "name": parts[1], "driver": parts[2], "memory": parts[3] if len(parts) > 3 else ""})
    hw = HardwareInfo(gpus=gpus)
    return env, hw


def _status_for_summary(summary: dict[str, Any]) -> str:
    if summary.get("guard") or summary.get("guard_report"):
        return "guarded"
    return "completed"


FORCE = False


def ingest_ace_collect_runs(registry: Registry) -> int:
    count = 0
    outputs_dir = ACE_ROOT / "outputs"
    for manifest_path in outputs_dir.rglob("manifest.json"):
        # Skip if this is a nested manifest inside a run we already handled.
        run_dir = manifest_path.parent
        try:
            data = json.loads(manifest_path.read_text(encoding="utf-8"))
        except Exception as exc:
            _write_log(f"SKIP {manifest_path}: parse error {exc}")
            continue

        run_id = f"legacy.ace.collect.{run_dir.name}"
        if not FORCE and registry.get(run_id):
            _write_log(f"SKIP {run_id}: already registered")
            continue

        env, hw = _env_and_hardware_from_ace_manifest(data)
        inputs = []
        for key, sha in data.get("file_hashes", {}).items():
            p = run_dir / key
            if p.exists():
                inputs.append(FileRecord(path=str(p.resolve()), sha256=sha, role=f"input:{key}"))

        artifacts = [FileRecord(path=str(p.resolve()), sha256=_sha256_file(p), role="artifact:generations.jsonl")
                     for p in [run_dir / "generations.jsonl"] if p.exists()]

        manifest = RunManifest(
            run_id=run_id,
            stage="collect",
            status="completed",
            created_at=data.get("created_at", _now()),
            started_at=data.get("created_at"),
            completed_at=data.get("created_at"),
            spec={
                "experiment": "ace",
                "model_name": data.get("model_name"),
                "provider": data.get("provider"),
                "max_model_len": data.get("max_model_len"),
                "max_output_tokens": data.get("max_output_tokens"),
                "planned_samples": data.get("planned_samples"),
                "prompt_count": data.get("prompt_count"),
            },
            git=_make_placeholder_git(),
            source=_make_placeholder_source(run_dir),
            summary={"original_output_dir": str(run_dir.resolve())},
            inputs=inputs,
            artifacts=artifacts,
            env=env,
            hardware=hw,
        )
        registry.ingest_manifest(manifest)
        _write_log(f"INGESTED {run_id}")
        count += 1
    return count


def ingest_dft_train_runs(registry: Registry) -> int:
    count = 0
    for out_dir in DFT_ROOT.glob("out_train*"):
        if not out_dir.is_dir():
            continue
        run_id = f"legacy.dft.train.{out_dir.name}"
        if not FORCE and registry.get(run_id):
            _write_log(f"SKIP {run_id}: already registered")
            continue

        train_config_path = out_dir / "train_config.json"
        summary_path = out_dir / "summary.json"
        summary = _summarize_dft_train(summary_path if summary_path.exists() else None,
                                       train_config_path if train_config_path.exists() else None)

        inputs = _input_records_from_dir(out_dir)
        artifacts = _artifact_records_from_dir(out_dir)
        for log_path in [out_dir / "logs" / "train.jsonl"]:
            if log_path.exists():
                artifacts.append(FileRecord(
                    path=str(log_path.resolve()),
                    sha256=_sha256_file(log_path),
                    role="artifact:log",
                    size_bytes=log_path.stat().st_size,
                ))

        manifest = RunManifest(
            run_id=run_id,
            stage="train",
            status=_status_for_summary(summary),
            created_at=_now(),
            spec={**summary.get("config", {}), "experiment": "dft"},
            git=_make_placeholder_git(),
            source=_make_placeholder_source(out_dir),
            inputs=inputs,
            artifacts=artifacts,
            env=EnvironmentInfo(),
            hardware=HardwareInfo(),
            summary={**summary, "original_output_dir": str(out_dir.resolve())},
        )
        registry.ingest_manifest(manifest)
        _write_log(f"INGESTED {run_id}")
        count += 1
    return count


def ingest_dft_eval_runs(registry: Registry) -> int:
    count = 0
    for out_dir in DFT_ROOT.glob("out_*"):
        if not out_dir.is_dir() or out_dir.name.startswith("out_train"):
            continue
        summary_path = out_dir / "summary.json"
        if not summary_path.exists():
            continue
        run_id = f"legacy.dft.eval.{out_dir.name}"
        if not FORCE and registry.get(run_id):
            _write_log(f"SKIP {run_id}: already registered")
            continue

        summary = _summarize_dft_eval(summary_path)
        inputs = _input_records_from_dir(out_dir)
        artifacts = []
        for comp in out_dir.glob("completions_*.jsonl"):
            artifacts.append(FileRecord(
                path=str(comp.resolve()),
                sha256=_sha256_file(comp),
                role=f"artifact:{comp.name}",
                size_bytes=comp.stat().st_size,
            ))

        manifest = RunManifest(
            run_id=run_id,
            stage="eval",
            status="completed",
            created_at=_now(),
            spec={**summary.get("config", {}), "experiment": "dft"},
            git=_make_placeholder_git(),
            source=_make_placeholder_source(out_dir),
            inputs=inputs,
            artifacts=artifacts,
            env=EnvironmentInfo(),
            hardware=HardwareInfo(),
            summary={**summary, "original_output_dir": str(out_dir.resolve())},
        )
        registry.ingest_manifest(manifest)
        _write_log(f"INGESTED {run_id}")
        count += 1
    return count


def main() -> int:
    import argparse

    global FORCE
    parser = argparse.ArgumentParser()
    parser.add_argument("--force", action="store_true", help="Re-register runs that already exist.")
    args = parser.parse_args()
    FORCE = args.force

    REGISTRY_PATH.parent.mkdir(parents=True, exist_ok=True)
    registry = Registry(REGISTRY_PATH)

    _write_log("Migration started (force=%s)" % FORCE)
    counts = {
        "ace_collect": ingest_ace_collect_runs(registry),
        "dft_train": ingest_dft_train_runs(registry),
        "dft_eval": ingest_dft_eval_runs(registry),
    }
    _write_log(f"Migration complete: {counts}")
    print(json.dumps(counts, indent=2))
    return 0


if __name__ == "__main__":
    sys.exit(main())
