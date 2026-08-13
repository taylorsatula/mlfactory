"""mlfactory wrapper for the staged CausalGraph MVP commands."""
from __future__ import annotations

import json
from pathlib import Path

import yaml

from mlfactory.core.manifest import FileRecord, sha256_file
from mlfactory.plugins.base import PLUGINS, StagePlugin


class CausalGraphPlugin(StagePlugin):
    stage = "causal-graph"

    def __init__(self, manifest):
        super().__init__(manifest)
        self.spec = manifest.spec

    def prepare(self) -> None:
        (self.run_dir / "artifacts").mkdir(parents=True, exist_ok=True)
        (self.run_dir / "logs").mkdir(parents=True, exist_ok=True)
        (self.run_dir / "artifacts" / "config.yaml").write_text(
            yaml.safe_dump(self.spec, sort_keys=True), encoding="utf-8"
        )

    def execute(self) -> None:
        from .experiment import run_analysis, run_coarse, validate_generator
        mode = str(self.spec.get("mode", "validate"))
        if mode == "validate":
            summary = validate_generator(int(self.spec.get("count", 10_000)), int(self.spec.get("seed", 20260811)), self.run_dir / "artifacts" / "generator_sample.jsonl")
        elif mode == "coarse":
            import argparse
            args = argparse.Namespace(
                output=str(self.run_dir / "artifacts"), seed=int(self.spec.get("seed", 20260811)),
                examples_per_depth=int(self.spec.get("examples_per_depth", 64)), base_url=self.spec.get("base_url", "http://127.0.0.1:3090/v1"),
                model=self.spec.get("model", "f16-jackrongds4qwen"), timeout=float(self.spec.get("timeout", 180)), max_tokens=int(self.spec.get("max_tokens", 768)),
            )
            summary = run_coarse(args)
        elif mode == "analyze":
            import argparse
            args = argparse.Namespace(records=str(Path(self.spec["records"]).resolve()), output=str(self.run_dir / "artifacts"), bootstrap=int(self.spec.get("bootstrap", 2000)), seed=int(self.spec.get("seed", 20260811)))
            summary = run_analysis(args)
        else:
            raise ValueError(f"unsupported causal-graph mode: {mode}")
        self.manifest.summary = summary
        (self.run_dir / "artifacts" / "summary.json").write_text(json.dumps(summary, indent=2, sort_keys=True) + "\n", encoding="utf-8")

    def finalize(self) -> None:
        self.manifest.artifacts = []
        artifacts = self.run_dir / "artifacts"
        for path in sorted(artifacts.rglob("*")):
            if path.is_file():
                self.manifest.artifacts.append(FileRecord(path=str(path.resolve()), sha256=sha256_file(path), role=f"artifact:{path.relative_to(artifacts)}", size_bytes=path.stat().st_size))
        self.manifest.write(self.run_dir / "manifest.json")


PLUGINS.register(CausalGraphPlugin)
