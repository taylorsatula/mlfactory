"""Factory plugin for the sample supervised fine-tuning stage."""
from __future__ import annotations

from pathlib import Path
from typing import Any

from mlfactory.core.artifacts import save_config, save_summary
from mlfactory.core.env import training_env
from mlfactory.core.finetune import build_causal_lm_examples, train_transformers_causal_lm
from mlfactory.core.manifest import FileRecord, sha256_file
from mlfactory.core.metrics import MetricsLogger
from mlfactory.plugins.base import PLUGINS, StagePlugin


class SampleTrainPlugin(StagePlugin):
    """Stage 3: fine-tune a classifier or causal LM on classification records."""

    stage = "train"

    def __init__(self, manifest):
        super().__init__(manifest)
        self.spec = manifest.spec

    def prepare(self) -> None:
        (self.run_dir / "artifacts").mkdir(parents=True, exist_ok=True)
        (self.run_dir / "logs").mkdir(parents=True, exist_ok=True)
        self._records_path = self._resolve_input()
        backend = self.spec.get("backend", "numpy_softmax")
        if backend not in {"numpy_softmax", "transformers_causal_lm"}:
            raise ValueError(
                "backend must be 'numpy_softmax' or 'transformers_causal_lm'"
            )
        self.manifest.inputs.append(FileRecord(
            path=str(self._records_path.resolve()),
            sha256=sha256_file(self._records_path),
            role="input:training_data",
            size_bytes=self._records_path.stat().st_size,
        ))

    def _resolve_input(self) -> Path:
        input_file = self.spec.get("input_file")
        if input_file:
            path = Path(input_file)
            if path.exists():
                return path
            raise FileNotFoundError(f"training input file not found: {path}")

        input_run = self.spec.get("input_run")
        if input_run:
            from mlfactory.core.registry import Registry

            parent = Registry(".mlfactory/registry.db").get(input_run)
            if parent is None:
                raise FileNotFoundError(f"parent run not found in registry: {input_run}")
            parent_dir = Path(parent.source.path).parent
            path = parent_dir / "artifacts" / "classifications.jsonl"
            if path.exists():
                return path
            raise FileNotFoundError(
                f"classifications.jsonl not found in parent run {input_run} at {path}"
            )
        raise ValueError(
            "train spec must have 'input_file' or 'input_run' pointing to "
            "classifications.jsonl"
        )

    def execute(self) -> None:
        from mlfactory.experiments.sample.train import (
            load_classification_records,
            save_softmax_model,
            train_softmax_classifier,
        )

        records = load_classification_records(self._records_path)
        excluded = {str(value) for value in self.spec.get("exclude_labels", [])}
        if excluded:
            records = [record for record in records if str(record["topic"]) not in excluded]
        if not records:
            raise ValueError("no training examples remain after exclude_labels filtering")

        backend = self.spec.get("backend", "numpy_softmax")
        training = dict(self.spec.get("training", {}))
        metrics = MetricsLogger(self.run_dir, run_id=self.manifest.run_id, echo=False)

        def log_step(step: int, values: dict[str, float]) -> None:
            metrics.step(step=step, **values)

        env = self.spec.get("env", {})
        with training_env(hf_home=env.get("HF_HOME"), extra=env.get("extra")):
            if backend == "numpy_softmax":
                model, report = train_softmax_classifier(
                    records,
                    epochs=int(training.get("epochs", 100)),
                    learning_rate=float(training.get("learning_rate", 0.5)),
                    l2=float(training.get("l2", 0.0)),
                    min_frequency=int(training.get("min_frequency", 1)),
                    seed=int(training.get("seed", 42)),
                    metric_callback=log_step,
                )
                checkpoint = save_softmax_model(
                    model, self.run_dir / "artifacts" / "checkpoint-final"
                )
            else:
                examples = build_causal_lm_examples(
                    records,
                    prompt_template=self.spec.get(
                        "prompt_template",
                        "Classify this text by topic:\n\n{text}\n\nAnswer:\n",
                    ),
                    response_template=self.spec.get(
                        "response_template",
                        '{{"topic": "{topic}", "confidence": {confidence}, '
                        '"reasoning": "{reasoning}"}}',
                    ),
                )
                checkpoint = self.run_dir / "artifacts" / "checkpoint-final"
                report = train_transformers_causal_lm(
                    examples, checkpoint, training, metric_callback=log_step
                )

        summary = {
            **report,
            "input_file": str(self._records_path),
            "excluded_labels": sorted(excluded),
            "checkpoint": str(checkpoint.resolve()),
        }
        save_config(self.run_dir, {
            "backend": backend,
            "training": training,
            "exclude_labels": sorted(excluded),
        }, name="training_config.json")
        save_summary(self.run_dir, summary, manifest=self.manifest)
        metrics.event("training_complete", {
            "backend": backend,
            "num_examples": len(records),
            "checkpoint": str(checkpoint),
        })

    def finalize(self) -> None:
        artifacts_dir = self.run_dir / "artifacts"
        for path in sorted(artifacts_dir.rglob("*")):
            if path.is_file():
                relative = path.relative_to(artifacts_dir)
                self.manifest.artifacts.append(FileRecord(
                    path=str(path.resolve()),
                    sha256=sha256_file(path),
                    role=f"artifact:{relative}",
                    size_bytes=path.stat().st_size,
                ))

        logs_dir = self.run_dir / "logs"
        for path in sorted(logs_dir.rglob("*")):
            if path.is_file():
                self.manifest.logs.append(FileRecord(
                    path=str(path.resolve()),
                    sha256=sha256_file(path),
                    role=f"log:{path.name}",
                    size_bytes=path.stat().st_size,
                ))
        self.manifest.write(self.run_dir / "manifest.json")


PLUGINS.register(SampleTrainPlugin)
