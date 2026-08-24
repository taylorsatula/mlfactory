"""Factory plugin for the sample classify stage.

Demonstrates:
- Model server via ``model()`` context manager (local GGUF inference)
- APIClient for external endpoints (OpenRouter, remote llama-server, etc.)
- ``inference_env()`` environment guard
- Reading artifacts from a parent run (multi-stage lineage)
- MetricsLogger with per-item events
"""
from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from mlfactory.core.api import APIClient, APIConfig, extract_json
from mlfactory.core.datasave import DataSaver, finalize_artifacts
from mlfactory.core.env import inference_env
from mlfactory.core.metrics import MetricsLogger
from mlfactory.plugins.base import PLUGINS, StagePlugin


class SampleClassifyPlugin(StagePlugin):
    """Stage 2: classify text chunks by topic via LLM inference."""

    stage = "classify"

    def __init__(self, manifest):
        super().__init__(manifest)
        self.spec = manifest.spec

    # ------------------------------------------------------------------
    def prepare(self) -> None:
        (self.run_dir / "artifacts").mkdir(parents=True, exist_ok=True)
        (self.run_dir / "logs").mkdir(parents=True, exist_ok=True)

        # Resolve input: either an explicit file path or a parent run's artifact.
        self._chunks_path = self._resolve_input()

    def _resolve_input(self) -> Path:
        """Find the chunks.jsonl to classify.

        Priority:
        1. Explicit ``input_file`` in spec (a path to chunks.jsonl).
        2. ``input_run`` in spec — look up that run's artifacts/chunks.jsonl
           via the registry.
        """
        spec = self.spec

        # Direct file path.
        input_file = spec.get("input_file")
        if input_file and Path(input_file).exists():
            return Path(input_file)

        # Parent run reference — resolve via registry.
        input_run = spec.get("input_run")
        if input_run:
            from mlfactory.core.registry import Registry
            registry = Registry(".mlfactory/registry.db")
            parent = registry.get(input_run)
            if parent is None:
                raise FileNotFoundError(f"parent run not found in registry: {input_run}")
            # Find chunks.jsonl in the parent's artifacts.
            parent_run_dir = Path(parent.source.path).parent
            chunks = parent_run_dir / "artifacts" / "chunks.jsonl"
            if chunks.exists():
                return chunks
            raise FileNotFoundError(
                f"chunks.jsonl not found in parent run {input_run} at {chunks}"
            )

        raise ValueError(
            "classify spec must have 'input_file' (path to chunks.jsonl) "
            "or 'input_run' (run_id of a transform run)"
        )

    # ------------------------------------------------------------------
    def execute(self) -> None:
        from mlfactory.experiments.sample.transform import (
            VALID_TOPICS,
            build_classify_messages,
            parse_classification,
        )

        spec = self.spec
        topics = spec.get("topics", VALID_TOPICS)
        max_chunks = spec.get("max_chunks")  # optional limit for smoke tests

        # Load chunks from the parent run.
        chunks: list[dict[str, Any]] = []
        with open(self._chunks_path, "r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if line:
                    chunks.append(json.loads(line))

        if max_chunks:
            chunks = chunks[:max_chunks]

        metrics = MetricsLogger(self.run_dir, run_id=self.manifest.run_id, echo=False)

        # Classify each chunk using an LLM.
        classifications: list[dict[str, Any]] = []

        with inference_env(hf_home=spec.get("env", {}).get("HF_HOME")):
            client = self._get_client()

            for i, chunk in enumerate(chunks):
                messages = build_classify_messages(chunk["text"], topics)
                try:
                    response = client.chat_completion(
                        messages=messages,
                        temperature=spec.get("temperature", 0.2),
                        max_tokens=spec.get("max_tokens", 256),
                    )
                    classification = parse_classification(response)
                    classification["chunk_index"] = chunk["index"]
                    classification["text"] = chunk["text"]
                    error = None
                except Exception as e:
                    classification = {
                        "chunk_index": chunk["index"],
                        "text": chunk["text"],
                        "topic": "error",
                        "confidence": 0.0,
                        "reasoning": "",
                    }
                    error = str(e)

                classifications.append(classification)

                # Log per-chunk metrics.
                metrics.step(
                    step=i,
                    confidence=classification["confidence"],
                    topic_assigned=1,  # count
                )
                if error:
                    metrics.event("classify_error", {"chunk": i, "error": error})

        # Write classifications.jsonl — consumed by the eval stage.
        saver = DataSaver(self.run_dir, self.manifest)
        out_path = saver.save(
            "classifications.jsonl",
            classifications,
            title="Topic classifications",
            description=(
                "Per-chunk topic classifications produced by LLM inference. "
                "Each row pairs a chunk with its assigned topic, confidence and reasoning."
            ),
            tags=["labels", "sample"],
            format="jsonl",
        )

        # Compute topic distribution for summary.
        topic_counts: dict[str, int] = {}
        for c in classifications:
            topic_counts[c["topic"]] = topic_counts.get(c["topic"], 0) + 1

        avg_conf = sum(c["confidence"] for c in classifications) / len(classifications) if classifications else 0

        summary = {
            "total_chunks": len(classifications),
            "avg_confidence": round(avg_conf, 4),
            "topic_distribution": topic_counts,
            "input_file": str(self._chunks_path),
            "classifications_artifact": str(out_path.resolve()),
        }

        from mlfactory.core.datasave import DataSaver
        # Save the summary with a lab label; mirror it onto manifest.summary.
        self.manifest.summary = summary
        DataSaver(self.run_dir, self.manifest).save(
            "summary.json",
            summary,
            title="Classify summary",
            description=(
                "Aggregate classification results for the run. Holds the "
                "topic distribution and mean confidence over all classified chunks."
            ),
            format="json",
        )

        metrics.event("classify_complete", {
            "total": len(classifications),
            "avg_confidence": round(avg_conf, 4),
        })

    # ------------------------------------------------------------------
    def _get_client(self) -> APIClient:
        """Build an APIClient from the spec.

        Supports two modes:
        1. Local model server: spec has ``model`` alias → start llama-server.
        2. External endpoint: spec has ``base_url`` → use directly.
        """
        spec = self.spec

        # Mode 1: local model server via model() context manager.
        model_alias = spec.get("model")
        if model_alias:
            from mlfactory.core.model_server import model
            gpu = spec.get("gpu", 0)
            srv = model(model_alias, gpu=gpu)
            srv.start()
            # Store server reference so we can stop it in finalize.
            self._model_server = srv
            return APIClient(APIConfig(
                base_url=srv.base_url,
                model=srv.spec.alias or model_alias,
                timeout=spec.get("timeout", 120.0),
            ))

        # Mode 2: external endpoint.
        base_url = spec.get("base_url")
        if not base_url:
            raise ValueError(
                "classify spec must have 'model' (local GGUF alias) "
                "or 'base_url' (OpenAI-compatible endpoint)"
            )
        return APIClient(APIConfig(
            base_url=base_url,
            api_key=spec.get("api_key", "none"),
            model=spec.get("model_name", "default"),
            timeout=spec.get("timeout", 120.0),
        ))

    # ------------------------------------------------------------------
    def finalize(self) -> None:
        """Stop the model server, hash any unregistered artifacts, persist manifest."""
        # Stop the model server if we started one.
        if hasattr(self, "_model_server") and self._model_server:
            try:
                self._model_server.stop()
            except Exception:
                pass

        finalize_artifacts(self.manifest, self.run_dir)


PLUGINS.register(SampleClassifyPlugin)
