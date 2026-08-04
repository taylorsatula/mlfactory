"""Factory plugin for the sample text-transform stage.

Demonstrates:
- Plugin lifecycle: prepare() → execute() → finalize()
- MetricsLogger for per-chunk metrics → dashboard.jsonl
- save_summary() from core.artifacts
- Artifact hashing in finalize()
"""
from __future__ import annotations

import json
from pathlib import Path

from mlfactory.core.artifacts import save_summary
from mlfactory.core.manifest import FileRecord, sha256_file
from mlfactory.core.metrics import MetricsLogger
from mlfactory.plugins.base import PLUGINS, StagePlugin


class SampleTransformPlugin(StagePlugin):
    """Stage 1: chunk text and compute per-chunk statistics."""

    stage = "transform"

    def __init__(self, manifest):
        super().__init__(manifest)
        self.spec = manifest.spec

    # ------------------------------------------------------------------
    def prepare(self) -> None:
        (self.run_dir / "artifacts").mkdir(parents=True, exist_ok=True)
        (self.run_dir / "logs").mkdir(parents=True, exist_ok=True)

        chunk_size = self.spec.get("chunk_size", 200)
        if not isinstance(chunk_size, int) or chunk_size < 10:
            raise ValueError(f"chunk_size must be int >= 10, got {chunk_size}")

    # ------------------------------------------------------------------
    def execute(self) -> None:
        from mlfactory.experiments.sample.transform import (
            generate_sample_corpus,
            run_transform,
        )

        spec = self.spec
        chunk_size = spec.get("chunk_size", 200)
        num_paragraphs = spec.get("num_paragraphs", 8)

        # Resolve input text: from file or generate sample corpus.
        input_path = spec.get("input_text")
        if input_path and Path(input_path).exists():
            input_text = Path(input_path).read_text(encoding="utf-8")
        else:
            input_text = generate_sample_corpus(num_paragraphs)

        # Run the domain logic.
        chunk_records = run_transform(input_text, chunk_size, num_paragraphs)

        # Write chunks.jsonl — the artifact consumed by the classify stage.
        chunks_path = self.run_dir / "artifacts" / "chunks.jsonl"
        with open(chunks_path, "w", encoding="utf-8") as f:
            for record in chunk_records:
                f.write(json.dumps(record, ensure_ascii=False) + "\n")

        # Log per-chunk metrics via MetricsLogger → dashboard.jsonl.
        metrics = MetricsLogger(self.run_dir, run_id=self.manifest.run_id, echo=False)
        for record in chunk_records:
            metrics.step(
                step=record["index"],
                word_count=record["word_count"],
                sentence_count=record["sentence_count"],
                unique_words=record["unique_words"],
                avg_word_length=record["avg_word_length"],
                vocabulary_richness=record["vocabulary_richness"],
            )

        # Compute and save aggregate summary via core.artifacts helper.
        total_words = sum(r["word_count"] for r in chunk_records)
        total_sentences = sum(r["sentence_count"] for r in chunk_records)
        total_chars = sum(r["character_count"] for r in chunk_records)
        avg_unique = sum(r["unique_words"] for r in chunk_records) / len(chunk_records) if chunk_records else 0
        avg_word_len = sum(r["avg_word_length"] for r in chunk_records) / len(chunk_records) if chunk_records else 0

        summary = {
            "num_chunks": len(chunk_records),
            "chunk_size": chunk_size,
            "total_words": total_words,
            "total_sentences": total_sentences,
            "total_characters": total_chars,
            "avg_unique_words_per_chunk": round(avg_unique, 1),
            "avg_word_length": round(avg_word_len, 2),
            "chunks_artifact": str(chunks_path.resolve()),
        }

        # save_summary writes artifacts/summary.json and updates manifest.summary.
        save_summary(self.run_dir, summary, manifest=self.manifest)

        metrics.event("transform_complete", {
            "num_chunks": len(chunk_records),
            "total_words": total_words,
        })

    # ------------------------------------------------------------------
    def finalize(self) -> None:
        """Hash all artifacts and logs, persist the manifest."""
        artifacts_dir = self.run_dir / "artifacts"
        for p in sorted(artifacts_dir.rglob("*")):
            if p.is_file():
                rel = p.relative_to(artifacts_dir)
                self.manifest.artifacts.append(
                    FileRecord(
                        path=str(p.resolve()),
                        sha256=sha256_file(p),
                        role=f"artifact:{rel}",
                        size_bytes=p.stat().st_size,
                    )
                )

        logs_dir = self.run_dir / "logs"
        for p in sorted(logs_dir.rglob("*")):
            if p.is_file():
                self.manifest.logs.append(
                    FileRecord(
                        path=str(p.resolve()),
                        sha256=sha256_file(p),
                        role=f"log:{p.name}",
                        size_bytes=p.stat().st_size,
                    )
                )

        self.manifest.write(self.run_dir / "manifest.json")


PLUGINS.register(SampleTransformPlugin)
