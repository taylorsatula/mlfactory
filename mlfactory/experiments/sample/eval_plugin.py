"""Factory plugin for the sample eval stage.

Demonstrates:
- APIClient for LLM-as-judge evaluation
- Judge.compare() for pairwise A/B comparison
- Guard logic: marks run as "guarded" when quality is below threshold
- Reading artifacts from a parent classify run (multi-stage lineage)
- save_summary and save_config from core.artifacts
"""
from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from mlfactory.core.api import APIClient, APIConfig, Judge, extract_json
from mlfactory.core.artifacts import save_config, save_summary
from mlfactory.core.manifest import FileRecord, sha256_file
from mlfactory.core.metrics import MetricsLogger
from mlfactory.plugins.base import PLUGINS, StagePlugin


class SampleEvalPlugin(StagePlugin):
    """Stage 3: evaluate classification quality via LLM-as-judge."""

    stage = "eval"

    def __init__(self, manifest):
        super().__init__(manifest)
        self.spec = manifest.spec

    # ------------------------------------------------------------------
    def prepare(self) -> None:
        (self.run_dir / "artifacts").mkdir(parents=True, exist_ok=True)
        (self.run_dir / "logs").mkdir(parents=True, exist_ok=True)

        self._classifications_path = self._resolve_input()

    def _resolve_input(self) -> Path:
        """Find the classifications.jsonl to evaluate.

        Priority:
        1. Explicit ``input_file`` in spec.
        2. ``input_run`` in spec — look up that run's artifacts.
        """
        spec = self.spec

        input_file = spec.get("input_file")
        if input_file and Path(input_file).exists():
            return Path(input_file)

        input_run = spec.get("input_run")
        if input_run:
            from mlfactory.core.registry import Registry
            registry = Registry(".mlfactory/registry.db")
            parent = registry.get(input_run)
            if parent is None:
                raise FileNotFoundError(f"parent run not found in registry: {input_run}")
            parent_run_dir = Path(parent.source.path).parent
            cls_path = parent_run_dir / "artifacts" / "classifications.jsonl"
            if cls_path.exists():
                return cls_path
            raise FileNotFoundError(
                f"classifications.jsonl not found in parent run {input_run} at {cls_path}"
            )

        raise ValueError(
            "eval spec must have 'input_file' (path to classifications.jsonl) "
            "or 'input_run' (run_id of a classify run)"
        )

    # ------------------------------------------------------------------
    def execute(self) -> None:
        from mlfactory.experiments.sample.transform import (
            build_eval_messages,
            compute_quality_report,
            parse_eval_response,
        )

        spec = self.spec

        # Load classifications from the parent run.
        classifications: list[dict[str, Any]] = []
        with open(self._classifications_path, "r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if line:
                    classifications.append(json.loads(line))

        max_samples = spec.get("max_samples")  # optional limit for smoke tests
        if max_samples:
            classifications = classifications[:max_samples]

        metrics = MetricsLogger(self.run_dir, run_id=self.manifest.run_id, echo=False)

        # Build the judge client.
        judge_client = self._get_judge()

        # Evaluate each classification.
        evaluations: list[dict[str, Any]] = []
        pairwise_results: list[dict[str, Any]] = []

        for i, cls_record in enumerate(classifications):
            # --- LLM-as-judge: rate the classification ---
            messages = build_eval_messages(
                chunk_text=cls_record.get("text", ""),
                assigned_topic=cls_record.get("topic", "unknown"),
                reasoning=cls_record.get("reasoning", ""),
            )
            try:
                response = judge_client.chat_completion(
                    messages=messages,
                    temperature=0.0,
                    max_tokens=spec.get("max_tokens", 256),
                )
                eval_result = parse_eval_response(response)
                error = None
            except Exception as e:
                eval_result = {"correct": False, "score": 0.0, "feedback": f"eval error: {e}"}
                error = str(e)

            eval_result["chunk_index"] = cls_record.get("chunk_index", i)
            eval_result["topic"] = cls_record.get("topic", "unknown")
            evaluations.append(eval_result)

            metrics.step(
                step=i,
                score=eval_result["score"],
                correct=int(eval_result["correct"]),
            )
            if error:
                metrics.event("eval_error", {"chunk": i, "error": error})

            # --- Pairwise comparison via Judge.compare() ---
            # Compare this classification's reasoning against a "baseline"
            # reasoning (we use the first chunk as the reference).
            if i > 0 and spec.get("pairwise", False):
                ref = classifications[0]
                try:
                    winner = judge_client.compare(
                        prompt="Which text is more clearly about its assigned topic?",
                        candidate_a=cls_record.get("text", ""),
                        candidate_b=ref.get("text", ""),
                        criterion="topic clarity",
                    )
                    pairwise_results.append({
                        "chunk_index": cls_record.get("chunk_index", i),
                        "winner": winner,
                        "reference_index": ref.get("chunk_index", 0),
                    })
                except Exception:
                    pass

        # Compute aggregate quality report.
        quality_report = compute_quality_report(evaluations)

        # Write per-item evaluations.
        eval_path = self.run_dir / "artifacts" / "quality_scores.jsonl"
        with open(eval_path, "w", encoding="utf-8") as f:
            for e in evaluations:
                f.write(json.dumps(e, ensure_ascii=False) + "\n")

        # Write pairwise results if we did any.
        if pairwise_results:
            pw_path = self.run_dir / "artifacts" / "pairwise_results.jsonl"
            with open(pw_path, "w", encoding="utf-8") as f:
                for pw in pairwise_results:
                    f.write(json.dumps(pw, ensure_ascii=False) + "\n")

        # Build and save the eval report.
        eval_report = {
            **quality_report,
            "input_file": str(self._classifications_path),
            "judge_model": spec.get("judge_model", "unknown"),
            "pairwise_comparisons": len(pairwise_results),
        }
        save_summary(self.run_dir, eval_report, manifest=self.manifest)

        # Save the eval config for reproducibility.
        save_config(self.run_dir, {
            "judge_model": spec.get("judge_model"),
            "judge_base_url": spec.get("judge_base_url"),
            "quality_threshold": spec.get("quality_threshold", 0.7),
            "pairwise": spec.get("pairwise", False),
            "max_samples": max_samples,
        }, name="eval_config.json")

        # ---- Guard logic ----
        # If quality is below the threshold, mark the run as "guarded".
        # This demonstrates how experiments can gate downstream stages.
        threshold = spec.get("quality_threshold", 0.7)
        if quality_report["accuracy"] < threshold:
            self.manifest.status = "guarded"
            self.manifest.guard_report = {
                "reason": f"accuracy {quality_report['accuracy']:.4f} below threshold {threshold}",
                "threshold": threshold,
                "accuracy": quality_report["accuracy"],
                "avg_score": quality_report["avg_score"],
            }
            metrics.event("guard_triggered", {
                "accuracy": quality_report["accuracy"],
                "threshold": threshold,
            })

        metrics.event("eval_complete", {
            "total": len(evaluations),
            "accuracy": quality_report["accuracy"],
            "avg_score": quality_report["avg_score"],
        })

    # ------------------------------------------------------------------
    def _get_judge(self) -> APIClient:
        """Build an APIClient for the judge model."""
        spec = self.spec
        base_url = spec.get("judge_base_url")
        if not base_url:
            raise ValueError(
                "eval spec must have 'judge_base_url' (OpenAI-compatible endpoint)"
            )
        return APIClient(APIConfig(
            base_url=base_url,
            api_key=spec.get("judge_api_key", "none"),
            model=spec.get("judge_model", "default"),
            timeout=spec.get("timeout", 120.0),
            max_retries=spec.get("max_retries", 3),
        ))

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


PLUGINS.register(SampleEvalPlugin)
