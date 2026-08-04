"""Tests for fine-tuning data preparation and the dependency-free backend."""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np

from mlfactory.core.finetune import build_causal_lm_examples
from mlfactory.experiments.sample.train import (
    load_softmax_model,
    predict_softmax,
    save_softmax_model,
    train_softmax_classifier,
)


def _records() -> list[dict]:
    return [
        {"text": "neural network learns from data", "topic": "ml", "confidence": 0.9, "reasoning": "ML terms"},
        {"text": "model training uses data", "topic": "ml", "confidence": 0.8, "reasoning": "training"},
        {"text": "tokens represent natural language", "topic": "nlp", "confidence": 0.9, "reasoning": "language"},
        {"text": "language model processes tokens", "topic": "nlp", "confidence": 0.8, "reasoning": "tokens"},
    ]


def test_numpy_finetune_updates_and_round_trips(tmp_path: Path) -> None:
    model, report = train_softmax_classifier(
        _records(), epochs=100, learning_rate=1.0, seed=7
    )
    assert report["optimizer_steps"] == 100
    assert report["final_loss"] < report["initial_loss"]
    assert report["train_accuracy"] >= 0.75

    checkpoint = save_softmax_model(model, tmp_path / "checkpoint-final")
    loaded = load_softmax_model(checkpoint)
    assert np.array_equal(model["weights"], loaded["weights"])
    assert predict_softmax(model, ["language tokens"]) == predict_softmax(
        loaded, ["language tokens"]
    )


def test_build_causal_lm_examples_uses_completion_labels() -> None:
    example = build_causal_lm_examples(_records()[:1])[0]
    assert "neural network" in example["prompt"]
    response = json.loads(example["completion"])
    assert response["topic"] == "ml"
    assert response["confidence"] == 0.9
