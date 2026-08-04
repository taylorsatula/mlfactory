"""Domain logic for the sample fine-tuning stage."""
from __future__ import annotations

import json
import re
from collections import Counter
from collections.abc import Callable
from pathlib import Path
from typing import Any

import numpy as np


_TOKEN_RE = re.compile(r"[A-Za-z0-9_]+")


def load_classification_records(path: str | Path) -> list[dict[str, Any]]:
    """Load and validate classification JSONL records."""
    records: list[dict[str, Any]] = []
    with Path(path).open("r", encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, 1):
            if not line.strip():
                continue
            record = json.loads(line)
            if not record.get("text") or not record.get("topic"):
                raise ValueError(
                    f"line {line_number} must contain non-empty 'text' and 'topic'"
                )
            records.append(record)
    if not records:
        raise ValueError("training input contains no examples")
    return records


def _tokens(text: str) -> list[str]:
    return _TOKEN_RE.findall(text.lower())


def build_vocabulary(records: list[dict[str, Any]], min_frequency: int = 1) -> list[str]:
    counts = Counter(token for record in records for token in _tokens(str(record["text"])))
    return sorted(token for token, count in counts.items() if count >= min_frequency)


def vectorize(texts: list[str], vocabulary: list[str]) -> np.ndarray:
    """Create L1-normalized bag-of-token vectors."""
    index = {token: i for i, token in enumerate(vocabulary)}
    matrix = np.zeros((len(texts), len(vocabulary)), dtype=np.float64)
    for row, text in enumerate(texts):
        for token in _tokens(text):
            column = index.get(token)
            if column is not None:
                matrix[row, column] += 1.0
        total = matrix[row].sum()
        if total:
            matrix[row] /= total
    return matrix


def _softmax(logits: np.ndarray) -> np.ndarray:
    shifted = logits - logits.max(axis=1, keepdims=True)
    exp = np.exp(shifted)
    return exp / exp.sum(axis=1, keepdims=True)


def train_softmax_classifier(
    records: list[dict[str, Any]],
    *,
    epochs: int = 100,
    learning_rate: float = 0.5,
    l2: float = 0.0,
    min_frequency: int = 1,
    seed: int = 42,
    metric_callback: Callable[[int, dict[str, float]], None] | None = None,
) -> tuple[dict[str, Any], dict[str, Any]]:
    """Fit a multinomial softmax classifier with full-batch gradient descent."""
    if epochs < 1:
        raise ValueError("epochs must be >= 1")
    if learning_rate <= 0:
        raise ValueError("learning_rate must be > 0")

    vocabulary = build_vocabulary(records, min_frequency=min_frequency)
    if not vocabulary:
        raise ValueError("training examples produced an empty vocabulary")
    labels = sorted({str(record["topic"]) for record in records})
    if len(labels) < 2:
        raise ValueError("training requires at least two distinct topic labels")

    texts = [str(record["text"]) for record in records]
    label_index = {label: i for i, label in enumerate(labels)}
    targets = np.array([label_index[str(record["topic"])] for record in records])
    features = vectorize(texts, vocabulary)
    target_matrix = np.eye(len(labels), dtype=np.float64)[targets]

    rng = np.random.default_rng(seed)
    weights = rng.normal(0.0, 0.01, size=(len(vocabulary), len(labels)))
    bias = np.zeros(len(labels), dtype=np.float64)
    losses: list[float] = []

    for epoch in range(1, epochs + 1):
        probabilities = _softmax(features @ weights + bias)
        loss = float(
            -np.mean(np.log(probabilities[np.arange(len(targets)), targets] + 1e-12))
            + 0.5 * l2 * np.sum(weights * weights)
        )
        error = (probabilities - target_matrix) / len(records)
        weights -= learning_rate * (features.T @ error + l2 * weights)
        bias -= learning_rate * error.sum(axis=0)
        accuracy = float(np.mean(np.argmax(probabilities, axis=1) == targets))
        losses.append(loss)
        if metric_callback:
            metric_callback(epoch, {"loss": loss, "train_accuracy": accuracy})

    final_probabilities = _softmax(features @ weights + bias)
    final_accuracy = float(np.mean(np.argmax(final_probabilities, axis=1) == targets))
    model = {
        "weights": weights,
        "bias": bias,
        "vocabulary": vocabulary,
        "labels": labels,
    }
    report = {
        "backend": "numpy_softmax",
        "num_examples": len(records),
        "num_labels": len(labels),
        "vocabulary_size": len(vocabulary),
        "epochs": epochs,
        "optimizer_steps": epochs,
        "initial_loss": losses[0],
        "final_loss": losses[-1],
        "train_accuracy": final_accuracy,
    }
    return model, report


def save_softmax_model(model: dict[str, Any], output_dir: str | Path) -> Path:
    """Persist a safe, reloadable model without pickle-based object arrays."""
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    np.save(output_dir / "weights.npy", model["weights"], allow_pickle=False)
    np.save(output_dir / "bias.npy", model["bias"], allow_pickle=False)
    metadata = {
        "format": "mlfactory.numpy_softmax",
        "version": 1,
        "vocabulary": model["vocabulary"],
        "labels": model["labels"],
    }
    (output_dir / "config.json").write_text(
        json.dumps(metadata, indent=2, ensure_ascii=False), encoding="utf-8"
    )
    return output_dir


def load_softmax_model(model_dir: str | Path) -> dict[str, Any]:
    model_dir = Path(model_dir)
    metadata = json.loads((model_dir / "config.json").read_text(encoding="utf-8"))
    if metadata.get("format") != "mlfactory.numpy_softmax":
        raise ValueError("not an mlfactory numpy softmax checkpoint")
    return {
        "weights": np.load(model_dir / "weights.npy", allow_pickle=False),
        "bias": np.load(model_dir / "bias.npy", allow_pickle=False),
        "vocabulary": metadata["vocabulary"],
        "labels": metadata["labels"],
    }


def predict_softmax(model: dict[str, Any], texts: list[str]) -> list[dict[str, Any]]:
    features = vectorize(texts, model["vocabulary"])
    probabilities = _softmax(features @ model["weights"] + model["bias"])
    predictions: list[dict[str, Any]] = []
    for row in probabilities:
        index = int(np.argmax(row))
        predictions.append({
            "topic": model["labels"][index],
            "confidence": float(row[index]),
        })
    return predictions
