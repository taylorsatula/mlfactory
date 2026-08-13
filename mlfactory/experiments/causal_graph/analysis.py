"""Frontier fitting, uncertainty, and student-aware acquisition utilities."""
from __future__ import annotations

import json
import math
from collections import defaultdict
from pathlib import Path
from typing import Any, Iterable

import numpy as np

FEATURE_NAMES = ["depth", "relevant_nodes", "distractor_nodes", "binary_gate_count", "negation_count", "source_update_count"]


def _sigmoid(x: np.ndarray) -> np.ndarray:
    return np.where(x >= 0, 1.0 / (1.0 + np.exp(-np.clip(x, -60, 60))), np.exp(np.clip(x, -60, 60)) / (1.0 + np.exp(np.clip(x, -60, 60))))


def _fit_logistic_matrix(X: np.ndarray, y: np.ndarray, ridge: float = 1e-4, max_iter: int = 100) -> np.ndarray:
    X = np.asarray(X, dtype=float)
    y = np.asarray(y, dtype=float)
    theta = np.zeros(X.shape[1], dtype=float)
    penalty = np.eye(X.shape[1]) * ridge
    penalty[0, 0] = 0.0
    for _ in range(max_iter):
        p = _sigmoid(X @ theta)
        w = np.maximum(p * (1.0 - p), 1e-7)
        hessian = X.T @ (w[:, None] * X) + penalty
        gradient = X.T @ (y - p) - penalty @ theta
        try:
            step = np.linalg.solve(hessian, gradient)
        except np.linalg.LinAlgError:
            step = np.linalg.pinv(hessian) @ gradient
        theta += step
        if float(np.max(np.abs(step))) < 1e-7:
            break
    return theta


def fit_depth_logistic(records: list[dict[str, Any]], ridge: float = 1e-4) -> dict[str, Any]:
    x = np.asarray([float(r["depth"]) for r in records])
    y = np.asarray([1.0 if r["correct"] else 0.0 for r in records])
    theta = _fit_logistic_matrix(np.column_stack([np.ones(len(x)), x]), y, ridge=ridge)
    b0, b1 = map(float, theta)
    result: dict[str, Any] = {"beta0": b0, "beta1": b1, "n": len(records), "expected_decreasing": b1 < 0}
    if b1 < -1e-8:
        for p in (0.8, 0.5, 0.2):
            result[f"d{int(p * 100)}"] = float((math.log(p / (1 - p)) - b0) / b1)
        result["falloff_width"] = result["d20"] - result["d80"]
    else:
        result.update({"d80": None, "d50": None, "d20": None, "falloff_width": None})
    return result


def wilson_interval(successes: int, total: int, z: float = 1.959963984540054) -> tuple[float, float]:
    if total <= 0:
        return 0.0, 0.0
    p = successes / total
    denom = 1.0 + z * z / total
    center = (p + z * z / (2 * total)) / denom
    half = z * math.sqrt(p * (1 - p) / total + z * z / (4 * total * total)) / denom
    return max(0.0, center - half), min(1.0, center + half)


def _isotonic_decreasing(values: list[float], weights: list[int]) -> list[float]:
    # Each block stores weighted sum, weight, fitted value, and number of
    # original depth bins.  The output therefore has one value per input bin,
    # not one value per individual example.
    blocks: list[list[float | int]] = []
    for value, weight in zip(values, weights):
        blocks.append([value * weight, weight, value, 1])
        while len(blocks) >= 2 and float(blocks[-2][2]) < float(blocks[-1][2]):
            left, right = blocks[-2], blocks[-1]
            total_weight = int(left[1]) + int(right[1])
            total_sum = float(left[0]) + float(right[0])
            merged = [total_sum, total_weight, total_sum / total_weight, int(left[3]) + int(right[3])]
            blocks[-2:] = [merged]
    result: list[float] = []
    for block in blocks:
        result.extend([float(block[2])] * int(block[3]))
    return result


def raw_depth_stats(records: list[dict[str, Any]]) -> list[dict[str, Any]]:
    grouped: dict[int, list[bool]] = defaultdict(list)
    for record in records:
        grouped[int(record["depth"])].append(bool(record["correct"]))
    result = []
    for depth in sorted(grouped):
        values = grouped[depth]
        successes = sum(values)
        lo, hi = wilson_interval(successes, len(values))
        result.append({"depth": depth, "n": len(values), "correct": successes, "accuracy": successes / len(values), "wilson_low": lo, "wilson_high": hi})
    if result:
        fitted = _isotonic_decreasing([float(r["accuracy"]) for r in result], [int(r["n"]) for r in result])
        for row, value in zip(result, fitted):
            row["isotonic_accuracy"] = value
    return result


def _bootstrap_one(records: list[dict[str, Any]], rng: np.random.Generator) -> dict[str, Any]:
    grouped: dict[int, list[dict[str, Any]]] = defaultdict(list)
    for record in records:
        grouped[int(record["depth"])].append(record)
    sampled: list[dict[str, Any]] = []
    for values in grouped.values():
        sampled.extend(values[int(i)] for i in rng.integers(0, len(values), size=len(values)))
    return fit_depth_logistic(sampled)


def bootstrap_frontier(records: list[dict[str, Any]], B: int = 2000, seed: int = 20260811) -> list[dict[str, Any]]:
    rng = np.random.default_rng(seed)
    samples = []
    for _ in range(int(B)):
        samples.append(_bootstrap_one(records, rng))
    return samples


def summarize_bootstrap(samples: list[dict[str, Any]]) -> dict[str, Any]:
    result: dict[str, Any] = {"replicates": len(samples)}
    for key in ("d80", "d50", "d20", "falloff_width"):
        values = np.asarray([s[key] for s in samples if s.get(key) is not None], dtype=float)
        if len(values):
            result[key] = {"median": float(np.median(values)), "ci95": [float(np.quantile(values, 0.025)), float(np.quantile(values, 0.975))], "valid_fraction": float(len(values) / len(samples))}
        else:
            result[key] = None
    return result


def fit_surface(records: list[dict[str, Any]], ridge: float = 0.1) -> dict[str, Any]:
    # Keep the model small and preregistered: main structural variables plus
    # depth-by-gate and depth-by-distractor interactions.
    base = np.asarray([[float(r[name]) for name in FEATURE_NAMES] for r in records], dtype=float)
    X = np.column_stack([np.ones(len(base)), base, base[:, 0] * base[:, 3], base[:, 0] * base[:, 2]])
    names = ["intercept", *FEATURE_NAMES, "depth_x_binary_gate", "depth_x_distractor"]
    y = np.asarray([1.0 if r["correct"] else 0.0 for r in records])
    theta = _fit_logistic_matrix(X, y, ridge=ridge)
    return {"feature_names": names, "coefficients": [float(v) for v in theta], "n": len(records), "training_accuracy": float(np.mean((_sigmoid(X @ theta) >= 0.5) == y))}


def predict_surface(surface: dict[str, Any], records: list[dict[str, Any]]) -> np.ndarray:
    names = surface["feature_names"]
    theta = np.asarray(surface["coefficients"], dtype=float)
    rows = []
    for r in records:
        values = [1.0] + [float(r[name]) for name in FEATURE_NAMES]
        values.extend([float(r["depth"]) * float(r["binary_gate_count"]), float(r["depth"]) * float(r["distractor_nodes"])])
        rows.append(values)
    return _sigmoid(np.asarray(rows) @ theta)


def structural_cell(record: dict[str, Any]) -> tuple[int, int, int, int, int]:
    return (int(record["depth"]), int(record["relevant_nodes"]) // 2, int(record["distractor_nodes"]) // 2, int(record["binary_gate_count"]) // 2, int(record["source_update_count"]))


def score_candidates(candidates: list[dict[str, Any]], surface: dict[str, Any]) -> list[dict[str, Any]]:
    probabilities = predict_surface(surface, candidates)
    scored = []
    for candidate, probability in zip(candidates, probabilities):
        row = dict(candidate)
        row["predicted_success_probability"] = float(probability)
        row["boundary_score"] = float(4 * probability * (1 - probability))
        row["structural_cell"] = list(structural_cell(row))
        row["selected_for_training"] = False
        scored.append(row)
    return scored


def select_frontier_batch(scored: list[dict[str, Any]], batch_size: int = 1000, cell_fraction: float = 0.10, seed: int = 20260811) -> list[dict[str, Any]]:
    rng = np.random.default_rng(seed)
    ordered = sorted(scored, key=lambda r: (-float(r["boundary_score"]), str(r["id"])))
    cap = max(1, int(math.ceil(batch_size * cell_fraction)))
    selected: list[dict[str, Any]] = []
    cell_counts: dict[tuple[Any, ...], int] = defaultdict(int)
    world_counts: dict[str, int] = defaultdict(int)
    world_cap = max(cap, int(math.ceil(batch_size * 0.35)))
    for row in ordered:
        cell = tuple(row["structural_cell"])
        world = str(row["world_id"])
        if cell_counts[cell] >= cap or world_counts[world] >= world_cap:
            continue
        selected.append(row)
        cell_counts[cell] += 1
        world_counts[world] += 1
        row["selected_for_training"] = True
        if len(selected) >= batch_size:
            break
    # If the balanced pass could not fill the batch, fill by score while
    # retaining the structural cap and relaxing only the world cap.
    if len(selected) < batch_size:
        for row in ordered:
            if row in selected:
                continue
            cell = tuple(row["structural_cell"])
            if cell_counts[cell] >= cap:
                continue
            selected.append(row)
            cell_counts[cell] += 1
            row["selected_for_training"] = True
            if len(selected) >= batch_size:
                break
    return selected


def write_json(path: str | Path, value: Any) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def write_jsonl(path: str | Path, rows: Iterable[dict[str, Any]]) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n")
