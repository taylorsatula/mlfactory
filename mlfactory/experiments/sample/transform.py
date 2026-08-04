"""Sample experiment: text analysis pipeline.

Three-stage domain logic that is wrapped by mlfactory plugins:
  1. transform  — chunk text and compute statistics
  2. classify   — assign topics to chunks via LLM
  3. eval       — judge classification quality, gate on threshold

This module contains only domain logic — it knows nothing about mlfactory.
The plugins (transform_plugin.py, classify_plugin.py, eval_plugin.py) bridge
between mlfactory and these functions.
"""
from __future__ import annotations

import json
import re
import string
from pathlib import Path
from typing import Any


# ---------------------------------------------------------------------------
# Sample corpus generation
# ---------------------------------------------------------------------------

SAMPLE_PARAGRAPHS = [
    "Machine learning is a subset of artificial intelligence that focuses on "
    "building systems that learn from data. These systems improve their "
    "performance on a specific task over time without being explicitly "
    "programmed to do so. The field has grown rapidly in recent years, driven "
    "by increases in computational power and the availability of large datasets.",

    "Reproducibility in machine learning research means that other researchers "
    "can re-run an experiment and obtain the same results given the same data, "
    "code, and environment. This requires careful tracking of software versions, "
    "random seeds, hardware configurations, and data preprocessing steps.",

    "Natural language processing enables computers to understand, interpret, and "
    "generate human language. Modern NLP systems are built on transformer "
    "architectures that process text as sequences of tokens. These models are "
    "pre-trained on large corpora and then fine-tuned for specific tasks.",

    "Experiment tracking systems record the parameters, metrics, and artifacts "
    "produced by each run of a machine learning pipeline. They allow researchers "
    "to compare different configurations, trace the lineage of results, and "
    "identify which changes led to improvements.",

    "GPU acceleration has transformed machine learning by enabling parallel "
    "computation on thousands of cores simultaneously. Modern GPUs provide "
    "tens of teraflops of compute capacity and high-bandwidth memory for "
    "loading large model parameters.",

    "Data preprocessing is often the most time-consuming part of a machine "
    "learning project. Raw data must be cleaned, normalized, tokenized, and "
    "split into training, validation, and test sets. The quality of "
    "preprocessing directly affects model performance.",

    "Transfer learning allows a model trained on one task to be adapted to a "
    "different but related task. This approach reduces the amount of task-specific "
    "data needed and speeds up training. Pre-trained language models like BERT, "
    "GPT, and Llama are examples of transfer learning applied at scale.",

    "Hyperparameter optimization searches the space of configuration choices to "
    "find the combination that produces the best model performance. Common "
    "strategies include grid search, random search, Bayesian optimization, and "
    "early-stopping methods.",
]

VALID_TOPICS = ["machine_learning", "nlp", "infrastructure", "data_engineering", "research_methods"]


def generate_sample_corpus(num_paragraphs: int = 8) -> str:
    """Generate a multi-paragraph sample text for analysis."""
    paragraphs = []
    for i in range(num_paragraphs):
        paragraphs.append(SAMPLE_PARAGRAPHS[i % len(SAMPLE_PARAGRAPHS)])
    return "\n\n".join(paragraphs)


# ---------------------------------------------------------------------------
# Stage 1: Transform — chunk text and compute statistics
# ---------------------------------------------------------------------------

def split_into_chunks(text: str, chunk_size: int = 200) -> list[str]:
    """Split text into chunks of approximately ``chunk_size`` characters."""
    sentences = re.split(r'(?<=[.!?])\s+', text)
    chunks: list[str] = []
    current = ""
    for sentence in sentences:
        if len(current) + len(sentence) > chunk_size and current:
            chunks.append(current.strip())
            current = sentence
        else:
            current = f"{current} {sentence}" if current else sentence
    if current.strip():
        chunks.append(current.strip())
    return chunks


def analyze_chunk(text: str) -> dict[str, Any]:
    """Compute statistics for a single text chunk."""
    words = text.split()
    sentences = re.split(r'[.!?]+', text)
    sentences = [s.strip() for s in sentences if s.strip()]
    unique_words = set(w.lower().strip(string.punctuation) for w in words)
    word_lengths = [len(w.strip(string.punctuation)) for w in words if w.strip(string.punctuation)]

    return {
        "word_count": len(words),
        "sentence_count": len(sentences),
        "unique_words": len(unique_words),
        "avg_word_length": round(sum(word_lengths) / len(word_lengths), 2) if word_lengths else 0.0,
        "character_count": len(text),
        "vocabulary_richness": round(len(unique_words) / len(words), 4) if words else 0.0,
    }


def run_transform(
    input_text: str | None,
    chunk_size: int,
    num_paragraphs: int = 8,
) -> list[dict[str, Any]]:
    """Split text into chunks with per-chunk statistics.

    Returns a list of chunk records: [{index, text, word_count, ...}, ...]
    """
    if input_text is None:
        input_text = generate_sample_corpus(num_paragraphs)

    chunks = split_into_chunks(input_text, chunk_size)
    records = []
    for i, chunk in enumerate(chunks):
        stats = analyze_chunk(chunk)
        records.append({"index": i, "text": chunk, **stats})
    return records


# ---------------------------------------------------------------------------
# Stage 2: Classify — assign topics to chunks via LLM
# ---------------------------------------------------------------------------

CLASSIFY_SYSTEM_PROMPT = """\
You are a text classifier. Given a text chunk, assign it to exactly one topic \
from this list: {topics}.

Respond with a JSON object:
{{"topic": "<topic_name>", "confidence": <0.0-1.0>, "reasoning": "<brief explanation>"}}
"""


def build_classify_messages(chunk_text: str, topics: list[str]) -> list[dict[str, str]]:
    """Build chat messages for classifying a single chunk."""
    system = CLASSIFY_SYSTEM_PROMPT.format(topics=", ".join(topics))
    return [
        {"role": "system", "content": system},
        {"role": "user", "content": f"Classify this text:\n\n{chunk_text}"},
    ]


def parse_classification(response_text: str) -> dict[str, Any]:
    """Parse a classification response into a structured record."""
    from mlfactory.core.api import extract_json
    data = extract_json(response_text)
    return {
        "topic": data.get("topic", "unknown"),
        "confidence": float(data.get("confidence", 0.0)),
        "reasoning": data.get("reasoning", ""),
    }


# ---------------------------------------------------------------------------
# Stage 3: Eval — judge classification quality
# ---------------------------------------------------------------------------

EVAL_SYSTEM_PROMPT = """\
You are an evaluation judge. Given a text chunk and its assigned topic, rate \
whether the classification is correct.

Respond with a JSON object:
{{"correct": <true/false>, "score": <0.0-1.0>, "feedback": "<brief feedback>"}}
"""


def build_eval_messages(
    chunk_text: str,
    assigned_topic: str,
    reasoning: str,
) -> list[dict[str, str]]:
    """Build chat messages for evaluating a single classification."""
    system = EVAL_SYSTEM_PROMPT
    user = (
        f"Text chunk:\n{chunk_text}\n\n"
        f"Assigned topic: {assigned_topic}\n"
        f"Classifier reasoning: {reasoning}\n\n"
        "Is this classification correct?"
    )
    return [
        {"role": "system", "content": system},
        {"role": "user", "content": user},
    ]


def parse_eval_response(response_text: str) -> dict[str, Any]:
    """Parse an evaluation response into a structured record."""
    from mlfactory.core.api import extract_json
    data = extract_json(response_text)
    return {
        "correct": bool(data.get("correct", False)),
        "score": float(data.get("score", 0.0)),
        "feedback": data.get("feedback", ""),
    }


def compute_quality_report(evaluations: list[dict[str, Any]]) -> dict[str, Any]:
    """Compute aggregate quality metrics from individual evaluations."""
    if not evaluations:
        return {"avg_score": 0.0, "accuracy": 0.0, "total": 0, "correct": 0}

    total = len(evaluations)
    correct = sum(1 for e in evaluations if e.get("correct"))
    avg_score = sum(e.get("score", 0.0) for e in evaluations) / total

    return {
        "total": total,
        "correct": correct,
        "accuracy": round(correct / total, 4),
        "avg_score": round(avg_score, 4),
        "per_topic": _per_topic_breakdown(evaluations),
    }


def _per_topic_breakdown(evaluations: list[dict[str, Any]]) -> dict[str, dict]:
    """Break down quality by assigned topic."""
    by_topic: dict[str, list[dict]] = {}
    for e in evaluations:
        topic = e.get("topic", "unknown")
        by_topic.setdefault(topic, []).append(e)
    return {
        topic: {
            "count": len(items),
            "correct": sum(1 for i in items if i.get("correct")),
            "avg_score": round(sum(i.get("score", 0.0) for i in items) / len(items), 4),
        }
        for topic, items in by_topic.items()
    }
