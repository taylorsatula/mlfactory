"""Shared schema, answer extraction, and helpers for ACE problem families.

Every family module exposes:
    make(rng: random.Random, knobs: dict) -> Problem
    check(completion: str, reference: str, knobs: dict | None) -> bool

`make` builds the instance with an internal solver, so the reference answer
is exact by construction. `check` is the STRICT verifier used by
calibrate.py — it re-scores probe completions and supersedes the probe
collector's advisory soft match.

Row schema matches data/madlibz_verifiable_frontier_30.jsonl so the
existing collector (collect_qwen_frontier_30.py --candidates ...) consumes
generated pools unchanged. Extra key `knobs` carries generation parameters
for calibration joins.
"""
from __future__ import annotations

import hashlib
import json
import re
from dataclasses import dataclass

ANSWER_RE = re.compile(r"answer\s*:\s*(.+)", re.IGNORECASE)


def canon_hash(*parts: str) -> str:
    h = hashlib.sha256("\x1e".join(parts).encode()).hexdigest()
    return h[:16]


@dataclass
class Problem:
    family: str
    prose: str
    question: str
    answer: str            # canonical reference, single line preferred
    verifier_kind: str
    objective_task: str
    search_topology: str
    knobs: dict
    seed: int

    def to_row(self, proposal_id: int) -> dict:
        return {
            "domain": self.family,
            "prose": self.prose,
            "surface_question": self.question,
            "problem": {
                "reference_answer": self.answer,
                "deterministic_verifier": f"ace_gen.{self.family}.check",
            },
            "envelope": {
                "objective_task": self.objective_task,
                "search_topology": self.search_topology,
                "verifier_kind": self.verifier_kind,
            },
            "envelope_hash": canon_hash(
                self.family, json.dumps(self.knobs, sort_keys=True)),
            "surface_hash": canon_hash(self.prose, self.question),
            "seed": self.seed,
            "knobs": self.knobs,
            "provenance": {"proposal_id": proposal_id,
                           "generator": "ace_gen.v1"},
        }


def answer_text(completion: str) -> str:
    """Last 'Answer: ...' line from the visible portion, else visible tail."""
    if "</think>" in completion:
        completion = completion.split("</think>")[-1]
    hits = ANSWER_RE.findall(completion)
    return (hits[-1] if hits else completion).strip()


def norm_ws(s: str) -> str:
    return re.sub(r"\s+", " ", s.strip().lower())


def money(cents: int) -> str:
    sign = "-" if cents < 0 else ""
    c = abs(cents)
    return f"{sign}${c // 100}.{c % 100:02d}"


def parse_money(s: str) -> int | None:
    m = re.search(r"-?\$?\s*(\d+)\.(\d{2})", s)
    if not m:
        return None
    v = int(m.group(1)) * 100 + int(m.group(2))
    return -v if s.strip().startswith("-") else v
