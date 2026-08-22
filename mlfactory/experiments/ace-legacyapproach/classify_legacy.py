#!/usr/bin/env python3
"""Trajectory-quality classifier harness.

Reads a generations JSONL, sends each reasoning trajectory to a judge model,
parses the structured JSON verdict, and routes the original record into bucket
files based on overall_recommendation.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import re
import sys
import time
import traceback
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from mlfactory.core.api import APIClient, APIConfig, extract_json


# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------
BASE_URL = os.environ.get("ACE_CLASSIFIER_URL", "https://openrouter.ai/api/v1")
API_KEY = os.environ.get("ACE_CLASSIFIER_API_KEY", "none")
MODEL_NAME = os.environ.get("ACE_CLASSIFIER_MODEL", "qwen/qwen3.6-27b")
REQUEST_TIMEOUT = float(os.environ.get("ACE_CLASSIFIER_TIMEOUT", "1200"))
MAX_RETRIES = int(os.environ.get("ACE_CLASSIFIER_RETRIES", "3"))
BACKOFF_BASE = float(os.environ.get("ACE_CLASSIFIER_BACKOFF", "2.0"))
DEFAULT_MAX_TOKENS = int(os.environ.get("ACE_CLASSIFIER_MAX_TOKENS", "65536"))
LOOP_DELAY = float(os.environ.get("ACE_CLASSIFIER_LOOP_DELAY", "0.3"))
SITE_URL = os.environ.get("ACE_CLASSIFIER_SITE_URL", "https://localhost")
APP_NAME = os.environ.get("ACE_CLASSIFIER_APP_NAME", "ace-baseline-classifier")
EXTRA_BODY = os.environ.get("ACE_CLASSIFIER_EXTRA_BODY", "")
NO_THINK_SUFFIX = os.environ.get("ACE_CLASSIFIER_NO_THINK_SUFFIX", "/nothink")

BUCKET_DIR = Path(os.environ.get("ACE_BUCKET_DIR", "buckets"))
MANIFEST_NAME = os.environ.get("ACE_MANIFEST_NAME", "classifier_manifest.json")
DEBUG_DIR = Path(os.environ.get("ACE_CLASSIFIER_DEBUG_DIR", "")) or None

DEFAULT_SYSTEM_PROMPT = """# ACE Baseline Trajectory Review

You are reviewing **one model-generated reasoning trajectory in isolation**.

The supplied text is the output of another model. It is not a request for you to solve the original problem.

Your task is to determine whether the trajectory is a valuable candidate for an experiment studying **trajectory-preserving reasoning editing**.

Return **only one valid JSON object** matching the required schema.

---

# Research Context

The experiment studies this hypothesis:

> Reasoning quality may depend not only on the information contained in a reasoning trace, but also on the quality of the evolving autoregressive working state constructed by that trace.

A future editor will attempt to improve selected trajectories by making the reasoning state advance more consistently.

The editor may:

* remove repeated reasoning;
* consolidate repeated working state;
* remove redundant planning or narration;
* eliminate duplicate calculations;
* reduce unnecessary verification;
* tighten inefficient exploration;
* preserve useful corrections while removing correction loops;
* improve continuity between reasoning steps.

The editor may not:

* invent a different solution;
* replace the trajectory with a hindsight-optimal derivation;
* fabricate missing reasoning;
* silently correct a fundamentally incorrect solution;
* remove genuine uncertainty that affected the reasoning;
* remove meaningful branch exploration;
* substantially change the discovery order;
* erase necessary verification or closure.

The desired source trajectory therefore contains a recognizable reasoning process that can be improved **without being replaced**.

---

# Critical Evaluation Rule

You must **measure first and decide second**.

Do not begin by deciding whether the example should be kept or rejected.

First characterize the actual trajectory. Only after those observations are fixed may you apply the corpus-selection policy and assign an overall recommendation.

Two questions must remain independent:

1. **What editable behavior exists in the emitted trajectory?**
2. **Is that behavior experimentally valuable enough to include?**

A trajectory may be highly redundant even when the underlying problem is trivial.

In that case:

* mark the redundancy and editorial opportunity accurately;
* then reject separately because the task is too trivial.

Do not lower `editorial_opportunity` merely to justify a REJECT decision.

Judge the actual emitted trajectory, not the minimum reasoning theoretically required to solve the problem.

---

# Evaluation Order

Perform the assessment in this order:

1. Determine task depth.
2. Identify the distinct reasoning advances.
3. Identify repeated or non-advancing portions.
4. Characterize the reasoning arc.
5. Identify observed behaviors and the dominant pathology.
6. Assess editorial opportunity.
7. Assess rewrite risk.
8. Apply the selection policy.
9. Assign KEEP, BORDERLINE, or REJECT.

The output schema follows this order intentionally.

---

# Definitions

## Distinct Reasoning Advance

A span makes a distinct reasoning advance when it does at least one of the following:

* identifies a necessary fact or constraint;
* creates a useful representation of the problem;
* derives a new intermediate result;
* performs a necessary calculation;
* eliminates a live alternative;
* selects or changes a strategy for a stated reason;
* incorporates new evidence;
* detects and corrects an actual mistake;
* verifies a materially uncertain conclusion;
* consolidates scattered information needed for later reasoning;
* establishes justified closure.

Merely rephrasing or announcing one of these actions does not count as a new advance.

## Non-Advancing Span

A span is non-advancing when it does not materially change, test, correct, or consolidate the active reasoning state.

Examples include:

* restating the prompt;
* relisting already-established values;
* repeating the same causal chain;
* announcing a plan after the plan is already evident;
* drafting and redrafting the same answer;
* repeating a calculation without a new reason;
* checking an already-settled conclusion multiple times;
* narrating obvious transitions;
* repeatedly confirming compliance with the prompt;
* reconstructing state that was already available.

## Editorial Opportunity

Editorial opportunity measures how much the emitted trajectory could be improved while preserving its existing reasoning strategy and discovery path.

It does **not** measure task difficulty.

A trivial task may still produce a highly redundant trajectory with high editorial opportunity.

## Rewrite Risk

Rewrite risk measures how likely it is that an editor would need to invent missing reasoning, replace the method, or repair a fundamentally broken solution.

---

# Stage 1: Objective Trajectory Characterization

## task_depth

Assess the reasoning demands of the original task itself.

Choose exactly one:

* `trivial`
* `moderate`
* `substantial`

Use `trivial` when the task can be solved through one obvious fact, one elementary calculation, or one direct causal link.

Use `moderate` when several dependent steps, constraints, or choices are required.

Use `substantial` when successful completion requires extended reasoning, meaningful search, multiple interacting constraints, non-obvious derivation, debugging, or long-horizon state maintenance.

---

## distinct_reasoning_moves

Return an integer estimate of the number of materially distinct reasoning advances in the trajectory.

Count conceptual advances, not sentences or numbered headings.

Several paragraphs that restate the same inference count as one move.

---

## nonadvancing_span_count

Return an integer estimate of the number of substantial spans that repeat, narrate, reconstruct, or verify already-settled state without making a new contribution.

A span may be a sentence, bullet group, paragraph, numbered section, or tightly connected block.

Do not count tiny stylistic repetitions individually.

---

## trajectory_redundancy

Choose exactly one:

* `none`
* `low`
* `moderate`
* `high`

Use:

* `none` when essentially every span contributes;
* `low` when only minor trimming is possible;
* `moderate` when several meaningful spans can be consolidated or removed;
* `high` when large portions repeat established reasoning, plans, calculations, verification, or answer construction.

---

## reasoning_arc

Assess whether the trajectory contains a recognizable path from initial uncertainty to a conclusion.

Choose exactly one:

* `strong`
* `moderate`
* `weak`
* `absent`

A `strong` arc contains a coherent progression and enough support for its conclusion, even if it is inefficient.

A `moderate` arc is mostly coherent but contains gaps, weak transitions, or incomplete closure.

A `weak` arc contains fragments of useful reasoning but lacks a reliably traversable progression.

Use `absent` when no meaningful reasoning process is present.

---

## arc_components

Return all applicable values:

* `problem_state_construction`
* `strategy_selection`
* `derivation_or_search`
* `branch_management`
* `productive_correction`
* `material_verification`
* `state_consolidation`
* `candidate_answer`
* `justified_closure`

Include a component only when it meaningfully appears in the trajectory.

Do not infer missing components merely because the final answer is correct.

---

## observed_behaviors

Return every behavior that applies.

Possible values:

* `repeated_reasoning`
* `repeated_state_reconstruction`
* `repeated_planning`
* `repeated_verification`
* `duplicate_calculation`
* `redundant_narration`
* `productive_self_correction`
* `unproductive_self_correction`
* `correction_spiral`
* `strategy_change`
* `strategy_oscillation`
* `branch_reopening`
* `under_verification`
* `over_verification`
* `premature_closure`
* `weak_closure`
* `state_inconsistency`
* `contradiction`
* `malformed_recovery_loop`
* `verbose_but_coherent`
* `already_concise`
* `little_reasoning_present`
* `other`

`already_concise` and `verbose_but_coherent` are mutually exclusive.

Do not use `already_concise` merely because the task or solution concept is simple. Use it only when the actual emitted trajectory contains little removable repetition, planning, verification, or narration.

---

## dominant_pathology

Choose the single most important trajectory behavior.

Possible values:

* `repeated_state_reconstruction`
* `repeated_reasoning`
* `repeated_planning`
* `repeated_verification`
* `duplicate_calculation`
* `redundant_narration`
* `productive_self_correction`
* `correction_spiral`
* `strategy_oscillation`
* `branch_reopening`
* `under_verification`
* `over_verification`
* `premature_closure`
* `weak_closure`
* `state_inconsistency`
* `malformed_recovery_loop`
* `minimal_editorial_opportunity`
* `little_reasoning_present`
* `other`

Choose the behavior that best explains the trajectory's principal editorial opportunity or limitation.

---

## editorial_evidence

Return an array containing **one to three short observations** grounded in the actual trajectory.

Each observation should identify a concrete repeated, advancing, correcting, verifying, or missing behavior.

Good examples:

* `"The same predator-prey causal chain is stated in sections 3, 4, 5, 7, 8, and 9."`
* `"The trajectory recalculates the same division three times after the result is already established."`
* `"The model abandons one implementation strategy and adopts a simpler one after identifying the coordinate mapping."`

Do not provide vague claims such as `"The reasoning is verbose."`

Do not quote more text than necessary.

---

## editorial_opportunity

Estimate how much meaningful improvement is possible without replacing the reasoning strategy.

Choose exactly one:

* `none`
* `low`
* `moderate`
* `high`

Use:

* `none` when no useful trajectory-preserving transformation is available;
* `low` when only superficial trimming is possible;
* `moderate` when several non-advancing spans can be consolidated or removed;
* `high` when substantial portions can be improved while preserving a coherent and useful reasoning arc.

Task triviality must not reduce this rating.

---

## rewrite_risk

Choose exactly one:

* `low`
* `moderate`
* `high`

Use:

* `low` when an editor can improve the trace while clearly preserving its method;
* `moderate` when some gaps or ambiguity make preservation uncertain;
* `high` when producing a coherent improved trajectory would require inventing major reasoning, changing strategy, or repairing a fundamentally broken solution.

---

# Stage 2: Corpus-Selection Decision

Only now apply the selection policy.

## KEEP

Choose `KEEP` when:

* task depth is moderate or substantial;
* the reasoning arc is strong or usable;
* editorial opportunity is moderate or high;
* rewrite risk is low or manageable;
* editing can preserve the trajectory rather than replace it.

## BORDERLINE

Choose `BORDERLINE` when:

* the example may contain useful behavior but has meaningful uncertainty;
* task depth, arc quality, verification, or rewrite risk is marginal;
* the example may be valuable for analysis, DPO, or corrective distillation rather than ordinary trajectory-preserving SFT;
* a second review would materially help.

## REJECT

Choose `REJECT` when any of the following dominates:

* the task is too trivial to provide meaningful experimental value;
* little reasoning is present;
* the trajectory is already close to optimal;
* the reasoning is irrecoverably incoherent;
* rewriting would require inventing a new solution;
* the result cannot be reasonably verified;
* the trajectory lacks a usable reasoning arc.

A REJECT trajectory may still have high redundancy or high editorial opportunity. Preserve those measurements accurately.

---

## overall_recommendation

Choose exactly one:

* `KEEP`
* `BORDERLINE`
* `REJECT`

---

## primary_reason

Choose exactly one:

* `strong_candidate`
* `too_trivial`
* `already_efficient`
* `irrecoverably_incoherent`
* `requires_new_solution`
* `unverifiable`
* `little_reasoning_present`
* `marginal_candidate`
* `other`

---

## selection_summary

Write two to four sentences.

Sentence requirements:

1. Describe the trajectory's actual reasoning and editable behavior.
2. State separately why it should be kept, rejected, or treated as borderline.

Do not describe a redundant trajectory as concise merely because the underlying task is easy.

---

## confidence

Choose exactly one:

* `high`
* `medium`
* `low`

---

# Consistency Rules

The JSON must satisfy all of these rules:

1. `already_concise` and `verbose_but_coherent` cannot both appear.
2. `trajectory_redundancy: high` cannot normally coexist with `editorial_opportunity: none` or `low`.
3. Several repeated plans, formulations, calculations, or checks require at least `trajectory_redundancy: moderate`.
4. `too_trivial` may determine `overall_recommendation`, but it must not lower the measured redundancy or editorial opportunity.
5. A correct final answer does not automatically imply a strong reasoning arc.
6. A long trajectory does not automatically imply high editorial opportunity.
7. Useful correction must not be labeled redundant merely because the final answer is known.
8. Verification is redundant only when it no longer addresses a plausible unresolved error.
9. High rewrite risk should generally prevent `KEEP`.
10. `minimal_editorial_opportunity` should be used only when the actual trajectory—not merely the ideal solution—is already efficient.
11. Your `editorial_evidence` must support the selected pathology and redundancy rating.
12. Do not alter observational fields to make them appear consistent with the final recommendation.

Before returning the JSON, check it against these rules and correct any contradiction.

---

# Required Output Schema

Return only valid JSON in this exact structure:

```json
{
  "task_depth": "trivial | moderate | substantial",
  "distinct_reasoning_moves": 0,
  "nonadvancing_span_count": 0,
  "trajectory_redundancy": "none | low | moderate | high",
  "reasoning_arc": "strong | moderate | weak | absent",
  "arc_components": [],
  "observed_behaviors": [],
  "dominant_pathology": "",
  "editorial_evidence": [],
  "editorial_opportunity": "none | low | moderate | high",
  "rewrite_risk": "low | moderate | high",
  "overall_recommendation": "KEEP | BORDERLINE | REJECT",
  "primary_reason": "",
  "selection_summary": "",
  "confidence": "high | medium | low"
}
```

Do not include markdown, commentary, analysis, headings, or text outside the JSON object.

/nothink
"""

REQUIRED_FIELDS = [
    "task_depth",
    "distinct_reasoning_moves",
    "nonadvancing_span_count",
    "trajectory_redundancy",
    "reasoning_arc",
    "arc_components",
    "observed_behaviors",
    "dominant_pathology",
    "editorial_evidence",
    "editorial_opportunity",
    "rewrite_risk",
    "overall_recommendation",
    "primary_reason",
    "selection_summary",
    "confidence",
]

INTEGER_FIELDS = {"distinct_reasoning_moves", "nonadvancing_span_count"}
FREE_TEXT_FIELDS = {"selection_summary"}

ENUMS = {
    "task_depth": {"trivial", "moderate", "substantial"},
    "trajectory_redundancy": {"none", "low", "moderate", "high"},
    "reasoning_arc": {"strong", "moderate", "weak", "absent"},
    "editorial_opportunity": {"none", "low", "moderate", "high"},
    "rewrite_risk": {"low", "moderate", "high"},
    "overall_recommendation": {"KEEP", "BORDERLINE", "REJECT"},
    "confidence": {"high", "medium", "low"},
    "primary_reason": {
        "strong_candidate", "too_trivial", "already_efficient",
        "irrecoverably_incoherent", "requires_new_solution", "unverifiable",
        "little_reasoning_present", "marginal_candidate", "other",
    },
    "arc_components": {
        "problem_state_construction", "strategy_selection", "derivation_or_search",
        "branch_management", "productive_correction", "material_verification",
        "state_consolidation", "candidate_answer", "justified_closure",
    },
    "observed_behaviors": {
        "repeated_reasoning", "repeated_state_reconstruction", "repeated_planning",
        "repeated_verification", "duplicate_calculation", "redundant_narration",
        "productive_self_correction", "unproductive_self_correction", "correction_spiral",
        "strategy_change", "strategy_oscillation", "branch_reopening",
        "under_verification", "over_verification", "premature_closure", "weak_closure",
        "state_inconsistency", "contradiction", "malformed_recovery_loop",
        "verbose_but_coherent", "already_concise", "little_reasoning_present", "other",
    },
    "dominant_pathology": {
        "repeated_state_reconstruction", "repeated_reasoning", "repeated_planning",
        "repeated_verification", "duplicate_calculation", "redundant_narration",
        "productive_self_correction", "correction_spiral", "strategy_oscillation",
        "branch_reopening", "under_verification", "over_verification",
        "premature_closure", "weak_closure", "state_inconsistency",
        "malformed_recovery_loop", "minimal_editorial_opportunity",
        "little_reasoning_present", "other",
    },
}


def now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


def load_jsonl(path: Path) -> list[dict]:
    records = []
    if not path.exists():
        return records
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            records.append(json.loads(line))
    return records


def append_jsonl(path: Path, record: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as f:
        f.write(json.dumps(record, ensure_ascii=False) + "\n")
        f.flush()
        os.fsync(f.fileno())


def write_json_atomic(path: Path, data: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    with tmp.open("w", encoding="utf-8") as f:
        json.dump(data, f, indent=2, ensure_ascii=False)
        f.flush()
        os.fsync(f.fileno())
    tmp.replace(path)


def extract_trajectory(record: dict) -> tuple[str | None, str | None]:
    """Return (original_user_prompt, trajectory_text) from a generation record."""
    messages = record.get("messages", [])
    user_prompt = None
    for m in messages:
        if m.get("role") == "user":
            user_prompt = m.get("content")
            break

    trajectory = record.get("raw_model_output")
    if trajectory is None:
        reasoning = record.get("reasoning_content") or ""
        final = record.get("final_answer_content") or ""
        if reasoning:
            trajectory = f"<think>\n{reasoning}\n</think>\n\n{final}"
        elif final:
            trajectory = final

    return user_prompt, trajectory


def build_user_message(record: dict) -> str:
    user_prompt, trajectory = extract_trajectory(record)
    if not trajectory:
        raise ValueError(f"Record {record.get('sample_id')} has no usable prompt or trajectory")

    parts = [
        "You are reviewing a single model-generation record. "
        "The text below is the model's own output, not a message from a human user.",
    ]

    if user_prompt:
        parts.append("## Original problem presented to the model\n\n```text\n" + user_prompt + "\n```")

    parts.append("## Model's reasoning trajectory (the item under review)\n\n```text\n" + trajectory + "\n```")

    # Include record metadata (without duplicating the trajectory).
    meta = {
        k: v for k, v in record.items()
        if k not in {"raw_model_output", "reasoning_content", "final_answer_content", "messages"}
    }
    parts.append("## Generation record metadata (JSON)\n\n```json\n" + json.dumps(meta, ensure_ascii=False, indent=2) + "\n```")

    return "\n\n".join(parts)


def parse_json_response(text: str | None) -> dict:
    """Extract and parse the JSON object from model output."""
    if text is None:
        raise ValueError("Model returned None content")
    text = text.strip()
    if not text:
        raise ValueError("Empty response")

    if text.startswith("```"):
        text = re.sub(r"^```(?:json)?\s*", "", text, flags=re.IGNORECASE)
        text = re.sub(r"\s*```$", "", text)
        text = text.strip()

    try:
        return json.loads(text)
    except json.JSONDecodeError:
        pass

    match = re.search(r"\{.*\}", text, re.DOTALL)
    if match:
        return json.loads(match.group(0))

    raise ValueError("No JSON object found in response")


def normalize_verdict(verdict: dict) -> dict:
    """Validate and normalize enum/integer fields; coerce invalid values safely."""
    normalized = {}
    for field in REQUIRED_FIELDS:
        value = verdict.get(field)
        if field in INTEGER_FIELDS:
            try:
                normalized[field] = int(value) if value is not None else 0
            except (TypeError, ValueError):
                normalized[field] = 0
        elif field in FREE_TEXT_FIELDS:
            normalized[field] = value if isinstance(value, str) else ""
        elif field == "editorial_evidence":
            if not isinstance(value, list):
                value = [value] if value else []
            normalized[field] = [v if isinstance(v, str) else str(v) for v in value]
        elif field in ("arc_components", "observed_behaviors"):
            if not isinstance(value, list):
                value = [value] if value else []
            allowed = ENUMS[field]
            value = [v if v in allowed else "other" for v in value]
            seen = set()
            value = [v for v in value if not (v in seen or seen.add(v))]
            normalized[field] = value
        else:
            allowed = ENUMS[field]
            if value not in allowed:
                value = "other"
            normalized[field] = value

    # Enforce consistency rule: already_concise and verbose_but_coherent are mutually exclusive.
    behaviors = set(normalized.get("observed_behaviors", []))
    if "already_concise" in behaviors and "verbose_but_coherent" in behaviors:
        # Prefer the more specific observation; remove already_concise.
        behaviors.remove("already_concise")
        normalized["observed_behaviors"] = [b for b in normalized["observed_behaviors"] if b != "already_concise"]

    return normalized


def classify_record(
    client: APIClient,
    record: dict,
    system_prompt: str,
    temperature: float,
    max_tokens: int,
    extra_body: dict | None,
    use_json_mode: bool,
) -> dict:
    """Call the classifier model and return the parsed + normalized verdict."""
    user_content = build_user_message(record)
    messages = [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": user_content},
    ]

    def log_debug(exc: Exception, raw_text: str | None = None) -> None:
        if not DEBUG_DIR:
            return
        DEBUG_DIR.mkdir(parents=True, exist_ok=True)
        sid = record.get("sample_id", "unknown")
        path = DEBUG_DIR / f"{sid}_attempt1.json"
        payload = {
            "sample_id": sid,
            "attempt": 1,
            "timestamp": now_iso(),
            "raw_content": raw_text,
            "error": {"type": type(exc).__name__, "message": str(exc)},
        }
        try:
            write_json_atomic(path, payload)
        except Exception:
            pass

    try:
        response_format = {"type": "json_object"} if use_json_mode else None
        text = client.chat_completion(
            messages=messages,
            temperature=temperature,
            max_tokens=max_tokens,
            response_format=response_format,
            extra_body=extra_body,
        )
        verdict = extract_json(text)
        return normalize_verdict(verdict)
    except Exception as e:
        log_debug(e)
        print(f"  ERROR for {record.get('sample_id')}: {e}", file=sys.stderr)
        raise


def bucket_path(bucket_dir: Path, label: str) -> Path:
    return bucket_dir / f"{label.upper()}.jsonl"


def main() -> None:
    parser = argparse.ArgumentParser(description="Classify reasoning trajectories into buckets.")
    parser.add_argument("input", type=Path, help="Input JSONL of generation records.")
    parser.add_argument("--bucket-dir", type=Path, default=BUCKET_DIR, help="Directory for bucket JSONL files.")
    parser.add_argument("--system-prompt", type=Path, default=None, help="Path to custom system prompt (txt/md).")
    parser.add_argument("--base-url", default=BASE_URL, help="OpenAI-compatible base URL.")
    parser.add_argument("--model", default=MODEL_NAME, help="Model name/alias.")
    parser.add_argument("--api-key", default=API_KEY, help="API key.")
    parser.add_argument("--temperature", type=float, default=0.6, help="Sampling temperature.")
    parser.add_argument("--max-tokens", type=int, default=DEFAULT_MAX_TOKENS, help="Max tokens for classifier response.")
    parser.add_argument("--no-json-mode", action="store_true", help="Do not request JSON response_format.")
    parser.add_argument("--site-url", default=SITE_URL, help="HTTP-Referer header for OpenRouter.")
    parser.add_argument("--app-name", default=APP_NAME, help="X-Title header for OpenRouter.")
    parser.add_argument("--extra-body", default=EXTRA_BODY, help='JSON extra_body to send.')
    parser.add_argument("--resume", action="store_true", help="Skip records already present in bucket files.")
    parser.add_argument("--max-records", type=int, default=None, help="Process only first N records.")
    parser.add_argument("--dry-run", action="store_true", help="Show prompts and planned buckets without calling the model.")
    parser.add_argument("--manifest-name", default=MANIFEST_NAME, help="Manifest filename.")
    args = parser.parse_args()

    system_prompt = DEFAULT_SYSTEM_PROMPT
    if args.system_prompt:
        system_prompt = args.system_prompt.read_text(encoding="utf-8")
    if NO_THINK_SUFFIX and not system_prompt.rstrip().endswith(NO_THINK_SUFFIX):
        system_prompt = system_prompt.rstrip() + "\n\n" + NO_THINK_SUFFIX

    bucket_dir = args.bucket_dir
    bucket_dir.mkdir(parents=True, exist_ok=True)

    records = load_jsonl(args.input)
    if args.max_records:
        records = records[: args.max_records]

    already_done: set[str] = set()
    if args.resume:
        for label in ENUMS["overall_recommendation"]:
            path = bucket_path(bucket_dir, label)
            for rec in load_jsonl(path):
                already_done.add(rec.get("sample_id", ""))
        print(f"Resuming: {len(already_done)} sample_ids already in buckets.")

    client = APIClient(
        APIConfig(
            base_url=args.base_url,
            api_key=args.api_key,
            model=args.model,
            timeout=REQUEST_TIMEOUT,
            max_retries=MAX_RETRIES,
            backoff_base=BACKOFF_BASE,
            site_url=args.site_url,
            app_name=args.app_name,
        )
    )

    counts = Counter()
    error_count = 0
    start_time = time.perf_counter()
    extra_body = None

    for idx, record in enumerate(records, 1):
        sample_id = record.get("sample_id", f"record_{idx}")
        if sample_id in already_done:
            print(f"[{idx}/{len(records)}] SKIP {sample_id}: already classified")
            continue

        if record.get("error") is not None:
            print(f"[{idx}/{len(records)}] SKIP {sample_id}: generation error ({record['error'].get('type', 'unknown')})")
            error_bucket = bucket_path(bucket_dir, "NO_TRAJECTORY")
            append_jsonl(error_bucket, record)
            counts["NO_TRAJECTORY"] += 1
            continue

        _, trajectory = extract_trajectory(record)
        if not trajectory:
            print(f"[{idx}/{len(records)}] SKIP {sample_id}: no trajectory")
            error_bucket = bucket_path(bucket_dir, "NO_TRAJECTORY")
            append_jsonl(error_bucket, record)
            counts["NO_TRAJECTORY"] += 1
            continue

        print(f"[{idx}/{len(records)}] CLASSIFY {sample_id} ...", end=" ", flush=True)

        if args.dry_run:
            user_msg = build_user_message(record)
            print("\n--- SYSTEM PROMPT ---")
            print(system_prompt[:500] + "...")
            print("--- USER MESSAGE ---")
            print(user_msg[:500] + "...")
            print("--- END ---")
            counts["DRY_RUN"] += 1
            continue

        try:
            extra_body = json.loads(args.extra_body) if args.extra_body else None
            verdict = classify_record(
                client, record, system_prompt,
                args.temperature, args.max_tokens,
                extra_body, not args.no_json_mode,
            )
            label = verdict["overall_recommendation"]
            out_path = bucket_path(bucket_dir, label)

            enriched = dict(record)
            enriched["classifier_verdict"] = verdict
            enriched["classified_at"] = now_iso()
            enriched["classifier_model"] = args.model

            append_jsonl(out_path, enriched)
            counts[label] += 1
            print(f"-> {label} ({verdict.get('confidence')}, {verdict.get('editorial_opportunity')})")
        except Exception as e:
            error_count += 1
            counts["ERROR"] += 1
            error_bucket = bucket_path(bucket_dir, "ERROR")
            enriched = dict(record)
            enriched["classifier_error"] = {
                "type": type(e).__name__,
                "message": str(e),
                "traceback": traceback.format_exc(),
            }
            enriched["classified_at"] = now_iso()
            append_jsonl(error_bucket, enriched)
            print(f"-> ERROR ({e})")

    elapsed = time.perf_counter() - start_time

    manifest = {
        "created_at": now_iso(),
        "input": str(args.input.resolve()),
        "bucket_dir": str(bucket_dir.resolve()),
        "model": args.model,
        "base_url": args.base_url,
        "temperature": args.temperature,
        "max_tokens": args.max_tokens,
        "system_prompt_length": len(system_prompt),
        "site_url": args.site_url,
        "app_name": args.app_name,
        "extra_body": extra_body,
        "json_mode": not args.no_json_mode,
        "records_processed": len(records),
        "counts": dict(counts),
        "errors": error_count,
        "duration_seconds": round(elapsed, 2),
    }
    manifest_path = bucket_dir / args.manifest_name
    write_json_atomic(manifest_path, manifest)

    print("\nDone.")
    print(f"Counts: {dict(counts)}")
    print(f"Errors: {error_count}")
    print(f"Manifest: {manifest_path}")


if __name__ == "__main__":
    main()
