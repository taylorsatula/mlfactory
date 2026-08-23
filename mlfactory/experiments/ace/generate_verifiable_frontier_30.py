#!/usr/bin/env python3
"""Author 30 objectively checkable Madlibz reasoning candidates.

This script deliberately performs no judging, solving, culling, or rewriting.
Each row is produced by sample_envelope -> authoring_messages -> JSON extraction
-> freeze_authored, and the envelope plan is written alongside the frozen rows.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import re
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import requests
import yaml

from mlfactory.core.api import extract_json
from mlfactory.core.madlibz import (
    VERIFIABLE_DOMAIN_PROFILES,
    authoring_messages,
    freeze_authored,
    sample_envelope,
)

ROOT = Path(__file__).resolve().parents[3]
DEFAULT_OUTPUT = ROOT / "mlfactory" / "experiments" / "ace" / "data" / "madlibz_verifiable_frontier_30.jsonl"
DEFAULT_PLAN = ROOT / "mlfactory" / "experiments" / "ace" / "data" / "madlibz_verifiable_frontier_30_manifest.json"
CHAT_URL = "https://gw.lunaroute.com/v1/chat/completions"
MODELS_URL = "https://gw.lunaroute.com/v1/models"
MODEL_PREFERENCE = ["kimi-k3-flex"]

# Deliberately explicit control mixture: no candidate is a blind sibling of
# another, and the mechanism is provenance rather than an instruction to the
# authoring model to script solver behavior.
PLAN: tuple[dict[str, str], ...] = (
    {"domain": "route_and_timetable", "task": "constraint_satisfaction", "topology": "delayed_constraint_conflict", "verifier": "constraint_checker", "mechanism": "A feasible-looking transfer consumes the only capacity needed by a later mandatory leg."},
    {"domain": "route_and_timetable", "task": "algorithm_trace", "topology": "dependency_chain", "verifier": "state_transition_replay", "mechanism": "Each departure choice changes the reachable arrival window for the next transfer."},
    {"domain": "route_and_timetable", "task": "combinatorial_reasoning", "topology": "deceptive_local_optimum", "verifier": "finite_count_or_optimum", "mechanism": "The shortest first leg blocks a globally better route through a scarce connection."},
    {"domain": "resource_scheduling", "task": "constraint_satisfaction", "topology": "delayed_constraint_conflict", "verifier": "exact_assignment", "mechanism": "A locally harmless slot assignment leaves an unavoidable collision at the final shared resource."},
    {"domain": "resource_scheduling", "task": "combinatorial_reasoning", "topology": "representation_change", "verifier": "finite_count_or_optimum", "mechanism": "Interval choices become tractable only after converting them into an overlap or precedence structure."},
    {"domain": "warehouse_operations", "task": "algorithm_trace", "topology": "bookkeeping_state_interaction", "verifier": "state_transition_replay", "mechanism": "Reservations, substitutions, and picks mutate shared stock, so independent order arithmetic gives the wrong state."},
    {"domain": "warehouse_operations", "task": "evidence_reconciliation", "topology": "competing_hypotheses", "verifier": "hypothesis_elimination", "mechanism": "Several inventory explanations fit the first scan, but later movement records eliminate all but one."},
    {"domain": "graph_networks", "task": "combinatorial_reasoning", "topology": "representation_change", "verifier": "finite_count_or_optimum", "mechanism": "The narrative links must be represented as a graph before the minimum or count is visible."},
    {"domain": "graph_networks", "task": "constraint_satisfaction", "topology": "deceptive_local_optimum", "verifier": "exact_assignment", "mechanism": "Choosing the apparently cheapest edge early prevents a complete assignment under a later degree constraint."},
    {"domain": "graph_networks", "task": "formal_logic", "topology": "representation_change", "verifier": "truth_table_or_model_check", "mechanism": "Connectivity claims must be translated into a finite set of logical conditions and checked together."},
    {"domain": "logic_records", "task": "formal_logic", "topology": "competing_hypotheses", "verifier": "truth_table_or_model_check", "mechanism": "Multiple record assignments satisfy the opening clues until a quantified exception separates them."},
    {"domain": "logic_records", "task": "constraint_satisfaction", "topology": "cross_source_reconciliation", "verifier": "exact_assignment", "mechanism": "Two imperfect label lists must be joined before the uniqueness constraints can be applied."},
    {"domain": "state_machine_workflow", "task": "algorithm_trace", "topology": "bookkeeping_state_interaction", "verifier": "state_transition_replay", "mechanism": "Repeated events and acknowledgements mutate the workflow, making an event-count shortcut invalid."},
    {"domain": "state_machine_workflow", "task": "stateful_planning", "topology": "dependency_chain", "verifier": "constraint_checker", "mechanism": "Actions unlock later guards in sequence, and one premature action makes the target state unreachable."},
    {"domain": "state_machine_workflow", "task": "formal_logic", "topology": "adversarial_edge_case", "verifier": "truth_table_or_model_check", "mechanism": "The ordinary transition rule fails only for a boundary event at an exact state and time."},
    {"domain": "ledger_reconciliation", "task": "evidence_reconciliation", "topology": "cross_source_reconciliation", "verifier": "hypothesis_elimination", "mechanism": "Overlapping exports disagree on identifiers and dates, and only one adjustment history balances every account."},
    {"domain": "ledger_reconciliation", "task": "combinatorial_reasoning", "topology": "bookkeeping_state_interaction", "verifier": "finite_count_or_optimum", "mechanism": "A small set of reversals and transfers interact, so treating each discrepancy independently double-counts state."},
    {"domain": "algorithm_data_structures", "task": "algorithm_trace", "topology": "adversarial_edge_case", "verifier": "state_transition_replay", "mechanism": "A boundary operation exposes an aliasing or ordering effect absent from ordinary inputs."},
    {"domain": "algorithm_data_structures", "task": "debugging_analysis", "topology": "representation_change", "verifier": "test_case_execution", "mechanism": "The visible output symptom disappears when the data is represented as the actual queue or stack state."},
    {"domain": "access_policy_rules", "task": "formal_logic", "topology": "competing_hypotheses", "verifier": "truth_table_or_model_check", "mechanism": "Several policy interpretations agree for normal users, while a paired exception distinguishes the valid one."},
    {"domain": "access_policy_rules", "task": "constraint_satisfaction", "topology": "delayed_constraint_conflict", "verifier": "exact_assignment", "mechanism": "Granting a role satisfies an immediate request but violates a later separation-of-duties rule."},
    {"domain": "debugging_fixture", "task": "debugging_analysis", "topology": "delayed_constraint_conflict", "verifier": "test_case_execution", "mechanism": "The obvious patch fixes the shown case but fails a supplied interaction test that runs later."},
    {"domain": "debugging_fixture", "task": "debugging_analysis", "topology": "competing_hypotheses", "verifier": "test_case_execution", "mechanism": "Two root causes explain the first failure, and a second deterministic test separates them."},
    {"domain": "set_cover_and_selection", "task": "combinatorial_reasoning", "topology": "deceptive_local_optimum", "verifier": "finite_count_or_optimum", "mechanism": "The item covering the most requirements first forces an extra item and loses to a less impressive pair."},
    {"domain": "set_cover_and_selection", "task": "constraint_satisfaction", "topology": "delayed_constraint_conflict", "verifier": "constraint_checker", "mechanism": "Coverage looks complete until budget, incompatibility, and mandatory inclusion are checked jointly."},
    {"domain": "adversarial_testing", "task": "adversarial_verification", "topology": "adversarial_edge_case", "verifier": "counterexample_checker", "mechanism": "The claimed invariant survives ordinary examples but fails at a smallest boundary witness."},
    {"domain": "adversarial_testing", "task": "adversarial_verification", "topology": "representation_change", "verifier": "counterexample_checker", "mechanism": "A counterexample is hard to see in the prose but immediate after converting the rule to a finite table."},
    {"domain": "route_and_timetable", "task": "evidence_reconciliation", "topology": "cross_source_reconciliation", "verifier": "hypothesis_elimination", "mechanism": "A dispatch log, passenger list, and delay notice each contain partial truth with one stale field."},
    {"domain": "state_machine_workflow", "task": "formal_logic", "topology": "competing_hypotheses", "verifier": "truth_table_or_model_check", "mechanism": "Two possible current states explain the visible event prefix until a guarded event is evaluated."},
    {"domain": "ledger_reconciliation", "task": "stateful_planning", "topology": "bookkeeping_state_interaction", "verifier": "constraint_checker", "mechanism": "The order of finite corrections changes intermediate balances and determines which correction remains legal."},
)


def now() -> str:
    return datetime.now(timezone.utc).isoformat()


def load_key() -> str:
    key = os.environ.get("LUNAROUTE_API_KEY")
    if key:
        return key
    secrets = ROOT / ".mlfactory" / "secrets.yaml"
    if secrets.exists():
        data = yaml.safe_load(secrets.read_text()) or {}
        key = data.get("LUNAROUTE_API_KEY")
        if key:
            return str(key)
    raise RuntimeError("LUNAROUTE_API_KEY is not set and .mlfactory/secrets.yaml has no key")


def choose_model(key: str) -> str:
    response = requests.get(MODELS_URL, headers={"Authorization": f"Bearer {key}"}, timeout=30)
    response.raise_for_status()
    active = {item["id"] for item in response.json().get("data", [])}
    for model in MODEL_PREFERENCE:
        if model in active:
            return model
    raise RuntimeError(f"none of the preferred authoring models are active: {sorted(active)}")


def call_author(key: str, model: str, messages: list[dict[str, str]]) -> tuple[str, str]:
    payload = {
        "model": model,
        "messages": messages,
        "temperature": 0.8,
        "max_tokens": 32768,
        "response_format": {"type": "json_object"},
        "stream": True,
    }
    response = requests.post(
        CHAT_URL,
        headers={"Authorization": f"Bearer {key}", "Content-Type": "application/json"},
        json=payload,
        stream=True,
        timeout=(60, 1800),
    )
    response.raise_for_status()
    content: list[str] = []
    reasoning: list[str] = []
    for line in response.iter_lines():
        if not line:
            continue
        text = line.decode("utf-8")
        if not text.startswith("data: "):
            continue
        value = text[6:]
        if value.strip() == "[DONE]":
            break
        try:
            chunk = json.loads(value)
        except json.JSONDecodeError:
            continue
        for choice in chunk.get("choices", []):
            delta = choice.get("delta", {})
            if delta.get("content"):
                content.append(delta["content"])
            if delta.get("reasoning"):
                reasoning.append(delta["reasoning"])
            if delta.get("reasoning_content"):
                reasoning.append(delta["reasoning_content"])
    return "".join(content), "".join(reasoning)


def extract_authored(content: str, reasoning: str) -> dict[str, Any]:
    errors: list[str] = []
    for source in (content, reasoning):
        if not source.strip():
            continue
        try:
            return extract_json(source)
        except ValueError as exc:
            errors.append(str(exc))
    raise ValueError("author response contained no valid JSON: " + "; ".join(errors))


def author_one(
    index: int,
    spec: dict[str, str],
    key: str,
    model: str,
    retries: int,
    output: Path,
    lane_count: int,
) -> dict[str, Any]:
    seed = 47000 + index
    env = sample_envelope(
        seed=seed,
        domain=spec["domain"],
        mode="verifiable",
        objective_task=spec["task"],
        search_topology=spec["topology"],
        verifier_kind=spec["verifier"],
    )
    messages = authoring_messages(env)
    last_error = ""
    lane = ((index - 1) % lane_count) + 1
    print(json.dumps({
        "starting": index,
        "lane": lane,
        "seed": seed,
        "domain": env.domain,
        "envelope_hash": env.envelope_hash,
        "model": model,
        "max_tokens": 32768,
    }), flush=True)
    for attempt in range(1, retries + 1):
        try:
            content, reasoning = call_author(key, model, messages)
            authored = extract_authored(content, reasoning)
            # This is structural validation only; no semantic quality check
            # or culling is performed here.
            record = freeze_authored(
                env,
                authored,
                model=model,
                corpus="madlibz-verifiable-frontier-30",
                proposal_id=index,
                search_pressure_mechanism=spec["mechanism"],
                authoring_temperature=0.8,
                authoring_max_tokens=32768,
                authoring_attempt=attempt,
                authoring_lane=lane,
                authoring_lane_count=lane_count,
                authoring_prompt_hash=hashlib.sha256(
                    json.dumps(messages, sort_keys=True).encode()
                ).hexdigest(),
            )
            print(json.dumps({
                "proposal_id": index,
                "lane": lane,
                "seed": seed,
                "domain": env.domain,
                "envelope_hash": env.envelope_hash,
                "task": env.objective_task,
                "topology": env.search_topology,
                "verifier": env.verifier_kind,
                "mechanism": spec["mechanism"],
                "output": str(output),
            }), flush=True)
            return record
        except (ValueError, requests.RequestException, RuntimeError) as exc:
            last_error = f"{type(exc).__name__}: {exc}"
            if attempt == retries:
                raise RuntimeError(f"proposal {index} failed after {retries} attempts: {last_error}") from exc
            time.sleep(min(30, 2 ** attempt))
    raise RuntimeError(f"proposal {index} did not freeze: {last_error}")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--manifest", type=Path, default=DEFAULT_PLAN)
    parser.add_argument("--retries", type=int, default=5)
    parser.add_argument("--lanes", type=int, default=6)
    args = parser.parse_args()
    if args.lanes < 1 or args.lanes > 6:
        raise ValueError("--lanes must be between 1 and 6")
    if len(PLAN) != 30:
        raise AssertionError(f"expected 30 plan entries, got {len(PLAN)}")
    if len({item["domain"] for item in PLAN}) < 10:
        raise AssertionError("plan lacks domain diversity")

    key = load_key()
    model = choose_model(key)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.manifest.parent.mkdir(parents=True, exist_ok=True)
    if args.output.exists() or args.manifest.exists():
        raise FileExistsError("refusing to overwrite an existing artifact")

    records: dict[int, dict[str, Any]] = {}
    with ThreadPoolExecutor(max_workers=args.lanes, thread_name_prefix="madlibz-lane") as pool:
        futures = {
            pool.submit(author_one, index, spec, key, model, args.retries, args.output, args.lanes): index
            for index, spec in enumerate(PLAN, start=1)
        }
        for future in as_completed(futures):
            index = futures[future]
            records[index] = future.result()

    rows = [records[index] for index in range(1, len(PLAN) + 1)]
    with args.output.open("w", encoding="utf-8") as handle:
        for record in rows:
            handle.write(json.dumps(record, ensure_ascii=False) + "\n")

    manifest = {
        "artifact": str(args.output),
        "created_at": now(),
        "count": len(rows),
        "authoring_model": model,
        "lanes": args.lanes,
        "generation_contract": "sample_envelope -> authoring_messages -> extract_json -> freeze_authored",
        "judged_or_culled": False,
        "qwen_run": False,
        "proposals": [
            {
                "proposal_id": index,
                "seed": 47000 + index,
                "domain": spec["domain"],
                "task": spec["task"],
                "search_topology": spec["topology"],
                "verifier": spec["verifier"],
                "search_pressure_mechanism": spec["mechanism"],
                "envelope_hash": rows[index - 1]["envelope_hash"],
                "output_artifact": str(args.output),
            }
            for index, spec in enumerate(PLAN, start=1)
        ],
        "diversity_summary": {
            "domains": sorted({spec["domain"] for spec in PLAN}),
            "domain_count": len({spec["domain"] for spec in PLAN}),
            "tasks": sorted({spec["task"] for spec in PLAN}),
            "task_count": len({spec["task"] for spec in PLAN}),
            "search_topologies": sorted({spec["topology"] for spec in PLAN}),
            "search_topology_count": len({spec["topology"] for spec in PLAN}),
            "verifiers": sorted({spec["verifier"] for spec in PLAN}),
            "verifier_count": len({spec["verifier"] for spec in PLAN}),
        },
    }
    args.manifest.write_text(json.dumps(manifest, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({"completed": len(rows), "artifact": str(args.output), "manifest": str(args.manifest), "model": model}), flush=True)


if __name__ == "__main__":
    main()
