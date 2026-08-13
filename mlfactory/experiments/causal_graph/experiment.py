"""Command-line entry points for staged CausalGraph MVP work."""
from __future__ import annotations

import argparse
import json
import random
from pathlib import Path
from typing import Any

from .analysis import bootstrap_frontier, fit_depth_logistic, fit_surface, raw_depth_stats, summarize_bootstrap, write_json, write_jsonl
from .generator import TRAIN_WORLDS, WORLD_IDS, generate_task, regenerate_matches, verify_task
from .probe import CausalGraphClient, evaluate_tasks, make_coarse_tasks


def validate_generator(count: int, seed: int, output: Path) -> dict[str, Any]:
    rng = random.Random(seed)
    valid = 0
    answers = {"YES": 0, "NO": 0}
    rows = []
    for index in range(count):
        depth = rng.randint(1, 32)
        gates = rng.randint(0, min(depth, 8))
        extra = rng.randint(0, gates)
        task = generate_task(
            seed + index, depth=depth, relevant_nodes=depth + 1 + extra,
            distractor_nodes=rng.randint(0, 6), binary_gate_count=gates,
            negation_count=rng.randint(0, depth - gates), source_update_count=rng.randint(0, 3),
            world_id=WORLD_IDS[index % len(WORLD_IDS)], render_template_id=index % 3,
        )
        verify_task(task)
        if not regenerate_matches(task):
            raise AssertionError(f"deterministic regeneration failed for {task['id']}")
        valid += 1
        answers[task["canonical_answer"]] += 1
        if len(rows) < 100:
            rows.append(task)
    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open("w", encoding="utf-8") as handle:
        for task in rows:
            handle.write(json.dumps(task, ensure_ascii=False, sort_keys=True) + "\n")
    summary = {"requested": count, "valid": valid, "answer_counts": answers, "sample_path": str(output)}
    write_json(output.with_name("generator_validation_summary.json"), summary)
    return summary


def run_coarse(args: argparse.Namespace) -> dict[str, Any]:
    root = Path(args.output)
    root.mkdir(parents=True, exist_ok=True)
    tasks = make_coarse_tasks(seed=args.seed, examples_per_depth=args.examples_per_depth, worlds=TRAIN_WORLDS)
    client = CausalGraphClient(base_url=args.base_url, model=args.model, timeout=args.timeout)
    write_jsonl(root / "coarse_probe_tasks.jsonl", tasks)
    records = evaluate_tasks(tasks, client, root / "probe_dev.jsonl", max_tokens=args.max_tokens)
    stats = raw_depth_stats(records)
    fit = fit_depth_logistic(records)
    surface = fit_surface(records)
    summary = {"model": args.model, "n": len(records), "raw_depth_stats": stats, "depth_logistic": fit, "surface": surface}
    write_json(root / "coarse_probe_summary.json", summary)
    write_jsonl(root / "per_example_metrics.jsonl", records)
    return summary


def run_analysis(args: argparse.Namespace) -> dict[str, Any]:
    records = []
    with Path(args.records).open(encoding="utf-8") as handle:
        for line in handle:
            if line.strip():
                value = json.loads(line)
                # Accept either metrics JSONL or probe records containing a
                # nested model_output object.
                records.append(value.get("model_output", value))
    fit = fit_depth_logistic(records)
    stats = raw_depth_stats(records)
    surface = fit_surface(records)
    samples = bootstrap_frontier(records, B=args.bootstrap, seed=args.seed)
    summary = {"n": len(records), "raw_depth_stats": stats, "depth_logistic": fit, "bootstrap": summarize_bootstrap(samples), "surface": surface}
    output = Path(args.output)
    output.mkdir(parents=True, exist_ok=True)
    write_json(output / "frontier_fit.json", summary)
    write_jsonl(output / "bootstrap_samples.jsonl", samples)
    return summary


def main() -> None:
    parser = argparse.ArgumentParser(description="CausalGraph MVP staged experiment")
    sub = parser.add_subparsers(dest="command", required=True)
    val = sub.add_parser("validate")
    val.add_argument("--count", type=int, default=10_000)
    val.add_argument("--seed", type=int, default=20260811)
    val.add_argument("--output", type=Path, default=Path("runs/causal-graph-generator-validation/sample.jsonl"))
    coarse = sub.add_parser("coarse")
    coarse.add_argument("--output", default="runs/causal-graph-baseline")
    coarse.add_argument("--seed", type=int, default=20260811)
    coarse.add_argument("--examples-per-depth", type=int, default=64)
    coarse.add_argument("--base-url", default="http://127.0.0.1:3090/v1")
    coarse.add_argument("--model", default="f16-jackrongds4qwen")
    coarse.add_argument("--timeout", type=float, default=180.0)
    coarse.add_argument("--max-tokens", type=int, default=256)
    analysis = sub.add_parser("analyze")
    analysis.add_argument("records")
    analysis.add_argument("--output", default="runs/causal-graph-analysis")
    analysis.add_argument("--bootstrap", type=int, default=2000)
    analysis.add_argument("--seed", type=int, default=20260811)
    args = parser.parse_args()
    if args.command == "validate":
        print(json.dumps(validate_generator(args.count, args.seed, args.output), indent=2))
    elif args.command == "coarse":
        print(json.dumps(run_coarse(args), indent=2))
    else:
        print(json.dumps(run_analysis(args), indent=2))


if __name__ == "__main__":
    main()
