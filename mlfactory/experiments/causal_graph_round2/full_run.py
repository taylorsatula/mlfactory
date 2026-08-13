"""End-to-end Round 2 experiment under one fixed `/nothink` contract."""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import shutil
import subprocess
from collections import defaultdict
from pathlib import Path
from typing import Any

import numpy as np
import yaml

from .acquisition import closest_token_match, depth_matched, token_matched_batch, targeted, token_summary
from .analysis import (
    bootstrap_frontier,
    fit_depth_logistic,
    fit_surface,
    raw_depth_stats,
    summarize_bootstrap,
    write_json,
    write_jsonl,
)
from .generator import HELDOUT_WORLDS, TRAIN_WORLDS, audit_task, candidate_pool, contract_anchors, probe_tasks
from .progress import emit

DEPTHS = [1, 2, 3, 4, 5, 6, 8, 10, 12, 16, 20, 24]
BRANCHES = ("TARGETED", "RANDOM", "DEPTH_MATCHED")


def _read_jsonl(path: Path) -> list[dict[str, Any]]:
    return [json.loads(line) for line in path.open(encoding="utf-8") if line.strip()]


def _run_command(command: list[str], log: Path, repo: Path) -> None:
    environment = os.environ.copy()
    environment["HF_HUB_OFFLINE"] = "1"
    environment["PYTHONPATH"] = str(repo) + os.pathsep + environment.get("PYTHONPATH", "")
    environment["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"
    log.parent.mkdir(parents=True, exist_ok=True)
    with log.open("a", encoding="utf-8") as handle:
        handle.write("COMMAND " + " ".join(command) + "\n")
        handle.flush()
        result = subprocess.run(command, stdout=handle, stderr=subprocess.STDOUT, env=environment)
    if result.returncode:
        raise RuntimeError(f"command failed ({result.returncode}); see {log}")


def _evaluate(args: argparse.Namespace, tasks: Path, output: Path, checkpoint: str, adapter: Path | None = None) -> list[dict[str, Any]]:
    command = [
        args.training_python, "-m", "mlfactory.experiments.causal_graph_round2.evaluate_hf",
        "--base-model", args.base_model,
        "--input", str(tasks),
        "--output", str(output),
        "--checkpoint", checkpoint,
        "--max-new-tokens", str(args.max_new_tokens),
        "--batch-size", "8",
        "--dashboard-file", str(Path(args.output) / "dashboard.jsonl"),
        "--stage", f"evaluate:{checkpoint}",
    ]
    if adapter:
        command += ["--adapter", str(adapter)]
    _run_command(command, Path(args.output) / "logs" / f"{checkpoint.lower()}_eval.log", Path(args.repo))
    return _read_jsonl(output)


def _train(args: argparse.Namespace, branch: str, round_index: int, data: Path, previous: Path | None) -> tuple[Path, dict[str, Any]]:
    output = Path(args.output) / "checkpoints" / branch.lower() / f"round_{round_index}"
    command = [
        args.training_python, "-m", "mlfactory.experiments.causal_graph_round2.train_adapter",
        "--base-model", args.base_model,
        "--train-file", str(data),
        "--output-dir", str(output),
        "--seed", str(args.seed + round_index),
        "--dashboard-file", str(Path(args.output) / "dashboard.jsonl"),
        "--stage", f"train:{branch}:round_{round_index}",
    ]
    if previous:
        command += ["--init-adapter", str(previous)]
    if args.max_steps:
        command += ["--max-steps", str(args.max_steps)]
    _run_command(command, Path(args.output) / "logs" / f"{branch.lower()}_round_{round_index}_train.log", Path(args.repo))
    return output, json.loads((output / "training_manifest.json").read_text(encoding="utf-8"))


def _audit(rows: list[dict[str, Any]], tokenizer: Any) -> tuple[list[dict[str, Any]], list[dict[str, str]]]:
    accepted: list[dict[str, Any]] = []
    rejected: list[dict[str, str]] = []
    for task in rows:
        row, reason = audit_task(task, tokenizer)
        if row is None:
            rejected.append({"id": task.get("id", "unknown"), "reason": reason or "unknown"})
        else:
            accepted.append(row)
    return accepted, rejected


def _summary(records: list[dict[str, Any]]) -> dict[str, Any]:
    parsed = [row for row in records if not row["parse_failure"]]
    return {
        "frontier": fit_depth_logistic(records),
        "raw": raw_depth_stats(records),
        "n": len(records),
        "accuracy": sum(row["correct"] for row in records) / max(1, len(records)),
        "terminal_accuracy": sum(row["terminal_correct"] for row in records) / max(1, len(records)),
        "trace_bit_accuracy": sum(row["trace_bit_accuracy"] for row in records) / max(1, len(records)),
        "conditional_accuracy": sum(row["correct"] for row in parsed) / max(1, len(parsed)),
        "parse_failures": sum(row["parse_failure"] for row in records),
        "thinking_markers": sum(row["thinking_marker"] for row in records),
        "budget_exhausted": sum(row["budget_exhausted"] for row in records),
    }


def _paired_bootstrap(left: list[dict[str, Any]], right: list[dict[str, Any]], replicates: int, seed: int) -> dict[str, Any]:
    left_map = {row["example_id"]: row for row in left}
    right_map = {row["example_id"]: row for row in right}
    groups: dict[int, list[str]] = defaultdict(list)
    for example_id, row in left_map.items():
        if example_id in right_map:
            groups[int(row["depth"])].append(example_id)
    rng = np.random.default_rng(seed)
    shifts: list[float] = []
    for _ in range(replicates):
        sample: list[str] = []
        for ids in groups.values():
            sample.extend(ids[int(index)] for index in rng.integers(0, len(ids), len(ids)))
        left_fit = fit_depth_logistic([left_map[item] for item in sample])
        right_fit = fit_depth_logistic([right_map[item] for item in sample])
        if left_fit.get("d50") is not None and right_fit.get("d50") is not None:
            shifts.append(float(right_fit["d50"] - left_fit["d50"]))
    values = np.asarray(shifts)
    return {
        "replicates": replicates,
        "valid": len(shifts),
        "median": float(np.median(values)) if len(values) else None,
        "ci95": [float(np.quantile(values, 0.025)), float(np.quantile(values, 0.975))] if len(values) else None,
        "bootstrap_positive_fraction": float(np.mean(values > 0)) if len(values) else None,
        "samples": shifts,
    }


def _write_config(args: argparse.Namespace, run_dir: Path) -> None:
    config = vars(args).copy()
    write_json(run_dir / "config.json", config)
    (run_dir / "config.yaml").write_text(yaml.safe_dump(config, sort_keys=True), encoding="utf-8")
    base_config = Path(args.base_model) / "config.json"
    write_json(run_dir / "model_manifest.json", {
        "base_model": str(Path(args.base_model).resolve()),
        "config_sha256": hashlib.sha256(base_config.read_bytes()).hexdigest(),
    })


def run(args: argparse.Namespace) -> dict[str, Any]:
    from transformers import AutoTokenizer

    run_dir = Path(args.output).resolve()
    run_dir.mkdir(parents=True, exist_ok=True)
    for name in ("data", "candidates", "batches", "outputs", "logs", "checkpoints", "audits", "bootstrap"):
        (run_dir / name).mkdir(exist_ok=True)
    (run_dir / "run.pid").write_text(str(os.getpid()) + "\n", encoding="utf-8")
    dashboard = run_dir / "dashboard.jsonl"
    emit(dashboard, "run_start", stage="initialization", current=0, total=1)
    _write_config(args, run_dir)
    tokenizer = AutoTokenizer.from_pretrained(args.base_model, trust_remote_code=True)

    # Generate and hash sealed data before training, but do not evaluate it.
    sealed_raw = probe_tasks(args.seed + 20_000_000, DEPTHS, args.sealed_per_depth, tuple(TRAIN_WORLDS))
    sealed_raw += probe_tasks(args.seed + 30_000_000, DEPTHS, args.sealed_per_depth, tuple(HELDOUT_WORLDS))
    sealed, rejected = _audit(sealed_raw, tokenizer)
    if rejected:
        raise RuntimeError(f"sealed contract audit rejected {len(rejected)} tasks")
    sealed_path = run_dir / "data" / "sealed.jsonl"
    write_jsonl(sealed_path, sealed)
    (run_dir / "data" / "sealed.sha256").write_text(hashlib.sha256(sealed_path.read_bytes()).hexdigest() + "\n")

    baseline_dev_raw = probe_tasks(args.seed + 1_000_000, DEPTHS, args.dev_per_depth, tuple(TRAIN_WORLDS))
    baseline_dev_tasks, rejected = _audit(baseline_dev_raw, tokenizer)
    if rejected:
        raise RuntimeError(f"baseline development audit rejected {len(rejected)} tasks")
    baseline_dev_path = run_dir / "data" / "dev_round_0.jsonl"
    write_jsonl(baseline_dev_path, baseline_dev_tasks)
    baseline_dev = _evaluate(args, baseline_dev_path, run_dir / "outputs" / "baseline_dev.jsonl", "BASELINE_DEV")
    baseline = _summary(baseline_dev)
    write_json(run_dir / "baseline_dev_summary.json", baseline)
    frontier = baseline["frontier"]
    if not frontier["expected_decreasing"] or frontier.get("d50") is None or not (min(DEPTHS) <= frontier["d50"] <= max(DEPTHS)):
        raise RuntimeError(f"baseline frontier is not identifiable: {frontier}")
    if baseline["parse_failures"] or baseline["thinking_markers"] or baseline["budget_exhausted"]:
        raise RuntimeError(f"baseline violated fixed contract: {baseline}")

    curriculum_count = max(2, int(round(args.batch_size * args.curriculum_fraction)))
    if curriculum_count % 2:
        curriculum_count += 1
    curriculum_raw = contract_anchors(args.seed + 40_000_000, curriculum_count)
    curriculum, curriculum_rejected = _audit(curriculum_raw, tokenizer)
    write_jsonl(run_dir / "audits" / "curriculum_rejected.jsonl", curriculum_rejected)
    if len(curriculum) != curriculum_count or curriculum_rejected:
        raise RuntimeError(f"curriculum audit rejected {len(curriculum_rejected)} tasks")
    write_jsonl(run_dir / "batches" / "curriculum.jsonl", curriculum)
    curriculum_hashes = {row["graph_hash"] for row in curriculum}
    curriculum_tokens = token_summary(curriculum)
    policy_size = args.batch_size - len(curriculum)
    if policy_size <= 0 or policy_size % 2:
        raise RuntimeError("curriculum leaves an invalid policy batch size")

    adapters: dict[str, Path | None] = {branch: None for branch in BRANCHES}
    development: dict[str, list[dict[str, Any]]] = {branch: baseline_dev for branch in BRANCHES}
    acquisition_history: dict[str, list[dict[str, Any]]] = defaultdict(list)

    for round_index in range(1, args.rounds + 1):
        fit = fit_depth_logistic(development["TARGETED"])
        center = float(fit["d50"])
        emit(dashboard, "acquisition_start", stage=f"acquisition:round_{round_index}", current=0, total=3)

        targeted_raw = candidate_pool(args.seed + round_index * 100_000, max(args.candidate_pool, policy_size * 4), center=center, max_depth=args.max_depth)
        control_raw = candidate_pool(
            args.seed + 500_000 + round_index * 100_000,
            max(args.candidate_pool, policy_size * 8),
            center=center,
            max_depth=args.max_depth,
        )
        targeted_pool, targeted_rejected = _audit(targeted_raw, tokenizer)
        control_pool, control_rejected = _audit(control_raw, tokenizer)
        write_jsonl(run_dir / "audits" / f"round_{round_index}_targeted_rejected.jsonl", targeted_rejected)
        write_jsonl(run_dir / "audits" / f"round_{round_index}_control_rejected.jsonl", control_rejected)
        targeted_pool = [row for row in targeted_pool if row["graph_hash"] not in curriculum_hashes]
        control_pool = [row for row in control_pool if row["graph_hash"] not in curriculum_hashes]
        if min(len(targeted_pool), len(control_pool)) < policy_size:
            raise RuntimeError("too few contract-valid candidates")

        targeted_rows, scored = targeted(targeted_pool, fit_surface(development["TARGETED"]), policy_size, args.seed + round_index)
        if len(control_pool) < policy_size:
            raise RuntimeError("too few candidates remain for controls")
        random_rows, random_budget = closest_token_match(
            lambda attempt: token_matched_batch(control_pool, targeted_rows, args.seed + round_index * 100 + attempt),
            targeted_rows,
            attempts=4,
        )
        depth_rows, depth_budget = closest_token_match(
            lambda attempt: depth_matched(control_pool, targeted_rows, args.seed + round_index * 100 + attempt),
            targeted_rows,
            attempts=32,
        )
        batches = {
            "TARGETED": curriculum + targeted_rows,
            "RANDOM": curriculum + random_rows,
            "DEPTH_MATCHED": curriculum + depth_rows,
        }
        budget_report = {
            "CURRICULUM": curriculum_tokens,
            "TARGETED_POLICY": token_summary(targeted_rows),
            "RANDOM_POLICY": random_budget,
            "DEPTH_MATCHED_POLICY": depth_budget,
            "TOTAL": {branch: token_summary(rows) for branch, rows in batches.items()},
        }
        for branch, report in (("RANDOM_POLICY", random_budget), ("DEPTH_MATCHED_POLICY", depth_budget)):
            if max(abs(value) for value in report["relative_deltas"].values()) > args.token_tolerance:
                raise RuntimeError(f"{branch} token budget mismatch: {report}")
        write_json(run_dir / "audits" / f"round_{round_index}_token_budgets.json", budget_report)
        write_jsonl(run_dir / "candidates" / f"round_{round_index}_targeted.jsonl", scored)
        emit(dashboard, "acquisition_complete", stage=f"acquisition:round_{round_index}", current=3, total=3)

        dev_raw = probe_tasks(args.seed + 2_000_000 + round_index * 100_000, DEPTHS, args.dev_per_depth, tuple(TRAIN_WORLDS))
        dev_tasks, rejected = _audit(dev_raw, tokenizer)
        if rejected:
            raise RuntimeError(f"round {round_index} development audit rejected {len(rejected)} tasks")
        dev_path = run_dir / "data" / f"dev_round_{round_index}.jsonl"
        write_jsonl(dev_path, dev_tasks)

        for branch in BRANCHES:
            data_path = run_dir / "batches" / f"round_{round_index}_{branch.lower()}.jsonl"
            write_jsonl(data_path, batches[branch])
            adapter, training = _train(args, branch, round_index, data_path, adapters[branch])
            adapters[branch] = adapter
            outputs = _evaluate(
                args,
                dev_path,
                run_dir / "outputs" / f"round_{round_index}_{branch.lower()}_dev.jsonl",
                f"{branch}_ROUND_{round_index}",
                adapter,
            )
            summary = _summary(outputs)
            if summary["parse_failures"] or summary["thinking_markers"] or summary["budget_exhausted"]:
                raise RuntimeError(f"{branch} round {round_index} violated fixed contract: {summary}")
            development[branch] = outputs
            acquisition_history[branch].append({
                "round": round_index,
                "frontier": summary["frontier"],
                "training": training,
                "tokens": {"curriculum": curriculum_tokens, "policy": token_summary(batches[branch][len(curriculum):]), "total": token_summary(batches[branch])},
            })
            write_json(run_dir / f"round_{round_index}_{branch.lower()}_dev_summary.json", summary)

    # First contact with sealed outcomes occurs here.
    sealed_outputs: dict[str, list[dict[str, Any]]] = {
        "BASELINE": _evaluate(args, sealed_path, run_dir / "outputs" / "baseline_sealed.jsonl", "BASELINE_FINAL")
    }
    for branch in BRANCHES:
        sealed_outputs[branch] = _evaluate(
            args,
            sealed_path,
            run_dir / "outputs" / f"{branch.lower()}_sealed.jsonl",
            f"{branch}_FINAL",
            adapters[branch],
        )

    train_worlds = set(TRAIN_WORLDS)
    heldout_worlds = set(HELDOUT_WORLDS)
    train_rows = {name: [row for row in rows if row["world"] in train_worlds] for name, rows in sealed_outputs.items()}
    heldout_rows = {name: [row for row in rows if row["world"] in heldout_worlds] for name, rows in sealed_outputs.items()}
    comparisons = {
        f"{branch}_vs_BASELINE": _paired_bootstrap(train_rows["BASELINE"], train_rows[branch], args.bootstrap, args.seed + index)
        for index, branch in enumerate(BRANCHES, 1)
    }
    comparisons["TARGETED_vs_RANDOM"] = _paired_bootstrap(train_rows["RANDOM"], train_rows["TARGETED"], args.bootstrap, args.seed + 10)
    comparisons["TARGETED_vs_DEPTH_MATCHED"] = _paired_bootstrap(train_rows["DEPTH_MATCHED"], train_rows["TARGETED"], args.bootstrap, args.seed + 11)
    comparisons["TARGETED_vs_BASELINE_HELDOUT"] = _paired_bootstrap(heldout_rows["BASELINE"], heldout_rows["TARGETED"], args.bootstrap, args.seed + 12)

    result = {
        "baseline": _summary(train_rows["BASELINE"]),
        "baseline_bootstrap": summarize_bootstrap(bootstrap_frontier(train_rows["BASELINE"], args.bootstrap, args.seed)),
        "branches": {branch: _summary(train_rows[branch]) for branch in BRANCHES},
        "heldout": {name: _summary(rows) for name, rows in heldout_rows.items()},
        "comparisons": comparisons,
        "acquisition": dict(acquisition_history),
        "easy_anchor": {
            name: sum(row["correct"] for row in rows if row["depth"] == min(DEPTHS)) / max(1, sum(row["depth"] == min(DEPTHS) for row in rows))
            for name, rows in train_rows.items()
        },
    }
    write_json(run_dir / "summary.json", result)
    for name, comparison in comparisons.items():
        write_jsonl(run_dir / "bootstrap" / f"{name.lower()}.jsonl", ({"replicate": index, "delta_d50": value} for index, value in enumerate(comparison["samples"])))
    (run_dir / "final_report.md").write_text(_report(result), encoding="utf-8")
    emit(dashboard, "run_complete", stage="complete", current=1, total=1)
    return result


def _report(result: dict[str, Any]) -> str:
    target = result["comparisons"]["TARGETED_vs_BASELINE"]
    random = result["comparisons"]["TARGETED_vs_RANDOM"]
    depth = result["comparisons"]["TARGETED_vs_DEPTH_MATCHED"]
    easy_drop = result["easy_anchor"]["TARGETED"] - result["easy_anchor"]["BASELINE"]
    passed = (
        target["median"] is not None and target["median"] >= 1
        and target["ci95"][0] > 0
        and target["bootstrap_positive_fraction"] >= 0.95
        and random["ci95"][0] > 0
        and depth["ci95"][0] > 0
        and easy_drop >= -0.02
    )
    status = "PASS" if passed else "PARTIAL" if target["median"] and target["median"] > 0 else "FAIL"
    return f"""RESULT: {status}

PRIMARY METRIC: exact query-relevant bit trace plus terminal answer
BASELINE d50: {result['baseline']['frontier']['d50']}
TARGETED d50: {result['branches']['TARGETED']['frontier']['d50']}
RANDOM d50: {result['branches']['RANDOM']['frontier']['d50']}
DEPTH_MATCHED d50: {result['branches']['DEPTH_MATCHED']['frontier']['d50']}

TARGETED - BASELINE: {target['median']}
95% CI: {target['ci95']}
positive fraction: {target['bootstrap_positive_fraction']}

TARGETED - RANDOM: {random['median']}
95% CI: {random['ci95']}

TARGETED - DEPTH_MATCHED: {depth['median']}
95% CI: {depth['ci95']}

EASY ANCHOR CHANGE: {easy_drop}

See summary.json for terminal accuracy, trace-bit accuracy, parse failures, held-out worlds, and acquisition history.
"""


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--repo", default=".")
    parser.add_argument("--output", default="runs/causal-graph-round2-contract")
    parser.add_argument("--base-model", default="/home/admin/models/hf/Qwen3.5-9B")
    parser.add_argument("--training-python", default="/home/admin/.venvs/causal-graph/bin/python")
    parser.add_argument("--seed", type=int, default=20260812)
    parser.add_argument("--dev-per-depth", type=int, default=32)
    parser.add_argument("--sealed-per-depth", type=int, default=64)
    parser.add_argument("--candidate-pool", type=int, default=4000)
    parser.add_argument("--batch-size", type=int, default=1000)
    parser.add_argument("--rounds", type=int, default=3)
    parser.add_argument("--max-depth", type=int, default=24)
    parser.add_argument("--bootstrap", type=int, default=2000)
    parser.add_argument("--token-tolerance", type=float, default=0.08)
    parser.add_argument("--curriculum-fraction", type=float, default=0.2)
    parser.add_argument("--max-new-tokens", type=int, default=384)
    parser.add_argument("--max-steps", type=int)
    args = parser.parse_args()
    try:
        result = run(args)
    except Exception as exc:
        emit(Path(args.output) / "dashboard.jsonl", "run_failed", stage="failed", error=f"{type(exc).__name__}: {exc}")
        raise
    print(json.dumps({"output": args.output, "comparisons": {name: {key: value for key, value in row.items() if key != "samples"} for name, row in result["comparisons"].items()}}, indent=2))


if __name__ == "__main__":
    main()
