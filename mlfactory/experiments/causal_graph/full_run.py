"""End-to-end local CausalGraph MVP runner.

The runner deliberately keeps development data separate from the fixed sealed
corpus and never reads sealed outcomes while choosing data or stopping.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import shutil
import subprocess
import sys
from collections import defaultdict
from pathlib import Path
from typing import Any

import numpy as np
import yaml

from .acquisition import coordinate_summary, generate_candidate_pool, select_depth_matched, select_random_batch
from .analysis import bootstrap_frontier, fit_depth_logistic, fit_surface, raw_depth_stats, score_candidates, select_frontier_batch, summarize_bootstrap, write_json, write_jsonl
from .generator import HELDOUT_WORLDS, TRAIN_WORLDS, WORLD_IDS, canonical_trace, generate_task, verify_task

DEPTHS = [2, 4, 6, 8, 10, 12, 16, 20, 24, 32]


def _balanced_tasks(seed: int, depths: list[int], per_depth: int, worlds: tuple[str, ...], controls: str = "fixed") -> list[dict[str, Any]]:
    tasks: list[dict[str, Any]] = []
    for depth in depths:
        gates = 1 if depth == 2 else (2 if controls == "fixed" else min(max(depth // 5, 1), 6))
        negations = 0 if depth == 2 else (1 if controls == "fixed" else min(max(depth // 9, 0), depth - gates))
        relevant = depth + 1 + min(gates, 2 if controls == "fixed" else 3)
        counts = {"YES": 0, "NO": 0}; index = 0
        target = per_depth // 2
        while counts["YES"] < target or counts["NO"] < target:
            task = generate_task(
                seed + depth * 100_000 + index, depth=depth, relevant_nodes=relevant,
                distractor_nodes=index % 3, binary_gate_count=gates, negation_count=negations,
                source_update_count=index % 2, world_id=worlds[(index + depth) % len(worlds)], render_template_id=index % 3,
            )
            index += 1
            if counts[task["canonical_answer"]] >= target: continue
            verify_task(task); task["canonical_trace"] = canonical_trace(task)
            counts[task["canonical_answer"]] += 1; tasks.append(task)
        while len([x for x in tasks if x["depth"] == depth]) < per_depth:
            task = generate_task(seed + depth * 100_000 + index, depth=depth, relevant_nodes=relevant, distractor_nodes=index % 3, binary_gate_count=gates, negation_count=negations, source_update_count=index % 2, world_id=worlds[(index + depth) % len(worlds)], render_template_id=index % 3)
            index += 1
            verify_task(task); task["canonical_trace"] = canonical_trace(task); tasks.append(task)
    return tasks


def _write_tasks(path: Path, tasks: list[dict[str, Any]]) -> None:
    write_jsonl(path, tasks)


def _run(cmd: list[str], log: Path, env: dict[str, str] | None = None) -> None:
    log.parent.mkdir(parents=True, exist_ok=True)
    with log.open("a", encoding="utf-8") as handle:
        handle.write("COMMAND " + " ".join(cmd) + "\n")
        handle.flush()
        result = subprocess.run(cmd, stdout=handle, stderr=subprocess.STDOUT, env=env or os.environ.copy())
    if result.returncode:
        raise RuntimeError(f"command failed ({result.returncode}); see {log}")


def _read_jsonl(path: Path) -> list[dict[str, Any]]:
    with path.open(encoding="utf-8") as handle:
        return [json.loads(line) for line in handle if line.strip()]


def _evaluate(base: Path, tasks: Path, output: Path, checkpoint: str, python: str, root: Path, log: Path, adapter: Path | None = None) -> list[dict[str, Any]]:
    cmd = [python, "-m", "mlfactory.experiments.causal_graph.evaluate_hf", "--base-model", str(base), "--input", str(tasks), "--output", str(output), "--checkpoint", checkpoint, "--max-length", "4096", "--max-new-tokens", "384", "--batch-size", "8"]
    if adapter: cmd += ["--adapter", str(adapter)]
    env = os.environ.copy(); env["HF_HUB_OFFLINE"] = "1"; env["PYTHONPATH"] = str(root) + os.pathsep + env.get("PYTHONPATH", "")
    _run(cmd, log, env); return _read_jsonl(output)


def _train(base: Path, train_file: Path, output: Path, python: str, root: Path, log: Path, init_adapter: Path | None, seed: int, max_steps: int | None) -> dict[str, Any]:
    cmd = [python, "-m", "mlfactory.experiments.causal_graph.train_adapter", "--base-model", str(base), "--train-file", str(train_file), "--output-dir", str(output), "--seed", str(seed), "--epochs", "1", "--batch-size", "2", "--gradient-accumulation-steps", "4", "--learning-rate", "1e-5", "--max-length", "1024", "--max-target-tokens", "160", "--lora-r", "8", "--lora-alpha", "16"]
    if init_adapter: cmd += ["--init-adapter", str(init_adapter)]
    if max_steps is not None: cmd += ["--max-steps", str(max_steps)]
    env = os.environ.copy(); env["HF_HUB_OFFLINE"] = "1"; env["PYTHONPATH"] = str(root) + os.pathsep + env.get("PYTHONPATH", ""); env["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"
    _run(cmd, log, env)
    return json.loads((output / "training_manifest.json").read_text(encoding="utf-8"))


def _summary(records: list[dict[str, Any]]) -> dict[str, Any]:
    fit = fit_depth_logistic(records)
    return {"frontier": fit, "raw": raw_depth_stats(records), "n": len(records), "accuracy": sum(r["correct"] for r in records) / max(1, len(records)), "parse_failures": sum(r["parse_failure"] for r in records)}


def _paired_bootstrap(baseline: list[dict[str, Any]], trained: list[dict[str, Any]], B: int, seed: int) -> dict[str, Any]:
    bmap = {r["example_id"]: r for r in baseline}; tmap = {r["example_id"]: r for r in trained}
    groups: dict[int, list[str]] = defaultdict(list)
    for example_id, row in bmap.items():
        if example_id in tmap: groups[int(row["depth"])].append(example_id)
    rng = np.random.default_rng(seed); shifts = []
    for _ in range(B):
        sample = []
        for ids in groups.values():
            sample.extend(ids[int(i)] for i in rng.integers(0, len(ids), size=len(ids)))
        br = [bmap[i] for i in sample]; tr = [tmap[i] for i in sample]
        bf = fit_depth_logistic(br); tf = fit_depth_logistic(tr)
        if bf.get("d50") is not None and tf.get("d50") is not None: shifts.append(float(tf["d50"] - bf["d50"]))
    values = np.asarray(shifts)
    return {"replicates": B, "valid": len(shifts), "median": float(np.median(values)) if len(values) else None, "ci95": [float(np.quantile(values, .025)), float(np.quantile(values, .975))] if len(values) else None, "bootstrap_positive_fraction": float(np.mean(values > 0)) if len(values) else None, "samples": shifts}


def run(args: argparse.Namespace) -> dict[str, Any]:
    root = Path(args.repo).resolve(); run_dir = Path(args.output).resolve(); run_dir.mkdir(parents=True, exist_ok=True)
    for sub in ["candidate_pools", "selected_training_batches", "model_outputs", "training_logs", "checkpoint_manifests", "plots", "source_snapshot"]: (run_dir / sub).mkdir(exist_ok=True)
    for source_name in ["generator.py", "probe.py", "analysis.py", "acquisition.py", "evaluate_hf.py", "train_adapter.py", "full_run.py", "plots.py"]:
        source_path = root / "mlfactory" / "experiments" / "causal_graph" / source_name
        if source_path.exists(): shutil.copy2(source_path, run_dir / "source_snapshot" / source_name)
    base = Path(args.base_model).resolve(); python = str(Path(args.training_python).expanduser())
    config = vars(args).copy(); config["base_model"] = str(base); config["repo"] = str(root)
    write_json(run_dir / "config.json", config)
    (run_dir / "config.yaml").write_text(yaml.safe_dump(config, sort_keys=True), encoding="utf-8")
    uv = shutil.which("uv") or "/home/admin/.local/bin/uv"
    freeze = subprocess.run([uv, "pip", "freeze", "--python", python], capture_output=True, text=True)
    if freeze.returncode != 0 or not freeze.stdout.strip():
        freeze = subprocess.run([python, "-m", "pip", "freeze"], capture_output=True, text=True)
    (run_dir / "environment.txt").write_text(freeze.stdout, encoding="utf-8")
    model_manifest = {"alias": args.model, "gguf_model": args.gguf_model, "hf_base_model": str(base), "hf_config_sha256": hashlib.sha256((base / "config.json").read_bytes()).hexdigest(), "seed": args.seed}
    write_json(run_dir / "model_manifest.json", model_manifest)

    # Fixed sealed data is generated before any adapter training and is never
    # read by acquisition or development stopping logic.
    sealed = _balanced_tasks(args.seed + 10_000_000, DEPTHS, args.sealed_per_depth, tuple(TRAIN_WORLDS), "fixed") + _balanced_tasks(args.seed + 20_000_000, DEPTHS, args.sealed_per_depth, tuple(HELDOUT_WORLDS), "fixed")
    _write_tasks(run_dir / "sealed_eval.jsonl", sealed)
    config["sealed_eval_sha256"] = hashlib.sha256((run_dir / "sealed_eval.jsonl").read_bytes()).hexdigest()
    write_json(run_dir / "config.json", config)
    (run_dir / "config.yaml").write_text(yaml.safe_dump(config, sort_keys=True), encoding="utf-8")
    dev0 = _balanced_tasks(args.seed + 1_000_000, DEPTHS, args.dev_per_depth, tuple(TRAIN_WORLDS), "fixed")
    _write_tasks(run_dir / "probe_dev.jsonl", dev0)
    baseline_output = run_dir / "model_outputs" / "baseline_sealed.jsonl"
    baseline_dev_output = run_dir / "model_outputs" / "baseline_dev.jsonl"
    baseline_sealed = _read_jsonl(baseline_output) if baseline_output.exists() and sum(1 for _ in baseline_output.open(encoding="utf-8")) == len(sealed) else _evaluate(base, run_dir / "sealed_eval.jsonl", baseline_output, "BASELINE", python, root, run_dir / "training_logs" / "baseline_eval.log")
    baseline_dev = _read_jsonl(baseline_dev_output) if baseline_dev_output.exists() and sum(1 for _ in baseline_dev_output.open(encoding="utf-8")) == len(dev0) else _evaluate(base, run_dir / "probe_dev.jsonl", baseline_dev_output, "BASELINE", python, root, run_dir / "training_logs" / "baseline_eval.log")
    write_json(run_dir / "frontier_fits_baseline.json", _summary(baseline_dev))
    baseline_fit = fit_depth_logistic(baseline_dev); baseline_surface = fit_surface(baseline_dev)

    branch_states: dict[str, dict[str, Any]] = {"TARGETED": {"adapter": None, "dev": baseline_dev, "sealed": baseline_sealed}, "RANDOM": {"adapter": None, "dev": baseline_dev, "sealed": baseline_sealed}}
    if args.depth_matched: branch_states["DEPTH_MATCHED"] = {"adapter": None, "dev": baseline_dev, "sealed": baseline_sealed}
    acquisition_history: dict[str, list[dict[str, Any]]] = defaultdict(list)
    checkpoints: dict[str, list[dict[str, Any]]] = defaultdict(list)

    for round_index in range(1, args.rounds + 1):
        pool_seed = args.seed + round_index * 100_000
        target_fit = fit_depth_logistic(branch_states["TARGETED"]["dev"])
        center = target_fit.get("d50") if target_fit.get("d50") is not None else baseline_fit.get("d50")
        target_pool = generate_candidate_pool(pool_seed, args.candidate_pool, d50=center, max_depth=args.max_depth)
        scored = score_candidates(target_pool, fit_surface(branch_states["TARGETED"]["dev"]))
        targeted = select_frontier_batch(scored, args.batch_size, seed=pool_seed)
        random_pool = generate_candidate_pool(args.seed + 500_000 + round_index * 100_000, args.candidate_pool, d50=baseline_fit.get("d50"), max_depth=args.max_depth)
        random_rows = select_random_batch(random_pool, args.batch_size, seed=pool_seed + 1)
        batches = {"TARGETED": targeted, "RANDOM": random_rows}
        if args.depth_matched: batches["DEPTH_MATCHED"] = select_depth_matched(random_pool, targeted, seed=pool_seed + 2)
        for branch, rows in batches.items():
            for row in rows: row["canonical_trace"] = row.get("canonical_trace") or canonical_trace(row); row["round"] = round_index; row["branch"] = branch
            pool_path = run_dir / "candidate_pools" / f"round_{round_index}_{branch.lower()}.jsonl"
            write_jsonl(pool_path, scored if branch == "TARGETED" else random_pool)
            train_path = run_dir / "selected_training_batches" / f"round_{round_index}_{branch.lower()}.jsonl"; _write_tasks(train_path, rows)
            acquisition_history[branch].append({"round": round_index, **coordinate_summary(rows), "d50_used": center})
            out_adapter = run_dir / "checkpoints" / branch.lower() / f"round_{round_index}"; out_adapter.parent.mkdir(parents=True, exist_ok=True)
            dev_round = _balanced_tasks(args.seed + 3_000_000 + round_index * 100_000, DEPTHS, args.dev_per_depth, tuple(TRAIN_WORLDS), "fixed")
            dev_path = run_dir / "model_outputs" / f"dev_round_{round_index}.jsonl"; _write_tasks(dev_path, dev_round)
            dev_out = run_dir / "model_outputs" / f"{branch.lower()}_round_{round_index}_dev.jsonl"
            manifest_path = run_dir / "checkpoint_manifests" / f"{branch.lower()}_round_{round_index}.json"
            if manifest_path.exists() and (out_adapter / "training_manifest.json").exists() and dev_out.exists():
                train_meta = json.loads(manifest_path.read_text(encoding="utf-8"))
                branch_states[branch]["adapter"] = out_adapter
                branch_states[branch]["dev"] = _read_jsonl(dev_out)
            else:
                train_meta = _train(base, train_path, out_adapter, python, root, run_dir / "training_logs" / f"{branch.lower()}_round_{round_index}.log", branch_states[branch]["adapter"], args.seed + round_index, args.max_steps)
                write_json(manifest_path, train_meta)
                branch_states[branch]["adapter"] = out_adapter
                branch_states[branch]["dev"] = _evaluate(base, dev_path, dev_out, f"{branch}_ROUND_{round_index}", python, root, run_dir / "training_logs" / f"{branch.lower()}_round_{round_index}_eval.log", out_adapter)
            checkpoints[branch].append({"round": round_index, "adapter": str(out_adapter), "training": train_meta, "dev_frontier": fit_depth_logistic(branch_states[branch]["dev"])})
            write_json(run_dir / "frontier_fits" / f"{branch.lower()}_round_{round_index}.json", _summary(branch_states[branch]["dev"])) if (run_dir / "frontier_fits").mkdir(exist_ok=True) is None else None
        # Development-only saturation guard.  The sealed set is untouched.
        if round_index >= 2:
            prior_fit = checkpoints["TARGETED"][-2].get("dev_frontier") or {}
            prior_d50 = prior_fit.get("d50") if isinstance(prior_fit, dict) else prior_fit
            current = fit_depth_logistic(branch_states["TARGETED"]["dev"])
            if prior_d50 is not None and current.get("d50") is not None and abs(current["d50"] - prior_d50) < .25:
                break

    # Final sealed evaluation happens only after all acquisition/training.
    for branch in branch_states:
        adapter = branch_states[branch]["adapter"]
        if not adapter: continue
        branch_states[branch]["sealed"] = _evaluate(base, run_dir / "sealed_eval.jsonl", run_dir / "model_outputs" / f"{branch.lower()}_final_sealed.jsonl", f"{branch}_FINAL", python, root, run_dir / "training_logs" / f"{branch.lower()}_final_eval.log", adapter)
    train_worlds = set(TRAIN_WORLDS); heldout_worlds = set(HELDOUT_WORLDS)
    baseline_train = [r for r in baseline_sealed if r["world"] in train_worlds]
    baseline_heldout = [r for r in baseline_sealed if r["world"] in heldout_worlds]
    baseline_bootstrap = summarize_bootstrap(bootstrap_frontier(baseline_train, B=args.bootstrap, seed=args.seed + 99))
    final = {
        "baseline": _summary(baseline_train),
        "baseline_bootstrap": baseline_bootstrap,
        "branches": {branch: _summary([r for r in state["sealed"] if r["world"] in train_worlds]) for branch, state in branch_states.items()},
        "world_transfer": {"BASELINE": {"train": _summary(baseline_train), "heldout": _summary(baseline_heldout)}},
        "acquisition": dict(acquisition_history),
    }
    for branch, state in branch_states.items():
        if not state["adapter"]: continue
        branch_train = [r for r in state["sealed"] if r["world"] in train_worlds]
        branch_heldout = [r for r in state["sealed"] if r["world"] in heldout_worlds]
        final["world_transfer"][branch] = {"train": _summary(branch_train), "heldout": _summary(branch_heldout)}
        final.setdefault("comparisons", {})[f"{branch}_vs_baseline"] = _paired_bootstrap(baseline_train, branch_train, args.bootstrap, args.seed)
        final.setdefault("comparisons", {})[f"{branch}_vs_baseline_heldout"] = _paired_bootstrap(baseline_heldout, branch_heldout, args.bootstrap, args.seed + 10)
    if "TARGETED" in branch_states and "RANDOM" in branch_states:
        final.setdefault("comparisons", {})["TARGETED_vs_RANDOM"] = _paired_bootstrap([r for r in branch_states["RANDOM"]["sealed"] if r["world"] in train_worlds], [r for r in branch_states["TARGETED"]["sealed"] if r["world"] in train_worlds], args.bootstrap, args.seed + 1)
    if "DEPTH_MATCHED" in branch_states:
        final.setdefault("comparisons", {})["TARGETED_vs_DEPTH_MATCHED"] = _paired_bootstrap([r for r in branch_states["DEPTH_MATCHED"]["sealed"] if r["world"] in train_worlds], [r for r in branch_states["TARGETED"]["sealed"] if r["world"] in train_worlds], args.bootstrap, args.seed + 2)
    anchor_target = int(round(final["baseline"]["frontier"].get("d80") or DEPTHS[0])) - 3
    anchor_depth = max(min(DEPTHS), min(max(DEPTHS), anchor_target))
    final["easy_anchor"] = {"depth": anchor_depth, "BASELINE": sum(r["correct"] for r in baseline_train if int(r["depth"]) == anchor_depth) / max(1, sum(1 for r in baseline_train if int(r["depth"]) == anchor_depth))}
    for branch, state in branch_states.items():
        if state["adapter"]:
            rows = [r for r in state["sealed"] if r["world"] in train_worlds and int(r["depth"]) == anchor_depth]
            final["easy_anchor"][branch] = sum(r["correct"] for r in rows) / max(1, len(rows))
    final["easy_anchor"]["changes"] = {branch: final["easy_anchor"].get(branch, 0) - final["easy_anchor"]["BASELINE"] for branch in final["easy_anchor"] if branch not in {"depth", "changes"}}
    final["model_manifest"] = model_manifest; final["sealed_examples_per_depth_per_world"] = args.sealed_per_depth
    write_json(run_dir / "frontier_fits.json", final); write_json(run_dir / "summary.json", final)
    write_json(run_dir / "generator_version.json", {"module": "mlfactory.experiments.causal_graph.generator", "version": "mvp-1", "seed": args.seed, "worlds_train": list(TRAIN_WORLDS), "worlds_heldout": list(HELDOUT_WORLDS)})
    write_json(run_dir / "rng_seeds.json", {"master": args.seed, "sealed_train": args.seed + 10_000_000, "sealed_heldout": args.seed + 20_000_000, "development": args.seed + 1_000_000, "round_seed_formula": "master + round * 100000"})
    # Preserve a columnar copy when pyarrow is available; JSONL remains the
    # canonical fallback for minimal environments.
    try:
        import pyarrow as pa
        import pyarrow.parquet as pq
        all_metrics = []
        for path in sorted((run_dir / "model_outputs").glob("*.jsonl")): all_metrics.extend(_read_jsonl(path))
        if all_metrics: pq.write_table(pa.Table.from_pylist(all_metrics), run_dir / "per_example_metrics.parquet")
        bootstrap_rows = []
        for name, value in final.get("comparisons", {}).items():
            for index, shift_value in enumerate(value.get("samples", [])): bootstrap_rows.append({"comparison": name, "replicate": index, "delta_d50": shift_value})
        if bootstrap_rows: pq.write_table(pa.Table.from_pylist(bootstrap_rows), run_dir / "bootstrap_samples.parquet")
    except Exception as exc:
        (run_dir / "parquet_unavailable.txt").write_text(f"parquet export unavailable: {type(exc).__name__}: {exc}\\n", encoding="utf-8")
    (run_dir / "summary.csv").write_text("branch,d50,accuracy\\n" + "\\n".join(f"{branch},{state.get('frontier', {}).get('d50')},{state.get('accuracy')}" for branch, state in [("BASELINE", final["baseline"]), *final.get("branches", {}).items()]) + "\\n", encoding="utf-8")
    (run_dir / "final_report.md").write_text(render_report(final, args), encoding="utf-8")
    try:
        from .plots import make_plots
        make_plots(run_dir)
    except Exception as exc:
        (run_dir / "plots" / "plot_error.txt").write_text(f"plot generation failed: {type(exc).__name__}: {exc}\\n", encoding="utf-8")
    return final


def render_report(result: dict[str, Any], args: argparse.Namespace) -> str:
    base = result["baseline"]["frontier"]; targeted = result["branches"].get("TARGETED", {}).get("frontier", {}); random = result["branches"].get("RANDOM", {}).get("frontier", {}); cmp = result.get("comparisons", {}).get("TARGETED_vs_RANDOM", {})
    shift = result.get("comparisons", {}).get("TARGETED_vs_baseline", {})
    positive = shift.get("bootstrap_positive_fraction")
    status = "PASS" if base.get("d50") is not None and shift.get("median") is not None and shift.get("median", 0) >= 1 and shift.get("ci95") and shift["ci95"][0] > 0 and (positive or 0) >= .95 and cmp.get("ci95") and cmp["ci95"][0] > 0 else "PARTIAL" if shift.get("median") is not None and shift.get("median", 0) > 0 else "FAIL"
    def v(obj: dict[str, Any], key: str) -> str: return str(obj.get(key, "unidentifiable"))
    baseline_ci = result.get("baseline_bootstrap", {}).get("d50", {}).get("ci95")
    transfer = result.get("world_transfer", {}).get("TARGETED", {})
    return f'''RESULT: {status}\n\nMODEL: {args.model}\nBASE CHECKPOINT: {args.base_model}\nTRAINING METHOD: QLoRA\n\nBASELINE FRONTIER\nd80: {v(base, "d80")}\nd50: {v(base, "d50")}\nd20: {v(base, "d20")}\nfalloff width: {v(base, "falloff_width")}\n95% CI for d50: shift bootstrap is {baseline_ci}\n\nTARGETED FINAL FRONTIER\nd80: {v(targeted, "d80")}\nd50: {v(targeted, "d50")}\nd20: {v(targeted, "d20")}\n\nRANDOM FINAL FRONTIER\nd80: {v(random, "d80")}\nd50: {v(random, "d50")}\nd20: {v(random, "d20")}\n\nTARGETED FRONTIER SHIFT\nDelta d50: {shift.get("median")}\n95% CI: {shift.get("ci95")}\nbootstrap_positive_fraction: {positive}\n\nTARGETED ADVANTAGE OVER RANDOM\nDelta d50: {cmp.get("median")}\n95% CI: {cmp.get("ci95")}\n\nEASY ANCHOR\ndepth: {result.get("easy_anchor", {}).get("depth")}\nbaseline: {result.get("easy_anchor", {}).get("BASELINE")}\ntargeted final: {result.get("easy_anchor", {}).get("TARGETED")}\nchange: {result.get("easy_anchor", {}).get("changes", {}).get("TARGETED")}\n\nHELD-OUT WORLD TRANSFER\nbaseline d50: {result.get("world_transfer", {}).get("BASELINE", {}).get("heldout", {}).get("frontier", {}).get("d50")}\ntargeted d50: {transfer.get("heldout", {}).get("frontier", {}).get("d50")}\nDelta d50: see TARGETED_vs_baseline_heldout\n\nACQUISITION MOVEMENT\n{json.dumps(result.get("acquisition", {}), indent=2, sort_keys=True)}\n\nDID A MEASURABLE BASELINE FRONTIER EXIST?\n{"yes" if base.get("d50") is not None else "no"}\n\nDID TARGETED TRAINING MOVE IT?\n{"yes" if shift.get("median", 0) > 0 else "no"}\n\nDID TARGETING BEAT EQUAL-BUDGET RANDOM SYNTHESIS?\n{"yes" if cmp.get("median", 0) > 0 and cmp.get("ci95") and cmp["ci95"][0] > 0 else "no"}\n\nDID ACQUISITION FOLLOW THE MOVING FRONTIER?\nSee acquisition coordinate summaries.\n\nDID THE GAIN TRANSFER TO HELD-OUT WORLDS?\nSee per-world sealed metrics.\n\nMAJOR CONFOUNDS:\nPrompt length is recorded by task structure; parse failures are separate in per-example outputs.\n\nOBSERVED RESULTS:\nSee summary.json and frontier_fits.json.\n\nSTATISTICAL INTERPRETATION:\nAll confidence intervals use paired, difficulty-stratified bootstrap; bootstrap_positive_fraction is not a posterior probability.\n\nWHAT THE RESULT DOES NOT ESTABLISH:\nThis controlled result does not establish automatic interpretation of arbitrary real-world datasets or universal curriculum generation.\n\nRECOMMENDED NEXT EXPERIMENT:\nBroaden held-out renderers only if the controlled acquisition comparison is positive.\n'''


def main() -> None:
    p = argparse.ArgumentParser(); p.add_argument("--repo", default="."); p.add_argument("--output", default="runs/causal-graph-mvp"); p.add_argument("--base-model", default="/home/admin/models/hf/Qwen3.5-9B"); p.add_argument("--training-python", default="/home/admin/.venvs/causal-graph/bin/python"); p.add_argument("--model", default="f16-jackrongds4qwen"); p.add_argument("--gguf-model", default="/home/admin/models/DeepSeek-V4-Pro-Qwen3.5-9B/DeepSeek-V4-Pro-Qwen3.5-9B.f16.gguf"); p.add_argument("--seed", type=int, default=20260811); p.add_argument("--dev-per-depth", type=int, default=32); p.add_argument("--sealed-per-depth", type=int, default=128); p.add_argument("--candidate-pool", type=int, default=4000); p.add_argument("--batch-size", type=int, default=1000); p.add_argument("--rounds", type=int, default=3); p.add_argument("--max-depth", type=int, default=40); p.add_argument("--bootstrap", type=int, default=2000); p.add_argument("--max-steps", type=int); p.add_argument("--depth-matched", action="store_true")
    result = run(p.parse_args()); print(json.dumps({"output": p.parse_args().output, "baseline": result["baseline"]["frontier"], "comparisons": {k: {x: v for x, v in value.items() if x != "samples"} for k, value in result.get("comparisons", {}).items()}}, indent=2))


if __name__ == "__main__": main()
