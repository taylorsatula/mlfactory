#!/usr/bin/env python3
"""Capture Qwen3.8 answers and reasoning for 400 anomaly ACE prompts.

The two source files are sampled evenly (200 records each) and processed as one
append-only run. A completed output line is the resume checkpoint, so stopping
and restarting continues at the next prompt without duplicate requests.
"""
from __future__ import annotations

import argparse
import json
import os
import signal
import subprocess
import sys
import time
from datetime import datetime
from pathlib import Path
from typing import Any

import requests

ACE_DIR = Path(__file__).parent
MODEL_URL = "http://127.0.0.1:3090/v1/chat/completions"
DEFAULT_MODEL = "Qwen3.8 27B MTP"
DEFAULT_API_KEY = "de401e0064756cf372297a5a4068a8ff3aa00e96efea7d85a2504f753e6d3763"
DEFAULT_GPU_POWER_LIMIT_W = 340
DEFAULT_POWER_THRESHOLD_HOURS = 2.0
DEFAULT_SECONDS_PER_PROMPT = 60.0

shutdown_requested = False


def now() -> str:
    return datetime.now().strftime("%Y-%m-%d %H:%M:%S")


def handle_signal(signum: int, _frame: Any) -> None:
    global shutdown_requested
    shutdown_requested = True
    print(f"\n[{now()}] Signal {signum}: stop after the current request", flush=True)


signal.signal(signal.SIGTERM, handle_signal)
signal.signal(signal.SIGINT, handle_signal)


def load_jsonl(path: Path) -> list[dict[str, Any]]:
    with path.open(encoding="utf-8") as stream:
        return [json.loads(line) for line in stream if line.strip()]


def append_jsonl(path: Path, record: dict[str, Any]) -> None:
    with path.open("a", encoding="utf-8") as stream:
        stream.write(json.dumps(record, ensure_ascii=False) + "\n")
        stream.flush()
        os.fsync(stream.fileno())


def completed_output_lines(path: Path) -> int:
    if not path.exists():
        return 0
    count = 0
    with path.open(encoding="utf-8") as stream:
        for line_number, line in enumerate(stream, 1):
            if not line.strip():
                continue
            try:
                json.loads(line)
            except json.JSONDecodeError as exc:
                raise RuntimeError(
                    f"output has invalid JSON at line {line_number}; repair it before resuming"
                ) from exc
            count += 1
    return count


def save_progress(path: Path, data: dict[str, Any]) -> None:
    temp = path.with_suffix(path.suffix + ".tmp")
    temp.write_text(json.dumps(data, indent=2) + "\n", encoding="utf-8")
    temp.replace(path)


def query_gpu_power_limits() -> dict[int, float]:
    """Return each GPU's current power limit in watts."""
    output = subprocess.check_output(
        [
            "nvidia-smi",
            "--query-gpu=index,power.limit",
            "--format=csv,noheader,nounits",
        ],
        text=True,
        stderr=subprocess.STDOUT,
    )
    limits: dict[int, float] = {}
    for line in output.splitlines():
        if not line.strip():
            continue
        index, limit = (part.strip() for part in line.split(",", 1))
        limits[int(index)] = float(limit)
    if not limits:
        raise RuntimeError("nvidia-smi returned no GPUs")
    return limits


def run_power_command(gpu_index: int, watts: float, sudo_password: str | None) -> None:
    """Set one GPU limit, using passwordless sudo or an explicitly supplied secret."""
    command = ["nvidia-smi", "-i", str(gpu_index), "-pl", str(watts)]
    result = subprocess.run(
        ["sudo", "-n", *command], capture_output=True, text=True
    )
    if result.returncode == 0:
        return
    if not sudo_password:
        detail = (result.stderr or result.stdout).strip()
        raise RuntimeError(
            "setting GPU power requires sudo; configure NOPASSWD for nvidia-smi "
            "or set ACE_GPU_SUDO_PASSWORD for the run (" + detail + ")"
        )
    result = subprocess.run(
        ["sudo", "-S", *command],
        input=sudo_password + "\n",
        capture_output=True,
        text=True,
    )
    if result.returncode != 0:
        detail = (result.stderr or result.stdout).strip()
        raise RuntimeError(f"could not set GPU {gpu_index} power to {watts} W: {detail}")


def set_gpu_power_limits(
    limits: dict[int, float], watts: float, sudo_password: str | None
) -> None:
    for gpu_index in limits:
        run_power_command(gpu_index, watts, sudo_password)


def estimate_remaining_seconds(
    output_path: Path, remaining: int, default_seconds: float
) -> tuple[float, float]:
    samples: list[float] = []
    if output_path.exists():
        with output_path.open(encoding="utf-8") as stream:
            for line in stream:
                try:
                    elapsed = float(json.loads(line).get("trace_elapsed_s", 0))
                except (ValueError, TypeError, json.JSONDecodeError):
                    continue
                if elapsed > 0:
                    samples.append(elapsed)
    per_prompt = sum(samples[-20:]) / len(samples[-20:]) if samples else default_seconds
    return remaining * per_prompt, per_prompt


def call_model(
    prose: str,
    model: str,
    api_key: str,
    max_tokens: int,
    timeout: int,
) -> dict[str, Any]:
    response = requests.post(
        MODEL_URL,
        headers={
            "Authorization": f"Bearer {api_key}",
            "Content-Type": "application/json",
        },
        json={
            "model": model,
            "messages": [{"role": "user", "content": prose}],
            "temperature": 0.7,
            "max_tokens": max_tokens,
            "stream": True,
        },
        stream=True,
        timeout=(15, timeout),
    )
    response.raise_for_status()

    content_parts: list[str] = []
    reasoning_parts: list[str] = []
    finish_reason = ""
    usage: dict[str, Any] = {}
    started = time.monotonic()

    # Decode the SSE bytes explicitly.  Requests may otherwise infer ISO-8859-1
    # for this endpoint and turn UTF-8 punctuation into mojibake.
    for raw_line in response.iter_lines(decode_unicode=False):
        if shutdown_requested:
            raise KeyboardInterrupt("shutdown requested")
        if not raw_line:
            continue
        line = raw_line.decode("utf-8", "replace") if isinstance(raw_line, bytes) else raw_line
        if not line.startswith("data: "):
            continue
        payload = line[6:]
        if payload.strip() == "[DONE]":
            break
        try:
            chunk = json.loads(payload)
        except json.JSONDecodeError:
            continue
        if chunk.get("usage"):
            usage = chunk["usage"]
        choices = chunk.get("choices") or []
        if not choices:
            continue
        choice = choices[0]
        if choice.get("finish_reason"):
            finish_reason = choice["finish_reason"]
        delta = choice.get("delta") or {}
        content = delta.get("content") or ""
        reasoning = delta.get("reasoning_content") or delta.get("reasoning") or ""
        if content:
            content_parts.append(content)
        if reasoning:
            reasoning_parts.append(reasoning)

    elapsed = round(time.monotonic() - started, 1)
    content = "".join(content_parts)
    reasoning = "".join(reasoning_parts)
    if not content and not reasoning:
        raise RuntimeError("empty response from local model")
    return {
        "content": content,
        "reasoning": reasoning,
        "finish_reason": finish_reason,
        "usage": usage,
        "elapsed_s": elapsed,
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", default=os.environ.get("ACE_MODEL", DEFAULT_MODEL))
    parser.add_argument("--max-tokens", type=int, default=16384)
    parser.add_argument("--timeout", type=int, default=900)
    parser.add_argument("--retries", type=int, default=3)
    parser.add_argument("--retry-backoff", type=int, default=15)
    parser.add_argument("--count", type=int, default=400)
    parser.add_argument("--output", type=Path, default=ACE_DIR / "data" / "anomaly_400_qwen38_traces.jsonl")
    parser.add_argument("--progress", type=Path, default=ACE_DIR / "data" / "anomaly_400_qwen38_progress.json")
    parser.add_argument("--log", type=Path, default=ACE_DIR / "data" / "anomaly_400_qwen38.log")
    parser.add_argument("--pid-file", type=Path, default=ACE_DIR / "data" / "anomaly_400_qwen38.pid")
    parser.add_argument("--gpu-power-limit-w", type=float, default=DEFAULT_GPU_POWER_LIMIT_W)
    parser.add_argument("--power-threshold-hours", type=float, default=DEFAULT_POWER_THRESHOLD_HOURS)
    parser.add_argument("--estimate-seconds-per-prompt", type=float, default=DEFAULT_SECONDS_PER_PROMPT)
    parser.add_argument(
        "--no-auto-gpu-power",
        action="store_true",
        help="do not lower/restore GPU power even when the estimate exceeds the threshold",
    )
    args = parser.parse_args()

    if args.count <= 0:
        raise ValueError("--count must be positive")
    inputs = [
        ("corpus_300_a", ACE_DIR / "data" / "corpus_300_a.jsonl"),
        ("corpus_300_b", ACE_DIR / "data" / "corpus_300_b.jsonl"),
    ]
    records: list[tuple[str, int, dict[str, Any]]] = []
    base_quota, remainder = divmod(args.count, len(inputs))
    for input_index, (source_name, source_path) in enumerate(inputs):
        source_records = load_jsonl(source_path)
        quota = base_quota + (1 if input_index < remainder else 0)
        if quota > len(source_records):
            raise RuntimeError(
                f"{source_name} has {len(source_records)} records but needs {quota}"
            )
        records.extend((source_name, index, record) for index, record in enumerate(source_records[:quota]))

    args.output.parent.mkdir(parents=True, exist_ok=True)
    start = completed_output_lines(args.output)
    if start > len(records):
        raise RuntimeError(f"output has {start} records but inputs contain only {len(records)}")

    api_key = os.environ.get("LLAMA_API_KEY", DEFAULT_API_KEY)
    successes = 0
    failures = 0
    if start:
        with args.output.open(encoding="utf-8") as stream:
            for line in stream:
                if line.strip():
                    try:
                        if "error" in (json.loads(line).get("trace") or {}):
                            failures += 1
                        else:
                            successes += 1
                    except json.JSONDecodeError:
                        pass

    def log(message: str) -> None:
        line = f"[{now()}] {message}"
        print(line, flush=True)
        with args.log.open("a", encoding="utf-8") as stream:
            stream.write(line + "\n")
            stream.flush()

    args.log.parent.mkdir(parents=True, exist_ok=True)
    args.log.touch(exist_ok=True)
    args.pid_file.write_text(str(os.getpid()) + "\n", encoding="utf-8")
    estimated_seconds, estimated_per_prompt = estimate_remaining_seconds(
        args.output, len(records) - start, args.estimate_seconds_per_prompt
    )
    threshold_seconds = args.power_threshold_hours * 3600
    should_manage_gpu_power = (
        not args.no_auto_gpu_power
        and estimated_seconds > threshold_seconds
        and len(records) > start
    )
    sudo_password = os.environ.get("ACE_GPU_SUDO_PASSWORD")
    progress = {
        "completed": start,
        "total": len(records),
        "successful": successes,
        "failed": failures,
        "current": None,
        "model": args.model,
        "estimated_hours": round(estimated_seconds / 3600, 2),
        "estimated_seconds_per_prompt": round(estimated_per_prompt, 1),
        "gpu_power_management": "pending" if should_manage_gpu_power else "not_needed",
        "gpu_power_limit_w": args.gpu_power_limit_w,
        "sources": {name: sum(1 for source, _, _ in records if source == name) for name, _ in inputs},
    }
    save_progress(args.progress, progress)

    log(f"=== Qwen3.8 ACE anomaly trace run: {len(records)} prompts (one continuous run) ===")
    log(f"Sources: corpus_300_a={progress['sources']['corpus_300_a']}, corpus_300_b={progress['sources']['corpus_300_b']}")
    log(f"Model: {args.model} | endpoint: {MODEL_URL} | max_tokens: {args.max_tokens}")
    if start:
        log(f"Resuming at {start + 1}/{len(records)}; {start} output records already present")
    log(
        f"Estimate: {estimated_seconds / 3600:.2f}h remaining "
        f"({estimated_per_prompt:.1f}s/prompt); power threshold: {args.power_threshold_hours:.2f}h"
    )

    previous_gpu_limits: dict[int, float] | None = None
    try:
        if should_manage_gpu_power:
            previous_gpu_limits = query_gpu_power_limits()
            log(f"Lowering GPU power to {args.gpu_power_limit_w:g} W: {previous_gpu_limits}")
            set_gpu_power_limits(previous_gpu_limits, args.gpu_power_limit_w, sudo_password)
            progress["gpu_power_management"] = "lowered"
            save_progress(args.progress, progress)
            log("GPU power limit lowered for long run")

        for global_index in range(start, len(records)):
            if shutdown_requested:
                break
            source_name, source_index, record = records[global_index]
            seed = record.get("seed", "?")
            log(
                f"CURRENT [{global_index + 1}/{len(records)}] source={source_name} "
                f"source_index={source_index} seed={seed} domain={record.get('domain', '?')}"
            )
            began = time.monotonic()
            trace: dict[str, Any] | None = None
            last_error = ""
            for attempt in range(1, args.retries + 1):
                try:
                    trace = call_model(record["prose"], args.model, api_key, args.max_tokens, args.timeout)
                    trace["attempts"] = attempt
                    break
                except KeyboardInterrupt:
                    raise
                except (requests.RequestException, RuntimeError) as exc:
                    last_error = f"{type(exc).__name__}: {exc}"
                    log(f"  attempt={attempt}/{args.retries} ERROR {last_error}")
                    if attempt < args.retries and not shutdown_requested:
                        time.sleep(args.retry_backoff)

            output_record = dict(record)
            output_record["trace_source"] = source_name
            output_record["trace_source_index"] = source_index
            output_record["trace_global_index"] = global_index
            if trace is None:
                trace = {"error": last_error or "request failed", "attempts": args.retries}
                failures += 1
                log(f"TRACE ERROR [{global_index + 1}/{len(records)}] {trace['error']}")
            else:
                successes += 1
                log(
                    f"TRACE OK [{global_index + 1}/{len(records)}] "
                    f"content={len(trace['content'])} reasoning={len(trace['reasoning'])} "
                    f"elapsed={trace['elapsed_s']}s"
                )
            output_record["trace"] = trace
            output_record["trace_elapsed_s"] = round(time.monotonic() - began, 1)
            append_jsonl(args.output, output_record)
            progress.update(
                completed=global_index + 1,
                successful=successes,
                failed=failures,
                current={"source": source_name, "source_index": source_index, "seed": seed},
            )
            save_progress(args.progress, progress)
    except KeyboardInterrupt:
        log("STOPPED before starting another prompt; output is resumable")
    finally:
        if previous_gpu_limits is not None:
            try:
                log(f"Restoring GPU power limits: {previous_gpu_limits}")
                for gpu_index, watts in previous_gpu_limits.items():
                    run_power_command(gpu_index, watts, sudo_password)
                progress["gpu_power_management"] = "restored"
                save_progress(args.progress, progress)
                log("GPU power limits restored")
            except Exception as exc:
                progress["gpu_power_management"] = f"restore_failed: {exc}"
                save_progress(args.progress, progress)
                log(f"ERROR restoring GPU power limits: {exc}")
        try:
            args.pid_file.unlink()
        except FileNotFoundError:
            pass

    progress["current"] = None
    save_progress(args.progress, progress)
    log(f"=== RUN END: completed={progress['completed']}/{progress['total']} successful={successes} failed={failures} ===")
    return 0 if progress["completed"] == progress["total"] and failures == 0 else 1


if __name__ == "__main__":
    sys.exit(main())
