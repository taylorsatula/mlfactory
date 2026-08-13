#!/usr/bin/env bash
set -euo pipefail
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../../.." && pwd)"
PYTHON="${PYTHON:-/home/admin/dft-eval-harness/.venv312/bin/python}"
SPEC="$ROOT/mlfactory/experiments/voice/specs/voice_qwen35_9b_grounded_v1.yaml"
REGISTRY="$ROOT/.mlfactory/registry.db"
RUN_ID="${1:-voice-qwen35-9b-grounded-v1-$(date -u +%Y%m%dT%H%M%SZ)}"
LOG="$ROOT/runs/${RUN_ID}.launcher.log"
mkdir -p "$ROOT/runs"
export PYTHONPATH="$ROOT"
export HF_HOME="/home/admin/.cache/huggingface"
export HF_HUB_OFFLINE=1
export PYTHONUNBUFFERED=1
nohup "$PYTHON" -m mlfactory.cli --registry "$REGISTRY" run "$SPEC" --runs-dir "$ROOT/runs" --run-id "$RUN_ID" >"$LOG" 2>&1 &
PID=$!
printf '%s\n' "$PID" > "$ROOT/runs/${RUN_ID}.pid"
printf 'run_id=%s\npid=%s\nlog=%s\ndashboard=cd %s && %s -m mlfactory.cli --registry %s dashboard --watch-run %s --refresh 2\n' "$RUN_ID" "$PID" "$LOG" "$ROOT" "$PYTHON" "$REGISTRY" "$RUN_ID"
