#!/bin/bash
# Disposable startup script for Qwen3.5-4B baseline collection on the local llama-server.
# This is intentionally NOT a systemd service; start it when you need it and Ctrl-C when done.
set -euo pipefail
cd "$(dirname "$0")"

MODEL="${ACE_MODEL_GGUF:-/home/admin/models/Qwen3.5-4B-UD-Q8_K_XL.gguf}"
PORT="${ACE_LLAMA_PORT:-3090}"
MAIN_GPU="${ACE_MAIN_GPU:-0}"
ALIAS="Qwen/Qwen3.5-4B"

if [ ! -f "$MODEL" ]; then
    echo "ERROR: GGUF not found: $MODEL" >&2
    exit 1
fi

# Stop any mutually exclusive llama systemd services so we don't OOM or collide on port 3090.
stop_svc() {
    local svc="$1"
    if systemctl is-active --quiet "$svc" 2>/dev/null; then
        echo ">> Stopping $svc ..."
        # This requires passwordless sudo or manual pre-stop. Do not hardcode credentials.
        sudo -n systemctl stop "$svc" || {
            echo "ERROR: could not stop $svc. Stop it manually or configure passwordless sudo." >&2
            exit 1
        }
        sudo -n systemctl disable "$svc" || true
        for i in $(seq 1 24); do
            systemctl is-active --quiet "$svc" 2>/dev/null || break
            sleep 5
        done
    fi
}
stop_svc llama-laguna
stop_svc llama-qwopus

# Safety: kill any leftover llama-server bound to the target port.
PID=$(lsof -ti tcp:"$PORT" 2>/dev/null || true)
if [ -n "${PID:-}" ]; then
    echo ">> Killing leftover PID $PID on port $PORT"
    kill "$PID" 2>/dev/null || true
    sleep 2
fi

echo ">> Starting llama-server with $MODEL"
exec /home/admin/llama.cpp/build/bin/llama-server \
    --alias "$ALIAS" \
    --jinja \
    --reasoning on \
    --reasoning-format none \
    --reasoning-budget -1 \
    --host 0.0.0.0 \
    --port "$PORT" \
    --model "$MODEL" \
    --n-gpu-layers 999 \
    --split-mode none \
    --main-gpu "$MAIN_GPU" \
    --ctx-size 32768 \
    --parallel 1 \
    --flash-attn on \
    --batch-size 2048 \
    --ubatch-size 512 \
    --cache-type-k q8_0 \
    --cache-type-v q8_0 \
    --cache-ram 0 \
    --no-cache-idle-slots \
    --spec-type draft-mtp \
    --spec-draft-n-max 3 \
    --metrics \
    --n-predict -1 \
    "$@"
