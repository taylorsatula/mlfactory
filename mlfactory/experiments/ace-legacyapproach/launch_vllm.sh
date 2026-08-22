#!/bin/bash
set -euo pipefail
cd "$(dirname "$0")"
source /home/admin/ace-baseline-trajectories/.venv/bin/activate
export HF_HOME="${HF_HOME:-/home/admin/.cache/huggingface}"
if [ -z "${HF_TOKEN:-}" ] && [ -f "${HF_HOME}/token" ]; then
  export HF_TOKEN="$(cat "${HF_HOME}/token")"
fi
export VLLM_WORKER_MULTIPROC_METHOD=spawn
export CUDA_DEVICE_ORDER=PCI_BUS_ID
exec vllm serve Qwen/Qwen3.5-4B \
  --revision 851bf6e806efd8d0a36b00ddf55e13ccb7b8cd0a \
  --served-model-name Qwen/Qwen3.5-4B \
  --trust-remote-code \
  --dtype bfloat16 \
  --max-model-len 16384 \
  --gpu-memory-utilization 0.92 \
  --max-num-seqs 1 \
  --port 8001 \
  --host 0.0.0.0 \
  --limit-mm-per-prompt '{"image": 0, "video": 0}' \
  "$@"
