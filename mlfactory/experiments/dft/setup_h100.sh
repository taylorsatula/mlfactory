#!/bin/bash
set -euo pipefail

PROJECT_DIR="${1:-/workspace/dft-eval-harness}"
cd "$PROJECT_DIR"

if [[ ! -d .venv ]]; then
  python3 -m venv .venv
fi
source .venv/bin/activate

python -m pip install --upgrade pip setuptools wheel packaging ninja

# Known-good Hopper/CUDA 12.9 build. Install torch before any CUDA extension.
python -m pip install \
  torch==2.13.0+cu129 \
  --index-url https://download.pytorch.org/whl/cu129

python -m pip install \
  transformers==5.14.1 \
  accelerate==1.14.0 \
  peft==0.19.1 \
  trl==1.9.1 \
  bitsandbytes==0.50.0 \
  sentence-transformers==5.6.1 \
  datasets==5.0.0 \
  openai numpy scipy scikit-learn tqdm

# Qwen3.5 hybrid linear-attention fast path. TileLang is required for the
# correct gated-delta backward backend on Hopper with modern Triton.
python -m pip install \
  flash-linear-attention==0.5.1 \
  tilelang==0.1.12

# Build against this venv's cu129 torch. Build isolation may inject a mismatched
# torch/CUDA wheel and raise a false CUDA-version mismatch.
python -m pip install --no-build-isolation \
  causal-conv1d==1.6.2.post1

python - <<'PY'
import importlib.metadata as metadata
import torch
import transformers
import bitsandbytes
import fla
import causal_conv1d
import tilelang

for package in [
    "torch", "transformers", "accelerate", "peft", "trl",
    "bitsandbytes", "sentence-transformers", "datasets",
    "flash-linear-attention", "causal-conv1d", "tilelang", "triton",
]:
    print(f"{package}=={metadata.version(package)}")

assert torch.cuda.is_available(), "PyTorch cannot see CUDA"
print("torch CUDA build:", torch.version.cuda)
print("GPUs:", [torch.cuda.get_device_name(i) for i in range(torch.cuda.device_count())])
x = torch.randn(1024, 1024, device="cuda:0")
print("CUDA tensor smoke:", x.square().mean().item())
PY

python -m pip freeze > environment.known-good.freeze.txt
printf '\nH100 environment ready: %s/.venv\n' "$PROJECT_DIR"
