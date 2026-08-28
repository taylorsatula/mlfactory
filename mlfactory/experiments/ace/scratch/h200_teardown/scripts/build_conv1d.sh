#!/bin/bash
cd /tmp
export TORCH_CUDA_ARCH_LIST=9.0
export MAX_JOBS=64
/venv/main/bin/python -m pip install causal-conv1d --no-build-isolation > /workspace/conv1d_build.log 2>&1
echo BUILD_RC=$? >> /workspace/conv1d_build.log
