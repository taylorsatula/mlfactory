#!/bin/bash
cd /tmp
export FLASH_ATTN_CUDA_ARCHS=90
export MAX_JOBS=96
/venv/main/bin/python -m pip install flash-attn --no-build-isolation > /workspace/flash_build2.log 2>&1
echo BUILD_RC=$? >> /workspace/flash_build2.log
