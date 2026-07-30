"""Environment context managers for reproducible ML runs.

Sets allocator, CUDA, and HF variables for the duration of an experiment and
restores the previous values on exit.
"""
from __future__ import annotations

import os
from contextlib import contextmanager
from typing import Iterator


DEFAULT_TRAINING_ENV = {
    "PYTORCH_ALLOC_CONF": "expandable_segments:True,roundup_power2_divisions:[32:256,64:128,256:64,>:32]",
    "PYTORCH_CUDA_ALLOC_CONF": "expandable_segments:True,roundup_power2_divisions:[32:256,64:128,256:64,>:32]",
    "TRITON_DISABLE_AUTOTUNING": "1",
    "PYTHONUNBUFFERED": "1",
}


@contextmanager
def env_guard(
    variables: dict[str, str | None],
    overwrite: bool = False,
) -> Iterator[None]:
    """Temporarily set environment variables, then restore them.

    If a value is ``None``, the variable is deleted.
    """
    previous: dict[str, str | None] = {}
    try:
        for key, value in variables.items():
            previous[key] = os.environ.get(key)
            if value is None:
                os.environ.pop(key, None)
            elif overwrite or key not in os.environ or not os.environ[key]:
                os.environ[key] = value
        yield
    finally:
        for key, old in previous.items():
            if old is None:
                os.environ.pop(key, None)
            else:
                os.environ[key] = old


@contextmanager
def training_env(
    hf_home: str | None = None,
    extra: dict[str, str] | None = None,
) -> Iterator[None]:
    """Standard training environment used by DFT and similar experiments."""
    env = dict(DEFAULT_TRAINING_ENV)
    if hf_home:
        env["HF_HOME"] = hf_home
    if extra:
        env.update(extra)
    with env_guard(env, overwrite=False):
        yield


@contextmanager
def inference_env(
    hf_home: str | None = None,
    extra: dict[str, str] | None = None,
) -> Iterator[None]:
    """Minimal inference environment: unbuffered output and optional HF cache."""
    env: dict[str, str] = {"PYTHONUNBUFFERED": "1"}
    if hf_home:
        env["HF_HOME"] = hf_home
    if extra:
        env.update(extra)
    with env_guard(env, overwrite=False):
        yield
