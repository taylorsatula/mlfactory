"""Reusable embedding model resource.

Wraps sentence-transformers with consistent dtype/device handling and provides
a context-manager interface for the DFT-style experiments.
"""
from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np


class Embedder:
    """Sentence-transformers wrapper with caching and normalized output."""

    _cache: dict[str, "Embedder"] = {}

    def __init__(self, model_name: str, device: str = "cuda:0", dtype: str = "float16"):
        self.model_name = model_name
        self.device = device
        self.dtype = dtype
        self._model: Any | None = None
        self.dim: int | None = None

    def _load(self) -> Any:
        if self._model is None:
            from sentence_transformers import SentenceTransformer

            self._model = SentenceTransformer(
                self.model_name,
                device=self.device,
                trust_remote_code=True,
                model_kwargs={"torch_dtype": getattr(__import__("torch"), self.dtype)},
            )
            self.dim = self._model.get_embedding_dimension()
        return self._model

    def encode(
        self,
        texts: list[str],
        batch_size: int = 32,
        normalize: bool = True,
        convert_to_numpy: bool = True,
    ) -> np.ndarray:
        model = self._load()
        return model.encode(
            texts,
            batch_size=batch_size,
            convert_to_numpy=convert_to_numpy,
            normalize_embeddings=normalize,
            show_progress_bar=False,
        )

    @classmethod
    def get(cls, model_name: str, device: str = "cuda:0") -> "Embedder":
        key = f"{model_name}@{device}"
        if key not in cls._cache:
            cls._cache[key] = cls(model_name, device=device)
        return cls._cache[key]


def embedder(model_name: str, device: str = "cuda:0") -> Embedder:
    """Return a cached Embedder instance."""
    return Embedder.get(model_name, device=device)
