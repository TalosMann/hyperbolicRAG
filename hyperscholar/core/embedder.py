"""Embedding functions.

`BgeM3Embedder` — production: local `BAAI/bge-m3` via sentence-transformers,
MPS-accelerated, async via run_in_executor (network-constraint-safe: no
runtime calls to blocked Western endpoints).

`HashEmbedder` — deterministic, dependency-free embedder for tests and CI.
Texts sharing vocabulary land near each other in cosine space, which is enough
to exercise retrieval logic end-to-end without downloading a model.

Both expose the attributes HyperRAG's storage layer expects on an embedding
function: `.embedding_dim`, `.max_token_size`, and `async __call__(texts)`.
"""
from __future__ import annotations

import asyncio
import hashlib
import re

import numpy as np


class BaseEmbedder:
    embedding_dim: int = 1024
    max_token_size: int = 8192

    async def __call__(self, texts: list[str]) -> np.ndarray:  # (n, dim)
        raise NotImplementedError


class BgeM3Embedder(BaseEmbedder):
    """Local bge-m3. Loaded lazily so importing this module never requires torch."""

    def __init__(self, model_name: str = "BAAI/bge-m3", device: str = "auto",
                 dim: int = 1024, max_token_size: int = 8192):
        self.model_name = model_name
        self.device = self._resolve_device(device)
        self.embedding_dim = dim
        self.max_token_size = max_token_size
        self._model = None

    @staticmethod
    def _resolve_device(device: str) -> str:
        if device != "auto":
            return device
        try:
            import torch
            if torch.cuda.is_available():
                return "cuda"
            if torch.backends.mps.is_available():
                return "mps"
        except ImportError:
            pass
        return "cpu"

    def _ensure_model(self):
        if self._model is None:
            from sentence_transformers import SentenceTransformer
            self._model = SentenceTransformer(self.model_name, device=self.device)

    async def __call__(self, texts: list[str]) -> np.ndarray:
        self._ensure_model()
        loop = asyncio.get_event_loop()

        def _encode():
            return self._model.encode(
                texts, normalize_embeddings=True, convert_to_numpy=True
            )

        vecs = await loop.run_in_executor(None, _encode)
        return np.asarray(vecs, dtype=np.float32)


class HashEmbedder(BaseEmbedder):
    """Deterministic bag-of-hashed-words embedding. Tests/CI only."""

    def __init__(self, dim: int = 256):
        self.embedding_dim = dim
        self.max_token_size = 8192

    def _embed_one(self, text: str) -> np.ndarray:
        vec = np.zeros(self.embedding_dim, dtype=np.float32)
        for tok in re.findall(r"[a-z0-9]+", text.lower()):
            h = int(hashlib.md5(tok.encode()).hexdigest(), 16)
            idx = h % self.embedding_dim
            sign = 1.0 if (h >> 8) % 2 == 0 else -1.0
            vec[idx] += sign
        n = np.linalg.norm(vec)
        return vec / n if n > 0 else vec

    async def __call__(self, texts: list[str]) -> np.ndarray:
        return np.stack([self._embed_one(t) for t in texts])


def build_embedder(cfg) -> BaseEmbedder:
    """cfg: core.config.EmbeddingConfig"""
    if cfg.model in ("hash", "test"):
        return HashEmbedder(dim=cfg.dim)
    return BgeM3Embedder(model_name=cfg.model, device=cfg.device, dim=cfg.dim,
                         max_token_size=cfg.max_token_size)
