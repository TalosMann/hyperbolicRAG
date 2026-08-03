"""Factory: config → wired (backend, router).

The one-line backend swap lives here. `config.yaml`:

    rag:
      backend: hyperrag        # ← change to hyperrag_light | hierarchical

Everything else — storage classes, embedder, LLM, namespaces — stays identical,
which is what makes backend comparisons fair.
"""
from __future__ import annotations

from ..core.config import Config, load_config
from ..core.embedder import build_embedder
from .base import RAGBackend
from .hierarchical_backend import HierarchicalRAGBackend
from .hyperrag_backend import HyperRAGBackend, HyperRAGLightBackend
from .pure_cograg_backend import PureCogRAGBackend
from .cograg_flash_backend import CogRagFlashBackend
from .cograg_official_backend import CogRAGOfficialBackend
from .router import RAGRouter

BACKENDS = {
    "hyperrag": HyperRAGBackend,
    "hyperrag_light": HyperRAGLightBackend,
    "hierarchical": HierarchicalRAGBackend,
    "pure_cograg": PureCogRAGBackend,
    "cograg_flash": CogRagFlashBackend,
    "cograg_official": CogRAGOfficialBackend,
    "cograg_safe": None,   # resolved lazily — cograg_safe.backend imports rag.base,
                           # so a top-level import here would be circular
}


def _backend_cls(name: str):
    cls = BACKENDS[name]
    if cls is None and name == "cograg_safe":
        from ..cograg_safe.backend import CogRAGSafeBackend
        BACKENDS[name] = cls = CogRAGSafeBackend
    return cls


def storage_classes(cfg: Config):
    """(kv_cls, vector_cls, hypergraph_cls, pg_dsn) for whichever store the
    config points at. Shared by build_backend and the eval/ scripts so every
    entrypoint agrees on where data lives — memory:// or Postgres, never a
    mix within one pipeline run."""
    if cfg.store.dsn.startswith("memory://"):
        from ..storage.memory import (
            MemoryHypergraphStorage, MemoryKVStorage, MemoryVectorStorage,
        )
        return MemoryKVStorage, MemoryVectorStorage, MemoryHypergraphStorage, None
    from ..storage.pg import PGHypergraphStorage, PGKVStorage, PGVectorStorage
    return PGKVStorage, PGVectorStorage, PGHypergraphStorage, cfg.store.dsn


def build_backend(cfg: Config | None = None, *, llm_func=None,
                  embedder=None) -> RAGBackend:
    cfg = cfg or load_config()
    name = cfg.rag.backend
    if name not in BACKENDS:
        raise ValueError(f"Unknown rag.backend '{name}'. "
                         f"Options: {', '.join(BACKENDS)}")

    kv_cls, vector_cls, hg_cls, pg_dsn = storage_classes(cfg)
    embedder = embedder or build_embedder(cfg.embedding)
    if llm_func is None:
        from ..core.llm import build_llm_func
        llm_func = build_llm_func(cfg.llm)

    if name in ("hyperrag", "hyperrag_light"):
        return BACKENDS[name](
            llm_func=llm_func, embedder=embedder,
            working_dir=cfg.working_dir,
            kv_cls=kv_cls, vector_cls=vector_cls, hypergraph_cls=hg_cls,
            pg_dsn=pg_dsn, fail_markers=cfg.rag.fail_markers)
    elif name == "pure_cograg":
        return BACKENDS[name](
            llm_func=llm_func, embedder=embedder,
            working_dir=cfg.working_dir,
            kv_cls=kv_cls, vector_cls=vector_cls,
            pg_dsn=pg_dsn, fail_markers=cfg.rag.fail_markers)
    elif name == "cograg_flash":
        from ..core.llm import build_llm_func
        llm_fast_func = build_llm_func(cfg.llm_fast) if hasattr(cfg, "llm_fast") and cfg.llm_fast else llm_func
        return BACKENDS[name](
            llm_func=llm_func, llm_fast_func=llm_fast_func, embedder=embedder,
            working_dir=cfg.working_dir,
            kv_cls=kv_cls, vector_cls=vector_cls,
            pg_dsn=pg_dsn, fail_markers=cfg.rag.fail_markers)
    elif name == "hierarchical":
        return BACKENDS[name](
            llm_func=llm_func, embedder=embedder,
            working_dir=cfg.working_dir,
            kv_cls=kv_cls, vector_cls=vector_cls,
            pg_dsn=pg_dsn, fail_markers=cfg.rag.fail_markers)
    elif name in ("cograg_official", "cograg_safe"):
        return _backend_cls(name)(
            llm_func=llm_func, embedder=embedder,
            working_dir=cfg.working_dir,
            fail_markers=cfg.rag.fail_markers)


def build_router(cfg: Config | None = None, *, llm_func=None,
                 embedder=None) -> RAGRouter:
    cfg = cfg or load_config()
    backend = build_backend(cfg, llm_func=llm_func, embedder=embedder)
    return RAGRouter(backend, min_sources=cfg.rag.min_sources)
