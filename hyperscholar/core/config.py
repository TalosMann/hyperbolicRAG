"""Config loader — one YAML file drives backend selection and all wiring.

Switching RAG backends is a one-line change:

    rag:
      backend: hyperrag        # hyperrag | hyperrag_light | hierarchical
"""
from __future__ import annotations

import os
import re
from dataclasses import dataclass, field
from pathlib import Path

import yaml
from dotenv import load_dotenv
load_dotenv(Path(__file__).resolve().parent.parent / ".env")

_ENV_RE = re.compile(r"\$\{([A-Z0-9_]+)(?::([^}]*))?\}")

DEFAULT_CONFIG_PATH = Path(__file__).resolve().parent.parent / "config.yaml"


def _interp(value):
    """${VAR} / ${VAR:default} environment interpolation, recursively."""
    if isinstance(value, str):
        def sub(m):
            return os.environ.get(m.group(1), m.group(2) if m.group(2) is not None else "")
        return _ENV_RE.sub(sub, value)
    if isinstance(value, dict):
        return {k: _interp(v) for k, v in value.items()}
    if isinstance(value, list):
        return [_interp(v) for v in value]
    return value


@dataclass
class RAGConfig:
    backend: str = "hyperrag"            # hyperrag | hyperrag_light | hierarchical
    top_k: int = 60
    min_sources: int = 1                 # router sufficiency threshold
    fail_markers: list = field(default_factory=lambda: [
        "Sorry, I'm not able to provide an answer to that question",
    ])


@dataclass
class EmbeddingConfig:
    model: str = "BAAI/bge-m3"
    dim: int = 1024
    device: str = "mps"
    batch_size: int = 8
    max_token_size: int = 8192


@dataclass
class LLMProviderConfig:
    name: str = "deepseek"
    base_url: str = "https://api.deepseek.com/v1"
    model: str = "deepseek-chat"
    api_key_env: str = "DEEPSEEK_API_KEY"


@dataclass
class LLMConfig:
    # Legacy single-provider fields (kept for backward compat)
    provider: str = "deepseek"
    base_url: str = "https://api.deepseek.com/v1"
    model: str = "deepseek-chat"
    api_key_env: str = "DEEPSEEK_API_KEY"
    # Multi-provider fallback list (new)
    providers: list = field(default_factory=list)


@dataclass
class StoreConfig:
    # "memory://"            → in-process stores (dev / tests, no Postgres needed)
    # "postgresql://..."     → pgvector, schemas: shared / hyperrag / hierarchical
    dsn: str = "memory://"
    vector_dim: int = 1024


@dataclass
class HyperRAGConfig:
    # Bounds asyncio.gather concurrency during entity extraction. Set low for
    # rate-limited free-tier providers (Gemini: 15 RPM → max_async: 3 is safe).
    # Maps to HyperRAG's `llm_model_max_async` constructor kwarg.
    max_async: int = 4
    # Fewer LLM "gleaning" passes per chunk during entity extraction — reduces
    # total LLM calls at some cost to extraction thoroughness.
    entity_extract_max_gleaning: int = 1


@dataclass
class Config:
    rag: RAGConfig = field(default_factory=RAGConfig)
    embedding: EmbeddingConfig = field(default_factory=EmbeddingConfig)
    llm: LLMConfig = field(default_factory=LLMConfig)
    store: StoreConfig = field(default_factory=StoreConfig)
    hyperrag: HyperRAGConfig = field(default_factory=HyperRAGConfig)
    working_dir: str = "./hyperscholar_runtime"


def load_config(path: str | Path | None = None) -> Config:
    p = Path(path) if path else DEFAULT_CONFIG_PATH
    raw = {}
    if p.exists():
        raw = _interp(yaml.safe_load(p.read_text()) or {})
    raw_llm = raw.get("llm", {})
    # Build LLMConfig — handle both old single-provider and new providers-list format
    providers_raw = raw_llm.pop("providers", [])
    providers = [LLMProviderConfig(**p) for p in providers_raw]
    llm_cfg = LLMConfig(**{k: v for k, v in raw_llm.items()
                           if k in LLMConfig.__dataclass_fields__
                           and k != "providers"})
    llm_cfg.providers = providers

    _raw_wd = raw.get("working_dir", "./hyperscholar_runtime")
    _abs_wd = str((Path(__file__).resolve().parent.parent / _raw_wd.lstrip("./")).resolve())

    cfg = Config(
        rag=RAGConfig(**raw.get("rag", {})),
        embedding=EmbeddingConfig(**raw.get("embedding", {})),
        llm=llm_cfg,
        store=StoreConfig(**raw.get("store", {})),
        hyperrag=HyperRAGConfig(**raw.get("hyperrag", {})),
        working_dir=_abs_wd,
    )
    # keep dims in sync: the vector store dimension must equal the embedder output
    cfg.store.vector_dim = cfg.embedding.dim
    return cfg
