"""RAGBackend — the contract every retrieval backend must satisfy.

Backends share content (chunks, embeddings, LLM cache) through the shared
storage layer and keep their structural overlays isolated per backend schema.
The router and everything above it only ever sees this interface, so swapping
backends is a config change, never a code change.
"""
from __future__ import annotations

from abc import ABC, abstractmethod

from ..core.types import ConceptGraph, Document, IndexResult, QueryResult, TenantNS


class RAGBackend(ABC):
    """One instance serves every tenant namespace, isolated by `namespace` args."""

    @property
    @abstractmethod
    def name(self) -> str:
        """Stable identifier: 'hyperrag' | 'hyperrag_light' | 'hierarchical'."""

    @abstractmethod
    async def index(self, namespace: TenantNS, documents: list[Document]) -> IndexResult:
        """Ensure chunks/embeddings exist in shared content, then build this
        backend's structural overlay for the namespace."""

    @abstractmethod
    async def query(self, namespace: TenantNS, text: str, top_k: int = 60) -> QueryResult:
        """Answer using this backend's retrieval strategy, grounded in the
        namespace's corpus. `ok=False` when no grounded answer was possible."""

    @abstractmethod
    async def get_concept_graph(self, namespace: TenantNS, center: str,
                                depth: int = 2) -> ConceptGraph:
        """Local concept topology around `center` — feeds the Poincaré sphere."""

    @abstractmethod
    async def delete_namespace(self, namespace: TenantNS) -> bool:
        """Remove this backend's structure for the namespace.
        Never touches shared content chunks."""
