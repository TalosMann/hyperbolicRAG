r"""CogRAGSafeBackend — RAGBackend adapter for the zero-hallucination pipeline.

Indexing, storage, and the concept graph delegate to the vendored Cog-RAG
(via CogRAGOfficialBackend) — nothing there changes. Only QUERY is replaced:
Cog-RAG's drift-prone synthesis prompts are dead code on this path; answers go
through bind -> hybrid retrieve -> assert -> verify (R1/R2) -> compose.
"""
from __future__ import annotations

import os
from typing import Callable

from ..core.types import (ConceptGraph, Document, IndexResult, QueryResult,
                          SourceRef, TenantNS)
from ..rag.base import RAGBackend
from ..rag.cograg_official_backend import CogRAGOfficialBackend
from .evidence import EvidencePack
from .pipeline import safe_answer

DECLINE_TEXT = "The source material does not contain an answer to this question."


class CogRAGSafeBackend(RAGBackend):
    _name = "cograg_safe"

    def __init__(self, *, llm_func: Callable, embedder,
                 working_dir: str = ".",
                 fail_markers: list[str] | None = None):
        self._llm = llm_func
        self._inner = CogRAGOfficialBackend(
            llm_func=llm_func, embedder=embedder, working_dir=working_dir,
            fail_markers=fail_markers)
        self._working_dir = working_dir
        self._packs: dict[str, EvidencePack] = {}

    @property
    def name(self) -> str:
        return self._name

    def _pack(self, namespace: TenantNS) -> EvidencePack:
        if namespace not in self._packs:
            runtime = os.path.join(self._working_dir, "cograg_official")
            self._packs[namespace] = EvidencePack.load(runtime, namespace)
        return self._packs[namespace]

    def invalidate(self, namespace: TenantNS) -> None:
        self._packs.pop(namespace, None)

    async def index(self, namespace: TenantNS, documents: list[Document]) -> IndexResult:
        res = await self._inner.index(namespace, documents)
        self.invalidate(namespace)          # pack is stale after new content
        return res

    async def query(self, namespace: TenantNS, text: str, top_k: int = 60) -> QueryResult:
        pack = self._pack(namespace)
        rag = self._inner._rag(namespace)
        try:
            r = await safe_answer(pack, rag, self._llm, text)
        except Exception as e:
            return QueryResult(answer=DECLINE_TEXT, backend=self.name,
                               namespace=namespace, mode="safe", ok=False,
                               raw={"error": f"{type(e).__name__}: {e}"})
        ok = r["decision"] == "SERVE"
        sources = [SourceRef(chunk_id=cid, doc_id="", score=0.0,
                             excerpt=pack.chunks.get(cid, "")[:240])
                   for cid in r.get("evidence_chunks", [])[:5]]
        return QueryResult(answer=r["answer"], sources=sources, backend=self.name,
                           namespace=namespace, mode="safe", ok=ok,
                           raw={k: r[k] for k in
                                ("intent", "mentions", "verified_relations",
                                 "rejected_relations", "verified_facts", "dropped_facts",
                                 "n_relations_asserted", "residual_flagged")
                                if k in r})

    async def get_concept_graph(self, namespace: TenantNS, center: str,
                                depth: int = 2) -> ConceptGraph:
        return await self._inner.get_concept_graph(namespace, center, depth)

    async def delete_namespace(self, namespace: TenantNS) -> bool:
        self.invalidate(namespace)
        return await self._inner.delete_namespace(namespace)
