"""CogRAGOfficialBackend — wraps the official haoohu/Cog-RAG implementation.

The official Cog-RAG (AAAI 2026, arXiv:2511.13201) is cloned at
Cog-RAG/ alongside hyperscholar/ and made importable via cograg_local.pth.

Storage uses Cog-RAG's native file-based backends (JsonKVStorage +
NanoVectorDBStorage + HypergraphStorage) under a per-namespace working dir.
Content-hash deduplication inside ainsert() makes re-runs idempotent.

Default query mode is "cog" (dual-hypergraph with theme alignment), which is
the paper's primary contribution. Other modes: cog-hybrid, cog-entity,
cog-theme, naive.
"""
from __future__ import annotations

import asyncio
import os
from typing import Callable

from ..core.types import (
    ConceptEdge, ConceptGraph, ConceptNode, Document, IndexResult,
    QueryResult, SourceRef, TenantNS,
)
from .base import RAGBackend

FAIL_DEFAULT = "I cannot answer based on the available information."


class CogRAGOfficialBackend(RAGBackend):
    _name = "cograg_official"

    def __init__(self, *, llm_func: Callable, embedder,
                 working_dir: str = ".",
                 query_mode: str = "cog",
                 fail_markers: list[str] | None = None,
                 cograg_kwargs: dict | None = None):
        self._llm = llm_func
        self._embedder = embedder
        self._working_dir = working_dir
        self._query_mode = query_mode
        self._fail_markers = fail_markers or [FAIL_DEFAULT]
        self._kw = cograg_kwargs or {}
        self._instances: dict[str, object] = {}

    @property
    def name(self) -> str:
        return self._name

    def _rag(self, namespace: TenantNS):
        if namespace not in self._instances:
            try:
                from cograg import CogRAG, QueryParam
                from cograg.utils import EmbeddingFunc
            except ImportError as e:
                raise RuntimeError(
                    "The `cograg` package is not importable. Ensure "
                    "Cog-RAG/ is cloned and cograg_local.pth points to it."
                ) from e

            # Wrap our LLM so it silently accepts the hashing_kv kwarg that
            # CogRAG's __post_init__ bakes in via functools.partial.
            _llm = self._llm

            async def _llm_adapter(prompt: str, hashing_kv=None, **kw) -> str:
                return await _llm(prompt)

            workdir = os.path.join(self._working_dir, self._name, namespace)
            os.makedirs(workdir, exist_ok=True)

            self._instances[namespace] = CogRAG(
                working_dir=workdir,
                llm_model_func=_llm_adapter,
                embedding_func=EmbeddingFunc(
                    embedding_dim=self._embedder.embedding_dim,
                    max_token_size=self._embedder.max_token_size,
                    func=self._embedder,
                ),
                # NanoVectorDBStorage batches with asyncio.gather over all
                # batches at once; limit to 1 concurrent embedding call so
                # the MPS/CPU encoder isn't flooded and segfaults.
                embedding_func_max_async=1,
                **self._kw,
            )
        return self._instances[namespace]

    async def index(self, namespace: TenantNS, documents: list[Document]) -> IndexResult:
        rag = self._rag(namespace)
        texts = [d.content for d in documents]
        await rag.ainsert(texts)
        n_chunks = len(await rag.text_chunks.all_keys())
        return IndexResult(namespace=namespace, documents=len(documents),
                           chunks=n_chunks, backend=self.name)

    async def query(self, namespace: TenantNS, text: str, top_k: int = 60) -> QueryResult:
        from cograg import QueryParam
        rag = self._rag(namespace)
        try:
            answer = await rag.aquery(text, QueryParam(mode=self._query_mode, top_k=top_k))
        except Exception as e:
            return QueryResult(answer=self._fail_markers[0], backend=self.name,
                               namespace=namespace, mode=self._query_mode, ok=False,
                               raw={"error": f"{type(e).__name__}: {e}"})
        ok = bool(answer) and not any(m in answer for m in self._fail_markers)
        return QueryResult(answer=answer or "", sources=await self._top_sources(rag, text),
                           backend=self.name, namespace=namespace,
                           mode=self._query_mode, ok=ok)

    async def _top_sources(self, rag, text: str, k: int = 5) -> list[SourceRef]:
        try:
            hits = await rag.chunks_vdb.query(text, top_k=k)
            ids = [h["id"] for h in hits]
            chunks = await rag.text_chunks.get_by_ids(ids)
            return [
                SourceRef(chunk_id=h["id"],
                          doc_id=(c or {}).get("full_doc_id", ""),
                          score=float(h.get("distance", 0.0)),
                          excerpt=((c or {}).get("content", "") or "")[:240])
                for h, c in zip(hits, chunks)
            ]
        except Exception:
            return []

    async def get_concept_graph(self, namespace: TenantNS, center: str,
                                depth: int = 2) -> ConceptGraph:
        rag = self._rag(namespace)
        hg = rag.chunk_entity_relation_hypergraph

        center_id = None
        if await hg.has_vertex(center.upper()):
            center_id = center.upper()
        elif await hg.has_vertex(center):
            center_id = center
        else:
            hits = await rag.entities_vdb.query(center, top_k=1)
            if hits:
                center_id = hits[0].get("entity_name", hits[0]["id"])
        if center_id is None or not await hg.has_vertex(center_id):
            return ConceptGraph(center=center)

        nodes: dict[str, ConceptNode] = {}
        edges: list[ConceptEdge] = []

        async def add_node(v_id: str, level: int, is_center: bool = False):
            if v_id in nodes:
                return
            data = await hg.get_vertex(v_id) or {}
            summary = data.get("description") or data.get("additional_properties") or ""
            nodes[v_id] = ConceptNode(
                id=v_id, label=v_id.strip('"'),
                summary=str(summary).split("<SEP>")[0][:280],
                level=level, is_center=is_center)

        await add_node(center_id, 0, True)
        frontier = [center_id]
        for d in range(1, depth + 1):
            nxt = []
            for v in frontier:
                for nb in await hg.get_nbr_v_of_vertex(v):
                    if nb not in nodes:
                        await add_node(nb, d)
                        nxt.append(nb)
                    edges.append(ConceptEdge(source=v, target=nb))
            frontier = nxt
        seen, uniq = set(), []
        for e in edges:
            k = tuple(sorted((e.source, e.target)))
            if k not in seen:
                seen.add(k)
                uniq.append(e)
        return ConceptGraph(center=center_id, nodes=list(nodes.values()), edges=uniq)

    async def delete_namespace(self, namespace: TenantNS) -> bool:
        rag = self._rag(namespace)
        hg = rag.chunk_entity_relation_hypergraph
        for e in list((await hg.get_all_hyperedges()).keys()):
            await hg.remove_hyperedge(e)
        for v in list((await hg.get_all_vertices()).keys()):
            await hg.remove_vertex(v)
        self._instances.pop(namespace, None)
        return True
