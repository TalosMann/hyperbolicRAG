"""CogRag Flash Backend — Cost-effective two-stage cascade variant."""
from __future__ import annotations

import os
import numpy as np

from ..core.types import (
    ConceptEdge, ConceptGraph, ConceptNode, Document, IndexResult,
    QueryResult, SourceRef, TenantNS,
)
from .base import RAGBackend
from .hierarchical_backend import (
    ANSWER_PROMPT, FAIL, SUMMARY_PROMPT, _chunk_text, _greedy_clusters, _hash_id,
)

TARGETING_RECON_PROMPT = """Based on the user query and the following thematic summaries, extract 4-5 concrete keywords that represent the most relevant aspects of the query within these themes. Output the keywords as a comma-separated list.
QUERY: {query}
SUMMARIES:
{summaries}"""


class CogRagFlashBackend(RAGBackend):
    _name = "cograg_flash"

    def __init__(self, *, llm_func, llm_fast_func=None, embedder, kv_cls, vector_cls,
                 working_dir: str = ".", pg_dsn: str | None = None,
                 chunk_size: int = 1200, chunk_overlap: int = 100,
                 cluster_threshold: float = 0.45, cosine_threshold: float | None = None,
                 fail_markers: list[str] | None = None):
        self._llm = llm_func
        self._llm_fast = llm_fast_func or llm_func
        self._embedder = embedder
        self._working_dir = working_dir
        self._kv_cls = kv_cls
        self._vector_cls = vector_cls
        self._pg_dsn = pg_dsn
        self._chunk_size = chunk_size
        self._chunk_overlap = chunk_overlap
        self._cluster_threshold = cluster_threshold
        self._cosine_threshold = cosine_threshold
        self._fail_markers = fail_markers or [FAIL]

    @property
    def name(self) -> str:
        return self._name

    def _cfg(self, namespace: TenantNS) -> dict:
        addon = {"tenant": namespace}
        if self._pg_dsn:
            addon["pg_dsn"] = self._pg_dsn
        if self._cosine_threshold is not None:
            addon["cosine_better_than_threshold"] = self._cosine_threshold
        return {"addon_params": addon, "embedding_batch_num": 8,
                "working_dir": os.path.join(self._working_dir, self._name, namespace)}

    def _stores(self, namespace: TenantNS):
        cfg = self._cfg(namespace)
        raw_chunks_kv = self._kv_cls(namespace="raw_chunks", global_config=cfg)
        raw_chunks_vdb = self._vector_cls(
            namespace="raw_chunks", global_config=cfg,
            embedding_func=self._embedder, meta_fields={"content", "full_doc_id"}
        )
        theme_targeting_vdb = self._vector_cls(
            namespace="theme_targeting", global_config=cfg,
            embedding_func=self._embedder, meta_fields={"content", "child_chunks"}
        )
        cache = self._kv_cls(namespace="llm_response_cache", global_config=cfg)
        return raw_chunks_kv, raw_chunks_vdb, theme_targeting_vdb, cache

    async def _embed_all(self, texts: list[str]) -> np.ndarray:
        out = []
        for i in range(0, len(texts), 8):
            out.append(await self._embedder(texts[i:i + 8]))
        return np.concatenate(out) if out else np.zeros((0, self._embedder.embedding_dim))

    async def index(self, namespace: TenantNS, documents: list[Document]) -> IndexResult:
        import pathlib
        pathlib.Path(self._cfg(namespace)["working_dir"]).mkdir(parents=True, exist_ok=True)
        
        raw_chunks_kv, raw_chunks_vdb, theme_targeting_vdb, cache = self._stores(namespace)

        all_chunks = {}
        chunk_texts = []
        chunk_ids = []
        
        for d in documents:
            # handle `d.id` vs `doc_id`, `hierarchical_backend` uses `_hash_id(d.content, "doc-")`. Let's use `d.id` if available, otherwise hash.
            doc_id = getattr(d, 'id', None) or _hash_id(d.content, "doc-")
            for i, piece in enumerate(_chunk_text(d.content, self._chunk_size, self._chunk_overlap)):
                cid = _hash_id(piece, "chunk-")
                all_chunks[cid] = {"content": piece, "full_doc_id": doc_id}
                chunk_texts.append(piece)
                chunk_ids.append(cid)

        if not all_chunks:
            return IndexResult(namespace=namespace, documents=len(documents), chunks=0, backend=self.name)

        await raw_chunks_vdb.upsert(all_chunks)
        await raw_chunks_kv.upsert(all_chunks)

        vecs = await self._embed_all(chunk_texts)
        clusters = _greedy_clusters(vecs, self._cluster_threshold)
        
        theme_payload = {}
        for cl in clusters:
            passages = "\n---\n".join(chunk_texts[i][:1500] for i in cl)
            summary = await self._llm(SUMMARY_PROMPT.format(passages=passages), hashing_kv=cache)
            tid = _hash_id(summary, "theme-")
            theme_payload[tid] = {
                "content": summary,
                "child_chunks": [chunk_ids[i] for i in cl]
            }
            
        await theme_targeting_vdb.upsert(theme_payload)

        for store in (raw_chunks_kv, raw_chunks_vdb, theme_targeting_vdb, cache):
            cb = getattr(store, "index_done_callback", None)
            if cb is not None:
                await cb()

        return IndexResult(
            namespace=namespace, documents=len(documents), chunks=len(chunk_ids),
            backend=self.name, detail={"themes": len(theme_payload)}
        )

    async def query(self, namespace: TenantNS, text: str, top_k: int = 60) -> QueryResult:
        raw_chunks_kv, raw_chunks_vdb, theme_targeting_vdb, cache = self._stores(namespace)
        
        # Stage 1: Targeting
        theme_hits = await theme_targeting_vdb.query(text, top_k=2)
        summaries = [h.get("content", "") for h in theme_hits if h.get("content")]
        
        if summaries:
            sum_str = "\n\n".join(summaries)
            target_kws_raw = await self._llm_fast(TARGETING_RECON_PROMPT.format(query=text, summaries=sum_str), hashing_kv=cache)
            target_kws = target_kws_raw.strip()
            enriched_query = f"{text} {target_kws}"
        else:
            enriched_query = text

        # Stage 2: Retrieval
        chunk_hits = await raw_chunks_vdb.query(enriched_query, top_k=8)
        
        passages, sources = [], []
        chunk_rows = await raw_chunks_kv.get_by_ids([h["id"] for h in chunk_hits])
        for h, row in zip(chunk_hits, chunk_rows):
            content = (row or {}).get("content", "")
            if content:
                passages.append((h.get("distance", 0), content))
                sources.append(
                    SourceRef(chunk_id=h["id"], doc_id=(row or {}).get("full_doc_id", ""),
                              score=float(h.get("distance", 0)), excerpt=content[:240])
                )
                
        if not passages:
            return QueryResult(answer=FAIL, backend=self.name, namespace=namespace, mode="cograg_flash", ok=False)

        passages.sort(key=lambda x: -x[0])
        context = "\n\n".join(p for _, p in passages[:8])[:24000]
        
        answer = await self._llm(ANSWER_PROMPT.format(context=context, question=text, fail=FAIL), hashing_kv=cache)
        ok = bool(answer) and not any(m in answer for m in self._fail_markers)
        return QueryResult(answer=answer, sources=sources, backend=self.name, namespace=namespace, mode="cograg_flash", ok=ok)

    async def get_concept_graph(self, namespace: TenantNS, center: str, depth: int = 2) -> ConceptGraph:
        return ConceptGraph(center=center, nodes=[], edges=[])

    async def delete_namespace(self, namespace: TenantNS) -> bool:
        raw_chunks_kv, raw_chunks_vdb, theme_targeting_vdb, cache = self._stores(namespace)
        await raw_chunks_kv.drop()
        if self._pg_dsn:
            from ..storage.pg import reset_backend_structures
            await reset_backend_structures(self._pg_dsn, self._name, tenant=namespace)
        return True
