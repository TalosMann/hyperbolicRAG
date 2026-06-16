"""HierarchicalRAGBackend — RAPTOR-style summary tree over the shared chunks.

Self-contained (no `hyperrag` import): it uses the same injected storage
classes, reading/writing the shared content layer (full_docs / text_chunks /
chunk vectors / LLM cache) and keeping its structural overlay — the summary
tree — in its own storage namespace (`tree_nodes` → hierarchical schema in PG).

Index:
  1. chunk documents (character-based, consistent with the project's
     tiktoken-free constraint) → shared content
  2. embed chunks → shared chunk vectors
  3. cluster chunk embeddings (greedy cosine clustering, dependency-free)
  4. LLM-summarize each cluster → level-1 tree nodes (embedded)
  5. recurse over node embeddings until few enough roots

Query (collapsed-tree retrieval):
  vector-search tree nodes + raw chunks together, take the best passages,
  build a context block, ask the LLM to answer strictly from it.
"""
from __future__ import annotations

import hashlib
import os

import numpy as np

from ..core.types import (
    ConceptEdge, ConceptGraph, ConceptNode, Document, IndexResult,
    QueryResult, SourceRef, TenantNS,
)
from .base import RAGBackend

SUMMARY_PROMPT = """You are building a hierarchical study index. Summarize the following passages into one cohesive paragraph that preserves key facts, names, and relationships. Output only the summary.

PASSAGES:
{passages}
"""

ANSWER_PROMPT = """Answer the question using ONLY the context below. If the context does not contain the answer, reply exactly: "{fail}"

CONTEXT:
{context}

QUESTION: {question}
"""

FAIL = "Sorry, I'm not able to provide an answer to that question."


def _hash_id(text: str, prefix: str) -> str:
    return prefix + hashlib.md5(text.encode()).hexdigest()


def _chunk_text(text: str, size: int = 1200, overlap: int = 100) -> list[str]:
    """Character-based chunking (project constraint: no tiktoken at runtime)."""
    text = text.strip()
    if not text:
        return []
    out, step = [], max(1, size - overlap)
    for start in range(0, len(text), step):
        piece = text[start:start + size]
        if piece.strip():
            out.append(piece)
        if start + size >= len(text):
            break
    return out


def _greedy_clusters(vectors: np.ndarray, threshold: float = 0.45,
                     max_size: int = 8) -> list[list[int]]:
    """Dependency-free clustering: assign each vector to the first cluster whose
    centroid is within cosine `threshold`, else start a new cluster."""
    clusters: list[list[int]] = []
    centroids: list[np.ndarray] = []
    for i, v in enumerate(vectors):
        nv = v / (np.linalg.norm(v) or 1.0)
        best, best_sim = None, threshold
        for ci, c in enumerate(centroids):
            if len(clusters[ci]) >= max_size:
                continue
            sim = float(np.dot(nv, c / (np.linalg.norm(c) or 1.0)))
            if sim > best_sim:
                best, best_sim = ci, sim
        if best is None:
            clusters.append([i])
            centroids.append(nv.copy())
        else:
            clusters[best].append(i)
            centroids[best] = (centroids[best] * (len(clusters[best]) - 1) + nv) / len(clusters[best])
    return clusters


class HierarchicalRAGBackend(RAGBackend):
    _name = "hierarchical"

    def __init__(self, *, llm_func, embedder, kv_cls, vector_cls,
                 pg_dsn: str | None = None,
                 chunk_size: int = 1200, chunk_overlap: int = 100,
                 cluster_threshold: float = 0.45, max_levels: int = 3,
                 cosine_threshold: float | None = None,
                 fail_markers: list[str] | None = None):
        self._llm = llm_func
        self._embedder = embedder
        self._kv_cls = kv_cls
        self._vector_cls = vector_cls
        self._pg_dsn = pg_dsn
        self._chunk_size = chunk_size
        self._chunk_overlap = chunk_overlap
        self._cluster_threshold = cluster_threshold
        self._max_levels = max_levels
        self._cosine_threshold = cosine_threshold
        self._fail_markers = fail_markers or [FAIL]

    @property
    def name(self) -> str:
        return self._name

    # ── storage handles per tenant ─────────────────────────────────────────────
    def _cfg(self, namespace: TenantNS) -> dict:
        addon = {"tenant": namespace}
        if self._pg_dsn:
            addon["pg_dsn"] = self._pg_dsn
        if self._cosine_threshold is not None:
            addon["cosine_better_than_threshold"] = self._cosine_threshold
        return {"addon_params": addon, "embedding_batch_num": 8,
                "working_dir": os.path.join("hierarchical_runtime", namespace)}

    def _stores(self, namespace: TenantNS):
        cfg = self._cfg(namespace)
        docs = self._kv_cls(namespace="full_docs", global_config=cfg)
        chunks = self._kv_cls(namespace="text_chunks", global_config=cfg)
        cache = self._kv_cls(namespace="llm_response_cache", global_config=cfg)
        chunks_vdb = self._vector_cls(namespace="chunks", global_config=cfg,
                                      embedding_func=self._embedder, meta_fields=set())
        tree_vdb = self._vector_cls(namespace="tree_nodes", global_config=cfg,
                                    embedding_func=self._embedder,
                                    meta_fields={"level", "children", "content"})
        tree_kv = self._kv_cls(namespace="tree_nodes", global_config=cfg)
        return docs, chunks, cache, chunks_vdb, tree_vdb, tree_kv

    # ── RAGBackend contract ───────────────────────────────────────────────────
    async def index(self, namespace: TenantNS, documents: list[Document]) -> IndexResult:
        docs, chunks, cache, chunks_vdb, tree_vdb, tree_kv = self._stores(namespace)

        # 1–2: shared content (skip work already present — shared across backends)
        new_docs = {_hash_id(d.content, "doc-"): {"content": d.content, "title": d.title}
                    for d in documents}
        new_doc_keys = await docs.filter_keys(list(new_docs.keys()))
        new_docs = {k: v for k, v in new_docs.items() if k in new_doc_keys}

        all_chunks = {}
        for doc_id, d in new_docs.items():
            for i, piece in enumerate(_chunk_text(d["content"], self._chunk_size,
                                                  self._chunk_overlap)):
                all_chunks[_hash_id(piece, "chunk-")] = {
                    "content": piece, "full_doc_id": doc_id, "chunk_order_index": i}
        new_chunk_keys = await chunks.filter_keys(list(all_chunks.keys()))
        new_chunks = {k: v for k, v in all_chunks.items() if k in new_chunk_keys}
        if new_chunks:
            await chunks_vdb.upsert(new_chunks)
            await chunks.upsert(new_chunks)
        await docs.upsert(new_docs)

        # 3–5: structural overlay — rebuild the tree over the namespace's full corpus
        chunk_ids = await chunks.all_keys()
        chunk_rows = await chunks.get_by_ids(chunk_ids)
        contents = [(cid, (row or {}).get("content", ""))
                    for cid, row in zip(chunk_ids, chunk_rows) if row]
        n_nodes = 0
        if contents:
            level_ids = [c[0] for c in contents]
            level_texts = [c[1] for c in contents]
            for level in range(1, self._max_levels + 1):
                if len(level_ids) <= 2:
                    break
                vecs = await self._embed_all(level_texts)
                clusters = _greedy_clusters(vecs, self._cluster_threshold)
                if len(clusters) >= len(level_ids):   # nothing merged → stop
                    break
                next_ids, next_texts, node_payload = [], [], {}
                for cl in clusters:
                    passages = "\n---\n".join(level_texts[i][:1500] for i in cl)
                    summary = await self._llm(
                        SUMMARY_PROMPT.format(passages=passages), hashing_kv=cache)
                    nid = _hash_id(summary + str(level), "tree-")
                    node_payload[nid] = {
                        "content": summary, "level": level,
                        "children": [level_ids[i] for i in cl]}
                    next_ids.append(nid)
                    next_texts.append(summary)
                await tree_vdb.upsert(node_payload)
                await tree_kv.upsert(node_payload)
                n_nodes += len(node_payload)
                level_ids, level_texts = next_ids, next_texts

        return IndexResult(namespace=namespace, documents=len(documents),
                           chunks=len(chunk_ids), backend=self.name,
                           detail={"tree_nodes": n_nodes,
                                   "new_chunks": len(new_chunks),
                                   "reused_shared_chunks": len(chunk_ids) - len(new_chunks)})

    async def _embed_all(self, texts: list[str]) -> np.ndarray:
        out = []
        for i in range(0, len(texts), 8):
            out.append(await self._embedder(texts[i:i + 8]))
        return np.concatenate(out) if out else np.zeros((0, self._embedder.embedding_dim))

    async def query(self, namespace: TenantNS, text: str, top_k: int = 60) -> QueryResult:
        docs, chunks, cache, chunks_vdb, tree_vdb, tree_kv = self._stores(namespace)
        k = max(4, min(top_k, 12))
        tree_hits = await tree_vdb.query(text, top_k=k)
        chunk_hits = await chunks_vdb.query(text, top_k=k)

        passages, sources = [], []
        for h in tree_hits:
            node = await tree_kv.get_by_id(h["id"]) or {}
            content = node.get("content") or h.get("content", "")
            if content:
                passages.append((h.get("distance", 0), f"[summary] {content}"))
        chunk_rows = await chunks.get_by_ids([h["id"] for h in chunk_hits])
        for h, row in zip(chunk_hits, chunk_rows):
            content = (row or {}).get("content", "")
            if content:
                passages.append((h.get("distance", 0), content))
                sources.append(SourceRef(chunk_id=h["id"],
                                         doc_id=(row or {}).get("full_doc_id", ""),
                                         score=float(h.get("distance", 0)),
                                         excerpt=content[:240]))
        if not passages:
            return QueryResult(answer=FAIL, backend=self.name, namespace=namespace,
                               mode="collapsed-tree", ok=False)

        passages.sort(key=lambda x: -x[0])
        context = "\n\n".join(p for _, p in passages[:k])[:8000]
        answer = await self._llm(
            ANSWER_PROMPT.format(context=context, question=text, fail=FAIL),
            hashing_kv=cache)
        ok = bool(answer) and not any(m in answer for m in self._fail_markers)
        return QueryResult(answer=answer, sources=sources, backend=self.name,
                           namespace=namespace, mode="collapsed-tree", ok=ok)

    async def get_concept_graph(self, namespace: TenantNS, center: str,
                                depth: int = 2) -> ConceptGraph:
        """Tree neighbourhood: best-matching node as center; parents = broader
        (level +1), children = more specific (level -1) — maps directly onto the
        Poincaré sphere's latitude convention."""
        docs, chunks, cache, chunks_vdb, tree_vdb, tree_kv = self._stores(namespace)
        hits = await tree_vdb.query(center, top_k=1)
        if not hits:
            return ConceptGraph(center=center)
        center_id = hits[0]["id"]
        all_ids = await tree_kv.all_keys()
        all_nodes = {i: n for i, n in
                     zip(all_ids, await tree_kv.get_by_ids(all_ids)) if n}

        nodes, edges = {}, []

        def label_of(node):
            return (node.get("content", "")[:60] + "…") if len(node.get("content", "")) > 60 \
                else node.get("content", "")

        def add(nid, node, level, is_center=False):
            if nid not in nodes:
                nodes[nid] = ConceptNode(id=nid, label=label_of(node),
                                         summary=node.get("content", "")[:280],
                                         level=level, is_center=is_center)

        center_node = all_nodes.get(center_id, {})
        add(center_id, center_node, 0, True)
        # children (more specific → negative levels)
        frontier = [(center_id, 0)]
        for _ in range(depth):
            nxt = []
            for nid, lvl in frontier:
                for ch in (all_nodes.get(nid, {}) or {}).get("children", []):
                    ch_node = all_nodes.get(ch)
                    if ch_node:
                        add(ch, ch_node, lvl - 1)
                        edges.append(ConceptEdge(source=nid, target=ch))
                        nxt.append((ch, lvl - 1))
            frontier = nxt
        # parents (broader → positive levels)
        frontier = [(center_id, 0)]
        for _ in range(depth):
            nxt = []
            for nid, lvl in frontier:
                for pid, pnode in all_nodes.items():
                    if nid in (pnode.get("children") or []):
                        add(pid, pnode, lvl + 1)
                        edges.append(ConceptEdge(source=pid, target=nid))
                        nxt.append((pid, lvl + 1))
            frontier = nxt
        return ConceptGraph(center=center_id, nodes=list(nodes.values()), edges=edges)

    async def delete_namespace(self, namespace: TenantNS) -> bool:
        docs, chunks, cache, chunks_vdb, tree_vdb, tree_kv = self._stores(namespace)
        await tree_kv.drop()
        if self._pg_dsn:
            from ..storage.pg import reset_backend_structures
            await reset_backend_structures(self._pg_dsn, "hierarchical", tenant=namespace)
        return True
