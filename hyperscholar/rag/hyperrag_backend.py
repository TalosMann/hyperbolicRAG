"""HyperRAGBackend — wraps the upstream (locally modified) HyperRAG package.

Zero upstream changes: HyperRAG already exposes storage injection points
(`key_string_value_json_storage_cls`, `vector_db_storage_cls`,
`hypergraph_storage_cls`). We pass our shared/structure storage classes and
route the tenant namespace through `addon_params`, which HyperRAG carries
into every storage constructor via `global_config`.

HyperRAG-light is *not* a separate index: upstream `mode="hyper-lite"` is a
lighter retrieval path over the same hypergraph (entities only, no
relationship vdb). So `HyperRAGLightBackend` subclasses this backend, shares
the same structure schema, and differs only in query mode — index once,
compare both.
"""
from __future__ import annotations

import os
from typing import Callable

from ..core.types import (
    ConceptEdge, ConceptGraph, ConceptNode, Document, IndexResult,
    QueryResult, SourceRef, TenantNS,
)
from .base import RAGBackend


class HyperRAGBackend(RAGBackend):
    _mode = "hyper"
    _name = "hyperrag"

    def __init__(self, *, llm_func: Callable, embedder, working_dir: str,
                 kv_cls, vector_cls, hypergraph_cls,
                 pg_dsn: str | None = None,
                 fail_markers: list[str] | None = None,
                 cosine_threshold: float | None = None,
                 hyperrag_kwargs: dict | None = None):
        self._llm = llm_func
        self._embedder = embedder
        self._working_dir = working_dir
        self._kv_cls = kv_cls
        self._vector_cls = vector_cls
        self._hg_cls = hypergraph_cls
        self._pg_dsn = pg_dsn
        self._cosine_threshold = cosine_threshold
        self._fail_markers = fail_markers or [
            "Sorry, I'm not able to provide an answer to that question"]
        self._kw = hyperrag_kwargs or {}
        self._instances: dict[str, object] = {}   # tenant → HyperRAG

    @property
    def name(self) -> str:
        return self._name

    # ── HyperRAG instance per tenant (storage is the real state; this is cheap) ──
    def _rag(self, namespace: TenantNS):
        if namespace not in self._instances:
            try:
                from hyperrag import HyperRAG
                from hyperrag.utils import EmbeddingFunc
            except ImportError as e:
                raise RuntimeError(
                    "The `hyperrag` package is not importable. Install / add to "
                    "PYTHONPATH your local Hyper-RAG checkout "
                    "(/Users/talosmann/Projects/moonlabs/Hyper-RAG)."
                ) from e

            addon = {"tenant": namespace}
            if self._pg_dsn:
                addon["pg_dsn"] = self._pg_dsn
            if self._cosine_threshold is not None:
                addon["cosine_better_than_threshold"] = self._cosine_threshold

            import os
            workdir = os.path.join(self._working_dir, self.name, namespace)
            os.makedirs(workdir, exist_ok=True)   # upstream opens its log before mkdir

            self._instances[namespace] = HyperRAG(
                working_dir=workdir,
                llm_model_func=self._llm,
                embedding_func=EmbeddingFunc(
                    embedding_dim=self._embedder.embedding_dim,
                    max_token_size=self._embedder.max_token_size,
                    func=self._embedder,
                ),
                key_string_value_json_storage_cls=self._kv_cls,
                vector_db_storage_cls=self._vector_cls,
                hypergraph_storage_cls=self._hg_cls,
                addon_params=addon,
                **self._kw,
            )
        return self._instances[namespace]

    # ── RAGBackend contract ───────────────────────────────────────────────────
    async def index(self, namespace: TenantNS, documents: list[Document]) -> IndexResult:
        rag = self._rag(namespace)
        texts = [d.content for d in documents]
        await rag.ainsert(texts)
        n_chunks = len(await rag.text_chunks.all_keys())
        return IndexResult(namespace=namespace, documents=len(documents),
                           chunks=n_chunks, backend=self.name)

    async def query(self, namespace: TenantNS, text: str, top_k: int = 60) -> QueryResult:
        from hyperrag import QueryParam
        rag = self._rag(namespace)
        try:
            answer = await rag.aquery(text, QueryParam(mode=self._mode, top_k=top_k))
        except (AttributeError, KeyError, TypeError, UnboundLocalError) as e:
            # Upstream hyper_query assumes non-empty retrieval (e.g.
            # _build_relation_query_context returns None on a miss, then
            # `.get` is called on it). An empty/unindexed namespace must be a
            # graceful "no grounded answer", not a crash — the abstraction
            # layer guarantees uniform behaviour here.
            return QueryResult(answer=self._fail_markers[0], backend=self.name,
                               namespace=namespace, mode=self._mode, ok=False,
                               raw={"error": f"{type(e).__name__}: {e}"})
        ok = bool(answer) and not any(m in answer for m in self._fail_markers)
        return QueryResult(answer=answer or "", sources=await self._top_sources(rag, text),
                           backend=self.name, namespace=namespace, mode=self._mode, ok=ok)

    async def _top_sources(self, rag, text: str, k: int = 5) -> list[SourceRef]:
        """Best-effort traceability: nearest chunks for the query."""
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
        """BFS over the entity hypergraph from the best-matching entity."""
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

        async def add_node(v_id: str, level: int, is_center=False):
            if v_id in nodes:
                return
            data = await hg.get_vertex(v_id) or {}
            summary = (data.get("description") or data.get("additional_properties") or "")
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
        # de-dup edges
        seen, uniq = set(), []
        for e in edges:
            k = tuple(sorted((e.source, e.target)))
            if k not in seen:
                seen.add(k)
                uniq.append(e)
        return ConceptGraph(center=center_id, nodes=list(nodes.values()), edges=uniq)

    async def delete_namespace(self, namespace: TenantNS) -> bool:
        """Drop this backend's structure for the tenant; shared chunks remain."""
        rag = self._rag(namespace)
        hg = rag.chunk_entity_relation_hypergraph
        for e in list((await hg.get_all_hyperedges()).keys()):
            await hg.remove_hyperedge(e)
        for v in list((await hg.get_all_vertices()).keys()):
            await hg.remove_vertex(v)
        if self._pg_dsn:
            from ..storage.pg import reset_backend_structures
            await reset_backend_structures(self._pg_dsn, "hyperrag", tenant=namespace)
        self._instances.pop(namespace, None)
        return True


class HyperRAGLightBackend(HyperRAGBackend):
    """Same hypergraph structure, lighter retrieval (`hyper-lite`): entity-vdb
    driven, no relationship-vector hop. Shares the index with HyperRAGBackend —
    index once under either, query under both, compare."""
    _mode = "hyper-lite"
    _name = "hyperrag_light"
