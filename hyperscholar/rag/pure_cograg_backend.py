"""Pure CogRAG Backend — Faithful to the Hu et al. paper."""
from __future__ import annotations

import asyncio
import json
import os

from ..core.types import (
    ConceptEdge, ConceptGraph, ConceptNode, Document, IndexResult,
    QueryResult, SourceRef, TenantNS,
)
from .base import RAGBackend
from .hierarchical_backend import FAIL, _chunk_text, _hash_id

P_EXT_THEME = """Summarise the primary theme of the following text chunk in one concise sentence. Output only the summary.
CHUNK: {chunk}"""

P_EXT_KEY = """Extract up to 5 key entity names related to the theme from the chunk. Output them as a comma-separated list.
THEME: {theme}
CHUNK: {chunk}"""

P_EXT_ENTITY = """Extract all fine-grained entities from the chunk. Format as a JSON list of objects, each with 'name' and 'description' keys.
CHUNK: {chunk}"""

P_EXT_REL = """Extract high-order relations (sets of interacting entities) and describe their relation. Format as a JSON list of objects, each with 'entities' (list of names) and 'description' keys.
CHUNK: {chunk}"""

P_KEYWORD_THEME = """Extract exactly 3 abstract theme keywords from the following query. Output them as a comma-separated list.
QUERY: {query}"""

P_ALIGN_ENTITY = """Extract exactly 3 concrete entity keywords related to this query and the provided initial theme answer. Output them as a comma-separated list.
QUERY: {query}
THEME_ANSWER: {theme_answer}"""

P_THEME_ANSWER = """Draft an initial conceptual answer to the query based on the following theme contexts.
QUERY: {query}
CONTEXTS:
{contexts}"""

P_FINAL_ANSWER = """Provide a final comprehensive answer to the question using the provided contexts. Incorporate details from the entity contexts to ground your response. If the contexts do not contain enough information to answer, reply exactly: "{fail}"

QUESTION: {query}
INITIAL THEME ANSWER:
{theme_answer}

ENTITY CONTEXTS:
{contexts}"""


class PureCogRAGBackend(RAGBackend):
    _name = "pure_cograg"

    def __init__(self, *, llm_func, embedder, kv_cls, vector_cls,
                 working_dir: str = ".", pg_dsn: str | None = None,
                 chunk_size: int = 1200, chunk_overlap: int = 100,
                 cosine_threshold: float | None = None,
                 fail_markers: list[str] | None = None):
        self._llm = llm_func
        self._embedder = embedder
        self._working_dir = working_dir
        self._kv_cls = kv_cls
        self._vector_cls = vector_cls
        self._pg_dsn = pg_dsn
        self._chunk_size = chunk_size
        self._chunk_overlap = chunk_overlap
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
        cache = self._kv_cls(namespace="llm_response_cache", global_config=cfg)
        theme_edges_vdb = self._vector_cls(
            namespace="theme_edges", global_config=cfg,
            embedding_func=self._embedder, meta_fields={"incident_nodes", "chunk_id", "content"}
        )
        theme_nodes_kv = self._kv_cls(namespace="theme_nodes", global_config=cfg)
        
        entity_nodes_vdb = self._vector_cls(
            namespace="entity_nodes", global_config=cfg,
            embedding_func=self._embedder, meta_fields={"incident_edges", "chunk_id", "name", "content"}
        )
        entity_edges_kv = self._kv_cls(namespace="entity_edges", global_config=cfg)
        
        return cache, theme_edges_vdb, theme_nodes_kv, entity_nodes_vdb, entity_edges_kv

    async def index(self, namespace: TenantNS, documents: list[Document]) -> IndexResult:
        import pathlib
        pathlib.Path(self._cfg(namespace)["working_dir"]).mkdir(parents=True, exist_ok=True)
        
        cache, theme_edges_vdb, theme_nodes_kv, entity_nodes_vdb, entity_edges_kv = self._stores(namespace)
        
        chunks = []
        for d in documents:
            for piece in _chunk_text(d.content, self._chunk_size, self._chunk_overlap):
                chunks.append(piece)
        
        n_theme_edges = 0
        n_entity_nodes = 0

        for chunk in chunks:
            chunk_id = _hash_id(chunk, "chunk-")
            
            # 1. Extract theme
            theme = await self._llm(P_EXT_THEME.format(chunk=chunk), hashing_kv=cache)
            theme_edge_id = _hash_id(theme, "tedge-")
            
            # 2. Extract key entities for theme
            key_ents_raw = await self._llm(P_EXT_KEY.format(theme=theme, chunk=chunk), hashing_kv=cache)
            key_ents = [e.strip() for e in key_ents_raw.split(',') if e.strip()]
            
            # Store theme edge
            await theme_edges_vdb.upsert({
                theme_edge_id: {
                    "content": theme,
                    "incident_nodes": key_ents,
                    "chunk_id": chunk_id
                }
            })
            n_theme_edges += 1
            
            # Store key entities in KV
            theme_nodes_payload = {}
            for name in key_ents:
                node_id = _hash_id(name, "tnode-")
                theme_nodes_payload[node_id] = {
                    "name": name,
                    "incident_edge": theme_edge_id,
                    "chunk_id": chunk_id
                }
            await theme_nodes_kv.upsert(theme_nodes_payload)

            # 3. Extract entities
            ents_raw = await self._llm(P_EXT_ENTITY.format(chunk=chunk), hashing_kv=cache)
            try:
                js_str = ents_raw.strip()
                if js_str.startswith("```json"): js_str = js_str[7:]
                if js_str.endswith("```"): js_str = js_str[:-3]
                entities = json.loads(js_str.strip())
            except Exception:
                entities = []
            
            # 4. Extract relations
            rels_raw = await self._llm(P_EXT_REL.format(chunk=chunk), hashing_kv=cache)
            try:
                js_str = rels_raw.strip()
                if js_str.startswith("```json"): js_str = js_str[7:]
                if js_str.endswith("```"): js_str = js_str[:-3]
                relations = json.loads(js_str.strip())
            except Exception:
                relations = []

            # Process relations
            edge_payload = {}
            ent_to_edges = {}
            for rel in relations:
                desc = rel.get("description", "")
                ents = rel.get("entities", [])
                if not desc or not ents:
                    continue
                edge_id = _hash_id(desc, "eedge-")
                edge_payload[edge_id] = {
                    "content": desc,
                    "chunk_id": chunk_id,
                    "entities": ents
                }
                for e in ents:
                    ent_to_edges.setdefault(e, []).append(edge_id)
            if edge_payload:
                await entity_edges_kv.upsert(edge_payload)

            # Store entity nodes
            ent_nodes_payload = {}
            for ent in entities:
                name = ent.get("name", "")
                desc = ent.get("description", "")
                if not name or not desc:
                    continue
                node_id = _hash_id(name+desc, "enode-")
                edges = ent_to_edges.get(name, [])
                
                ent_nodes_payload[node_id] = {
                    "content": desc,
                    "name": name,
                    "chunk_id": chunk_id,
                    "incident_edges": edges
                }
            if ent_nodes_payload:
                await entity_nodes_vdb.upsert(ent_nodes_payload)
                n_entity_nodes += len(ent_nodes_payload)

        for store in (cache, theme_edges_vdb, theme_nodes_kv, entity_nodes_vdb, entity_edges_kv):
            cb = getattr(store, "index_done_callback", None)
            if cb is not None:
                await cb()

        return IndexResult(
            namespace=namespace, documents=len(documents), chunks=len(chunks),
            backend=self.name, detail={"theme_edges": n_theme_edges, "entity_nodes": n_entity_nodes}
        )

    async def query(self, namespace: TenantNS, text: str, top_k: int = 60) -> QueryResult:
        cache, theme_edges_vdb, theme_nodes_kv, entity_nodes_vdb, entity_edges_kv = self._stores(namespace)
        
        # Stage 1: Theme
        theme_kws_raw = await self._llm(P_KEYWORD_THEME.format(query=text), hashing_kv=cache)
        theme_kws = theme_kws_raw.strip()
        
        theme_hits = await theme_edges_vdb.query(theme_kws, top_k=3)
        theme_contexts = []
        for h in theme_hits:
            content = h.get("content", "")
            if content:
                inc_nodes = h.get("incident_nodes", [])
                node_names = []
                for n in inc_nodes:
                    nid = _hash_id(n, "tnode-")
                    nd = await theme_nodes_kv.get_by_id(nid)
                    if nd:
                        node_names.append(nd.get("name", n))
                
                theme_contexts.append(f"Theme: {content}\nEntities: {', '.join(node_names)}")
                
        if not theme_contexts:
            return QueryResult(answer=FAIL, backend=self.name, namespace=namespace, mode="pure_cograg", ok=False)
            
        theme_ctx_str = "\n\n".join(theme_contexts)
        a_theme = await self._llm(P_THEME_ANSWER.format(query=text, contexts=theme_ctx_str), hashing_kv=cache)
        
        # Stage 2: Entity
        ent_kws_raw = await self._llm(P_ALIGN_ENTITY.format(query=text, theme_answer=a_theme), hashing_kv=cache)
        ent_kws = ent_kws_raw.strip()
        
        ent_hits = await entity_nodes_vdb.query(ent_kws, top_k=6)
        ent_contexts = []
        sources = []
        for h in ent_hits:
            content = h.get("content", "")
            if content:
                inc_edges = h.get("incident_edges", [])
                edge_descs = []
                for eid in inc_edges:
                    ed = await entity_edges_kv.get_by_id(eid)
                    if ed:
                        edge_descs.append(ed.get("content", ""))
                
                ent_contexts.append(f"Entity Info: {content}\nRelations: {'; '.join(edge_descs)}")
                sources.append(
                    SourceRef(chunk_id=h.get("chunk_id", ""), score=float(h.get("distance", 0)), excerpt=content[:240])
                )
        
        if not ent_contexts:
            return QueryResult(answer=FAIL, backend=self.name, namespace=namespace, mode="pure_cograg", ok=False)
            
        ent_ctx_str = "\n\n".join(ent_contexts)
        final_answer = await self._llm(
            P_FINAL_ANSWER.format(query=text, theme_answer=a_theme, contexts=ent_ctx_str, fail=FAIL),
            hashing_kv=cache
        )
        
        ok = bool(final_answer) and not any(m in final_answer for m in self._fail_markers)
        return QueryResult(answer=final_answer, sources=sources, backend=self.name, namespace=namespace, mode="pure_cograg", ok=ok)

    async def get_concept_graph(self, namespace: TenantNS, center: str, depth: int = 2) -> ConceptGraph:
        return ConceptGraph(center=center, nodes=[], edges=[])

    async def delete_namespace(self, namespace: TenantNS) -> bool:
        cache, theme_edges_vdb, theme_nodes_kv, entity_nodes_vdb, entity_edges_kv = self._stores(namespace)
        await theme_nodes_kv.drop()
        await entity_edges_kv.drop()
        if self._pg_dsn:
            from ..storage.pg import reset_backend_structures
            await reset_backend_structures(self._pg_dsn, self._name, tenant=namespace)
        return True
