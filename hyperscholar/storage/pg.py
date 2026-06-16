"""PostgreSQL + pgvector storage classes implementing HyperRAG's storage contracts.

Design (per HYPERSCHOLAR_ROADMAP.md Decision 2):

  shared schema       → kv_store (docs / text_chunks / llm cache), chunk vectors
  hyperrag schema     → entity & relationship vectors, vertices, hyperedges
  hierarchical schema → summary-tree nodes

Each class honours HyperRAG's constructor contract
(`cls(namespace=..., global_config=..., [embedding_func=..., meta_fields=...])`)
so they slot straight into HyperRAG's `*_storage_cls` injection points without
touching upstream code.

Routing of HyperRAG's internal namespaces to schemas:

  full_docs / text_chunks / llm_response_cache  → shared.kv_store
  chunks (vector)                               → shared.vectors
  entities / relationships (vector)             → hyperrag.vectors
  chunk_entity_relation (hypergraph)            → hyperrag.vertices + hyperedges

Tenant comes from global_config["addon_params"]["tenant"]; the DSN from
global_config["addon_params"]["pg_dsn"]. A process-wide pool per DSN is shared
by all storage instances.
"""
from __future__ import annotations

import asyncio
import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional, Set, Tuple, Union

import numpy as np

_POOLS: dict = {}
_SCHEMA_SQL = (Path(__file__).resolve().parent.parent / "db" / "schema.sql").read_text()

EDGE_SEP = "|#|"

# HyperRAG internal namespace → (schema, kind)
_VECTOR_SCHEMA = {
    "chunks": "shared",
    "entities": "hyperrag",
    "relationships": "hyperrag",
}


def _edge_key(e_tuple: Union[List, Set, Tuple]) -> str:
    return EDGE_SEP.join(sorted(map(str, e_tuple)))


def _vec_literal(v) -> str:
    return "[" + ",".join(f"{float(x):.7g}" for x in np.asarray(v).ravel()) + "]"


async def get_pool(dsn: str):
    """One asyncpg pool per DSN, bootstrapping the schema on first connect."""
    if dsn not in _POOLS:
        import asyncpg
        pool = await asyncpg.create_pool(dsn=dsn, min_size=1, max_size=8)
        async with pool.acquire() as conn:
            await conn.execute(_SCHEMA_SQL)
        _POOLS[dsn] = pool
    return _POOLS[dsn]


def _addon(global_config: dict) -> dict:
    return (global_config or {}).get("addon_params", {})


def _run(coro):
    """Bridge for sync __post_init__ paths if ever needed (storage is async-first)."""
    return asyncio.get_event_loop().run_until_complete(coro)


# ──────────────────────────────────────────────────────────────────────────────
@dataclass
class PGKVStorage:
    """BaseKVStorage on shared.kv_store. Serves full_docs / text_chunks / llm cache."""
    namespace: str
    global_config: dict

    def __post_init__(self):
        ap = _addon(self.global_config)
        self._tenant = ap.get("tenant", "global")
        self._dsn = ap["pg_dsn"]

    async def _pool(self):
        return await get_pool(self._dsn)

    async def all_keys(self) -> list[str]:
        pool = await self._pool()
        rows = await pool.fetch(
            "SELECT id FROM shared.kv_store WHERE tenant=$1 AND ns=$2",
            self._tenant, self.namespace)
        return [r["id"] for r in rows]

    async def get_by_id(self, id: str):
        pool = await self._pool()
        row = await pool.fetchrow(
            "SELECT data FROM shared.kv_store WHERE tenant=$1 AND ns=$2 AND id=$3",
            self._tenant, self.namespace, id)
        return json.loads(row["data"]) if row else None

    async def get_by_ids(self, ids: list[str], fields=None):
        pool = await self._pool()
        rows = await pool.fetch(
            "SELECT id, data FROM shared.kv_store WHERE tenant=$1 AND ns=$2 AND id = ANY($3)",
            self._tenant, self.namespace, ids)
        by_id = {r["id"]: json.loads(r["data"]) for r in rows}
        out = []
        for i in ids:
            d = by_id.get(i)
            if d is not None and fields is not None:
                d = {k: v for k, v in d.items() if k in fields}
            out.append(d)
        return out

    async def filter_keys(self, data: list[str]) -> set[str]:
        pool = await self._pool()
        rows = await pool.fetch(
            "SELECT id FROM shared.kv_store WHERE tenant=$1 AND ns=$2 AND id = ANY($3)",
            self._tenant, self.namespace, data)
        existing = {r["id"] for r in rows}
        return set(d for d in data if d not in existing)

    async def upsert(self, data: dict):
        if not data:
            return {}
        pool = await self._pool()
        new_keys = await self.filter_keys(list(data.keys()))
        left = {k: v for k, v in data.items() if k in new_keys}
        if left:
            await pool.executemany(
                """INSERT INTO shared.kv_store (tenant, ns, id, data)
                   VALUES ($1,$2,$3,$4::jsonb)
                   ON CONFLICT (tenant, ns, id) DO NOTHING""",
                [(self._tenant, self.namespace, k, json.dumps(v)) for k, v in left.items()])
        return left

    async def drop(self):
        pool = await self._pool()
        await pool.execute(
            "DELETE FROM shared.kv_store WHERE tenant=$1 AND ns=$2",
            self._tenant, self.namespace)

    async def index_done_callback(self):
        pass

    async def query_done_callback(self):
        pass


# ──────────────────────────────────────────────────────────────────────────────
@dataclass
class PGVectorStorage:
    """BaseVectorStorage on pgvector.

    chunks → shared.vectors; entities/relationships → hyperrag.vectors.
    Query semantics mirror NanoVectorDBStorage: cosine similarity,
    `better_than_threshold` filter, results as {**meta, "id", "distance"}.
    """
    namespace: str
    global_config: dict
    embedding_func: Any = None
    meta_fields: set = field(default_factory=set)
    cosine_better_than_threshold: float = 0.2

    def __post_init__(self):
        ap = _addon(self.global_config)
        self._tenant = ap.get("tenant", "global")
        self._dsn = ap["pg_dsn"]
        self._schema = _VECTOR_SCHEMA.get(self.namespace, "hyperrag")
        self._batch = (self.global_config or {}).get("embedding_batch_num", 8)
        self.cosine_better_than_threshold = (self.global_config or {}).get(
            "cosine_better_than_threshold",
            ap.get("cosine_better_than_threshold", self.cosine_better_than_threshold))

    async def _pool(self):
        return await get_pool(self._dsn)

    async def upsert(self, data: dict[str, dict]):
        if not data:
            return []
        items = list(data.items())
        contents = [v["content"] for _, v in items]
        batches = [contents[i:i + self._batch] for i in range(0, len(contents), self._batch)]
        embs = await asyncio.gather(*[self.embedding_func(b) for b in batches])
        embeddings = np.concatenate(embs)
        pool = await self._pool()
        await pool.executemany(
            f"""INSERT INTO {self._schema}.vectors (tenant, ns, id, meta, embedding)
                VALUES ($1,$2,$3,$4::jsonb,$5::vector)
                ON CONFLICT (tenant, ns, id)
                DO UPDATE SET meta = EXCLUDED.meta, embedding = EXCLUDED.embedding""",
            [(self._tenant, self.namespace, k,
              json.dumps({mk: v[mk] for mk in self.meta_fields if mk in v}),
              _vec_literal(vec))
             for (k, v), vec in zip(items, embeddings)])
        return [k for k, _ in items]

    async def query(self, query: str, top_k: int = 5) -> list[dict]:
        qv = (await self.embedding_func([query]))[0]
        pool = await self._pool()
        rows = await pool.fetch(
            f"""SELECT id, meta, 1 - (embedding <=> $3::vector) AS sim
                FROM {self._schema}.vectors
                WHERE tenant=$1 AND ns=$2
                  AND 1 - (embedding <=> $3::vector) > $4
                ORDER BY embedding <=> $3::vector
                LIMIT $5""",
            self._tenant, self.namespace, _vec_literal(qv),
            self.cosine_better_than_threshold, top_k)
        return [
            {**json.loads(r["meta"]), "id": r["id"], "distance": float(r["sim"])}
            for r in rows
        ]

    async def index_done_callback(self):
        pass

    async def query_done_callback(self):
        pass


# ──────────────────────────────────────────────────────────────────────────────
@dataclass
class PGHypergraphStorage:
    """BaseHypergraphStorage on hyperrag.vertices + hyperrag.hyperedges."""
    namespace: str
    global_config: dict

    def __post_init__(self):
        ap = _addon(self.global_config)
        self._tenant = ap.get("tenant", "global")
        self._dsn = ap["pg_dsn"]

    async def _pool(self):
        return await get_pool(self._dsn)

    # vertices -----------------------------------------------------------------
    async def has_vertex(self, v_id) -> bool:
        pool = await self._pool()
        return bool(await pool.fetchval(
            "SELECT 1 FROM hyperrag.vertices WHERE tenant=$1 AND v_id=$2",
            self._tenant, str(v_id)))

    async def get_vertex(self, v_id, default=None):
        pool = await self._pool()
        row = await pool.fetchrow(
            "SELECT data FROM hyperrag.vertices WHERE tenant=$1 AND v_id=$2",
            self._tenant, str(v_id))
        return json.loads(row["data"]) if row else default

    async def get_all_vertices(self):
        pool = await self._pool()
        rows = await pool.fetch(
            "SELECT v_id, data FROM hyperrag.vertices WHERE tenant=$1", self._tenant)
        return {r["v_id"]: json.loads(r["data"]) for r in rows}

    async def get_num_of_vertices(self):
        pool = await self._pool()
        return await pool.fetchval(
            "SELECT count(*) FROM hyperrag.vertices WHERE tenant=$1", self._tenant)

    async def upsert_vertex(self, v_id, v_data: Optional[Dict] = None):
        pool = await self._pool()
        await pool.execute(
            """INSERT INTO hyperrag.vertices (tenant, v_id, data)
               VALUES ($1,$2,$3::jsonb)
               ON CONFLICT (tenant, v_id)
               DO UPDATE SET data = hyperrag.vertices.data || EXCLUDED.data""",
            self._tenant, str(v_id), json.dumps(v_data or {}))

    async def remove_vertex(self, v_id):
        pool = await self._pool()
        await pool.execute(
            "DELETE FROM hyperrag.vertices WHERE tenant=$1 AND v_id=$2",
            self._tenant, str(v_id))
        await pool.execute(
            "DELETE FROM hyperrag.hyperedges WHERE tenant=$1 AND $2 = ANY(members)",
            self._tenant, str(v_id))

    async def vertex_degree(self, v_id) -> int:
        pool = await self._pool()
        return await pool.fetchval(
            "SELECT count(*) FROM hyperrag.hyperedges WHERE tenant=$1 AND $2 = ANY(members)",
            self._tenant, str(v_id))

    # hyperedges ----------------------------------------------------------------
    async def has_hyperedge(self, e_tuple) -> bool:
        pool = await self._pool()
        return bool(await pool.fetchval(
            "SELECT 1 FROM hyperrag.hyperedges WHERE tenant=$1 AND e_key=$2",
            self._tenant, _edge_key(e_tuple)))

    async def get_hyperedge(self, e_tuple, default=None):
        pool = await self._pool()
        row = await pool.fetchrow(
            "SELECT data FROM hyperrag.hyperedges WHERE tenant=$1 AND e_key=$2",
            self._tenant, _edge_key(e_tuple))
        return json.loads(row["data"]) if row else default

    async def get_all_hyperedges(self):
        pool = await self._pool()
        rows = await pool.fetch(
            "SELECT e_key, data FROM hyperrag.hyperedges WHERE tenant=$1", self._tenant)
        return {tuple(r["e_key"].split(EDGE_SEP)): json.loads(r["data"]) for r in rows}

    async def get_num_of_hyperedges(self):
        pool = await self._pool()
        return await pool.fetchval(
            "SELECT count(*) FROM hyperrag.hyperedges WHERE tenant=$1", self._tenant)

    async def upsert_hyperedge(self, e_tuple, e_data: Optional[Dict] = None):
        pool = await self._pool()
        members = sorted(map(str, e_tuple))
        await pool.execute(
            """INSERT INTO hyperrag.hyperedges (tenant, e_key, members, data)
               VALUES ($1,$2,$3,$4::jsonb)
               ON CONFLICT (tenant, e_key)
               DO UPDATE SET data = hyperrag.hyperedges.data || EXCLUDED.data""",
            self._tenant, EDGE_SEP.join(members), members, json.dumps(e_data or {}))

    async def remove_hyperedge(self, e_tuple):
        pool = await self._pool()
        await pool.execute(
            "DELETE FROM hyperrag.hyperedges WHERE tenant=$1 AND e_key=$2",
            self._tenant, _edge_key(e_tuple))

    async def hyperedge_degree(self, e_tuple) -> int:
        return len(set(map(str, e_tuple)))

    # neighbourhood ---------------------------------------------------------------
    async def get_nbr_e_of_vertex(self, v_id) -> list:
        pool = await self._pool()
        rows = await pool.fetch(
            "SELECT e_key FROM hyperrag.hyperedges WHERE tenant=$1 AND $2 = ANY(members)",
            self._tenant, str(v_id))
        return [tuple(r["e_key"].split(EDGE_SEP)) for r in rows]

    async def get_nbr_v_of_hyperedge(self, e_tuple) -> list:
        return sorted(map(str, e_tuple))

    async def get_nbr_v_of_vertex(self, v_id, exclude_self=True) -> list:
        pool = await self._pool()
        rows = await pool.fetch(
            "SELECT members FROM hyperrag.hyperedges WHERE tenant=$1 AND $2 = ANY(members)",
            self._tenant, str(v_id))
        seen, out = set(), []
        for r in rows:
            for v in r["members"]:
                if exclude_self and v == str(v_id):
                    continue
                if v not in seen:
                    seen.add(v)
                    out.append(v)
        return out

    async def index_done_callback(self):
        pass

    async def query_done_callback(self):
        pass


async def reset_backend_structures(dsn: str, backend_schema: str, tenant: str | None = None):
    """Wipe a backend's structural overlay (optionally one tenant) without
    touching shared content — the cheap-rebuild path for A/B testing."""
    pool = await get_pool(dsn)
    if backend_schema == "hyperrag":
        tables = ["hyperrag.vectors", "hyperrag.vertices", "hyperrag.hyperedges"]
    elif backend_schema == "hierarchical":
        tables = ["hierarchical.nodes"]
    else:
        raise ValueError(f"unknown backend schema {backend_schema}")
    for t in tables:
        if tenant:
            await pool.execute(f"DELETE FROM {t} WHERE tenant=$1", tenant)
        else:
            await pool.execute(f"TRUNCATE {t}")
