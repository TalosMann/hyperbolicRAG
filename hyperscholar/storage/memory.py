"""In-memory storage classes implementing HyperRAG's storage contracts.

Activated with `store.dsn: memory://`. They mirror the semantics of the
upstream JsonKVStorage / NanoVectorDBStorage / HypergraphStorage closely
enough to run the whole pipeline offline (tests, quick experiments), while
honouring the same constructor signature HyperRAG uses to instantiate its
storage classes:

    cls(namespace=..., global_config=...)                       # KV / hypergraph
    cls(namespace=..., global_config=..., embedding_func=..., meta_fields=...)  # vector

Tenant isolation: the tenant namespace is read from
`global_config["addon_params"]["tenant"]`, exactly as the PG classes do.
All instances share a process-wide registry keyed by (tenant, kind, namespace),
so separate HyperRAG instances over the same tenant see the same data —
mimicking a database.
"""
from __future__ import annotations

import asyncio
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Set, Tuple, Union

import numpy as np

# process-wide "database"
_REGISTRY: dict = {}


def reset_memory_store():
    """Wipe everything (tests)."""
    _REGISTRY.clear()


def _bucket(tenant: str, kind: str, namespace: str, default):
    key = (tenant, kind, namespace)
    if key not in _REGISTRY:
        _REGISTRY[key] = default()
    return _REGISTRY[key]


def _tenant_addon(global_config: dict) -> dict:
    return (global_config or {}).get("addon_params", {})


def _tenant_of(global_config: dict) -> str:
    return _tenant_addon(global_config).get("tenant", "global")


def _edge_key(e_tuple: Union[List, Set, Tuple]) -> tuple:
    return tuple(sorted(map(str, e_tuple)))


@dataclass
class MemoryKVStorage:
    namespace: str
    global_config: dict

    def __post_init__(self):
        self._tenant = _tenant_of(self.global_config)
        self._data: dict = _bucket(self._tenant, "kv", self.namespace, dict)

    async def all_keys(self) -> list[str]:
        return list(self._data.keys())

    async def get_by_id(self, id):
        return self._data.get(id, None)

    async def get_by_ids(self, ids, fields=None):
        if fields is None:
            return [self._data.get(i, None) for i in ids]
        return [
            ({k: v for k, v in self._data[i].items() if k in fields}
             if self._data.get(i, None) else None)
            for i in ids
        ]

    async def filter_keys(self, data: list[str]) -> set[str]:
        return set(s for s in data if s not in self._data)

    async def upsert(self, data: dict):
        left = {k: v for k, v in data.items() if k not in self._data}
        self._data.update(left)
        return left

    async def drop(self):
        self._data.clear()

    async def index_done_callback(self):
        pass

    async def query_done_callback(self):
        pass


@dataclass
class MemoryVectorStorage:
    namespace: str
    global_config: dict
    embedding_func: Any = None
    meta_fields: set = field(default_factory=set)
    cosine_better_than_threshold: float = 0.2

    def __post_init__(self):
        self._tenant = _tenant_of(self.global_config)
        self._rows: dict = _bucket(self._tenant, "vec", self.namespace, dict)
        self._batch = (self.global_config or {}).get("embedding_batch_num", 8)
        ap = _tenant_addon(self.global_config)
        self.cosine_better_than_threshold = (self.global_config or {}).get(
            "cosine_better_than_threshold",
            ap.get("cosine_better_than_threshold", self.cosine_better_than_threshold))

    async def upsert(self, data: dict[str, dict]):
        if not data:
            return []
        items = list(data.items())
        contents = [v["content"] for _, v in items]
        batches = [contents[i:i + self._batch] for i in range(0, len(contents), self._batch)]
        embs = await asyncio.gather(*[self.embedding_func(b) for b in batches])
        embeddings = np.concatenate(embs)
        for (k, v), vec in zip(items, embeddings):
            self._rows[k] = {
                "meta": {mk: v[mk] for mk in self.meta_fields if mk in v},
                "vector": np.asarray(vec, dtype=np.float32),
            }
        return [k for k, _ in items]

    async def query(self, query: str, top_k: int = 5) -> list[dict]:
        if not self._rows:
            return []
        qv = (await self.embedding_func([query]))[0].astype(np.float32)
        qn = np.linalg.norm(qv) or 1.0
        scored = []
        for k, row in self._rows.items():
            v = row["vector"]
            sim = float(np.dot(qv, v) / (qn * (np.linalg.norm(v) or 1.0)))
            if sim > self.cosine_better_than_threshold:
                scored.append((sim, k, row))
        scored.sort(key=lambda x: -x[0])
        return [
            {**row["meta"], "id": k, "distance": sim}
            for sim, k, row in scored[:top_k]
        ]

    async def index_done_callback(self):
        pass

    async def query_done_callback(self):
        pass


@dataclass
class MemoryHypergraphStorage:
    namespace: str
    global_config: dict

    def __post_init__(self):
        self._tenant = _tenant_of(self.global_config)
        store = _bucket(self._tenant, "hg", self.namespace,
                        lambda: {"v": {}, "e": {}})
        self._v: dict = store["v"]
        self._e: dict = store["e"]

    # vertices ---------------------------------------------------------------
    async def has_vertex(self, v_id) -> bool:
        return str(v_id) in self._v

    async def get_vertex(self, v_id, default=None):
        return self._v.get(str(v_id), default)

    async def get_all_vertices(self):
        return dict(self._v)

    async def get_num_of_vertices(self):
        return len(self._v)

    async def upsert_vertex(self, v_id, v_data: Optional[Dict] = None):
        key = str(v_id)
        cur = self._v.get(key, {})
        cur.update(v_data or {})
        self._v[key] = cur

    async def remove_vertex(self, v_id):
        key = str(v_id)
        self._v.pop(key, None)
        for ek in [e for e in self._e if key in e]:
            self._e.pop(ek, None)

    async def vertex_degree(self, v_id) -> int:
        key = str(v_id)
        return sum(1 for e in self._e if key in e)

    # hyperedges ---------------------------------------------------------------
    async def has_hyperedge(self, e_tuple) -> bool:
        return _edge_key(e_tuple) in self._e

    async def get_hyperedge(self, e_tuple, default=None):
        return self._e.get(_edge_key(e_tuple), default)

    async def get_all_hyperedges(self):
        return dict(self._e)

    async def get_num_of_hyperedges(self):
        return len(self._e)

    async def upsert_hyperedge(self, e_tuple, e_data: Optional[Dict] = None):
        key = _edge_key(e_tuple)
        cur = self._e.get(key, {})
        cur.update(e_data or {})
        self._e[key] = cur

    async def remove_hyperedge(self, e_tuple):
        self._e.pop(_edge_key(e_tuple), None)

    async def hyperedge_degree(self, e_tuple) -> int:
        return len(_edge_key(e_tuple))

    # neighbourhood ------------------------------------------------------------
    async def get_nbr_e_of_vertex(self, v_id) -> list:
        key = str(v_id)
        return [e for e in self._e if key in e]

    async def get_nbr_v_of_hyperedge(self, e_tuple) -> list:
        return list(_edge_key(e_tuple))

    async def get_nbr_v_of_vertex(self, v_id, exclude_self=True) -> list:
        key = str(v_id)
        out = []
        for e in self._e:
            if key in e:
                out.extend(v for v in e if not (exclude_self and v == key))
        # de-dup preserving order
        seen, uniq = set(), []
        for v in out:
            if v not in seen:
                seen.add(v)
                uniq.append(v)
        return uniq

    async def index_done_callback(self):
        pass

    async def query_done_callback(self):
        pass
