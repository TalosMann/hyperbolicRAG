-- HyperScholar Phase 1 schema
-- One database, four schemas:
--   shared        → backend-agnostic content + LLM cache   [SHARED by all backends]
--   hyperrag      → entities, relationships, hypergraph    [HyperRAG + HyperRAG-light]
--   hierarchical  → summary-tree nodes                      [HierarchicalRAG]
--
-- Every table is partitioned by tenant namespace:
--   'global' | 'inst_{id}' | 'personal_{uid}'
--
-- Embedding dimension is fixed at bootstrap (bge-m3 → 1024). Switching
-- embedding models requires recreating the vector columns (see roadmap caveat).

CREATE EXTENSION IF NOT EXISTS vector;

CREATE SCHEMA IF NOT EXISTS shared;
CREATE SCHEMA IF NOT EXISTS hyperrag;
CREATE SCHEMA IF NOT EXISTS hierarchical;

-- ─── shared: content layer ───────────────────────────────────────────────────
-- Generic KV store: serves HyperRAG's full_docs / text_chunks / llm_response_cache
-- (ns column = HyperRAG's internal storage namespace).
CREATE TABLE IF NOT EXISTS shared.kv_store (
    tenant      TEXT NOT NULL,
    ns          TEXT NOT NULL,
    id          TEXT NOT NULL,
    data        JSONB NOT NULL,
    updated_at  TIMESTAMPTZ NOT NULL DEFAULT now(),
    PRIMARY KEY (tenant, ns, id)
);
CREATE INDEX IF NOT EXISTS kv_store_tenant_ns ON shared.kv_store (tenant, ns);

-- Chunk vectors (the only vector table in shared: chunks are backend-agnostic).
CREATE TABLE IF NOT EXISTS shared.vectors (
    tenant      TEXT NOT NULL,
    ns          TEXT NOT NULL,          -- 'chunks'
    id          TEXT NOT NULL,
    meta        JSONB NOT NULL DEFAULT '{}',
    embedding   vector(1024) NOT NULL,
    PRIMARY KEY (tenant, ns, id)
);
CREATE INDEX IF NOT EXISTS shared_vectors_hnsw
    ON shared.vectors USING hnsw (embedding vector_cosine_ops);

-- ─── hyperrag: structure layer (also serves hyperrag_light queries) ─────────
CREATE TABLE IF NOT EXISTS hyperrag.vectors (
    tenant      TEXT NOT NULL,
    ns          TEXT NOT NULL,          -- 'entities' | 'relationships'
    id          TEXT NOT NULL,
    meta        JSONB NOT NULL DEFAULT '{}',
    embedding   vector(1024) NOT NULL,
    PRIMARY KEY (tenant, ns, id)
);
CREATE INDEX IF NOT EXISTS hyperrag_vectors_hnsw
    ON hyperrag.vectors USING hnsw (embedding vector_cosine_ops);

CREATE TABLE IF NOT EXISTS hyperrag.vertices (
    tenant      TEXT NOT NULL,
    v_id        TEXT NOT NULL,
    data        JSONB NOT NULL DEFAULT '{}',
    PRIMARY KEY (tenant, v_id)
);

CREATE TABLE IF NOT EXISTS hyperrag.hyperedges (
    tenant      TEXT NOT NULL,
    e_key       TEXT NOT NULL,          -- canonical: sorted members joined by '|#|'
    members     TEXT[] NOT NULL,
    data        JSONB NOT NULL DEFAULT '{}',
    PRIMARY KEY (tenant, e_key)
);
CREATE INDEX IF NOT EXISTS hyperedges_members_gin
    ON hyperrag.hyperedges USING gin (members);

-- ─── hierarchical: structure layer ───────────────────────────────────────────
CREATE TABLE IF NOT EXISTS hierarchical.nodes (
    tenant      TEXT NOT NULL,
    id          TEXT NOT NULL,
    level       INT  NOT NULL,          -- 0 = leaf summary over chunks; higher = broader
    children    TEXT[] NOT NULL DEFAULT '{}',   -- chunk-ids (level 0) or node-ids
    content     TEXT NOT NULL,
    embedding   vector(1024) NOT NULL,
    PRIMARY KEY (tenant, id)
);
CREATE INDEX IF NOT EXISTS hier_nodes_hnsw
    ON hierarchical.nodes USING hnsw (embedding vector_cosine_ops);
