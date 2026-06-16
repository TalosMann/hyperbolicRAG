# HyperScholar — Build Roadmap

**Status:** Architecture finalized, entering build phase
**Focus:** (1) Malleable hallucination-free retrieval layer, (2) Classroom system
**Last updated:** 2026

This document is the single source of truth for the build. It is written to be injectable into any AI coding session as full context.

---

## 1. Goals for this phase

Two workstreams, built in dependency order:

1. **A swappable RAG abstraction layer** so HyperRAG, HyperRAG-light, and HierarchicalRAG can be exchanged with a one-line config change for development testing and head-to-head comparison.
2. **The classroom system** — institutional accounts with teacher and student interfaces, corpus upload, access-code onboarding, and a corpus approval workflow.

Everything else (Teaching Styles, Content Synthesis, frontend) sits downstream and is scoped at the end.

---

## 2. Locked architecture decisions

### Decision 1 — Backend selection via simple config

A single config file declares the active backend. Switching is one line. The `RAGRouter` reads this at startup and instantiates the chosen backend.

```yaml
# config.yaml
rag:
  backend: hyperrag        # hyperrag | hyperrag_light | hierarchical

embedding:
  model: BAAI/bge-m3
  dim: 1024
  device: mps              # Apple Silicon

llm:
  provider: deepseek
  base_url: https://api.deepseek.com/v1
  model: deepseek-chat

store:
  type: pgvector
  dsn: postgresql://localhost/hyperscholar
```

Backends in scope: **HyperRAG**, **HyperRAG-light**, **HierarchicalRAG**. (No GraphRAG.)

### Decision 2 — Shared content, isolated per-backend structures

There are three layers, with different sharing rules:

```
Layer        Contents                              Sharing
───────      ──────────────────────────────────   ─────────────────────────────
Content      documents, chunks, embeddings         SHARED (one store, all backends)
Cache        LLM responses (extraction, summaries) SHARED (no repeated expensive calls)
Structure    hyperedges / summary tree / etc.      ISOLATED (one schema per backend)
```

**Why this split.** For a fair comparison you hold the content constant and vary only the structure. If each backend chunked and embedded independently, a comparison would conflate chunking + embedding + structure and you couldn't tell which moved the needle. Shared content means identical chunks and identical vectors enter all three backends, and the *only* difference is how each one organizes and traverses them. Embedding on MPS is expensive, so doing it once also speeds iteration.

**Storage mechanism — separate Postgres schemas in one database.**

```
Database: hyperscholar
├── shared schema          ← documents, chunks (+embeddings), llm_cache   [SHARED]
├── hyperrag schema        ← entities (+embeddings), hyperedges            [ISOLATED]
├── hyperrag_light schema  ← entities, reduced hyperedges                  [ISOLATED]
└── hierarchical schema    ← summary tree nodes (+embeddings)              [ISOLATED]
```

One connection, one DSN. Resetting a backend to re-test (e.g. a chunk-overlap change) is `DROP SCHEMA hyperrag CASCADE` then rebuild from the shared chunks — no re-embedding, no impact on other backends. Structure tables reference shared content by `chunk_id` / `document_id` and can join across schemas when convenient.

> Entities live in the per-backend schema, not in shared content. HyperRAG and HyperRAG-light both extract entities, but the underlying LLM calls hit the **shared** cache, so extraction isn't recomputed — the structures stay isolated while the expensive work stays shared. HierarchicalRAG, which summarizes rather than extracting entities, simply doesn't populate an entities table.

**Namespaces** partition every schema by tenant:
```
global            → curated textbook baseline
inst_{id}         → institutional DB
personal_{uid}    → personal DB
```
So a chunk is keyed by `(namespace, chunk_id)` in shared content, and a hyperedge is keyed by `(namespace, …)` in the `hyperrag` schema.

**Caveat:** embedding dimension is hardcoded to the model output (bge-m3 → 1024). Switching embedding models requires clearing the shared content store to avoid silent dimension mismatch.

---

## 3. Core abstractions

### 3.1 Shared content store (`shared` schema)

Holds only the backend-agnostic layer: documents, chunks (+embeddings), and the LLM cache. It does **not** own entities or any structural data — those live per-backend.

```python
class SharedContentStore:
    """pgvector-backed `shared` schema. Read/written by every backend + the LLM layer."""
    async def add_documents(self, namespace, docs) -> list[str]: ...
    async def add_chunks(self, namespace, chunks) -> None: ...          # content + embedding
    async def search_chunks(self, namespace, query_vec, top_k) -> list[Chunk]: ...
    async def get_chunks(self, namespace, chunk_ids) -> list[Chunk]: ... # resolve refs from structure
    async def cache_get(self, prompt_hash) -> str | None: ...           # shared LLM cache
    async def cache_set(self, prompt_hash, response) -> None: ...
    async def delete_namespace(self, namespace) -> None: ...
```

### 3.2 Per-backend structure store

Each backend owns an isolated schema (`hyperrag`, `hyperrag_light`, `hierarchical`) for its entities and structural artifacts. The concrete shape differs per backend, but each manages its own writes, traversal, and clean teardown (`DROP SCHEMA ... CASCADE`). Structure rows reference shared content by `(namespace, chunk_id)` and resolve the actual text via `SharedContentStore.get_chunks`.

```python
class StructureStore(ABC):
    """Owns one isolated per-backend schema. Stores entities + structural edges."""
    @abstractmethod
    async def build(self, namespace, chunks, llm, embedder) -> None: ...
    @abstractmethod
    async def retrieve(self, namespace, query_vec, top_k) -> list[StructureHit]: ...
    @abstractmethod
    async def local_topology(self, namespace, center, depth) -> ConceptGraph: ...
    @abstractmethod
    async def reset(self, namespace) -> None: ...
```

### 3.3 RAG backend interface

The contract every backend must satisfy. Each backend receives the shared content store, its own structure store, the LLM, and the embedder by injection — so all backends share content + cache while keeping structures isolated.

```python
class RAGBackend(ABC):
    def __init__(self, content: SharedContentStore, structure: StructureStore,
                 llm: BaseLLM, embedder: BaseEmbedder):
        ...

    @property
    @abstractmethod
    def name(self) -> str: ...

    @abstractmethod
    async def index(self, namespace: str, documents: list[Document]) -> IndexResult:
        """Ensure chunks/entities exist in the shared store, then build this
        backend's structural overlay for the namespace."""

    @abstractmethod
    async def query(self, namespace: str, text: str, top_k: int = 10) -> QueryResult:
        """Retrieve context using this backend's strategy. Returns ranked passages
        with source attribution for traceability."""

    @abstractmethod
    async def get_concept_graph(self, namespace: str, center: str, depth: int = 2) -> ConceptGraph:
        """Return local concept topology for the Poincaré sphere."""

    @abstractmethod
    async def delete_namespace(self, namespace: str) -> bool:
        """Remove this backend's structure (its schema) for a namespace.
        Does not touch shared content chunks."""
```

### 3.4 RAG router (three-tier query strategy)

Sits above the backend. Backend-agnostic — only calls `query(namespace, text)`.

```python
class RAGRouter:
    def __init__(self, backend: RAGBackend, global_ns: str = "global"):
        self.backend = backend
        self.global_ns = global_ns

    async def query_classroom(self, text, inst_ns) -> RouterResult:
        # Institutional DB is primary
        result = await self.backend.query(inst_ns, text)
        if self._sufficient(result):
            return RouterResult(result, source="institutional", out_of_scope=False)
        # Global DB fallback, flagged for disclaimer
        result = await self.backend.query(self.global_ns, text)
        return RouterResult(result, source="global", out_of_scope=True)
        # out_of_scope=True → UI shows:
        # "This is outside your institution's database — please verify."

    async def query_personal(self, text, personal_ns, category) -> RouterResult:
        if category in ("exam", "research_paper"):
            result = await self.backend.query(personal_ns, text)
            return RouterResult(result, source="personal", out_of_scope=False)
        # textbook / general → blend personal + global
        personal = await self.backend.query(personal_ns, text)
        glob     = await self.backend.query(self.global_ns, text)
        return RouterResult(self._merge(personal, glob), source="blended", out_of_scope=False)
```

---

## 4. Phases

Build in this order. Phases 1–5 deliver a working "teacher uploads corpus → student queries it → student submits paper → teacher approves" loop. Phase 6 can run in parallel with Phase 5.

### Phase 1 — RAG abstraction layer  *(highest priority)*
- `RAGBackend` ABC with the four methods above
- `SharedContentStore` on the `shared` pgvector schema (documents, chunks+embeddings, llm_cache)
- `StructureStore` ABC + one isolated schema per backend (`hyperrag`, `hyperrag_light`, `hierarchical`)
- `BaseLLM` (DeepSeek wrapper, reachable; SiliconFlow as alternate endpoint) and `BaseEmbedder` (bge-m3, MPS); the LLM writes through the shared cache
- `HyperRAGBackend` — refactor existing HyperRAG off NanoVectorDB so chunks/embeddings/cache come from `SharedContentStore`, and entities + hyperedges live in the `hyperrag` schema
- `HyperRAGLightBackend` and `HierarchicalRAGBackend` — initially stubs that pass interface tests, then fleshed out
- `RAGRouter` with the three-tier strategy
- **Interface conformance test suite** — any backend that implements `RAGBackend` is auto-validated
- Deliverable: swap `backend:` in config, re-run the same query, compare outputs

### Phase 2 — PostgreSQL schema + auth
- Tables: `institutions`, `users` (teacher/student role), `corpuses`, `access_codes`, `memberships`, `corpus_submissions`, `student_profiles`
- Access-code generation + validation
- Session/JWT auth with role separation (teacher vs student endpoints)
- API skeleton, no frontend

### Phase 3 — Corpus ingestion pipeline
- Teacher uploads → `pdfplumber` extraction → chunking → entity extraction → shared store + `HyperRAGBackend.index()` into `inst_{id}`
- Corpus metadata row in Postgres (status: indexing → ready)
- Reuses existing HyperRAG indexing work; new part is namespace routing + the DB record

### Phase 4 — Classroom query layer
- Student query → `RAGRouter.query_classroom(text, inst_ns)` → result + `out_of_scope` flag
- API endpoint packaging the result for the frontend
- **First end-to-end slice:** corpus-grounded, hallucination-free answers

### Phase 5 — Corpus approval workflow
- Student submits doc → `corpus_submission` row, status `pending`
- Teacher endpoint: list queue, approve / reject
- On approval → same ingestion pipeline into the institution namespace
- Student status polling (push notifications later)

### Phase 6 — Teaching Styles rule engine  *(parallelizable with Phase 5)*
- `student_profiles` CRUD (Kolb position, declared disabilities, interaction history)
- Disability declaration endpoint (teacher or student writable; declared only, never inferred)
- Rule engine priority: disability overrides → Kolb routing → modality preference
- Onboarding diagnostic question set
- Output: ranked delivery-preference object

### Phase 7 — Content Synthesis
- RAGRouter output + delivery preference → DeepSeek call → text script
- Q&A pair extraction from script (bubble shooter source)
- Cache script per `(query_hash, student_id, backend)`

### Phase 8 — Frontend integration
- Landing page + Poincaré sphere (topology from `get_concept_graph()`)
- Classroom student interface; teacher interface (upload, queue, analytics)
- Lesson interface (text + games)

---

## 5. Proposed file structure

```
hyperscholar/
├── config.yaml
├── core/
│   ├── content_store.py  # SharedContentStore (shared pgvector schema)
│   ├── structure_store.py# StructureStore ABC + per-backend schema helpers
│   ├── llm.py            # BaseLLM (DeepSeek / SiliconFlow), shared cache
│   ├── embedder.py       # BaseEmbedder (bge-m3, MPS)
│   └── types.py          # Document, Chunk, Entity, QueryResult, ConceptGraph
├── rag/
│   ├── base.py           # RAGBackend ABC
│   ├── router.py         # RAGRouter (three-tier strategy)
│   ├── hyperrag.py       # HyperRAGBackend
│   ├── hyperrag_light.py # HyperRAGLightBackend
│   └── hierarchical.py   # HierarchicalRAGBackend
├── db/
│   ├── schema.sql        # PostgreSQL schema
│   └── models.py         # ORM models
├── api/
│   ├── auth.py
│   ├── teacher.py        # corpus upload, queue, analytics
│   └── student.py        # join via code, query, submit
└── tests/
    └── test_backends.py  # interface conformance across all backends
```

---

## 6. Tech stack (confirmed)

- **LLM:** DeepSeek (primary) via `openai_complete_if_cache` → `https://api.deepseek.com/v1`; SiliconFlow (`api.siliconflow.cn`) as alternate reachable endpoint
- **Embeddings:** `BAAI/bge-m3` via `sentence-transformers`, local, MPS-accelerated, 1024-dim, async via `loop.run_in_executor`
- **Vector + relational store:** PostgreSQL + pgvector
- **PDF processing:** `pdfplumber`
- **Network constraint:** TLS blocked to OpenAI / tiktoken / Western APIs at runtime → tiktoken already replaced with character-level stubs; any runtime dependency on blocked endpoints needs a local fallback

---

## 7. First task

Begin Phase 1 with the `RAGBackend` ABC, the `SharedContentStore` (`shared` schema), and the `StructureStore` ABC. Then wrap existing HyperRAG as `HyperRAGBackend` — moving chunks/embeddings/cache onto the shared store and entities/hyperedges into the `hyperrag` schema. Validate by indexing one corpus once and running the same query through all three backends via a config switch, comparing outputs on identical content.
