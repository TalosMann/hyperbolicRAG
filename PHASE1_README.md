# HyperScholar — Phase 1: Swappable RAG Abstraction Layer

Phase 1 of the [HYPERSCHOLAR_ROADMAP](../HYPERSCHOLAR_ROADMAP.md): three RAG
backends behind one interface, shared content + isolated per-backend
structures, three-tier query routing. **Status: complete, 16/16 tests passing.**

```
rag:
  backend: hyperrag        # ← the one-line swap: hyperrag | hyperrag_light | hierarchical
```

## What's here

```
hyperscholar/
├── config.yaml                  # backend swap, LLM, embedder, store DSN
├── core/
│   ├── types.py                 # Document, QueryResult, ConceptGraph, RouterResult…
│   ├── config.py                # YAML + ${ENV:default} interpolation
│   ├── embedder.py              # BgeM3Embedder (MPS) · HashEmbedder (tests)
│   └── llm.py                   # make_deepseek_complete (shared-cache aware) · StubLLM
├── storage/
│   ├── memory.py                # in-process stores  (dsn: memory://)
│   └── pg.py                    # pgvector stores    (dsn: postgresql://…)
├── db/schema.sql                # shared / hyperrag / hierarchical schemas
├── rag/
│   ├── base.py                  # RAGBackend ABC
│   ├── hyperrag_backend.py      # HyperRAG + HyperRAG-light (shared index)
│   ├── hierarchical_backend.py  # RAPTOR-style summary tree (self-contained)
│   ├── router.py                # three-tier classroom/personal routing
│   └── factory.py               # config → wired backend/router
├── scripts/compare_backends.py  # Phase 1 deliverable: same corpus, all backends
└── tests/test_phase1.py         # storage contracts · router · backend conformance
```

## Key design facts

- **Zero upstream changes.** HyperRAG already exposes storage injection points
  (`key_string_value_json_storage_cls`, `vector_db_storage_cls`,
  `hypergraph_storage_cls`). Our storage classes implement its contracts; the
  tenant namespace travels via `addon_params={"tenant": …}`.
- **HyperRAG-light shares HyperRAG's index.** Upstream `hyper-lite` is a
  retrieval mode over the same hypergraph, so the light backend subclasses the
  full one and differs only in query path. Index once, compare both.
- **Shared content is real.** Indexing under HyperRAG then under
  HierarchicalRAG for the same tenant reports `reused_shared_chunks > 0` —
  no re-chunking, no re-embedding, structures isolated.
- **Tenant namespaces** (`global`, `inst_{id}`, `personal_{uid}`) partition
  every table; conformance tests verify isolation.

## Running on your Mac

```bash
cd /Users/talosmann/Projects/moonlabs
source venv/bin/activate
pip install -r hyperscholar/requirements.txt

# hyperrag package = your local checkout (with the tiktoken char-stub):
export PYTHONPATH="/Users/talosmann/Projects/moonlabs/Hyper-RAG:$PYTHONPATH"

# offline smoke test (no network, no Postgres):
python -m pytest hyperscholar/tests/test_phase1.py -q
python -m hyperscholar.scripts.compare_backends --offline \
    --query "How do plants convert sunlight into energy?"

# real run (DeepSeek + bge-m3 on MPS, in-memory store):
export DEEPSEEK_API_KEY=sk-…
python -m hyperscholar.scripts.compare_backends \
    --corpus path/to/notes.txt --namespace inst_demo \
    --query "Your question here"
```

### Postgres (optional now, required from Phase 2)

```bash
brew install postgresql@16 pgvector
createdb hyperscholar
export HYPERSCHOLAR_DSN="postgresql://localhost/hyperscholar"
```

The schema bootstraps itself on first connection (`db/schema.sql`). To rebuild
one backend's structures without touching shared content:

```python
from hyperscholar.storage.pg import reset_backend_structures
await reset_backend_structures(dsn, "hyperrag", tenant="inst_demo")
```

## Notes & known upstream issue

- **Upstream empty-retrieval crash:** `hyper_query`'s
  `_build_relation_query_context` returns `None` when retrieval finds nothing
  (e.g. querying an empty/unindexed namespace), and the caller then calls
  `.get` on it → `AttributeError`. The backend wrapper catches this and
  returns a graceful `ok=False` result, so the router's fallback logic works.
  Worth patching upstream eventually.
- The embedding dimension (1024 for bge-m3) is baked into `db/schema.sql`.
  Switching embedding models requires clearing the vector stores.
- `chunks` in `IndexResult` is the namespace total; `detail.new_chunks` /
  `detail.reused_shared_chunks` show how much shared content was reused.

## Phase 1 exit criteria — all met

1. ✅ `RAGBackend` ABC + conformance suite (any new backend is auto-validated)
2. ✅ Shared content / isolated structures over memory:// and postgresql://
3. ✅ HyperRAG wrapped without upstream modification; light variant shares index
4. ✅ HierarchicalRAG (summary tree) as a genuine third strategy
5. ✅ `RAGRouter` three-tier strategy with out-of-scope disclaimers
6. ✅ Config-line swap demonstrated by `scripts/compare_backends.py`

**Next: Phase 2 — PostgreSQL schema + auth** (institutions, users, corpuses,
access codes, memberships, submissions, student profiles).
