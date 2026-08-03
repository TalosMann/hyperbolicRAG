# Schola AI — Build Specification

**Status:** Spec ready for build. Intended to be given as full context to an AI build session (Claude Fable 5) to construct the project end-to-end, in phases.
**Relationship to this repo:** This repo (`hyperbolic/`, containing `hyperscholar/`, `Hyper-RAG/`, `Cog-RAG/`) becomes a **permanent RAG-strategy R&D testbed** — it stays exactly as-is, for indexing corpora and running head-to-head comparisons (HyperRAG vs HierarchicalRAG vs Cog-RAG vs whatever comes next) under `hyperscholar/eval/`. **Schola AI is a new, separate project** that consumes whichever RAG strategy the testbed validates, through a stable plugin interface — it never re-litigates RAG research, it just plugs in the winner.

---

## 0. How to use this document

Build in the phase order in §10. After each phase: run its tests, confirm the phase's exit criteria, then stop and check in before starting the next phase — do not silently continue through multiple phases in one pass. Sections marked **LOCKED** are architecture decisions already validated (in HyperScholar or by explicit user choice) — do not redesign them without asking. Sections marked **OPEN** are deliberately left for the build session to decide, using the stated constraints; make a decision, write it down in the repo (e.g. an ADR file), and move on.

---

## 1. Vision

Schola AI is a hallucination-resistant, personalized-education platform: a teacher uploads a corpus, students query it and get answers grounded only in that corpus (with a flagged fallback to a global baseline), submit their own material for approval, and receive lesson content adapted to their learning style, disability accommodations, and preferred delivery format (plain text, narrated script, or a game).

Three things must be swappable **independently of each other, at runtime, via config** — this is the central design constraint of the whole project, more important than any individual feature:

1. **RAG strategy** — how a query gets turned into a grounded answer (HyperRAG, HierarchicalRAG, Cog-RAG, or a future method).
2. **Delivery mode** — how a grounded answer becomes a learning experience (static text lesson, narrated audio, interactive quiz, bubble-shooter-style game, future modes).
3. **Pedagogy/disability strategy** — how content gets adapted to a specific student (Kolb learning-style routing, declared-disability accommodations, modality preference, future rule sets or ML-driven personalization).

None of these three should ever require touching another's code to change. A teacher's corpus, a student's profile, and a lesson's content must be representable independent of which plugin is currently active in each of the three dimensions.

---

## 2. Product scope

**Roles:** teacher, student, institution admin (manages teacher accounts + institution settings). No public/anonymous querying in v1 — every query is scoped to an authenticated user's institution or personal namespace.

**Core loop (v1, self-hosted, single institution):**
1. Institution admin creates the institution; teacher accounts are created/invited under it.
2. Teacher uploads a corpus (PDF/TXT/MD/JSON/JSONL) → ingested → indexed into `inst_{id}` namespace via the active RAG strategy.
3. Student joins via an access code, is scoped to that institution.
4. Student queries the classroom corpus → gets a grounded answer, or a flagged global-fallback answer, never a silent hallucination.
5. Student can submit their own document for the corpus → teacher approves/rejects → approved docs get ingested the same way.
6. Student requests a lesson on a topic → grounded answer + student's pedagogy profile → Content Synthesis produces a script → rendered through the student's preferred (or teacher-assigned) delivery mode.
7. Student and teacher both see a concept-graph visualization (Poincaré/hyperbolic layout) of the corpus topology around any topic.

**Explicitly out of scope for v1** (note in the spec, do not build): multi-institution SaaS billing, public marketplace of third-party plugins, real-time multiplayer games, mobile native apps. The architecture must not preclude these later — see §5 for how.

---

## 3. Core architecture principle: three plugin surfaces

Every one of the three swap points (RAG strategy, delivery mode, pedagogy strategy) follows the **same shape**, because this shape was validated in HyperScholar's Phase 1 and held up across six different RAG backend implementations without changing the router, API, or eval code above it:

- An abstract base class with a small number of methods (aim for ≤5) — the smallest interface that lets everything above it stay ignorant of which plugin is active.
- A **registry** (a plain dict, `{name: constructor}`) that config selects from by string key — no dynamic plugin discovery/marketplace machinery for v1, just an explicit registry that a developer edits to add a new plugin (one line).
- A **factory function** that reads config and constructs the wired object graph — this is the *only* place that knows concrete plugin classes exist; routers, API handlers, and tests interact only through the ABC.
- An **interface conformance test suite** that any new plugin must pass before it's considered wired in — catches "implements the ABC but doesn't actually behave correctly" bugs early (HyperScholar's Phase 1 had 16 of these and they caught real bugs).
- Config selects all three independently, in one file:

```yaml
# config.yaml (excerpt)
rag:
  strategy: hyperrag          # from the RAG strategy registry
delivery:
  default_mode: text_lesson   # from the delivery mode registry — student prefs can override per-request
pedagogy:
  strategy: kolb_disability_v1  # from the pedagogy registry
```

**LOCKED.** Do not add a fourth, more "flexible" plugin mechanism (e.g. loading arbitrary Python from a directory, or a full plugin-marketplace system) for v1 — the registry pattern above is deliberately simple and was chosen because it's what already worked.

---

## 4. Decisions carried over from HyperScholar (LOCKED)

These were validated in the testbed and should be ported as-is, not redesigned:

- **Three-layer content model.** *Content* (documents, chunks, embeddings) and *Cache* (LLM responses) are **shared** across every RAG strategy in one `shared` Postgres/pgvector schema. *Structure* (hyperedges, summary trees, entities — whatever a given RAG strategy needs) is **isolated** per strategy in its own schema. This is what makes swapping RAG strategies a config change instead of a re-ingest: identical chunks and embeddings feed every strategy, only the structural overlay differs.
- **Tenant namespace scheme:** `global` (curated baseline), `inst_{id}` (institutional), `personal_{uid}` (personal). Every chunk/entity/structure row is keyed by `(namespace, id)`. Keep this even though v1 deploys single-institution — retrofitting multi-tenancy into a schema that didn't have it from day one is expensive; including the namespace column from day one is nearly free.
- **The `RAGStrategy` interface must be identical in shape to HyperScholar's `RAGBackend` ABC**: `name` (property), `index(namespace, documents) -> IndexResult`, `query(namespace, text, top_k) -> QueryResult`, `get_concept_graph(namespace, center, depth) -> ConceptGraph`, `delete_namespace(namespace) -> bool`. Whichever backend wins in the `hyperscholar/eval/` testbed should port with **zero interface changes** — only the storage wiring changes (production always uses Postgres, never the testbed's `memory://` dev mode).
- **LLM provider fallback chain**: ordered list of providers (DeepSeek primary, others as fallback), first with a usable API key wins. Port `hyperscholar/core/llm.py`'s pattern directly.
- **Embedder**: `BAAI/bge-m3` via `sentence-transformers`, 1024-dim, device auto-detect (cuda/mps/cpu). Port `hyperscholar/core/embedder.py` directly. Changing embedding models requires clearing the shared content store (dimension mismatch is silent otherwise) — keep this documented constraint, don't try to solve multi-dimension support in v1.
- **Corpus ingestion**: `pdfplumber` for PDFs, plus TXT/MD/JSON/JSONL parsing with the same field-name fallback chain (`content → context → text → passage → abstract`) HyperScholar's `ingestion.py` uses. Port directly.
- **Router three-tier strategy**: classroom query tries institutional namespace first, falls back to global with an explicit `out_of_scope=True` flag the UI must surface as a disclaimer — never silently answer from the wrong corpus.

---

## 5. New architecture decisions for production (LOCKED unless marked OPEN)

- **Postgres-only.** No `memory://` dev-mode store in the product code — that pattern stays in the HyperScholar testbed only. Production tests use a real (test) Postgres database (e.g. via a Dockerized Postgres in CI), not an in-memory fake, so tests exercise the real schema/migrations.
- **Self-hosted, single-institution first, multi-tenant-ready schema.** Auth and the data model support multiple institutions from day one (the `institutions` table already exists in the schema), but v1 ships configured for one institution's operators to run it on their own infra. Do not build SaaS billing, plan tiers, or cross-institution admin tooling in v1.
- **Auth:** JWT-based sessions, role field (`institution_admin | teacher | student`) on the user row, password hashing via a standard library (e.g. `passlib`/`argon2`), access-code-based student self-enrollment (matches HyperScholar's Phase 2 design). **OPEN:** whether to add OAuth/SSO — not required for v1, but don't design the `users` table in a way that precludes adding an `auth_provider` column later.
- **Delivery modes need browser-side interactivity that plain Python cannot provide** — this is the one place "Python end-to-end" needs an explicit, scoped exception. Recommendation: FastAPI backend serves server-rendered pages (Jinja2 + HTMX) for everything that's forms/CRUD/text (teacher dashboard, corpus upload, classroom text-lesson view, approval queue) — this keeps ~90% of the frontend in Python with minimal JS. For the two genuinely interactive surfaces:
  - **Game delivery modes** (e.g. bubble-shooter Q&A) — an isolated JS layer per delivery-mode plugin (e.g. a small Phaser.js or plain HTML5-canvas bundle), loaded only when that delivery mode is active, communicating with the backend only through the `DeliveryPayload` JSON contract (§6.2). The game code must not need to know anything about RAG strategies, pedagogy strategies, or auth beyond "here is my payload and my submit-answer endpoint."
  - **Poincaré/hyperbolic concept-graph visualization** — an isolated JS visualization (e.g. D3.js or Three.js) consuming `get_concept_graph()`'s JSON output as its only interface to the backend.

  Both isolated JS surfaces are additive, sandboxed, and swappable independently of the Python core — treat them the same way as a RAG strategy or delivery mode: a small, replaceable module behind a fixed data contract.
- **Background jobs:** corpus ingestion (chunking, embedding, entity/structure extraction) must not block an HTTP request — use a task queue (e.g. Celery or RQ with Redis, or FastAPI `BackgroundTasks` if v1's ingestion volume is low enough; **OPEN** — decide based on expected corpus size/frequency, document the choice). Corpus status (`indexing → ready → failed`) is a column the frontend polls or gets pushed via websocket/SSE (**OPEN** — pick the simpler one, polling, unless there's a clear need for push).
- **Observability:** structured logging (not print statements) from day one, since this is explicitly the gap between the HyperScholar testbed (console-only) and a production system. Minimum: request logging with a request-id, error logging with stack traces, and a health-check endpoint.
- **Secrets:** all API keys and DB credentials via environment variables (`.env` for local dev, real secret storage for deployment), never committed. Mirror HyperScholar's `${VAR:default}` YAML interpolation pattern for config.
- **Migrations:** use a real migration tool (Alembic) from the first schema commit — do not hand-write ad-hoc `CREATE TABLE` scripts that drift from what's actually deployed.

---

## 6. Core abstractions

### 6.1 RAG Strategy plugin

```python
class RAGStrategy(ABC):
    """One instance serves every tenant namespace, isolated by `namespace` args.
    Identical shape to HyperScholar's RAGBackend — ported strategies need zero
    interface changes, only production (Postgres-only) storage wiring."""

    @property
    @abstractmethod
    def name(self) -> str: ...

    @abstractmethod
    async def index(self, namespace: str, documents: list[Document]) -> IndexResult: ...

    @abstractmethod
    async def query(self, namespace: str, text: str, top_k: int = 60) -> QueryResult: ...

    @abstractmethod
    async def get_concept_graph(self, namespace: str, center: str, depth: int = 2) -> ConceptGraph: ...

    @abstractmethod
    async def delete_namespace(self, namespace: str) -> bool: ...
```

### 6.2 Delivery Mode plugin

```python
class DeliveryMode(ABC):
    """Transforms a grounded answer + synthesized content into a presentation
    format. Must not know anything about RAG strategies or pedagogy rules —
    it receives fully-adapted content and renders it."""

    @property
    @abstractmethod
    def name(self) -> str: ...

    @abstractmethod
    async def render(self, content: LessonContent) -> DeliveryPayload:
        """LessonContent is delivery-mode-agnostic (script text, Q&A pairs,
        source citations, difficulty level). DeliveryPayload is a tagged
        union the frontend switches on to pick a renderer:
        TextLessonPayload | AudioScriptPayload | QuizPayload | GamePayload."""
```

### 6.3 Pedagogy/Disability Strategy plugin

```python
class PedagogyStrategy(ABC):
    """Adapts content to one student. Disability accommodations always take
    priority over learning-style routing over modality preference — this
    ordering is a product decision (accessibility is non-negotiable), not
    left to individual strategy implementations to reorder."""

    @property
    @abstractmethod
    def name(self) -> str: ...

    @abstractmethod
    def adapt(self, student_profile: StudentProfile, content: LessonContent) -> LessonContent: ...
```

`StudentProfile` fields (minimum): `kolb_position`, `declared_disabilities` (declared only — teacher- or student-written, **never inferred** from behavior, this was an explicit HyperScholar decision worth keeping), `modality_preference`, `interaction_history`.

### 6.4 Registries + factory

One `factory.py` per plugin surface (`rag/factory.py`, `delivery/factory.py`, `pedagogy/factory.py`), each holding a `REGISTRY: dict[str, Callable[..., T]]` and a `build_x(cfg) -> T` function, mirroring `hyperscholar/rag/factory.py`'s pattern exactly.

---

## 7. Data model (Postgres, sketch)

```
institutions        (id, name, created_at, settings jsonb)
users                (id, institution_id nullable, role, email, password_hash, created_at)
access_codes         (id, institution_id, code, role, uses_remaining, expires_at)
memberships          (user_id, institution_id, role)
corpuses             (id, institution_id, name, status, created_at)
corpus_documents      (id, corpus_id, source_ref, ingested_at)
corpus_submissions    (id, corpus_id, student_id, doc_ref, status, reviewed_by, reviewed_at)
student_profiles      (user_id, kolb_position, declared_disabilities jsonb, modality_preference, interaction_history jsonb)
queries               (id, user_id, namespace, text, rag_strategy, out_of_scope, created_at)
lessons               (id, query_id, delivery_mode, pedagogy_strategy, content jsonb, created_at)

# shared schema (per §4 — content shared across RAG strategies)
shared.documents, shared.chunks (+ embedding vector), shared.llm_cache

# one isolated schema per active RAG strategy, e.g.:
hyperrag.entities, hyperrag.hyperedges
hierarchical.tree_nodes
cograg.entities, cograg.themes
```

Namespace column (`global | inst_{id} | personal_{uid}`) present on every content/structure row per §4.

---

## 8. Tech stack

- **Backend:** Python 3.12, FastAPI, SQLAlchemy + Alembic, Pydantic for request/response models.
- **RAG/ML core:** ported from HyperScholar — `sentence-transformers` (bge-m3), OpenAI-compatible client for DeepSeek/SiliconFlow, whichever RAG strategy package(s) the testbed validates (Hyper-RAG, Cog-RAG, or successors) vendored the same way HyperScholar vendors them.
- **Data:** PostgreSQL + pgvector, Redis (if a task queue is chosen — see §5 OPEN item).
- **Frontend:** Jinja2 + HTMX server-rendered pages for CRUD/dashboard/text-lesson surfaces; isolated JS bundles (Phaser.js or canvas, and D3.js/Three.js) only for game delivery modes and the concept-graph visualization, per §5.
- **Testing:** pytest + pytest-asyncio, interface-conformance suites for all three plugin surfaces, a real (containerized) Postgres for integration tests.
- **Deployment:** Docker Compose (app + Postgres [+ Redis]) for self-hosted v1; document the multi-tenant-SaaS migration path in an ADR but do not build it.

---

## 9. Repo/file structure

```
schola-ai/
├── config.yaml
├── docker-compose.yml
├── alembic/
├── core/
│   ├── config.py            # YAML loader + ${ENV:default} interpolation
│   ├── embedder.py           # ported from hyperscholar
│   ├── llm.py                 # ported from hyperscholar (provider fallback chain)
│   └── types.py               # Document, Chunk, LessonContent, DeliveryPayload, etc.
├── rag/
│   ├── base.py                 # RAGStrategy ABC
│   ├── factory.py              # registry + build_rag_strategy(cfg)
│   ├── router.py                # three-tier query strategy (ported)
│   └── strategies/              # one file per ported/implemented strategy
├── delivery/
│   ├── base.py                  # DeliveryMode ABC
│   ├── factory.py
│   └── modes/                   # text_lesson.py, quiz.py, game_bubble_shooter.py, ...
├── pedagogy/
│   ├── base.py                  # PedagogyStrategy ABC
│   ├── factory.py
│   └── strategies/               # kolb_disability_v1.py, ...
├── ingestion/                     # ported from hyperscholar/ingestion.py
├── db/
│   ├── models.py                  # SQLAlchemy models (§7)
│   └── schema.sql (generated by Alembic, not hand-authored)
├── api/
│   ├── auth.py
│   ├── admin.py                    # institution admin endpoints
│   ├── teacher.py                    # corpus upload, approval queue, analytics
│   └── student.py                     # join, query, submit, lesson request
├── frontend/
│   ├── templates/                       # Jinja2
│   ├── static/js/games/                  # isolated per-delivery-mode JS
│   └── static/js/concept_graph/            # Poincaré/hyperbolic visualization
└── tests/
    ├── test_rag_conformance.py
    ├── test_delivery_conformance.py
    ├── test_pedagogy_conformance.py
    └── test_api.py
```

---

## 10. Build phases

Build in this order; each phase has explicit exit criteria. Stop and confirm with the user between phases.

**Phase 0 — Scaffolding.** Repo skeleton above, config loader, the three ABCs + empty registries + factories, Alembic wired to an empty Postgres. *Exit: `docker compose up` boots the app against a real Postgres with no tables yet, config loads and validates.*

**Phase 1 — Data model + auth.** Full schema (§7) via Alembic migrations, JWT auth, role-based endpoint protection, access-code enrollment. *Exit: institution admin can create a teacher account; a student can self-enroll via access code; role-gated endpoints reject wrong roles.*

**Phase 2 — RAG strategy layer.** Port the shared content store + at least one concrete `RAGStrategy` (whichever the testbed currently recommends) + the interface conformance suite. *Exit: a corpus indexes and queries correctly through the ported strategy against real Postgres; conformance suite passes.*

**Phase 3 — Corpus ingestion pipeline.** Port `ingestion.py`; wire teacher upload → background job → chunk/embed/structure-build → `corpuses.status` transitions. *Exit: teacher uploads a PDF, corpus status moves `indexing → ready`, content is queryable.*

**Phase 4 — Classroom query layer.** `RAGRouter.query_classroom` behind an API endpoint; `out_of_scope` flag surfaced. *Exit: student query against an indexed corpus returns a grounded answer; a query outside the corpus returns a global-fallback answer flagged `out_of_scope=True`.*

**Phase 5 — Corpus approval workflow.** Student submission endpoint, teacher approve/reject queue, approved docs flow into the same ingestion pipeline. *Exit: full submit → approve → re-index loop works.*

**Phase 6 — Pedagogy/disability plugin layer.** `student_profiles` CRUD, disability declaration endpoint (teacher- or student-writable, never inferred), at least one concrete `PedagogyStrategy` implementing the priority order in §6.3, conformance suite. *Exit: two students with different profiles get differently-adapted content for the same query.*

**Phase 7 — Delivery mode plugin layer + Content Synthesis.** `LessonContent` generation (RAGRouter output + pedagogy-adapted content → LLM script + Q&A extraction, cached per `(query_hash, student_id, rag_strategy)`), at least two concrete `DeliveryMode`s (one text-based, one game), conformance suite. *Exit: same lesson content renders correctly through both delivery modes.*

**Phase 8 — Frontend.** Jinja2/HTMX teacher dashboard (upload, approval queue, analytics), student interface (join, query, lesson view), isolated JS for the chosen game delivery mode and the concept-graph visualization. *Exit: a teacher and a student can complete the full core loop (§2) through the UI, not just the API.*

**Phase 9 — Production hardening.** Structured logging, health checks, rate limiting on public-ish endpoints (access-code redemption, login), backup/restore runbook for Postgres, deployment docs (Docker Compose runbook). *Exit: a fresh clone + documented setup steps gets a working instance without tribal knowledge.*

---

## 11. Porting guide from HyperScholar

**Port directly, minimal changes:** `core/embedder.py`, `core/llm.py`, `ingestion.py`, the `RAGRouter` three-tier logic, whichever `rag/*_backend.py` the testbed currently recommends (rename to fit `rag/strategies/`), the namespace/tenant scheme, the shared-content/isolated-structure schema split.

**Generalize, don't port as-is:** `rag/factory.py`'s `BACKENDS` dict → `rag/factory.py`'s `REGISTRY`, same pattern but named for a strategy layer that's meant to keep growing. `HYPERSCHOLAR_ROADMAP.md`'s Phase 6 (Teaching Styles rule engine) → generalize into the `PedagogyStrategy` ABC (§6.3) rather than one hardcoded rule engine. Phase 7 (Content Synthesis) → generalize into `DeliveryMode` + a synthesis step that's mode-agnostic (produces `LessonContent`, not a specific script format).

**Do not port:** `hyperscholar/eval/` (question generation, LLM-judge, comparison reports), `hyperscholar/scripts/compare_backends.py`, the `memory://` storage backend, anything offline-stub/test-only. These stay in the testbed permanently — Schola AI never re-runs RAG comparisons itself, it just consumes the testbed's conclusions.

---

## 12. Non-functional requirements

- Every plugin surface (§6) must ship with a conformance test suite before it's considered "wired in" — no plugin merges without one, mirroring HyperScholar's 16/16 Phase 1 tests.
- No hardcoded fail-marker string matching for refusal detection (a bug class found in the HyperScholar testbed — a RAG strategy's actual refusal phrasing may not match another's canned string). Detect "insufficient grounding" structurally (e.g. an explicit `ok: bool` / `sources: []` on `QueryResult`), not by substring-matching English sentences.
- Disability declarations are write-only by teacher/student, read by the pedagogy layer, never written by any inference/ML process — this is a product/ethics requirement, not just a technical one.
- All secrets via environment variables; no API keys or DB passwords in version control.
- Structured logs with request IDs from Phase 0 onward, not retrofitted in Phase 9.

---

## 13. Explicitly open decisions (OPEN — build session decides, document the choice)

- Task queue technology for background ingestion (Celery/RQ/FastAPI BackgroundTasks) — decide based on expected ingestion volume.
- Polling vs push (websocket/SSE) for corpus-status and submission-status updates.
- OAuth/SSO support — not required for v1, don't preclude it.
- Exact game engine for the first game delivery mode (Phaser.js vs plain canvas vs other) — pick the simplest one that satisfies the `DeliveryPayload` contract.
- Whether Content Synthesis caching (§10 Phase 7) uses Postgres or a separate cache store (Redis) — reuse whatever task-queue infra decision was already made if possible.
