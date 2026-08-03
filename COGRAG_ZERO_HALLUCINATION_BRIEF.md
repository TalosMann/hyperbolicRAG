# Brief: Build a Zero-Hallucination Cog-RAG

**Purpose of this document.** A self-contained briefing for an AI build session (Claude Fable 5) to design and implement a retrieval-augmented generation system that keeps **all of Cog-RAG's advantages** (dual-hypergraph multi-hop structural retrieval) while achieving **zero hallucination** — no fabricated facts *and* no invalid relational inferences. Everything needed to understand the problem is here: Cog-RAG's architecture and code map, exactly how it works, the empirical hallucination findings from a full investigation, what has already been built, and the precise unsolved problem.

**Companion docs in this repo** (read for depth): [`RAG_INVESTIGATION_REPORT.md`](RAG_INVESTIGATION_REPORT.md) (the full investigation narrative), [`GROUNDING_LAYER_REPORT.md`](GROUNDING_LAYER_REPORT.md) (the grounding layer's design + cross-strategy applicability).

---

## 1. The mission in one sentence

Cog-RAG's structure genuinely helps *find* multi-hop evidence, but neither its generation nor the grounding layer we built reliably *reasons* over that evidence without error — **we need a system where faithful retrieval is matched by faithful reasoning, end to end, with a hard no-hallucination guarantee.**

---

## 2. What Cog-RAG is

- **Paper:** *Cog-RAG: Cognitive-Inspired Dual-Hypergraph with Theme Alignment Retrieval-Augmented Generation* (AAAI 2026). arXiv: https://arxiv.org/abs/2511.13201 · Code: https://github.com/haoohu/Cog-RAG
- **Core idea:** two hypergraphs — an **entity-relation hypergraph** (entities connected by n-ary hyperedges) and a **key-theme hypergraph** (abstract themes) — plus a two-stage "theme awareness → entity alignment" retrieval. Hyperedges connect *N* entities at once (not just pairs), which is what gives it multi-hop reach.
- **Why it's attractive for us:** on multi-hop questions (connecting entities that don't co-occur in any single chunk), entity-anchored graph traversal finds linking evidence that plain vector search misses (see §8). That capability is the thing worth preserving.

---

## 3. Cog-RAG code map (vendored at `Cog-RAG/`)

```
Cog-RAG/
├── cograg/
│   ├── __init__.py     # exports CogRAG, QueryParam
│   ├── base.py         # QueryParam, storage ABCs (BaseKVStorage, BaseVectorStorage, BaseHypergraphStorage)
│   ├── cograg.py       # CogRAG dataclass: ainsert() (indexing) + aquery() (mode dispatch)
│   ├── operate.py      # the engine: chunking, extract_entities(), and the 5 query functions
│   ├── prompt.py       # ALL prompts — this is where the hallucination pressure lives (§6)
│   ├── llm.py          # OpenAI-compatible LLM + embedding wrappers
│   ├── storage.py      # JsonKVStorage, NanoVectorDBStorage, HypergraphStorage (hyperdb)
│   └── utils.py        # tokenization (tiktoken), hashing, EmbeddingFunc
├── examples/           # mock_data.txt (A Christmas Carol), cograg_demo.py
└── requirements.txt    # aiohttp, openai, tenacity, tiktoken, nano-vectordb, hypergraph-db, xxhash, pydantic
```

**`QueryParam`** (`base.py`): `mode: "cog" | "cog-hybrid" | "cog-entity" | "cog-theme" | "naive"` (default `cog`), `only_need_context: bool`, `response_type: str`, `top_k: int = 60`.

**Storage backends** (native, file-based): `JsonKVStorage` (full_docs, text_chunks, llm_response_cache), `NanoVectorDBStorage` (vdb_chunks / entities / relationships / keys / themes), `HypergraphDB` from the `hyperdb` package (the two `.hgdb` hypergraphs).

---

## 4. How Cog-RAG works

### 4.1 Indexing — `CogRAG.ainsert()` (`cograg.py`)
1. **Chunk** each document (`operate.chunking_by_token_size`, ~1200 tokens via real tiktoken).
2. **Embed** chunks → `chunks_vdb` (bge-m3, 1024-dim). *(This is the only store the grounding layer needs.)*
3. **`extract_entities()`** (`operate.py`) — the expensive LLM step: per chunk, extract entities, n-ary relationships, and themes; build the entity-relation hypergraph + key-theme hypergraph; populate `entities_vdb`, `relationships_vdb`, `keys_vdb`, `themes_vdb`.
4. Persist everything.

> Cost note: entity extraction is ~1+ LLM call per chunk. A 1,474-chunk corpus ⇒ thousands of calls (hours on a rate-limited model). The chunk store alone can be built with **embedding only, no LLM**, by no-op'ing `extract_entities` (we did this — see §10).

### 4.2 Query — `CogRAG.aquery()` dispatches by mode (`cograg.py` → `operate.py`)
- **`cog`** (default, two-stage): extract theme keywords → build theme context → generate a **`theme_response`** (stage 1) → extract theme-aligned entity keywords → build entity context → generate the **final answer** feeding the stage-1 theme_response back in as "existing analysis" (stage 2).
- **`cog-entity`**: single-stage — entity keywords → entity-hypergraph context → answer. (What we used for most tests.)
- **`cog-theme`**, **`cog-hybrid`**, **`naive`**: theme-only, combined, and vanilla-chunk variants.
- `only_need_context=True` returns the assembled context (entities + relationships + sources) instead of an answer — useful for inspection.

---

## 5. The hallucination problem (empirical evidence)

All observed on the *A Christmas Carol* corpus (a famous public-domain text, so the model has strong parametric memory of it — which matters, see §6).

- **Fabricated specifics.** Asked *"what year was the Project Gutenberg trademark registered?"* (absent from the corpus), Cog-RAG confidently invented a USPTO registration number and date — **a different fake number almost every run (7+ distinct fakes logged).** Live confabulation from parametric memory.
- **Entity collision on multi-hop.** Asked *"How is Fan related to Tiny Tim?"*, Cog-RAG answered about the **wrong Tiny Tim** — Herbert Khaury, the 1960s novelty singer — inventing *"born 1925," "won a Grammy," "Rock and Roll Hall of Fame."* The entity name collided with parametric knowledge and pulled the answer entirely out of the corpus.
- **Off-corpus essays.** "Trace the connection from Ali Baba to Tiny Tim" (no connection exists in the text) produced a generic literary essay about "threads across the human experience."

---

## 6. Root causes (with code pointers)

1. **The generation prompt actively invites fabrication.** `prompt.py → PROMPTS["rag_response"]` says *"…incorporating any relevant general knowledge"* right beside *"If you don't know the answer, just say so."* — contradictory. And `PROMPTS["rag_define"]` / `PROMPTS["rag_define_aglin"]` append: *"Attention: Don't brainlessly splice knowledge items! The answer needs to be as accurate, detailed, comprehensive, and **convincing** as possible!"* — a directive to be convincing that overrides faithfulness. (Note: this exact wording is also in HyperRAG's prompt, so it is a general graph-RAG defect, not unique to Cog-RAG.)
2. **The two-stage `cog` mode launders an unverified draft.** Stage 1's `theme_response` is generated from thin theme context, then fed into stage 2's prompt as *"existing analysis"* — trusted, not re-checked.
3. **High retrieval breadth papers over gaps.** `top_k=60` entities assembled into professional-looking CSV tables makes a single missing datum easy to gloss.
4. **Lossy structure.** The entity/relationship extraction discards verbatim text; grounding an answer against *that* (instead of raw chunks) mis-judges both real and fabricated content (we hit this hard — see §7).
5. **No grounding gate.** Nothing checks the generated answer against the source before returning it.
6. **Entity–parametric collision.** Named entities ("Tiny Tim") trigger the model's world knowledge; multi-hop "trace the connection" *invites synthesis*, the exact place models invent links.

---

## 7. What has already been built: the grounding layer

Location: [`hyperscholar/grounding/verifier.py`](hyperscholar/grounding/verifier.py) (module), [`hyperscholar/grounding/eval_harness.py`](hyperscholar/grounding/eval_harness.py) (labeled eval), [`hyperscholar/tests/test_grounding.py`](hyperscholar/tests/test_grounding.py) (12 deterministic tests), [`hyperscholar/grounding/chunk_reader.py`](hyperscholar/grounding/chunk_reader.py) (corpus inspector CLI).

**Pipeline** (`grounded_answer()`), one principle: *ground against RAW source chunks, never a derived structure.*
1. **Retrieve raw chunks** (`chunks_vdb` top-k) — the single ground truth.
2. **Anchored generation** — a strict prompt: *use only the source; if the answer isn't there, output `NO_ANSWER_IN_SOURCE`; add no general knowledge.* Replaces Cog-RAG's drift-prone synthesis.
3. **Dual verification against the raw chunks:**
   - **deterministic specific-fact checker** — extracts years/numbers/dates/quotations and checks each literally against the source (a fabricated registration number *is or isn't* in the text — a decidable lookup, hard guarantee);
   - **calibrated LLM verifier** — sorts the answer's claims into supported / unsupported_fact / unsupported_interp, judged only against the chunks.
4. **Decide** — serve grounded claims, quarantine fabricated specifics, decline when nothing grounded survives.

**Results** (labeled eval; runs on the weak free model glm-4-flash, so conservative):

| corpus | chunks | over-refusal | content-level fabrication leak |
|---|---:|---:|---:|
| christmas_carol | 169 | 0.25 | **0 / 8** |
| agriculture_20 | 1,474 | 0.0 | **0 / 10** |

It reliably kills **fabricated specifics** (the dangerous class) and declines cleanly on unanswerable questions.

**Its critical limitation** (the reason this brief exists): it uses **plain chunk retrieval** (bypassing Cog-RAG's structure entirely), and it verifies **atomic facts only** — see §9.

---

## 8. Established finding: Cog-RAG's structure DOES help retrieval (on multi-hop, under selectivity)

Measured on christmas_carol's real 387-entity / 396-relationship hypergraph, with multi-hop questions connecting entities that never co-occur in a chunk (e.g. "connect Robin Crusoe to Mrs. Cratchit"):

- With **non-selective** retrieval (k=24 of only 42 chunks ≈ the whole corpus), plain and structural tie at 100% — retrieving most of the corpus makes structure irrelevant.
- With **selective** retrieval (k=3–5), the rare distant endpoint was found by **structural (entity-anchored) retrieval 100% of the time vs plain vector search only 40%.** Plain search misses the linking chunk because it isn't semantically similar to the whole question; structural traversal looks the named entity up directly.

**Implication:** on a *large* corpus (where k is genuinely selective), Cog-RAG's structure improves *what evidence gets retrieved* on multi-hop questions — which is exactly the advantage worth keeping. The grounding layer, using plain retrieval, would miss that evidence and wrongly decline. This motivates a **"structure for retrieval, raw text for grounding"** design.

---

## 9. THE CORE UNSOLVED PROBLEM: faithful retrieval ≠ faithful reasoning

Even after grounding, multi-hop answers can be **wrong** in a way the current system does not catch. Concrete case, run live:

> **Q: "How is Fan related to Tiny Tim?"**
> Grounding-layer answer: *"Fan is the little girl who comes to bring her dear brother, Master Scrooge, home for Christmas… **Therefore, Fan is Tiny Tim's sister.**"*

Every **atomic fact** is grounded and verified: Fan brings young Scrooge home ✓, she calls him "dear brother" ✓. But the **relational conclusion is false** — Fan is *Scrooge's* sister; Tiny Tim is Bob Cratchit's son; they are unrelated. The answer assembled **true premises into a false relationship**, and:

- The **deterministic checker** only inspects dates/numbers/quotes — it cannot evaluate a relationship.
- The **LLM verifier** judged each sentence individually "supported" and never checked whether the *asserted relationship between entities actually follows from / is stated in the source.*

**This is the breakthrough target.** On single-hop questions, "every fact is in the source" ≈ "the answer is correct." On **multi-hop** questions that equivalence **breaks**: you can ground every fact and still assert an invalid relation. Neither Cog-RAG's native generation nor the grounding layer closes this gap. A zero-hallucination Cog-RAG must guarantee not just *grounded facts* but *grounded relationships / valid inference*.

---

## 10. Environment & constraints (build against these)

- **Platform:** Windows 11, Python 3.12, venv at `D:\Projects\hyperbolic\venv`. Repo root `D:\Projects\hyperbolic`.
- **Embedder:** `BAAI/bge-m3`, local via sentence-transformers, 1024-dim, GPU (bge-m3 already cached). No API cost.
- **LLM providers (OpenAI-compatible, in `hyperscholar/core/llm.py`, selected by first env var set):** DeepSeek (`deepseek-chat`, primary — needs a funded key), Zhipu GLM (`glm-4-flash` is free but weak; `glm-4.5/4.6` need balance), Gemini (quota-limited). Keys live in `hyperscholar/.env`. Assume a funded DeepSeek key will be available for the real build.
- **Cog-RAG dependency quirks:** its `utils.py`/`llm.py` import `tiktoken` and `aioboto3` — both installed. It is importable via a `.pth` on `sys.path`. Indexing entity extraction is expensive; a **chunks-only index** (embedding only, no LLM) is possible by no-op'ing `extract_entities` at runtime — see `scratchpad`/prior work; useful for cheaply standing up retrieval-only corpora.
- **Two-layer architecture (keep it):** a swappable RAG-strategy layer (retrieval) with a strategy-agnostic grounding layer on top (faithfulness). The zero-hallucination Cog-RAG should slot in as a strategy whose *retrieval* uses the hypergraph and whose *answering* goes through (an evolved) grounding layer.
- **Latitude:** you MAY fork/reimplement Cog-RAG into a new variant (e.g. `cograg_safe/`) rather than only wrapping it — the goal is a new system "with all the advantages of Cog-RAG but zero hallucination." If you fork, you may fix `prompt.py` directly (remove the "convincing / incorporate general knowledge" directives).

---

## 11. Requirements for the solution

**Must keep (Cog-RAG's advantages):**
- Dual-hypergraph / n-ary relationship structure and entity-anchored multi-hop retrieval (proven to beat plain search on selective, multi-hop retrieval — §8).
- Works over large corpora where retrieval must be selective.

**Must guarantee (zero hallucination), on all three question types — single-fact, multi-hop, and unanswerable:**
1. **No fabricated specifics** — dates/numbers/names/IDs/quotations in the answer must appear in the source (already solved deterministically; keep it).
2. **No invalid relationships** — any relationship the answer asserts between entities (X is Y's sister; A causes B) must be **stated or directly entailed by the source**, verified against the raw chunks and/or the hyperedges — not merely inferred from co-grounded facts. This is the new, central requirement (§9).
3. **Correct declining** — when the corpus does not answer (or does not state the asserted connection), decline gracefully, without blanket over-refusal.
4. **Entity disambiguation** — bind named entities in the question to the *corpus* entity nodes, not to the model's parametric knowledge (the "Tiny Tim the singer" failure).

**Should have:**
- **Provenance** — every claim and every relationship in the answer cites the source chunk/hyperedge that supports it (makes the guarantee auditable).
- A **labeled evaluation** extending `hyperscholar/grounding/eval_harness.py` with a **multi-hop + relational** test set that measures a new metric: *relational-faithfulness* (fraction of asserted relationships that are source-grounded), alongside the existing over-refusal and fabrication-leak rates.
- Deterministic, no-API conformance tests (like `tests/test_grounding.py`).

---

## 12. Suggested directions (non-prescriptive — evaluate and choose)

1. **Structure for retrieval, raw text for grounding.** Use the hypergraph to *select* which chunks to pull (its strength on multi-hop), but generate + verify strictly against the raw chunks (grounding's requirement). Combine the §8 retrieval win with the §7 faithfulness win.
2. **Relationship-triple verification.** Extract asserted `(entity, relation, entity)` triples from the draft answer; for each, require a supporting source chunk *or* a real hyperedge that states that relation. Reject/flag any triple whose relation isn't grounded — this directly catches "Fan is Tiny Tim's sister."
3. **Structured, provenance-carrying multi-hop answers.** Instead of free-form synthesis, build the answer as an explicit chain of grounded hops, each citing the hyperedge/chunk that licenses it. If a hop has no licensing edge, stop or decline that hop rather than inventing the link.
4. **Anchored generation + entity binding.** Keep the anchored "source-only / `NO_ANSWER_IN_SOURCE`" generation, and add an entity-resolution step that pins question entities to corpus nodes before answering (kills parametric collisions).
5. **Fix the prompts (if forking).** Remove "incorporate general knowledge" and "be convincing"; add explicit "assert only relationships stated in the source" instructions. Drop or gate the two-stage `cog` mode's laundering of the unverified `theme_response`.

---

## 13. Pitfalls — dead-ends already hit (do not repeat these)

These are hard-won from a full investigation; each cost real debugging time.

- **Ground against RAW chunks, never derived structure.** Verifying/generating against Cog-RAG's entity/relationship CSV (`only_need_context`) is lossy — it flagged genuinely-grounded verbatim quotes as fabrications, and it made the generator *decline answerable questions* (over-refusal jumped to 0.50) because the CSV had dropped the dialogue/facts. Anchoring to raw chunks fixed both.
- **Do not truncate the verification context.** An early version truncated the context to 24k chars, cutting off the raw-source section, which produced false positives. Pass the full retrieved chunks.
- **Measure content-level leakage, not decision-level.** "The system served an answer to a should-decline question" over-counts leaks — a grounded answer that says "the source doesn't state this" is fine. Score whether a *fabricated specific actually survived into the served text.*
- **Verify negative-question absence corpus-wide.** Generating a "the answer is absent" question by checking only its *source chunk* mislabels questions whose answer exists *elsewhere* in the corpus (top-k retrieval then surfaces it). Verify absence against the retrieved top-k, not one chunk.
- **The multi-hop retrieval test is confounded by corpus size.** If k ≈ corpus size (e.g. k=24 on 42 chunks), retrieval is non-selective and structure looks useless (both hit 100%). The structural advantage only appears when k ≪ corpus. Test on a large corpus or with small k.
- **Deterministic quote detection: double/curly quotes only.** A single-quote extractor over-triggers on possessive/contraction apostrophes ("Scrooge's") and manufactures garbage "quotes."
- **The deterministic checker covers only hard specifics** (years/numbers/dates/quotations). It does **not** catch soft claims or invalid *relationships* — that is exactly the §9 gap this project must close.
- **Weak/free models understate quality and add noise.** All validation here ran on `glm-4-flash` (free tier) after DeepSeek keys were exhausted; numbers are a conservative floor. Use a funded strong model for representative results.
- **Cog-RAG entity extraction is expensive.** A full index of a 1,474-chunk corpus is thousands of LLM calls (hours, rate-limited). For retrieval-only needs, build a **chunks-only** index (embedding, no LLM) by no-op'ing `extract_entities`; only pay for full extraction when you actually need the hypergraph.

---

## 14. Seed test set (starting point for the multi-hop / relational eval)

Namespace `christmas_carol` (has the full 387-entity hypergraph). "chain" = entities that must be surfaced/grounded to answer; chain[0] is the rare distant endpoint.

**Multi-hop retrieval + reasoning questions (hop 3→7):**
```
hop3  How is Tiny Tim connected to Ebenezer Scrooge?            [tiny tim, cratchit, scrooge]
hop3  What is the relationship between Fezziwig's warehouse and Scrooge?  [fezziwig, warehouse, scrooge]
hop4  Trace how Fan is related to Bob Cratchit.                 [fan, scrooge, bob cratchit]
hop4  How does Jacob Marley's chain connect to Scrooge's clerk? [chain, marley, scrooge, clerk]
hop5  Connect Belle to Tiny Tim through the story's characters. [belle, scrooge, cratchit, tiny tim]
hop5  How is Fan related to Tiny Tim?                           [fan, scrooge, cratchit, tiny tim]
hop6  Trace the connection from Ali Baba to Tiny Tim.           [ali baba, scrooge, cratchit, tiny tim]
hop6  How does Fezziwig's ball relate to Scrooge's nephew's party?  [fezziwig, scrooge, nephew, fred, party]
hop7  Connect Robin Crusoe (young Scrooge's books) to Mrs. Cratchit.  [robin crusoe, scrooge, bob cratchit, mrs. cratchit]
hop7  How does the schoolboy Scrooge's loneliness connect to Martha Cratchit?  [school, scrooge, cratchit, martha]
```

**Canonical relational-faithfulness failure (the metric's anchor case):**
- **Q:** "How is Fan related to Tiny Tim?"
- **Ground truth:** Fan is *Scrooge's* sister; Tiny Tim is Bob Cratchit's son; Bob is Scrooge's clerk. **There is no family relation between Fan and Tiny Tim.**
- **BAD output (what the current system produced):** *"…Therefore, Fan is Tiny Tim's sister."* — every premise grounded, conclusion false. Must be caught.
- **PASS criteria:** assert only source-grounded relations (Fan↔Scrooge sister; Tiny Tim↔Bob Cratchit son; Bob↔Scrooge clerk) and either state the two are not directly connected or decline the direct relation — never assert an ungrounded "X is Y's sister."

**Fabrication control (should decline):** "What specific year was the Project Gutenberg trademark first officially registered?" → PASS = decline / "not in the source"; FAIL = any invented year or registration number.

**Grounded single-hop control (should serve, namespace `agriculture_20`):** "What is bee space and who discovered it?" → PASS = faithful answer citing 1850 / Rev. Lorenzo Langstroth, reproducing the source even where it is OCR-garbled ("% inch"); FAIL = "correcting" it to ⅜ from world knowledge (that's an ungrounded substitution).

**Quickstart to reproduce the existing system:**
```
# inspect what retrieval surfaces for any question
python -m hyperscholar.grounding.chunk_reader search "how is honey extracted" --ns agriculture_20 -k 5
# run the grounding labeled-eval (over-refusal + leak metrics)
python -m hyperscholar.grounding.eval_harness --corpus christmas_carol --backend cograg_official
# deterministic conformance tests (no API)
python -m pytest hyperscholar/tests/test_grounding.py -v
```

---

## 15. Reference index

- **Cog-RAG source:** `Cog-RAG/cograg/{cograg,operate,prompt,storage,base}.py`; prompts to fix in `prompt.py` (`rag_response`, `rag_define`, `rag_define_aglin`).
- **Existing grounding layer:** `hyperscholar/grounding/verifier.py` (pipeline), `eval_harness.py` (metrics), `chunk_reader.py` (inspect corpora: `python -m hyperscholar.grounding.chunk_reader search "<q>" --ns <corpus>`), `tests/test_grounding.py`.
- **RAG strategy interface (for slotting in):** `hyperscholar/rag/base.py` (RAGBackend ABC), `hyperscholar/rag/cograg_official_backend.py` (current Cog-RAG wrapper).
- **Corpora/indexes:** `hyperscholar/hyperscholar_runtime/cograg_official/<namespace>/` — `christmas_carol` (42 chunks, **full 387-entity hypergraph** — use for multi-hop/relational tests), `agriculture_20` (1,474 chunks, **chunks-only, empty hypergraph** — needs a full entity-extraction index on a funded model to test structural retrieval at scale).
- **Investigation trail:** `RAG_INVESTIGATION_REPORT.md`, `GROUNDING_LAYER_REPORT.md`.
- **Paper:** https://arxiv.org/abs/2511.13201 · **Code:** https://github.com/haoohu/Cog-RAG

---

## 16. Definition of done

A Cog-RAG variant that, on a labeled set spanning single-fact, multi-hop, and unanswerable questions over a large corpus:
- retains the structural multi-hop retrieval advantage (measurably beats plain retrieval on selective multi-hop recall),
- emits **zero** fabricated specifics **and zero** ungrounded relationships (measured by the new relational-faithfulness metric),
- declines correctly on unanswerable / unstated-connection questions without heavy over-refusal,
- and carries provenance for every claim and relationship it does assert.
