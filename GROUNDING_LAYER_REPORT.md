# Hallucination-Prevention (Grounding) Layer — Design & Cross-Strategy Applicability

**Scope:** how the grounding layer works, what it does and doesn't guarantee, and why it applies over HyperRAG, HierarchicalRAG, and any future RAG strategy — not just Cog-RAG.

**Status:** Prototyped and validated in the testbed. Fabrication prevention **confirmed** (leak rate 0.0) and the raw-chunk anchoring fix **validated** (over-refusal 0.50→0.25 on a weak free model, likely better on a stronger one). Promoted into the repo at `hyperscholar/grounding/` (`verifier.py`, `eval_harness.py`) with a deterministic conformance suite at `hyperscholar/tests/test_grounding.py` (12 tests, no API). Run the labeled eval with `python -m hyperscholar.grounding.eval_harness --corpus christmas_carol --backend cograg_official`.

---

## 1. The two-layer picture

The testbed now separates two independently-swappable, independently-testable concerns:

```
          ┌──────────────────────────────────────────────┐
  query → │  GROUNDING LAYER  (strategy-agnostic)         │ → safe answer | decline
          │  anchored generation + verification + decide  │
          └───────────────────────┬──────────────────────┘
                                  │ needs only: raw source chunks for a query
          ┌───────────────────────┴──────────────────────┐
          │  RAG STRATEGY LAYER  (swappable)              │
          │  HyperRAG | HierarchicalRAG | Cog-RAG | …      │
          │  job: retrieval + (optional) draft answer     │
          └──────────────────────────────────────────────┘
```

**Key claim, defended in §5:** the grounding layer sits *above* the RAG strategy and depends on it for exactly one thing — the ability to retrieve **raw source chunks** for a query. Every strategy in the testbed already provides this, so the layer is backend-independent by construction.

---

## 2. The problem it solves

Observed repeatedly with Cog-RAG on the *A Christmas Carol* corpus (but not unique to it):

- Asked for a specific it didn't have (e.g. "what year was the Project Gutenberg trademark registered?"), the backend produced **confident, fabricated specifics** — a fake USPTO registration number and date, invented fresh on **every run** (we logged at least seven distinct fake registration numbers across runs). Live confabulation, not corpus content.
- Two root causes, both general to generative RAG:
  1. **Generation invited it.** The backend's answer prompt said, in effect, "incorporate relevant general knowledge" and "be detailed, comprehensive, and convincing" — directives that fight against "don't answer what isn't in the source." (This exact wording is byte-for-byte identical in HyperRAG's prompt too — so it is *not* a Cog-RAG-specific defect.)
  2. **Grounding was checked against the wrong thing, or not at all.** A large, professional-looking retrieved context makes a missing datum easy to paper over.

The grounding layer attacks both.

---

## 3. How the layer works

Four stages. The load-bearing principle runs through all of them: **ground against the raw source chunks — never against a derived structure** (entity/relationship CSVs, summary-tree nodes). Derived structures are lossy; a claim that is genuinely in the source can look "unsupported" against them, and vice-versa. This was the single most important lesson of the build (it caused false positives in early versions on *both* the verification and generation sides).

### 3.1 Retrieve raw chunks (ground truth)
Pull the top-k raw source passages for the query from the strategy's chunk store (`k=24` chosen because some answer chunks rank at 12–24). These serve as the **single ground truth** for both generation and verification.

### 3.2 Anchored generation
Instead of trusting the backend's own (drift-prone) synthesis, generate the answer with a strict prompt over the raw chunks:

> Use ONLY facts/names/dates/numbers/quotations that actually appear in the source. If the source does not contain the answer, reply exactly `NO_ANSWER_IN_SOURCE`. Do not pad with general knowledge or speculation. Stay close to the source's wording.

This directly neutralizes the "incorporate general knowledge / be convincing" behavior. `NO_ANSWER_IN_SOURCE` is a clean, structural decline signal (not an English phrase we later string-match).

### 3.3 Dual verification (against the raw chunks)
Every claim in the generated answer is checked two ways:

- **Deterministic specific-fact checker (no LLM, hard guarantee on the dangerous class).** Regex-extract the *hard specifics* — years, comma/long numbers, month-day dates, and double/curly-quoted quotations — and check each literally (numbers/dates) or by 6-gram overlap (quotations) against the raw chunks. A fabricated registration number, date, or quotation *literally is not in the source text* — that is decidable without an LLM, so this class is caught with no model fallibility. (Quote extraction is restricted to double/curly quotes; single-quote extraction was removed because it over-triggered on possessive/contraction apostrophes.)
- **Calibrated LLM verifier (for fuzzy/semantic claims).** Sorts the answer's *main* claims (not every clause — over-decomposition was a failure mode) into `supported` / `unsupported_fact` / `unsupported_interp`, judged **only** against the raw chunks, with faithful paraphrase counting as supported.

The deterministic checker also **re-filters the LLM's `supported` bucket**: if the LLM waved through a claim that carries a fabricated specific, the matcher demotes it. Belt-and-suspenders for the dangerous class.

### 3.4 Decision
- Grounded claims survive → **serve** them (plus reasonable interpretation), with fabricated specifics quarantined.
- Nothing grounded survives → **decline** ("not in your materials"). Crucially this is *derived* ("nothing grounded remained"), not a blanket refusal — grounded answers pass through intact.
- Policy hook: an *exam/strict* mode drops interpretation and unverifiable specifics; an *exploratory* mode can show quarantined material clearly labelled "not from your corpus — verify." (The decline / quarantine / serve split is what makes this configurable per delivery context.)

---

## 4. What we measured

Labeled set: 16 questions — 8 answerable (chunk-anchored) + 8 negatives (each verified absent from its source chunk by a fixed generator). Metrics: **over-refusal rate** (answerable wrongly declined) and **fabrication-leak rate** (fabrication served).

| Configuration | model | fabrication-leak | over-refusal | note |
|---|---|---:|---:|---|
| Backend answer + post-hoc verify | deepseek-chat | ~0.12 (1/8 real; 0.625 by a too-crude first metric) | 0.25 | verification alone reduces but doesn't prevent |
| **Anchored to entity CSV** (leak-focused) | deepseek-chat | **0.00** | 0.50 | fabrication eliminated; over-refused (see below) |
| **Anchored to raw chunks, k=24** (the fix) | glm-4-flash | **0.00** | **0.25** | validated; best config — Q6/Q7/Q8 recovered |

*The last row was validated on Zhipu `glm-4-flash` (a lightweight free model, weaker than the deepseek-chat the prompts were tuned on) — so its 0.25 over-refusal is likely pessimistic. Of the two residual over-refusals, one (Fezziwig) has its answer provably present in the raw chunks and is attributable to model weakness, not the layer; a stronger-model re-run should improve it.*

### Cross-corpus check — scale + non-memorized domain

Re-ran on **agriculture_20** (20 real technical documents — beekeeping, soil, etc. — indexed chunks-only to **1,474 chunks**, ~9x christmas_carol; a domain the model is far less likely to have memorized). 10 answerable + 10 negative questions, glm-4-flash.

| corpus | chunks | over-refusal | decision-level leak | **content-level fabrication leak** |
|---|---:|---:|---:|---:|
| christmas_carol | 169 | 0.25 | 0.0 | **0 / 8** |
| agriculture_20 | 1,474 | **0.0** | 0.2 | **0 / 10** |

The headline: **zero fabricated specifics reached output on either corpus** — the grounding guarantee held at scale on non-memorized content, and over-refusal was actually *lower* on the larger corpus (all 10 answerable served). The two agriculture "leaks" were **not fabrications** on inspection: one was a mislabeled negative (the asked detail was absent from its source chunk but groundable elsewhere in the corpus, and k=24 retrieval surfaced it), the other served grounded but off-target content without fabricating the asked specific. Both are **eval-harness precision issues**, not grounding failures, and they empirically motivate two §7 to-dos: (1) verify negative-question absence **corpus-wide** (against retrieved top-k), not just the source chunk; (2) score **content-level** fabrication (did a flagged specific survive into the served text?) rather than decision-level serve/decline.

Practical note: because the grounding layer uses only chunk-vector retrieval, a corpus can be indexed **chunks-only** (embedding, no LLM entity extraction) for this eval — agriculture_20 indexed in ~20 min of local GPU time with **zero API calls**, vs. hours of entity-extraction LLM calls for a full Cog-RAG build.

Two honest points:
- **Fabrication prevention is real and measured** — anchored generation drove the leak rate to **0.0** (all 8 negatives correctly declined). That is the headline result.
- The 0.50 over-refusal in the middle row was a **bug, not a wall**: generation had been anchored to Cog-RAG's *lossy entity CSV* rather than raw chunks. A no-LLM diagnostic proved the answers to all four over-refused questions were present in the raw chunks (k=12 for three, k=24 for the fourth). The fix (anchor generation to raw chunks, k=24) is in `grounding.py`; it needs one re-run to confirm the over-refusal drops while the leak rate stays at zero.

The over-refusal ↔ leak relationship is the fundamental **precision/recall dial of grounding**, tunable via prompt strictness, `k`, and the decline threshold — inherent to any grounded system, not specific to Cog-RAG.

---

## 5. Does it apply over HyperRAG and other strategies? — Yes, by construction

The layer needs exactly **one capability** from a strategy: *given a query, return the raw source chunks relevant to it.* It never touches the strategy's internal structure. So applicability reduces to "does the strategy expose raw chunks?" — and every strategy in the testbed does, because the locked architecture keeps **content (documents/chunks/embeddings) shared and identical across backends** while only the structural overlay differs.

Concretely:

| Strategy | Raw-chunk access | Already exposed today? |
|---|---|---|
| **Cog-RAG** | `chunks_vdb.query` → `text_chunks.get_by_ids` | Yes (used by the prototype) |
| **HyperRAG** | retrieval returns `text_units`; chunk store present | Yes — `eval/provenance.py::hyperrag_query_with_provenance` already extracts them |
| **HierarchicalRAG** | leaf chunks under tree nodes (`chunks_accessed`) | Yes — `eval/provenance.py::hierarchical_query_with_provenance` already records them |

The eval framework **already pulls raw retrieved chunks out of every backend** for provenance capture. That is precisely the input the grounding layer needs — so wiring the layer over HyperRAG or HierarchicalRAG is not new plumbing, it's reusing an interface that already exists.

Two modes of operation, both strategy-agnostic:
1. **Anchored (recommended):** ignore the backend's synthesized answer, retrieve its raw chunks, and generate + verify ourselves. Uniform behavior across all strategies; the backend is used purely as a retriever.
2. **Verify-only (lighter):** keep the backend's own answer, verify it against its raw chunks, quarantine/decline as needed. Cheaper, but inherits the backend's drift — best where the backend's generation is already conservative.

**Implication for strategy comparison:** because grounding is a uniform layer above the `RAGBackend` interface, you can run the *same* grounding layer over every strategy and compare them on a level field — "which retriever surfaces the chunks that let the grounded answer be served, without over-refusing?" That is a cleaner, safety-relevant comparison metric than the raw LLM-judge scores, and it slots directly into the existing `hyperscholar/eval/` harness.

A genuine finding from this exercise, worth stating: on a famous public-domain text, Cog-RAG's apparent answer quality was substantially its LLM's *memorized* knowledge, not its retrieval — anchoring exposed that, and showed its lossy entity extraction actively *hurts* faithful grounding. Running the grounding layer across all strategies will tell us, per strategy, how much of its answer quality is retrieval vs. parametric memory. That is exactly the kind of thing the two-layer split now lets us measure.

---

## 6. Limits (honest)

- **No absolute 0%.** "Prevent hallucination" realistically means *drive confident fabrication to near-zero and make the residual detectable or declinable* — not mathematical elimination. The deterministic checker gives a **hard** guarantee only on the classes it extracts (years, numbers, dates, double-quoted quotations); soft/vague claims ("late 1810s", "in his late twenties") depend on the LLM verifier and can still slip. Extending the deterministic extractor (decade-with-suffix, spelled-out quantities) closes known gaps incrementally.
- **The LLM verifier is fallible.** It is strong risk-reduction, not a proof. Hardening options: a second independent verifier, or an NLI/exact-match pass for the quotation sub-case.
- **Cost.** Anchored mode adds ~2 LLM calls per query (generate + verify) beyond retrieval. Gate it: run the cheap deterministic pre-check first and only invoke the LLM verifier when specifics are present; cache by `(query, answer)` hash.
- **Retrieval is the ceiling.** Grounding can only serve what retrieval surfaces. Over-refusal that traces to a retrieval miss is a *retriever* problem, not a grounding problem — which is exactly why keeping the two layers separate and measurable matters.

---

## 7. How we test and improve it over time

The layer has a standing evaluation harness (`eval_harness.py` / `harness2.py`), parallel to the RAG-comparison harness:

- **Inputs:** a labeled set — answerable questions (correct behavior = serve) + verified-absent negatives (correct behavior = decline). The negative generator now self-verifies absence (fixed in `hyperscholar/eval/question_generator.py`); a remaining improvement is to verify absence corpus-wide (against retrieved top-k) rather than only the source chunk, to eliminate mislabeled negatives.
- **Metrics:** over-refusal rate, fabrication-leak rate (content-aware: a serve that says "not in the source" is an honest non-answer, not a leak), and serve-recall.
- **Regression use:** run the harness on every change to the anchored prompt, verifier calibration, `k`, or deterministic extractor — and run it **per RAG strategy** to compare retrievers on grounded-answer quality.

Suggested next steps, in order: (1) restore LLM access and re-validate the raw-chunk fix; (2) promote the scratchpad prototype into `hyperscholar/grounding/` with the conformance test; (3) run the harness over HyperRAG and HierarchicalRAG to confirm cross-strategy behavior; (4) extend the deterministic extractor's coverage; (5) add the cost-gating pre-check.

---

## 8. One-paragraph summary

The grounding layer prevents hallucination by refusing to trust the RAG backend's synthesized answer: it retrieves the **raw source chunks**, generates an answer **anchored strictly to them**, and then **verifies every claim against those same chunks** — deterministically for hard specifics (dates, numbers, quotations, which either literally appear in the source or don't), and with a calibrated LLM check for fuzzy claims — serving only what survives and cleanly declining when nothing does. Because it depends on the backend for **only** raw-chunk retrieval (which every strategy already exposes, and which the eval framework already extracts), it is **strategy-agnostic**: the same layer works unchanged over HyperRAG, HierarchicalRAG, Cog-RAG, and anything added later, and lets you compare those strategies on a safety-relevant axis. Fabrication prevention is measured (leak rate driven to 0.0); the remaining work is tuning the precision/recall dial so answerable questions stay answered — a bug we've diagnosed and fixed, pending one re-run.
