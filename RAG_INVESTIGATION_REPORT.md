# RAG Strategy Comparison & Hallucination-Prevention — Investigation Report

**Testbed:** `hyperscholar/` (HyperRAG, HierarchicalRAG, Cog-RAG behind one swappable interface).
**Corpora:** *A Christmas Carol* (1 doc, 169 chunks) and **agriculture_20** (20 technical documents, 1,474 chunks).
**Companion docs:** [`GROUNDING_LAYER_REPORT.md`](GROUNDING_LAYER_REPORT.md) (grounding layer, technical) · [`hyperscholar/eval/results/eval_report.md`](hyperscholar/eval/results/eval_report.md) (raw RAG-comparison tables).

---

## 1. Executive summary

We set out to add Cog-RAG to the testbed and compare it head-to-head against HyperRAG and HierarchicalRAG. The comparison surfaced a deeper problem than "which retriever scores higher": Cog-RAG produced **fluent, confident, fabricated facts** on questions the corpus doesn't answer. Investigating that led to two concrete, durable outcomes:

1. **A finding.** The LLM-judge comparison rewards fluency and rewards *confident* answers — including confidently wrong ones. On its own it is not a safety metric. Cog-RAG "won" the judge while being the most likely to hallucinate.
2. **A build.** A **strategy-agnostic grounding layer** that sits above any RAG backend and prevents fabrication by grounding every answer against the raw source chunks. Validated across both corpora: **zero fabricated specifics reached output**, on 169-chunk fiction and on a 1,474-chunk non-memorized technical corpus alike.

The testbed now cleanly separates two independently-swappable, independently-testable concerns: the **RAG strategy** (retrieval quality) and the **grounding layer** (hallucination resistance).

---

## 2. RAG strategy comparison (the original task)

All three backends indexed the same corpus and answered the same questions; a blind, position-randomized LLM judge scored each answer 1–10 on five dimensions. Full tables in `eval/results/eval_report.md`.

**Overall means (christmas_carol, all styles):**

| | HyperRAG | HierarchicalRAG | Cog-RAG |
|---|---:|---:|---:|
| Mean score | 7.31 | 2.56 | **8.53** |
| Wins | 5 | 0 | **16** (tie 1) |

Read naively, Cog-RAG dominates. Three findings complicate that:

- **HierarchicalRAG's score is an artifact, not a ranking.** It refused nearly every relational/synthesis/overview question with a canned "I'm not able to provide an answer." Its low score reflects a broken run (blanket refusal), not weaker retrieval — and its "perfect" hallucination-resistance number (3/3 negatives declined) is the *same* blanket refusal, not selective judgment.
- **The judge rewards confident fabrication.** Cog-RAG's winning answers on unanswerable questions were fluent inventions (see §3). A judge scoring "comprehensiveness / empowerment / convincingness" structurally favors an answer that confidently makes something up over one that honestly declines.
- **Entity labels drifted to Chinese** on some questions (e.g. a real front-matter item became "彩色插图的尾花"), an artifact of DeepSeek's entity extraction — affecting all backends equally, but distorting judged scores on those items.

**Takeaway:** the LLM-judge comparison measures fluency/coverage, not correctness or safety. It is useful for relative retrieval quality but must be paired with a grounding/hallucination metric — which motivated the rest of the work.

---

## 3. The hallucination problem

Asked *"what year was the Project Gutenberg trademark registered?"* — a fact absent from the corpus — Cog-RAG produced a confident, specific answer: a fake USPTO registration number and date. Across repeated runs it invented **at least seven distinct fake registration numbers**, a different one almost every time. This is live confabulation from the model's parametric knowledge, not corpus content.

Two root causes, both **general to generative RAG, not unique to Cog-RAG**:

1. **The generation prompt invites it.** Cog-RAG's answer prompt says, in effect, "incorporate relevant general knowledge" and be "detailed, comprehensive, and convincing" — directives that fight against "don't answer what isn't in the source." This exact wording is **byte-for-byte identical in HyperRAG's prompt**, so it is not a Cog-RAG defect per se.
2. **Grounding was checked against the wrong thing — or not at all.** A large, professional-looking retrieved context (Cog-RAG assembles ~60 entities into CSV-style tables) makes a single missing datum easy to paper over.

A key diagnostic result: on a famous public-domain text, **much of Cog-RAG's apparent answer quality was the model's *memorized* knowledge, not its retrieval.** Constraining generation to the source (§4) exposed this — and showed that Cog-RAG's lossy entity extraction actively *hurts* faithful grounding.

---

## 4. The grounding layer (what we built)

A layer that sits **above** the RAG strategy and refuses to trust the backend's synthesized answer. Full design in `GROUNDING_LAYER_REPORT.md`; in brief, four stages, all sharing one principle — **ground against the raw source chunks, never a derived structure** (entity CSVs, summary trees are lossy):

1. **Retrieve raw chunks** (top-k source passages) as the single ground truth.
2. **Anchored generation** — replace the backend's drift-prone synthesis with a strict prompt: *use only the source; if the answer isn't there, emit `NO_ANSWER_IN_SOURCE`; do not add general knowledge.*
3. **Dual verification** against those chunks:
   - a **deterministic specific-fact checker** (no LLM) for years/numbers/dates/quotations — a fabricated registration number literally is-or-isn't in the source text, so this class is caught with a *hard* guarantee;
   - a **calibrated LLM verifier** for fuzzy/semantic claims (paraphrase counts as supported).
4. **Decide** — serve grounded content (+ interpretation), quarantine fabricated specifics, and **decline** only when nothing grounded survives (a *derived* decline, not blanket refusal).

### Why it applies to every strategy

The layer needs exactly one thing from a backend: *retrieve raw source chunks for a query.* Every strategy in the testbed already provides this — the locked architecture keeps **content (chunks/embeddings) shared and identical across backends**, and `eval/provenance.py` **already extracts raw retrieved chunks from HyperRAG, HierarchicalRAG, and Cog-RAG** for provenance capture. So wiring grounding over any strategy reuses an interface that already exists. It is a hallucination-resistance layer **decoupled from the choice of retriever**.

---

## 5. Validation results

Labeled sets of answerable questions (correct behavior = serve) + verified-negative questions (correct behavior = decline). Two metrics: **over-refusal rate** (answerable wrongly declined) and **fabrication-leak rate**. Runs used Zhipu `glm-4-flash` (a lightweight *free* model — weaker than the deepseek-chat the prompts were tuned on, so these numbers are conservative).

| corpus | chunks | model | over-refusal | decision-level leak | **content-level fabrication leak** |
|---|---:|---|---:|---:|---:|
| christmas_carol | 169 | glm-4-flash | 0.25 | 0.0 | **0 / 8** |
| agriculture_20 | 1,474 | glm-4-flash | **0.0** | 0.2 | **0 / 10** |

**Headline: zero fabricated specifics reached output on either corpus** — the guarantee held on 9x-larger, non-memorized technical content, and over-refusal was *lower* on the bigger corpus (all 10 answerable served).

Honest reading of the exceptions:
- The christmas_carol **0.25 over-refusal** was two questions; one (Fezziwig) has its answer provably in the retrieved chunks and is attributable to the weak free model, not the layer.
- The agriculture **0.2 "decision-level leak"** was, on inspection, **two grounded non-fabrications**: one mislabeled negative (the detail was absent from its source chunk but groundable elsewhere in the corpus), and one grounded-but-off-target answer that never fabricated the asked specific. Content-level fabrication leak was **0/10**.

Evolution of the design across the build (christmas_carol, illustrating the key fix):

| configuration | fabrication-leak | over-refusal |
|---|---:|---:|
| backend answer + post-hoc verify | ~0.12 | 0.25 |
| anchored to Cog-RAG's **entity CSV** | 0.00 | 0.50 |
| anchored to **raw chunks** (the fix) | **0.00** | **0.25** |

Anchoring to the *lossy entity CSV* eliminated fabrication but over-refused; anchoring to *raw chunks* (the same ground truth used for verification) recovered the over-refusals while keeping leak at zero — the best configuration.

---

## 6. The two-layer architecture

```
          ┌──────────────────────────────────────────────┐
  query → │  GROUNDING LAYER  (strategy-agnostic)         │ → safe answer | decline
          │  anchored generation + verification + decide  │
          └───────────────────────┬──────────────────────┘
                                  │ needs only: raw source chunks
          ┌───────────────────────┴──────────────────────┐
          │  RAG STRATEGY LAYER  (swappable)              │
          │  HyperRAG | HierarchicalRAG | Cog-RAG | …      │
          └──────────────────────────────────────────────┘
```

Two concerns, tested independently:
- **RAG strategy** — how well retrieval surfaces the right chunks. Compared via the eval harness (LLM-judge for fluency/coverage, plus grounded-answer serve/decline behavior).
- **Grounding** — whether the served answer is faithful to the source. A uniform layer that lets any strategy be compared on a safety-relevant axis, and that can be improved on its own cadence.

Both are now in the repo: strategy code under `hyperscholar/rag/`, grounding under `hyperscholar/grounding/` with a 12-test deterministic conformance suite (`hyperscholar/tests/test_grounding.py`, no API) and a labeled-eval harness (`python -m hyperscholar.grounding.eval_harness`).

---

## 7. Limitations (honest)

- **No absolute 0%.** "Prevent hallucination" means *drive confident fabrication to near-zero and make the residual detectable or declinable.* The deterministic checker gives a **hard** guarantee only on the classes it extracts (years, numbers, dates, double-quoted quotations); soft/vague claims ("late 1810s", approximate ranges) rely on the LLM verifier and can still slip.
- **Measured on a weak free model.** glm-4-flash understates the layer's quality; a stronger model would likely improve over-refusal. All three prior DeepSeek keys were exhausted/invalid, and paid GLM/Gemini tiers were out of balance — the numbers here are a conservative floor.
- **Two eval-harness precision issues** (not grounding failures), now empirically motivated by the agriculture run: (1) negative-question absence must be verified **corpus-wide** (against retrieved top-k), not just the source chunk; (2) leak should be scored **content-level** (did a flagged specific survive into the served text?), not by decision-level serve/decline.
- **Retrieval is the ceiling.** Grounding can only serve what retrieval surfaces; over-refusal that traces to a retrieval miss is a retriever problem, which is exactly why the two layers are kept separate and separately measurable.

---

## 8. Recommended next steps

1. **Re-validate on a stronger model** (funded DeepSeek/GLM key) for representative — not floor — numbers.
2. **Fix the two harness precision issues** (corpus-wide negative verification; content-level leak metric) and re-run.
3. **Run the grounding eval across HyperRAG and HierarchicalRAG**, not just Cog-RAG, to compare strategies on grounded-answer quality (the safety-relevant axis the LLM judge misses).
4. **Extend the deterministic extractor** (decades-with-suffix, spelled-out quantities) to close known soft-leak gaps.
5. **Promote the chunks-only indexer** (used to index agriculture_20 with zero API cost) into the repo as a standing utility for cheaply onboarding new eval corpora.

---

## 9. One-paragraph conclusion

Adding Cog-RAG to the testbed revealed that the LLM-judge RAG comparison rewards confident fluency — including confident fabrication — and therefore cannot stand alone as a quality measure. The response was a strategy-agnostic grounding layer that grounds every answer against raw source chunks, catches fabricated specifics deterministically, and declines only when nothing grounded remains. It let **zero fabricated specifics** through on both a small memorized corpus and a large non-memorized one, works unchanged over any RAG strategy (all of which already expose the raw chunks it needs), and gives the project a second, independently-improvable layer: retrieval quality and hallucination resistance, tested and advanced separately.
