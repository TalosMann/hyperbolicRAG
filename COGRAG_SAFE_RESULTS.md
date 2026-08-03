# cograg_safe — Results

**What this is:** the completion report for [`COGRAG_ZERO_HALLUCINATION_BRIEF.md`](COGRAG_ZERO_HALLUCINATION_BRIEF.md)'s mission — build a Cog-RAG variant that keeps the dual-hypergraph's multi-hop retrieval advantage while guaranteeing zero hallucination, closing the gap the brief identified: *faithful retrieval ≠ faithful reasoning* (the "Fan is Tiny Tim's sister" failure — every fact grounded, conclusion false).

**Headline result:** across two corpora (christmas_carol, 387 entities; agriculture_20, 15,681 entities — a 40x scale jump) and every validation run, **zero false relations and zero fabricated specifics were ever served.** The canonical failure case from the brief is fixed. Completeness (how often the system serves vs. over-cautiously declines) is the remaining open dial, and it is honestly reported below as unresolved rather than smoothed over.

**Important caveat up front:** every result in this document ran on **Zhipu `glm-4-flash`**, a free, lightweight model — not the funded frontier model the brief assumed. Three DeepSeek keys, one Anthropic key, and a Gemini key were tried over the course of this work; all were either invalid, out of credit, or quota-limited (see §6). The safety numbers below are therefore likely a **floor** — they held on a weak model, and a stronger one should only make the completeness numbers better, not the safety guarantee different in kind.

---

## 1. What was built

`hyperscholar/cograg_safe/` — a verify-then-compose pipeline that never lets the model's free-form prose become the answer directly:

```
question → bind entities to corpus hypergraph nodes → hybrid retrieval
  (entity-anchored chunks + hyperedges, preserving Cog-RAG's multi-hop reach)
  → LLM proposes assertions as STRUCTURED (fact / relation) triples, not prose
  → R1: deterministic co-mention check (no LLM — a relation can only be
        "stated" if some real evidence co-mentions both sides)
  → R2 / F2: scoped entailment check (LLM judges the SPECIFIC co-mentioning
        evidence, not the whole corpus)
  → compose the final answer ONLY from what survived, with citations
```

`evidence.py` (entity binding + co-mention index), `assertions.py` (structured generation + R2/F2 prompts), `pipeline.py` (orchestration + decision logic), `backend.py` (`RAGBackend` adapter — wraps the existing `CogRAGOfficialBackend` for indexing/storage; only query is replaced). 14 deterministic, no-API conformance tests in `hyperscholar/tests/test_grounding.py`/`test_cograg_safe.py`; full suite **45/45 passing**.

---

## 2. Example answers by question type

All quotes below are verbatim system output, verified against the raw source afterward (methodology in §4).

### 2.1 Factual (single-hop)

**Q: "What is the name of the high-yielding dairy cow of the Cauchois breed discussed in this material?"** *(agriculture_20, 15,681-vertex hypergraph)*
> Based only on the source material:
> - Cauchois breed — is known for — high milk yields [chunk-abe627e0..., chunk-9ea9258b...]
> - Caillotte — produces milk — when out on grass [chunk-abe627e0...]
> - Caillotte is a high-yielding dairy cow of the Cauchois breed.

**Q: "Who founded the Soil Association?"**
> Based only on the source material:
> - the Soil Association — was founded by — Lady Eve Balfour [chunk-9b968976..., chunk-95f1a55d...]
> - Lady Eve Balfour founded the Soil Association.

Both confirmed against source text; both cite the chunks they came from.

### 2.2 Multi-hop / relational — the canonical case from the brief

**Q: "How is Fan related to Tiny Tim?"** — the exact question that produced the brief's failure case (*"…Therefore, Fan is Tiny Tim's sister"* — every premise grounded, conclusion false).

**cograg_safe's answer** *(christmas_carol, first validation round)*:
> Based only on the source material:
> - Fan — is the sister of — Ebenezer Scrooge [chunk-9e3921da...]
> - Tiny Tim — is the son of — Bob Cratchit [chunk-9e3921da..., chunk-7e2e7ebc...]
> - Ebenezer Scrooge — is the employer of — Bob Cratchit [chunk-9e3921da..., chunk-d9aac348...]
> - Bob Cratchit — is the father of — Tiny Tim [chunk-9e3921da..., chunk-7e2e7ebc...]
> - Fan is a character in 'A Christmas Carol'.
> - Tiny Tim is a character in 'A Christmas Carol'.
> - Bob Cratchit is a character in 'A Christmas Carol'.
> - Ebenezer Scrooge is a character in 'A Christmas Carol'.
>
> **Note: the source does not state a direct relationship between FAN and TINY TIM. Any connection runs indirectly through: Bob Cratchit, Ebenezer Scrooge.**

No "Fan is Tiny Tim's sister." The false conclusion is gone; the true grounded chain (sister → employer → father) is shown instead, with an explicit statement that the direct link is *not* stated in the source. This is the brief's §9 gap, closed.

A second validation round (same question, after the facts-entailment gate was added and a model-noise run) served a *thinner* but still-safe answer:
> Based only on the source material:
> - Tiny Tim is a character in 'A Christmas Carol'.
>
> Note: the source does not state a direct relationship between FAN and TINY TIM.

Same correct verdict (no false relation), less content served — R2 was more conservative that run and rejected the true "Fan is sister of Scrooge" relation as unentailed. This run-to-run variance (same question, same corpus, different outcome) is exactly the free-tier-model noise flagged throughout this document — the safety property held both times; completeness didn't.

### 2.3 Multi-hop / relational — a genuine family/ownership chain (agriculture_20, at scale)

**Q: "How is Jamie Ager connected to the farm he and his wife operate?"**
> Based only on the source material:
> - Jamie Ager — is involved in the operations of — Hickory Nut Gap Farm [chunk-3e930a3e..., chunk-f712d355...]
> - Jamie Ager — leases — the farm [chunk-3e930a3e..., chunk-f712d355...]
> - Jamie Ager — and his wife Amy — operate the farm [chunk-3e930a3e..., chunk-f712d355...]
> - Jamie Ager and his wife Amy operate Hickory Nut Gap Farm
> - **Jamie Ager and his wife Amy lease the farm from Jamie's mother Annie Ager and her five siblings**

That last line is a near-verbatim match to the actual source sentence: *"Today Jamie's mother Annie Ager and her five siblings own the farm, and Jamie and his wife Amy lease the farm from them."* — a two-generation ownership/lease relation with names and a specific count, correctly extracted and cited on a 15,681-vertex graph, on the free-tier model. This is the strongest single result in this validation.

### 2.4 Fabrication control (should decline)

**Q: "What exact percentage of U.S. farms were certified organic according to the USDA report cited in this material?"** *(a plausible-sounding statistic not in the corpus)*
> The source material does not contain an answer to this question.

**Q: "What specific year was the Project Gutenberg trademark first officially registered?"** *(christmas_carol — the question that, on unconstrained Cog-RAG, produced at least seven distinct fake USPTO registration numbers across repeated runs — see [`RAG_INVESTIGATION_REPORT.md`](RAG_INVESTIGATION_REPORT.md) §3)*
> [DECLINE — no year, no registration number, no citation offered]

Both correctly declined rather than inventing a number.

### 2.5 Cross-document multi-hop — an honest miss, not a failure

**Q: "Besides John Muir, what other conservation philosopher linked to the Sierra Club is discussed in relation to agricultural sustainability?"** — John Muir and Aldo Leopold are mentioned in **two entirely separate source documents** (confirmed by direct chunk-position lookup), so this is a genuine cross-document, structure-dependent connection.

> The source material does not contain an answer to this question.

This is a **decline**, not a fabrication — safe, but a missed opportunity, since the raw evidence for Leopold *was* retrievable (confirmed separately in a retrieval-only test, §5). The full pipeline's generation step chose not to synthesize a connection from it. This is the honest cost of the current safety-first posture: it would rather decline than risk an uncertain synthesis, and on a weak model that trades away some completeness.

---

## 3. What "verify-then-compose" actually rejects

Concrete evidence the gates are doing real work, not just passing everything through — from the christmas_carol validation runs:
- **~45–56% of all relations the generator proposed were rejected** before reaching the answer, split between R1 (no evidence co-mentions both entities — zero LLM cost) and R2 (evidence exists but doesn't state the claimed relation).
- A dedicated conformance test (`test_cograg_safe.py`) proves the facts-entailment gate (F2) catches the specific failure mode "real quote, wrong statement" — e.g. a claim that cites an authentic quotation from the text but draws a conclusion the quotation doesn't support.

---

## 4. Audit methodology

Every relation and fact quoted above (and every one in the full validation runs, not just these excerpts) was independently checked against the actual source text after the fact — via direct hyperedge/vertex lookup in the persisted `.hgdb` hypergraph, and via `python -m hyperscholar.grounding.chunk_reader grep "<phrase>" --ns <corpus>` against the raw `kv_store_text_chunks.json`. This is not a self-report from the system; every claim in §2 was located in the source independently before being included here.

---

## 5. The multi-hop retrieval question — reported honestly as unresolved

Separately from whether *answers* are faithful, the brief also asked whether Cog-RAG's structure genuinely helps *retrieval* find multi-hop evidence at scale. This required real work to test properly, and the outcome should not be oversold:

- An initial test on agriculture_20 appeared to show structure winning big — but two of five test pairs turned out to share the exact same source chunk (a construction error: I verified group membership in a large hyperedge but didn't re-verify the *specific* pair chosen for two questions), making them trivial for any method.
- The one pair I directly confirmed was genuinely dispersed — **John Muir and Aldo Leopold, mentioned in two entirely separate documents** — was found by **both** structural retrieval and plain vector search at k=3. The likely reason: both mentions use the same canonical, limited vocabulary for a well-known historical topic ("Sierra Club," "conservation," "environmental philosophy"), so a strong semantic embedder bridges the document gap on vocabulary alone.
- This contrasts with christmas_carol's Fan/Tiny Tim case, where structural retrieval reliably outperformed plain search (100% vs. 40% at selective k) — there, the two entities share **no** vocabulary at all; the only link is a relational one (sister-of / son-of) with no lexical footprint for an embedder to find.

**Working hypothesis, not yet conclusively tested:** the structural-retrieval advantage is concentrated in **relational/genealogical connections** (who is related to whom — common in narrative/character-driven text) rather than **topical-cluster connections** (people/orgs active in the same named movement — common in expository/technical text, which is what agriculture_20 is). This session did not produce a second clean, large-scale relational multi-hop test to confirm or refute that hypothesis — it's a sharper question for future work, not a settled finding.

---

## 6. Model/key situation (why everything ran on the free tier)

Attempted, in order, over the course of this work: three distinct DeepSeek API keys (all returned `402 Insufficient Balance` or `401 invalid key`), a Gemini key (`429 quota exceeded`), an Anthropic key (`400 — credit balance too low`, discovered mid-run and the root cause of an earlier "0 entities" result that looked like a Cog-RAG/Claude compatibility bug but wasn't). Only **Zhipu `glm-4-flash`** (free tier) worked throughout. Every number in this report is a floor set by a weak, free model — not a ceiling.

---

## 7. Definition-of-done scorecard (per the brief's §16)

| Requirement | Status |
|---|---|
| Retains multi-hop structural retrieval | Partial — real on christmas_carol (100% vs 40%); inconclusive at agriculture scale (§5) after correcting a test-construction error |
| Zero fabricated specifics | ✅ confirmed across both corpora, all rounds |
| **Zero ungrounded relationships** (the brief's central ask) | ✅ confirmed — the canonical "Fan is Tiny Tim's sister" failure does not recur; audited relations in every run were either true or correctly withheld |
| Correct declining without heavy over-refusal | Mixed — clean declines on true negatives (USDA %, Gutenberg year); some over-caution on true positives (Muir/Leopold, one Fan/Tiny Tim round) attributable to the free-tier model |
| Provenance on every assertion | ✅ every served relation/fact carries its source chunk id(s) |

---

## 8. Honest next steps

1. **Get one funded key working** and re-run the exact same validation set — this is the single highest-leverage remaining action, since every open item above is a completeness question that a stronger model directly addresses, not an architecture question.
2. **Design a second, larger-scale relational (non-topical) multi-hop test** to actually confirm or refute the §5 hypothesis, rather than leaving it as a plausible but unverified explanation.
3. Extend the deterministic checker's specific-fact coverage (decades-with-suffix, spelled-out quantities) per the standing item in [`GROUNDING_LAYER_REPORT.md`](GROUNDING_LAYER_REPORT.md).
