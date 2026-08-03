r"""Structured assertion generation + Gate R2 (scoped entailment).

The generator does NOT write prose. It outputs relation assertions as parseable
triples, each citing evidence, plus atomic facts. Prose is composed later, ONLY
from assertions that survive both gates — so unverified relations cannot reach
the user (verify-then-compose).

Also: intent classification + the general-knowledge generation prompt for
SYNTHESIS-mode questions (comparisons/analysis reaching outside the corpus).
These never touch R1/R2 — that machinery only ever checks claims that purport
to be FROM the corpus. General-knowledge content is instead kept in its own,
clearly labeled section (see pipeline.compose_synthesis) so a reader always
knows which half of the answer is corpus-verified and which is not.
"""
from __future__ import annotations

import json
import re

ASSERTION_PROMPT = """You are answering a question using ONLY the SOURCE EVIDENCE below.
Do NOT write an essay. Output a JSON object with your findings as structured claims.

Rules:
- Use ONLY the SOURCE EVIDENCE. Never use outside knowledge. Entity names refer to
  the entities IN THIS SOURCE, not to any real-world namesakes.
- "relations": each item asserts ONE relationship between two named entities that the
  evidence STATES. Give the two entity names, the relation phrase, and the sentence
  from the evidence that states it. Do not infer relations the evidence does not state.
- If the question asks how two entities are connected and the evidence states no direct
  relation, DO NOT invent one — instead provide the chain of individually-stated
  relations that indirectly connects them (each hop separately evidenced).
- "facts": atomic non-relational facts needed for the answer, each with its evidence sentence.
- If the evidence contains nothing relevant, output {{"relations": [], "facts": []}}.

QUESTION: {question}

SOURCE EVIDENCE:
{evidence}

Output ONLY JSON in exactly this form:
{{
  "relations": [
    {{"a": "<entity name>", "relation": "<short relation phrase>", "b": "<entity name>",
      "evidence_quote": "<the sentence from the source that states this>"}}
  ],
  "facts": [
    {{"statement": "<atomic fact>", "evidence_quote": "<source sentence>"}}
  ]
}}"""

R2_PROMPT = """You are a strict relation verifier. Below are SOURCE EXCERPTS and ONE claimed
relationship. Decide whether the excerpts STATE or DIRECTLY ENTAIL the relationship.

- Judge ONLY from the excerpts. Outside knowledge is forbidden.
- "Directly entail" means a reader of these excerpts alone would agree the relationship
  is stated — not merely that both entities appear, and not a plausible guess.
- If the excerpts state a DIFFERENT relationship between these entities, answer NO.

SOURCE EXCERPTS:
{evidence}

CLAIMED RELATIONSHIP: {a} — {relation} — {b}

Answer with exactly one word: YES or NO."""


FACT_R2_PROMPT = """You are a strict fact verifier. Below are SOURCE EXCERPTS and ONE claimed
statement. Decide whether the excerpts STATE or DIRECTLY ENTAIL the statement.

- Judge ONLY from the excerpts. Outside knowledge is forbidden.
- A quote appearing in the excerpts does NOT by itself make the statement true — the
  statement's actual meaning must be what the excerpts say.
- If the excerpts say something DIFFERENT from the statement, answer NO.

SOURCE EXCERPTS:
{evidence}

CLAIMED STATEMENT: {statement}

Answer with exactly one word: YES or NO."""


INTENT_PROMPT = """Classify the QUESTION below as exactly one category.

- CORPUS: the question asks what a specific text/corpus itself says, states, or
  shows — facts, events, or relationships that exist WITHIN that corpus.
  Examples: "What happened when X did Y?", "How is A related to B in this story?",
  "What does the text say about Z?"
- SYNTHESIS: the question asks for comparison, analysis, or connections to
  things OUTSIDE the corpus — other works, other authors, broader concepts,
  real-world parallels — or interpretation that goes beyond what the text states.
  Examples: "How does this compare to...", "What does this remind you of...",
  "Why might an author choose...", "How does X mirror Y in a different work?"

If in doubt, prefer CORPUS.

QUESTION: {question}

Respond with exactly one word: CORPUS or SYNTHESIS."""

SYNTHESIS_PROMPT = """A student has asked a comparative/analytical question. You have
already been given VERIFIED FACTS from their course text below (already shown to the
student separately — do not repeat them verbatim). Your job here is ONLY to address
the part of the question that requires knowledge beyond this text — comparisons to
other works, broader literary/historical context, or general analysis.

VERIFIED FACTS FROM THE TEXT (context only, already shown to the student):
{corpus_summary}

STUDENT'S FULL QUESTION: {question}

Write a clear, direct answer to the general-knowledge / comparative part of this
question, drawing on your own knowledge. Do NOT claim any of this content comes from
the student's course text — it will be labeled separately as general knowledge, not
verified against any source. Be accurate and substantive; this is a legitimate
educational synthesis task, not something to hedge excessively about."""


async def classify_intent(llm, question: str) -> str:
    """'CORPUS' (strict pipeline, unchanged) or 'SYNTHESIS' (comparative/
    analytical — permits a separate, clearly labeled general-knowledge section).
    Defaults to CORPUS on any ambiguous/malformed reply — the strict path is
    the safe default when classification is uncertain."""
    reply = (await llm(INTENT_PROMPT.format(question=question))).strip().upper()
    return "SYNTHESIS" if reply.startswith("SYNTH") else "CORPUS"


def parse_assertions(raw: str) -> dict:
    """Parse the generator's JSON; tolerate code fences. Returns
    {"relations": [...], "facts": [...]} with malformed items dropped."""
    if not raw:
        return {"relations": [], "facts": []}
    m = re.search(r"\{.*\}", raw.replace("```json", "").replace("```", ""), re.DOTALL)
    if not m:
        return {"relations": [], "facts": []}
    try:
        obj = json.loads(m.group(0))
    except json.JSONDecodeError:
        return {"relations": [], "facts": []}
    rels = [r for r in obj.get("relations", [])
            if isinstance(r, dict) and r.get("a") and r.get("b") and r.get("relation")]
    facts = [f for f in obj.get("facts", [])
             if isinstance(f, dict) and f.get("statement")]
    return {"relations": rels, "facts": facts}


async def verify_fact_r2(llm, texts: list[str], statement: str,
                         max_chars: int = 9000) -> bool:
    """Scoped entailment for an atomic fact: judged ONLY against the chunks
    that contain its cited quote. Catches 'real quote, wrong statement'."""
    evidence = "\n\n".join(texts)[:max_chars]
    if not evidence.strip():
        return False
    reply = await llm(FACT_R2_PROMPT.format(evidence=evidence, statement=statement))
    return (reply or "").strip().upper().startswith("Y")


async def verify_relation_r2(llm, pack, triple: dict, co_ids: list[str],
                             max_items: int = 4, max_chars: int = 9000) -> bool:
    """Gate R2: given ONLY the co-mentioning evidence items (from Gate R1),
    does the source state/entail (a, relation, b)? Deterministic scope, LLM verdict."""
    texts = []
    for iid in co_ids[:max_items]:
        t = pack.evidence_text(iid)
        if t:
            texts.append(f"[{iid}]\n{t}")
    evidence = "\n\n".join(texts)[:max_chars]
    if not evidence.strip():
        return False
    reply = await llm(R2_PROMPT.format(evidence=evidence, a=triple["a"],
                                       relation=triple["relation"], b=triple["b"]))
    return (reply or "").strip().upper().startswith("Y")
