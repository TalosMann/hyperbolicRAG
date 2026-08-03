r"""Grounding verifier — anchored generation + dual verification.

Design (no RAG-strategy edits required):
  - Ground against RAW source chunks (the actual evidence), never a derived
    structure (entity CSVs, summary-tree nodes) — derived structures are lossy
    and mis-judge both real and fabricated content.
  - Anchored generation: replace the backend's drift-prone synthesis with a
    strict "answer only from the source, else NO_ANSWER_IN_SOURCE" prompt.
  - Dual verification against the raw chunks:
      * deterministic specific-fact checker (hard guarantee on the dangerous
        class: years/numbers/dates/double-quoted quotations)
      * calibrated LLM verifier (fuzzy/semantic claims; paraphrase = supported)
  - Decision: serve grounded (+ interpretation), quarantine fabricated
    specifics, decline when nothing grounded survives.

The one strategy-specific seam is `retrieve_raw_chunks` (how to pull raw source
passages from a given backend's chunk store). Everything else is backend-neutral.
"""
from __future__ import annotations

import json
import re

MONTHS = ("january february march april may june july august september october "
          "november december").split()

ANCHORED_GEN_PROMPT = """You are answering a student's question using ONLY the SOURCE MATERIAL provided.

Strict rules:
- Use ONLY facts, names, dates, numbers, and quotations that actually appear in the SOURCE MATERIAL. NEVER add information from outside/general knowledge.
- If the SOURCE MATERIAL does not contain the answer, reply with EXACTLY this token and nothing else: NO_ANSWER_IN_SOURCE
- Do not pad with background, speculation, or "as is well known" commentary.
- Stay close to the source's wording; quote it where helpful. Be concise.

SOURCE MATERIAL:
{context}

QUESTION: {question}

Answer using only the source material:"""

VERIFY_PROMPT = """You are a grounding verifier. Given SOURCE PASSAGES (the only evidence) and a
DRAFT ANSWER, extract the draft's MAIN factual claims (a handful of substantive claims — NOT
every clause) and sort each into:
1. supported          - the passages state it, OR the claim faithfully paraphrases/restates
                        something in the passages (different wording still counts as supported).
2. unsupported_fact   - a specific detail (date/number/name/ID/quotation) NOT in the passages.
3. unsupported_interp - a reasonable interpretation/inference asserting no new specific fact.
Bias toward 'supported' when the passages substantively back the claim, even if worded
differently. Ignore meta/hedging sentences.

SOURCE PASSAGES:
{context}

DRAFT ANSWER:
{answer}

Respond with ONLY JSON: {{"supported":[...],"unsupported_fact":[...],"unsupported_interp":[...]}}"""


# ── deterministic specific-fact checker (double/curly-double quotes only) ─────

def build_source_index(raw: str) -> dict:
    words = re.sub(r"[^a-z0-9 ]+", " ", raw.lower())
    words = re.sub(r"\s+", " ", words).strip().split()
    six = {" ".join(words[i:i + 6]) for i in range(len(words) - 5)}
    return {"text_norm": " " + " ".join(words) + " ",
            "digits": " " + re.sub(r"\s+", " ", re.sub(r"[^0-9]", " ", raw)).strip() + " ",
            "sixgrams": six}


def extract_specifics(answer: str) -> list[tuple[str, str]]:
    """Hard specifics worth grounding: years, numbers, month-day dates, and
    double/curly-double quotations. Single-quote extraction is intentionally
    omitted — it over-triggers on possessive/contraction apostrophes."""
    out: list[tuple[str, str]] = []
    for y in re.findall(r"\b(?:1[0-9]{3}|20[0-9]{2})\b", answer):
        out.append(("year", y))
    for n in re.findall(r"\b\d{1,3}(?:,\d{3})+\b", answer):
        out.append(("number", n))
    for n in re.findall(r"\b\d{4,}\b", answer):
        out.append(("number", n))
    for m, d in re.findall(rf"\b({'|'.join(MONTHS)})\s+(\d{{1,2}})\b", answer, re.I):
        out.append(("date", f"{m.lower()} {d}"))
    for q in re.findall(r'"([^"\n]{12,})"', answer):
        out.append(("quote", q))
    for q in re.findall(r"[“]([^“”\n]{12,})[”]", answer):
        out.append(("quote", q))
    return out


def specific_grounded(kind: str, value: str, idx: dict) -> bool:
    if kind in ("year", "number"):
        return f" {value.replace(',', '')} " in idx["digits"]
    if kind == "date":
        return value.lower() in idx["text_norm"]
    if kind == "quote":
        qw = re.sub(r"[^a-z0-9 ]+", " ", value.lower())
        qw = re.sub(r"\s+", " ", qw).strip().split()
        if len(qw) < 6:
            return " " + " ".join(qw) + " " in idx["text_norm"]
        return any(" ".join(qw[i:i + 6]) in idx["sixgrams"] for i in range(len(qw) - 5))
    return True


def flagged_specifics(text: str, idx: dict) -> list[tuple[str, str]]:
    """Hard specifics in `text` that are NOT grounded in the source index."""
    return [(k, v) for (k, v) in extract_specifics(text)
            if not specific_grounded(k, v, idx)]


# ── LLM calls ────────────────────────────────────────────────────────────────

def _parse_json(raw: str) -> dict:
    if not raw:
        return {}
    m = re.search(r"\{.*\}", raw.replace("```json", "").replace("```", ""), re.DOTALL)
    if not m:
        return {}
    try:
        return json.loads(m.group(0))
    except json.JSONDecodeError:
        return {}


async def llm_verify(llm, context: str, answer: str) -> dict:
    p = _parse_json(await llm(VERIFY_PROMPT.format(context=context, answer=answer)))
    return {"supported": p.get("supported", []),
            "unsupported_fact": p.get("unsupported_fact", []),
            "unsupported_interp": p.get("unsupported_interp", [])}


async def retrieve_raw_chunks(rag, text: str, k: int = 24) -> str:
    """Strategy-specific seam: pull raw source passages from a backend's chunk
    store. Works for any backend exposing a chunk vector store + text_chunks
    (Cog-RAG, HyperRAG, HierarchicalRAG all do)."""
    hits = await rag.chunks_vdb.query(text, top_k=k)
    ids = [h["id"] for h in hits]
    chunks = await rag.text_chunks.get_by_ids(ids)
    return "\n\n---\n\n".join((c or {}).get("content", "") for c in chunks if c)


# ── decision ─────────────────────────────────────────────────────────────────

def decide(v: dict, idx: dict) -> dict:
    """Turn verifier buckets into a serve/quarantine/decline decision.
    The deterministic checker re-filters the LLM's 'supported' bucket: any
    approved claim carrying a fabricated specific is demoted."""
    clean_supported, demoted = [], []
    for c in v["supported"]:
        (demoted if flagged_specifics(c, idx) else clean_supported).append(c)
    clean_interp = [c for c in v["unsupported_interp"] if not flagged_specifics(c, idx)]
    all_flagged = flagged_specifics(
        " ".join(v["supported"] + v["unsupported_fact"] + v["unsupported_interp"]), idx)
    decision = ("SERVE" if clean_supported
                else "SERVE_INTERP_ONLY" if clean_interp else "DECLINE")
    return {"decision": decision, "clean_supported": clean_supported,
            "clean_interp": clean_interp, "quarantined": v["unsupported_fact"] + demoted,
            "demoted": demoted, "flagged": all_flagged}


async def grounded_answer(be, rag, llm, namespace: str, question: str, k: int = 24) -> dict:
    """Full pipeline: raw-chunk retrieval -> anchored generation -> calibrated
    verify -> decide. Generation and verification share the SAME raw-chunk ground
    truth (not the backend's lossy derived context)."""
    raw = await retrieve_raw_chunks(rag, question, k)
    if not raw.strip():
        return {"decision": "DECLINE", "answered": False, "draft": "",
                "clean_supported": [], "clean_interp": [], "flagged": [],
                "quarantined": [], "demoted": [], "reason": "no_context"}

    answer = (await llm(ANCHORED_GEN_PROMPT.format(context=raw, question=question))).strip()
    if "NO_ANSWER_IN_SOURCE" in answer or not answer:
        return {"decision": "DECLINE", "answered": False, "draft": answer,
                "clean_supported": [], "clean_interp": [], "flagged": [],
                "quarantined": [], "demoted": [], "reason": "anchored_declined"}

    idx = build_source_index(raw)
    v = await llm_verify(llm, raw, answer)
    d = decide(v, idx)
    d["draft"] = answer
    d["answered"] = d["decision"] in ("SERVE", "SERVE_INTERP_ONLY")
    d["reason"] = "verified"
    return d
