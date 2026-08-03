r"""End-to-end pipeline: bind -> retrieve -> assert -> verify (R1+R2) -> compose.

The composed prose is built ONLY from assertions that survived both gates, so an
unverified relationship cannot appear in the answer. Facts are verified
deterministically (their cited evidence quote must exist in the retrieved text;
their statement must carry no ungrounded hard specifics).

LLM calls per query: 1 (assertion generation) + 1 per surviving triple (Gate R2).
Entity binding and Gate R1 are fully deterministic.
"""
from __future__ import annotations

import re

from ..grounding.verifier import build_source_index, flagged_specifics, specific_grounded
from .assertions import (ASSERTION_PROMPT, SYNTHESIS_PROMPT, classify_intent,
                         parse_assertions, verify_fact_r2, verify_relation_r2)
from .evidence import EvidencePack, _norm

MAX_EVIDENCE_CHARS = 80_000


# ── stage 1: entity binding (deterministic) ──────────────────────────────────

def bind_question_entities(pack: EvidencePack, question: str,
                           min_len: int = 3) -> list[str]:
    """Corpus entity names whose surface form occurs in the question, in
    question order. Binds mentions to CORPUS nodes (not parametric namesakes).
    Returns the longest-match mention list (e.g. 'TINY TIM' suppresses 'TIM')."""
    qn = _norm(question)
    hits: list[tuple[int, str]] = []
    for name in pack.vertices:
        nn = _norm(name).strip()
        if len(nn) < min_len:
            continue
        pos = qn.find(f" {nn} ")
        if pos >= 0:
            hits.append((pos, name))
    hits.sort()
    # longest-match suppression: drop a name whose tokens are a strict subset of
    # another hit's tokens ('TIM' suppressed by 'TINY TIM' / 'TIM CRATCHIT')
    all_names = [n for _, n in hits]
    out: list[str] = []
    for _, name in hits:
        toks = set(_norm(name).split())
        if any(toks < set(_norm(o).split()) for o in all_names if o != name):
            continue
        if name not in out:
            out.append(name)
    return out


# ── stage 2: hybrid retrieval ────────────────────────────────────────────────

async def hybrid_retrieve(pack: EvidencePack, rag, question: str,
                          mentions: list[str], k_sem: int = 8,
                          k_ent: int = 6) -> tuple[list[str], list[dict]]:
    """Chunk ids = semantic top-k  ∪  entity-anchored chunks per mention;
    relations = hyperedges touching the mentions. The entity-anchored part is
    what preserves the multi-hop advantage (rare distant endpoints)."""
    chunk_ids: list[str] = []
    if rag is not None:
        hits = await rag.chunks_vdb.query(question, top_k=k_sem)
        chunk_ids = [h["id"] for h in hits if h["id"] in pack.chunks]
    for m in mentions:
        anchored = sorted(pack.entity_chunks(m))[:k_ent]
        for cid in anchored:
            if cid not in chunk_ids:
                chunk_ids.append(cid)
    relations = pack.relations_among(mentions) if mentions else []
    return chunk_ids, relations


def render_evidence(pack: EvidencePack, chunk_ids: list[str],
                    relations: list[dict]) -> str:
    parts = []
    for e in relations[:25]:
        parts.append(f"[{e['id']}] relation among ({', '.join(e['entities'])}): "
                     f"{e['description']}")
    for cid in chunk_ids:
        parts.append(f"[{cid}]\n{pack.chunks[cid]}")
    return "\n\n".join(parts)[:MAX_EVIDENCE_CHARS]


# ── stages 4-5: verify assertions ────────────────────────────────────────────

def _quote_grounded(quote: str, source_idx) -> bool:
    q = (quote or "").strip()
    if len(q.split()) < 3:      # too short to check meaningfully -> require presence anyway
        return bool(q) and f" {_norm(q).strip()} " in source_idx["text_norm"]
    return specific_grounded("quote", q, source_idx)


def _chunks_with_quote(pack: EvidencePack, chunk_ids: list[str],
                       quote: str) -> list[str]:
    """Chunk ids (among the retrieved set) whose text contains the quote —
    the deterministic scope for the fact-entailment check."""
    qw = _norm(quote).strip().split()
    if not qw:
        return []
    needles = ([" ".join(qw)] if len(qw) < 6
               else [" ".join(qw[i:i + 6]) for i in range(len(qw) - 5)])
    out = []
    for cid in chunk_ids:
        tn = _norm(pack.chunks.get(cid, ""))
        if any(f" {n} " in tn for n in needles):
            out.append(cid)
    return out


async def verify_all(llm, pack: EvidencePack, parsed: dict,
                     source_idx, chunk_ids: list[str] | None = None) -> dict:
    verified_rel, rejected_rel = [], []
    for t in parsed["relations"]:
        co = pack.co_mention(t["a"], t["b"])
        if not co:
            rejected_rel.append({**t, "gate": "R1",
                                 "reason": "no evidence item co-mentions both entities"})
            continue
        ok = await verify_relation_r2(llm, pack, t, co)
        if ok:
            verified_rel.append({**t, "provenance": co[:3]})
        else:
            rejected_rel.append({**t, "gate": "R2",
                                 "reason": "co-mentioning evidence does not state this relation",
                                 "checked_against": co[:3]})

    verified_facts, dropped_facts = [], []
    for f in parsed["facts"]:
        quote = f.get("evidence_quote", "")
        if not _quote_grounded(quote, source_idx):
            dropped_facts.append({**f, "reason": "quote not found in source"})
            continue
        if flagged_specifics(f["statement"], source_idx):
            dropped_facts.append({**f, "reason": "statement carries ungrounded specific"})
            continue
        # facts-entailment gate: judge the STATEMENT against the chunks that
        # contain its quote (catches 'real quote, wrong statement'); if the
        # quote isn't locatable in a chunk, fall back to the top retrieved
        # chunks — the gate always runs, never passes by default
        scope = _chunks_with_quote(pack, chunk_ids or [], quote)
        if not scope:
            scope = list(chunk_ids or [])[:3]
        texts = [pack.chunks[c] for c in scope[:3] if c in pack.chunks]
        ok = await verify_fact_r2(llm, texts, f["statement"])
        if not ok:
            dropped_facts.append({**f, "gate": "F2", "reason":
                                  "source chunks do not entail the statement",
                                  "checked_against": scope[:3]})
            continue
        verified_facts.append(f)
    return {"verified_relations": verified_rel, "rejected_relations": rejected_rel,
            "verified_facts": verified_facts, "dropped_facts": dropped_facts}


# ── stage 6: compose (deterministic templating) ──────────────────────────────

def _direct_link(pack: EvidencePack, a: str, b: str, rels: list[dict]) -> bool:
    fa, fb = pack.forms(a), pack.forms(b)
    for r in rels:
        ra, rb = r["a"].lower().strip(), r["b"].lower().strip()
        if (ra in fa and rb in fb) or (ra in fb and rb in fa):
            return True
    return False


def _corpus_lines_and_endnote(pack: EvidencePack, mentions: list[str],
                              v: dict) -> tuple[list[str], str]:
    """Shared by compose() and compose_synthesis(): the bulleted, cited lines
    for every verified relation/fact, plus the templated 'no direct
    relationship' endnote when the two asked-about entities never verify-link.
    Pure deterministic assembly — no LLM call, nothing here is 'reasoning'."""
    rels, facts = v["verified_relations"], v["verified_facts"]
    lines, seen = [], set()
    for r in rels:
        key = (r["a"].lower().strip(), r["relation"].lower().strip(),
               r["b"].lower().strip())
        if key in seen:
            continue
        seen.add(key)
        prov = ", ".join(r.get("provenance", [])[:2])
        lines.append(f"- {r['a']} — {r['relation']} — {r['b']}  [{prov}]")
    for f in facts:
        if f["statement"].lower().strip() in seen:
            continue
        seen.add(f["statement"].lower().strip())
        lines.append(f"- {f['statement']}")

    endnote = ""
    if len(mentions) >= 2:
        a, b = mentions[0], mentions[1]
        if not _direct_link(pack, a, b, rels):
            via = ""
            endpoint_forms = pack.forms(a) | pack.forms(b)
            intermediates = sorted({n for r in rels for n in (r["a"], r["b"])
                                    if n.lower().strip() not in endpoint_forms})
            if intermediates:
                via = (" Any connection runs indirectly through: "
                       + ", ".join(intermediates) + ".")
            endnote = (f"\n\nNote: the source does not state a direct relationship "
                       f"between {a} and {b}.{via}")
    return lines, endnote


def compose(pack: EvidencePack, question: str, mentions: list[str],
            v: dict) -> dict:
    rels, facts = v["verified_relations"], v["verified_facts"]
    if not rels and not facts:
        return {"decision": "DECLINE", "answer":
                "The source material does not contain an answer to this question."}
    lines, endnote = _corpus_lines_and_endnote(pack, mentions, v)
    answer = ("Based only on the source material:\n" + "\n".join(lines) + endnote)
    return {"decision": "SERVE", "answer": answer}


def compose_synthesis(pack: EvidencePack, question: str, mentions: list[str],
                      v: dict, general_knowledge: str) -> dict:
    """SYNTHESIS mode: two clearly headed, separately-provenanced sections.
    The corpus section is built EXACTLY as in compose() (same verified-only
    guarantee, same templating, no LLM). The general-knowledge section is
    unverified by construction — it never claims to be from the corpus, so
    R1/R2 do not and should not apply to it — and is labeled as such so a
    reader always knows which half to trust outright and which to check."""
    lines, endnote = _corpus_lines_and_endnote(pack, mentions, v)
    corpus_block = ("From this text:\n" + "\n".join(lines) + endnote) if lines else (
        "From this text: the source material does not directly address this.")
    gk_block = ("\n\nGeneral literary knowledge (not from this corpus — "
               "please verify independently):\n" + (general_knowledge or "").strip())
    return {"decision": "SERVE", "answer": corpus_block + gk_block}


# ── shared: retrieve + assert + verify (used by both CORPUS and SYNTHESIS modes)

async def _gather_verified(pack: EvidencePack, rag, llm, question: str,
                           k_sem: int) -> dict:
    """Everything through Gate R1/R2/F2 verification. Identical for both
    modes — SYNTHESIS mode's general-knowledge track is generated separately
    and never touches this; only the composition step differs by mode."""
    mentions = bind_question_entities(pack, question)
    chunk_ids, relations = await hybrid_retrieve(pack, rag, question, mentions,
                                                 k_sem=k_sem)
    if not chunk_ids and not relations:
        return {"mentions": mentions, "chunk_ids": chunk_ids, "relations": relations,
                "verified_relations": [], "rejected_relations": [],
                "verified_facts": [], "dropped_facts": [], "n_relations_asserted": 0,
                "source_idx": build_source_index(""), "empty": True}

    evidence = render_evidence(pack, chunk_ids, relations)
    raw = await llm(ASSERTION_PROMPT.format(question=question, evidence=evidence))
    parsed = parse_assertions(raw)

    source_idx = build_source_index(
        "\n".join(pack.chunks[c] for c in chunk_ids) + "\n" +
        "\n".join(e["description"] for e in relations))
    v = await verify_all(llm, pack, parsed, source_idx, chunk_ids)
    return {"mentions": mentions, "chunk_ids": chunk_ids, "relations": relations,
            "source_idx": source_idx, "n_relations_asserted": len(parsed["relations"]),
            "empty": False, **v}


_INTENT_STOPWORDS = {"how", "what", "why", "when", "where", "who", "which",
                    "besides", "does", "did", "the", "based", "according"}


def _has_external_reference(pack: EvidencePack, question: str,
                            mentions: list[str]) -> bool:
    """Deterministic guard on top of classify_intent(): does the question name
    something that does NOT resolve to a corpus entity? SYNTHESIS mode is only
    trustworthy when it is — otherwise the LLM classifier's "SYNTHESIS" verdict
    is overridden back to CORPUS. Without this, a misclassified purely-corpus
    question (e.g. "How is Fan related to Tiny Tim?") can produce an unverified
    general-knowledge section that reasons about the SAME corpus's characters
    under a "not from this corpus" label -- honest about ITS provenance, but
    exactly the kind of corpus-adjacent claim the strict path exists to prevent.
    Same principle as Gate R1: never trust a single LLM judgment when a cheap
    deterministic check can validate it."""
    bound = set()
    for m in mentions:
        bound |= pack.forms(m)
    for cand in re.findall(r"\b[A-Z][a-zA-Z']+(?:\s+[A-Z][a-zA-Z']+)*\b", question):
        if cand.lower() in _INTENT_STOPWORDS or len(cand) <= 2:
            continue
        if _norm(cand).strip() not in bound:
            return True
    return False


def _corpus_summary_text(v: dict) -> str:
    """Plain-text digest of verified corpus content, fed to the SYNTHESIS-mode
    general-knowledge prompt as context ONLY — never re-verified, since it is
    already-verified output being summarized for a different generation call."""
    lines = [f"{r['a']} {r['relation']} {r['b']}" for r in v.get("verified_relations", [])]
    lines += [f["statement"] for f in v.get("verified_facts", [])]
    return "\n".join(lines) if lines else "(no directly relevant facts found in the text)"


# ── public entry point ───────────────────────────────────────────────────────

async def safe_answer(pack: EvidencePack, rag, llm, question: str,
                      k_sem: int = 8) -> dict:
    """Routes on intent (classify_intent): CORPUS -> the original strict,
    fully-cited pipeline, unchanged. SYNTHESIS -> the same verified corpus
    facts PLUS a separately-generated, clearly labeled general-knowledge
    section for comparisons/analysis that legitimately reach outside the
    corpus. The corpus-fact guarantee (verify-then-compose, R1/R2/F2) is
    identical in both modes — only whether an extra labeled section is
    appended differs."""
    intent = await classify_intent(llm, question)
    g = await _gather_verified(pack, rag, llm, question, k_sem)
    if intent == "SYNTHESIS" and not _has_external_reference(pack, question, g["mentions"]):
        intent = "CORPUS"   # deterministic override -- see _has_external_reference

    if g["empty"]:
        base = {"decision": "DECLINE", "answer":
                "The source material does not contain an answer to this question.",
                "n_llm_calls": 1}
    elif intent == "SYNTHESIS":
        gk_text = await llm(SYNTHESIS_PROMPT.format(
            corpus_summary=_corpus_summary_text(g), question=question))
        out = compose_synthesis(pack, question, g["mentions"], g, gk_text)
        base = {**out, "n_llm_calls": 2 + g["n_relations_asserted"] -
                       sum(1 for r in g["rejected_relations"] if r["gate"] == "R1")}
    else:
        out = compose(pack, question, g["mentions"], g)
        # final deterministic safety net: composed prose must carry no ungrounded specifics
        residual = flagged_specifics(out["answer"], g["source_idx"])
        if residual:
            keep = [ln for ln in out["answer"].splitlines()
                    if not flagged_specifics(ln, g["source_idx"])]
            out["answer"] = "\n".join(keep) or \
                "The source material does not contain an answer to this question."
        base = {**out, "residual_flagged": len(residual),
                "n_llm_calls": 1 + g["n_relations_asserted"] -
                               sum(1 for r in g["rejected_relations"] if r["gate"] == "R1")}

    return {**g, **base, "intent": intent, "evidence_chunks": g.get("chunk_ids", [])}
