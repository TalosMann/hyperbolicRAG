r"""eval/provenance.py

Provenance capture for both backends, WITHOUT modifying upstream HyperRAG.

HyperRAG already assembles an entity/hyperedge/text-unit context bundle inside
`hyper_query` and returns the whole dict (instead of just the answer string)
when `QueryParam.return_type == "json"`. We exploit that — no upstream patch.

HierarchicalRAG is ours, so we add a parallel `query_with_provenance` that
records which tree nodes (and at which levels) were touched during collapsed-
tree retrieval.

Both produce the same normalized shape:

    {
        "answer": str,
        "provenance": { ... backend-specific ... },
        "ok": bool,
    }

so the runner can treat them uniformly.
"""
from __future__ import annotations

import os
import sys
from pathlib import Path
from typing import Any

# Make `hyperrag` importable — it lives in Hyper-RAG/ one level above hyperscholar/.
# Respects the HYPERRAG_PATH env var; falls back to the standard layout.
_default_hyperrag = str(
    Path(__file__).resolve().parent.parent.parent / "Hyper-RAG"
)
_hyperrag_path = os.environ.get("HYPERRAG_PATH", _default_hyperrag)
if _hyperrag_path not in sys.path:
    sys.path.insert(0, _hyperrag_path)


# ── HyperRAG ──────────────────────────────────────────────────────────────────

async def hyperrag_query_with_provenance(backend, namespace: str, text: str,
                                         top_k: int = 60, mode: str | None = None) -> dict:
    """Query a HyperRAGBackend (or HyperRAGLightBackend) and capture the entity/
    hyperedge/text-unit bundle the upstream engine already builds.

    `mode` overrides the backend's default ('hyper' / 'hyper-lite'); if None the
    backend's own `_mode` is used, so the same function serves both backends.
    """
    from hyperrag import QueryParam

    rag = backend._rag(namespace)
    use_mode = mode or backend._mode

    # return_type="json" → hyper_query returns the full contextJson dict:
    #   {"entities": [...], "hyperedges": [...], "text_units": [...], "response": "..."}
    param = QueryParam(mode=use_mode, top_k=top_k, return_type="json")

    try:
        result = await rag.aquery(text, param)
    except (AttributeError, KeyError, TypeError, UnboundLocalError) as e:
        return {
            "answer": backend._fail_markers[0],
            "ok": False,
            "provenance": {"entities": [], "hyperedges": [], "text_units": [],
                           "error": f"{type(e).__name__}: {e}"},
        }

    # If upstream returned a bare string (e.g. fail_response), normalize it.
    if isinstance(result, str):
        return {
            "answer": result,
            "ok": not any(m in result for m in backend._fail_markers),
            "provenance": {"entities": [], "hyperedges": [], "text_units": []},
        }

    answer = result.get("response", "") or ""
    entities = [
        {
            "entity_name": e.get("entity_name", ""),
            "entity_type": e.get("entity_type", "UNKNOWN"),
            "description": (e.get("description", "") or "")[:400],
            "rank": e.get("rank", 0),
        }
        for e in result.get("entities", [])
    ]
    hyperedges = [
        {
            "entity_set": h.get("entity_set", []),
            "description": (h.get("description", "") or "")[:400],
            "keywords": h.get("keywords", ""),
            "weight": h.get("weight", 0),
            "rank": h.get("rank", 0),
        }
        for h in result.get("hyperedges", [])
    ]
    text_units = [
        {"content": (t.get("content", "") or "")[:500]}
        for t in result.get("text_units", [])
    ]

    ok = bool(answer) and not any(m in answer for m in backend._fail_markers)
    return {
        "answer": answer,
        "ok": ok,
        "provenance": {
            "entities": entities,
            "hyperedges": hyperedges,
            "text_units": text_units,
            "counts": {
                "entities": len(entities),
                "hyperedges": len(hyperedges),
                "text_units": len(text_units),
            },
        },
    }


# ── HierarchicalRAG ───────────────────────────────────────────────────────────

async def hierarchical_query_with_provenance(backend, namespace: str, text: str,
                                             top_k: int = 60) -> dict:
    """Re-implements HierarchicalRAGBackend.query, but records which tree nodes
    and raw chunks were retrieved, plus the set of tree levels touched.

    Kept here (not on the backend) so the backend's public contract stays the
    minimal 4-method ABC. If the hierarchical design changes, only this function
    needs updating alongside it.
    """
    docs, chunks, cache, chunks_vdb, tree_vdb, tree_kv = backend._stores(namespace)
    k = max(4, min(top_k, 12))
    tree_hits = await tree_vdb.query(text, top_k=k)
    chunk_hits = await chunks_vdb.query(text, top_k=k)

    passages = []
    nodes_accessed = []
    levels = set()

    for h in tree_hits:
        node = await tree_kv.get_by_id(h["id"]) or {}
        content = node.get("content") or h.get("content", "")
        level = node.get("level", h.get("level", -1))
        if content:
            passages.append((h.get("distance", 0), f"[summary] {content}"))
            levels.add(level)
            nodes_accessed.append({
                "id": h["id"],
                "level": level,
                "summary": content[:400],
                "n_children": len(node.get("children", [])),
                "distance": float(h.get("distance", 0)),
            })

    chunks_accessed = []
    chunk_rows = await chunks.get_by_ids([h["id"] for h in chunk_hits])
    for h, row in zip(chunk_hits, chunk_rows):
        content = (row or {}).get("content", "")
        if content:
            passages.append((h.get("distance", 0), content))
            levels.add(0)  # leaf level
            chunks_accessed.append({
                "chunk_id": h["id"],
                "doc_id": (row or {}).get("full_doc_id", ""),
                "excerpt": content[:300],
                "distance": float(h.get("distance", 0)),
            })

    if not passages:
        return {
            "answer": "Sorry, I'm not able to provide an answer to that question.",
            "ok": False,
            "provenance": {"nodes_accessed": [], "chunks_accessed": [],
                           "levels_accessed": []},
        }

    passages.sort(key=lambda x: -x[0])
    context = "\n\n".join(p for _, p in passages[:k])[:24000]

    from .hierarchical_prompts import ANSWER_PROMPT, FAIL
    answer = await backend._llm(
        ANSWER_PROMPT.format(context=context, question=text, fail=FAIL),
        hashing_kv=cache)
    ok = bool(answer) and not any(m in answer for m in backend._fail_markers)

    return {
        "answer": answer or "",
        "ok": ok,
        "provenance": {
            "nodes_accessed": nodes_accessed,
            "chunks_accessed": chunks_accessed,
            "levels_accessed": sorted(levels),
            "counts": {
                "tree_nodes": len(nodes_accessed),
                "leaf_chunks": len(chunks_accessed),
                "levels": len(levels),
            },
        },
    }
