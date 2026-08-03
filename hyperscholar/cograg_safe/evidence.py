r"""EvidencePack — the deterministic substrate for relational grounding.

Holds, for one namespace:
  chunks    : {chunk_id: raw text}                       (ground truth)
  vertices  : {NAME: {"source_ids": set, "description": str}}
  edges     : [{"id", "entities": [NAME,...], "description", "source_ids": set}]

and answers the deterministic questions the pipeline needs:
  - surface_forms(mention)   -> alias set for a question mention (permissive)
  - co_mention(a, b)         -> evidence item ids where a and b co-occur (Gate R1)
  - entity_chunks(mention)   -> chunk ids anchored to an entity (structural retrieval)
  - relations_among(mentions)-> hyperedges touching >=2 of the mentioned entities

Alias policy (validated against the christmas_carol hypergraph): entity extraction
splits aliases across vertices (TINY TIM vs TIM CRATCHIT; SCROOGE vs EBENEZER
SCROOGE), so a question mention expands to EVERY vertex name that token-contains
it (plus the mention itself for raw-text matching). Over-inclusion is sound here:
R1 is a NECESSARY-condition gate (reject only when no plausible surface pairing
co-occurs anywhere); Gate R2 does the strict filtering on the co-mentioning
evidence. Permissive R1 = fewer false rejections, never false acceptances of the
final answer.
"""
from __future__ import annotations

import json
import re
from pathlib import Path

SEP = "<SEP>"

_WORD = re.compile(r"[a-z0-9']+")


def _tokens(name: str) -> set[str]:
    return set(_WORD.findall(name.lower()))


def _norm(text: str) -> str:
    """Lowercased, punctuation-stripped, whitespace-COLLAPSED, padded — so a
    single-spaced needle always matches ('Fan, the' -> ' fan the ')."""
    t = re.sub(r"[^a-z0-9 ]+", " ", text.lower())
    return " " + re.sub(r"\s+", " ", t).strip() + " "


def surface_forms(mention: str, vertex_names: list[str]) -> set[str]:
    """All surface forms a question mention may appear under.

    A vertex name is an alias candidate if the mention's tokens are a subset of
    the vertex's tokens or vice versa (TINY TIM ~ TIM ~ TIM CRATCHIT;
    SCROOGE ~ EBENEZER SCROOGE). Always includes the raw mention itself.
    """
    m = _tokens(mention)
    if not m:
        return {mention.lower()}
    out = {mention.lower().strip()}
    for name in vertex_names:
        v = _tokens(name)
        if not v:
            continue
        if m <= v or v <= m:
            out.add(name.lower().strip())
    return out


class EvidencePack:
    def __init__(self, chunks: dict[str, str],
                 vertices: dict[str, dict],
                 edges: list[dict]):
        self.chunks = chunks
        self.vertices = vertices          # NAME -> {"source_ids": set[str], "description": str}
        self.edges = edges                # [{"id","entities","description","source_ids"}]
        self._chunk_norm = {cid: _norm(t) for cid, t in chunks.items()}
        self._vnames = list(vertices.keys())

    # ── loading ──────────────────────────────────────────────────────────────

    @classmethod
    def load(cls, runtime_dir: str | Path, namespace: str) -> "EvidencePack":
        """Load from a cograg_official namespace dir (chunks JSON + .hgdb)."""
        base = Path(runtime_dir) / namespace
        raw = json.loads((base / "kv_store_text_chunks.json").read_text(encoding="utf-8"))
        chunks = {cid: c.get("content", "") for cid, c in raw.items()}

        vertices: dict[str, dict] = {}
        edges: list[dict] = []
        hgdb = base / "hypergraph_chunk_entity_relation.hgdb"
        if hgdb.exists():
            from hyperdb import HypergraphDB
            hg = HypergraphDB(storage_file=str(hgdb))
            for v in hg.all_v:
                d = hg.v(v) or {}
                vertices[str(v)] = {
                    "source_ids": set(str(d.get("source_id", "")).split(SEP)) - {""},
                    "description": str(d.get("description", ""))[:600],
                }
            for i, e in enumerate(hg.all_e):
                d = hg.e(e) or {}
                edges.append({
                    "id": f"rel-{i}",
                    "entities": [str(x) for x in e],
                    "description": str(d.get("description", ""))[:600],
                    "source_ids": set(str(d.get("source_id", "")).split(SEP)) - {""},
                })
        return cls(chunks, vertices, edges)

    # ── alias + anchoring ────────────────────────────────────────────────────

    def forms(self, mention: str) -> set[str]:
        return surface_forms(mention, self._vnames)

    def bound_vertices(self, mention: str) -> list[str]:
        """Vertex names this mention binds to (corpus entity disambiguation)."""
        f = self.forms(mention)
        return [n for n in self._vnames if n.lower().strip() in f]

    def entity_chunks(self, mention: str) -> set[str]:
        """Chunk ids anchored to a mention: its vertices' source_ids plus any
        chunk whose raw text contains one of its surface forms."""
        out: set[str] = set()
        forms = self.forms(mention)
        for v in self.bound_vertices(mention):
            out |= self.vertices[v]["source_ids"]
        for cid, tnorm in self._chunk_norm.items():
            if any(f" {f} " in tnorm for f in (_norm(f).strip() for f in forms) if f):
                out.add(cid)
        return out & set(self.chunks)

    # ── Gate R1: deterministic co-mention ────────────────────────────────────

    def co_mention(self, a: str, b: str) -> list[str]:
        """Evidence item ids (chunk ids and/or edge ids) where mentions a and b
        BOTH occur, under any alias surface pairing. Empty list => the relation
        (a, *, b) cannot be stated by any single evidence item => R1 REJECT."""
        fa, fb = self.forms(a), self.forms(b)
        fa_n = [_norm(f).strip() for f in fa if f.strip()]
        fb_n = [_norm(f).strip() for f in fb if f.strip()]
        hits: list[str] = []

        # (1) raw-chunk surface co-mention
        for cid, tnorm in self._chunk_norm.items():
            if any(f" {x} " in tnorm for x in fa_n) and any(f" {y} " in tnorm for y in fb_n):
                hits.append(cid)

        va, vb = set(self.bound_vertices(a)), set(self.bound_vertices(b))

        # (2) shared hyperedge membership
        for e in self.edges:
            ents = set(e["entities"])
            if ents & va and ents & vb:
                hits.append(e["id"])

        # (3) shared vertex source_id (both entities extracted from same chunk)
        sa = set().union(*[self.vertices[v]["source_ids"] for v in va]) if va else set()
        sb = set().union(*[self.vertices[v]["source_ids"] for v in vb]) if vb else set()
        for cid in (sa & sb):
            if cid in self.chunks and cid not in hits:
                hits.append(cid)

        return hits

    # ── structural relation retrieval ────────────────────────────────────────

    def relations_among(self, mentions: list[str], max_edges: int = 40) -> list[dict]:
        """Hyperedges touching at least one bound vertex of >=1 mention, ranked by
        how many distinct mentions they touch (multi-mention edges first)."""
        bound = {m: set(self.bound_vertices(m)) for m in mentions}
        scored = []
        for e in self.edges:
            ents = set(e["entities"])
            touch = sum(1 for m, vs in bound.items() if ents & vs)
            if touch:
                scored.append((touch, len(e["entities"]), e))
        scored.sort(key=lambda t: (-t[0], -t[1]))
        return [e for _, _, e in scored[:max_edges]]

    def evidence_text(self, item_id: str) -> str:
        """Raw text of an evidence item (chunk content, or edge description +
        its source chunks' text) — what Gate R2 judges against."""
        if item_id in self.chunks:
            return self.chunks[item_id]
        for e in self.edges:
            if e["id"] == item_id:
                src = "\n".join(self.chunks.get(c, "") for c in list(e["source_ids"])[:3])
                return (f"[extracted relation among {', '.join(e['entities'])}]: "
                        f"{e['description']}\n--- source text ---\n{src}")
        return ""
