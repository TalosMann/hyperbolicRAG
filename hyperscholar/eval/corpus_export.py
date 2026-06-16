r"""eval/corpus_export.py

Dumps the FULL structural overlay of an indexed corpus to JSON + Markdown.
Run once after preindex.py has completed.

Uses file-backed storage (JsonKVStorage, NanoVectorDBStorage, HypergraphStorage)
to read from the same persisted files that preindex.py wrote.

Usage
-----
    cd D:\Projects\hyperbolic
    python -m hyperscholar.eval.corpus_export --corpus demo --backend hyperrag
    python -m hyperscholar.eval.corpus_export --corpus demo --backend hierarchical
    python -m hyperscholar.eval.corpus_export --corpus neurology --backend hyperrag
"""
from __future__ import annotations

import argparse
import asyncio
import json
from pathlib import Path


async def export_hyperrag(corpus: str, namespace: str, results_dir: Path) -> None:
    from hyperscholar.core.config import load_config
    from hyperscholar.core.embedder import build_embedder
    from hyperscholar.core.llm import build_llm_func
    from hyperscholar.rag.hyperrag_backend import HyperRAGBackend
    from hyperrag.storage import JsonKVStorage, NanoVectorDBStorage, HypergraphStorage

    cfg = load_config()
    embedder = build_embedder(cfg.embedding)
    llm = build_llm_func(cfg.llm)

    backend = HyperRAGBackend(
        llm_func=llm, embedder=embedder, working_dir=cfg.working_dir,
        kv_cls=JsonKVStorage,
        vector_cls=NanoVectorDBStorage,
        hypergraph_cls=HypergraphStorage,
        pg_dsn=None, fail_markers=cfg.rag.fail_markers)

    rag = backend._rag(namespace)
    hg = rag.chunk_entity_relation_hypergraph

    vertex_ids = await hg.get_all_vertices()   # returns a set of IDs
    edge_ids = await hg.get_all_hyperedges()   # returns a set of edge tuples

    entities = []
    for vid in vertex_ids:
        vdata = await hg.get_vertex(vid) or {}
        degree = await hg.vertex_degree(vid)
        entities.append({
            "id": vid,
            "type": vdata.get("entity_type", "UNKNOWN"),
            "description": (vdata.get("description", "") or "").split("<SEP>")[0][:400],
            "degree": degree,
        })
    entities.sort(key=lambda e: -e["degree"])

    edges = []
    for eid in edge_ids:
        edata = await hg.get_hyperedge(eid) or {}
        edges.append({
            "id": str(eid),
            "entity_set": edata.get("id_set", list(eid) if isinstance(eid, (set, frozenset, tuple)) else [str(eid)]),
            "description": (edata.get("description", "") or "")[:400],
            "keywords": edata.get("keywords", ""),
            "weight": edata.get("weight", 0),
        })

    chunk_ids = await rag.text_chunks.all_keys()
    out = {
        "corpus": corpus,
        "backend": "hyperrag",
        "stats": {
            "entities": len(entities),
            "hyperedges": len(edges),
            "chunks": len(chunk_ids),
        },
        "entities": entities,
        "hyperedges": edges,
    }
    out_dir = results_dir / corpus
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / "corpus_hyperrag.json").write_text(
        json.dumps(out, indent=2, ensure_ascii=False), encoding="utf-8")

    md = [f"# HyperRAG corpus structure — {corpus}\n",
          f"**Entities:** {len(entities)} · **Hyperedges:** {len(edges)} "
          f"· **Chunks:** {len(chunk_ids)}\n",
          "## Top entities by degree\n"]
    for e in entities[:50]:
        md.append(f"- **{e['id']}** ({e['type']}, degree {e['degree']}): {e['description']}")
    md.append("\n## Hyperedges (sample)\n")
    for e in edges[:50]:
        members = ", ".join(e["entity_set"]) if isinstance(e["entity_set"], list) else str(e["entity_set"])
        md.append(f"- [{members}] (w={e['weight']}): {e['description']}")
    (out_dir / "corpus_hyperrag.md").write_text("\n".join(md), encoding="utf-8")
    print(f"✓ hyperrag export → {out_dir}/corpus_hyperrag.[json|md]")
    print(f"  {len(entities)} entities, {len(edges)} hyperedges, {len(chunk_ids)} chunks")


async def export_hierarchical(corpus: str, namespace: str, results_dir: Path) -> None:
    from hyperscholar.core.config import load_config
    from hyperscholar.core.embedder import build_embedder
    from hyperscholar.core.llm import build_llm_func
    from hyperscholar.rag.hierarchical_backend import HierarchicalRAGBackend
    from hyperrag.storage import JsonKVStorage, NanoVectorDBStorage

    cfg = load_config()
    embedder = build_embedder(cfg.embedding)
    llm = build_llm_func(cfg.llm)

    backend = HierarchicalRAGBackend(
        llm_func=llm, embedder=embedder,
        kv_cls=JsonKVStorage,
        vector_cls=NanoVectorDBStorage,
        pg_dsn=None, fail_markers=cfg.rag.fail_markers)

    docs, chunks, cache, chunks_vdb, tree_vdb, tree_kv = backend._stores(namespace)

    all_ids = await tree_kv.all_keys()
    all_nodes = {i: n for i, n in
                 zip(all_ids, await tree_kv.get_by_ids(all_ids)) if n}
    chunk_ids = await chunks.all_keys()

    max_level = max((n.get("level", 0) for n in all_nodes.values()), default=0)

    def build_subtree(nid: str) -> dict:
        node = all_nodes.get(nid)
        if node is None:
            return {"id": nid, "level": 0, "type": "chunk"}
        kids = [build_subtree(ch) for ch in node.get("children", [])]
        return {
            "id": nid,
            "level": node.get("level", 0),
            "summary": (node.get("content", "") or "")[:400],
            "children": kids,
        }

    roots = [nid for nid, node in all_nodes.items()
             if node.get("level", 0) == max_level]
    tree = [build_subtree(r) for r in roots]

    out = {
        "corpus": corpus,
        "backend": "hierarchical",
        "stats": {
            "levels": max_level,
            "total_nodes": len(all_nodes),
            "leaf_chunks": len(chunk_ids),
            "roots": len(roots),
        },
        "tree": tree,
    }
    out_dir = results_dir / corpus
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / "corpus_hierarchical.json").write_text(
        json.dumps(out, indent=2, ensure_ascii=False), encoding="utf-8")

    md = [f"# HierarchicalRAG corpus structure — {corpus}\n",
          f"**Levels:** {max_level} · **Tree nodes:** {len(all_nodes)} "
          f"· **Leaf chunks:** {len(chunk_ids)} · **Roots:** {len(roots)}\n",
          "## Summary tree\n"]

    def render(node: dict, indent: int = 0):
        pad = "  " * indent
        if node.get("type") == "chunk":
            md.append(f"{pad}- _(chunk {node['id'][:16]}…)_")
        else:
            md.append(f"{pad}- **L{node['level']}** {node.get('summary', '')[:120]}")
            for ch in node.get("children", []):
                render(ch, indent + 1)

    for r in tree:
        render(r)
    (out_dir / "corpus_hierarchical.md").write_text("\n".join(md), encoding="utf-8")
    print(f"✓ hierarchical export → {out_dir}/corpus_hierarchical.[json|md]")
    print(f"  {len(all_nodes)} tree nodes across {max_level} levels, {len(chunk_ids)} chunks")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--corpus", required=True)
    ap.add_argument("--namespace", default=None)
    ap.add_argument("--backend", required=True, choices=["hyperrag", "hierarchical"])
    ap.add_argument("--results-dir",
                    default=str(Path(__file__).resolve().parent / "results"))
    args = ap.parse_args()
    namespace = args.namespace or args.corpus
    results_dir = Path(args.results_dir)
    if args.backend == "hyperrag":
        asyncio.run(export_hyperrag(args.corpus, namespace, results_dir))
    else:
        asyncio.run(export_hierarchical(args.corpus, namespace, results_dir))


if __name__ == "__main__":
    main()
