r"""eval/repair_entities_vdb.py

One-off repair: rebuilds a corrupted vdb_entities.json from the intact
hypergraph data (hypergraph_chunk_entity_relation.hgdb). The hypergraph
already holds every entity's name + description; we just re-embed and
re-populate the entity vector store.

Use when a JSONDecodeError on vdb_entities.json indicates a truncated write
(e.g. from a Ctrl+C during index_done_callback), but the hypergraph file
itself loaded fine.

Usage
-----
    cd D:\Projects\hyperbolic
    python -m hyperscholar.eval.repair_entities_vdb --corpus neurology
"""
from __future__ import annotations

import argparse
import asyncio
import shutil
from pathlib import Path


async def repair(corpus: str, namespace: str) -> None:
    from hyperscholar.core.config import load_config
    from hyperscholar.core.embedder import build_embedder
    from hyperscholar.core.llm import build_llm_func
    from hyperscholar.rag.hyperrag_backend import HyperRAGBackend
    from hyperrag.storage import JsonKVStorage, NanoVectorDBStorage, HypergraphStorage

    cfg = load_config()
    embedder = build_embedder(cfg.embedding)
    llm = build_llm_func(cfg.llm)

    workdir = Path(cfg.working_dir) / "hyperrag" / namespace
    bad_file = workdir / "vdb_entities.json"

    if bad_file.exists():
        backup = workdir / "vdb_entities.json.corrupted"
        shutil.move(str(bad_file), str(backup))
        print(f"[backup] moved corrupted file → {backup}")

    backend = HyperRAGBackend(
        llm_func=llm, embedder=embedder, working_dir=cfg.working_dir,
        kv_cls=JsonKVStorage, vector_cls=NanoVectorDBStorage,
        hypergraph_cls=HypergraphStorage, pg_dsn=None,
        fail_markers=cfg.rag.fail_markers)
    rag = backend._rag(namespace)
    hg = rag.chunk_entity_relation_hypergraph

    vertex_ids = await hg.get_all_vertices()
    print(f"[repair] {len(vertex_ids)} vertices found in hypergraph")

    # Rebuild the entity vector payloads in the EXACT shape upstream HyperRAG
    # uses (operate.py extract_entities → data_for_vdb, line ~595-601):
    #   key = compute_mdhash_id(entity_name, prefix="ent-")
    #   content = entity_name + description  (concatenated, no separator)
    from hyperrag.utils import compute_mdhash_id

    payload = {}
    for vid in vertex_ids:
        vdata = await hg.get_vertex(vid) or {}
        description = vdata.get("description", "") or ""
        key = compute_mdhash_id(vid, prefix="ent-")
        payload[key] = {
            "content": vid + description,
            "entity_name": vid,
        }

    print(f"[repair] re-embedding and upserting {len(payload)} entities…")
    await rag.entities_vdb.upsert(payload)
    await rag.entities_vdb.index_done_callback()
    print(f"[repair] done — vdb_entities.json rebuilt at {workdir}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--corpus", required=True)
    ap.add_argument("--namespace", default=None)
    args = ap.parse_args()
    namespace = args.namespace or args.corpus
    asyncio.run(repair(args.corpus, namespace))


if __name__ == "__main__":
    main()
