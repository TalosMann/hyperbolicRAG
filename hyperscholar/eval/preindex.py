r"""eval/preindex.py

Headless corpus indexing — no GUI. Indexes into both backends under a named
namespace using file-backed storage (JsonKVStorage, NanoVectorDBStorage,
HypergraphStorage) so data persists between processes.

Run overnight for large corpora with device:auto on a CUDA GPU.

Usage
-----
    cd D:\Projects\hyperbolic
    python -m hyperscholar.eval.preindex --corpus demo --file demo_small.jsonl --backend both
    python -m hyperscholar.eval.preindex --corpus neurology --file D:\Datasets\neurology\neurology.jsonl --backend both
"""
from __future__ import annotations

import argparse
import asyncio
import time
from pathlib import Path


async def preindex(corpus: str, namespace: str, file: str, backend: str) -> None:
    from hyperscholar.core.config import load_config
    from hyperscholar.core.embedder import build_embedder
    from hyperscholar.core.llm import build_llm_func
    from hyperscholar.ingestion import load_corpus, corpus_summary
    from hyperscholar.rag.hyperrag_backend import HyperRAGBackend
    from hyperscholar.rag.hierarchical_backend import HierarchicalRAGBackend
    from hyperrag.storage import JsonKVStorage, NanoVectorDBStorage, HypergraphStorage

    cfg = load_config()
    print(f"[config] working_dir: {cfg.working_dir}")
    print(f"[config] embedding device: {cfg.embedding.device}  model: {cfg.embedding.model}")
    print(f"[config] llm providers: {[p.name for p in cfg.llm.providers]}")

    embedder = build_embedder(cfg.embedding)
    llm = build_llm_func(cfg.llm)

    docs = load_corpus(file)
    print(f"[corpus] {corpus_summary(docs)}")
    if not docs:
        raise RuntimeError(f"No documents loaded from {file}")

    backends = []
    if backend in ("hyperrag", "both"):
        backends.append(("hyperrag", HyperRAGBackend(
            llm_func=llm, embedder=embedder, working_dir=cfg.working_dir,
            kv_cls=JsonKVStorage,
            vector_cls=NanoVectorDBStorage,
            hypergraph_cls=HypergraphStorage,
            pg_dsn=None, fail_markers=cfg.rag.fail_markers)))
    if backend in ("hierarchical", "both"):
        backends.append(("hierarchical", HierarchicalRAGBackend(
            llm_func=llm, embedder=embedder,
            working_dir=cfg.working_dir,
            kv_cls=JsonKVStorage,
            vector_cls=NanoVectorDBStorage,
            pg_dsn=None, fail_markers=cfg.rag.fail_markers)))

    for name, b in backends:
        print(f"\n[index] {name} → namespace '{namespace}' ({len(docs)} docs)…")
        t0 = time.time()
        result = await b.index(namespace, docs)
        dt = time.time() - t0
        print(f"[index] {name} done — {result.chunks} chunks in {dt:.1f}s "
              f"({result.detail})")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--corpus", required=True)
    ap.add_argument("--namespace", default=None)
    ap.add_argument("--file", required=True)
    ap.add_argument("--backend", default="both",
                    choices=["hyperrag", "hierarchical", "both"])
    args = ap.parse_args()
    namespace = args.namespace or args.corpus
    asyncio.run(preindex(args.corpus, namespace, args.file, args.backend))


if __name__ == "__main__":
    main()
