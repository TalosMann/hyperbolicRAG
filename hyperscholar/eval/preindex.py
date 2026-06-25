r"""eval/preindex.py

Headless corpus indexing — no GUI. Indexes into both backends under a named
namespace using file-backed storage (JsonKVStorage, NanoVectorDBStorage,
HypergraphStorage) so data persists between processes.

BATCHED INDEXING (important): HyperRAG's entity extraction runs one
asyncio.gather() per ainsert() call across ALL chunks in that call. If any
single chunk's LLM call fails (rate limit, insufficient balance, transient
network error), the ENTIRE gather aborts and none of that call's extraction
work is merged into the hypergraph — even chunks that already succeeded.
Indexing the whole corpus in one call means one failure anywhere wastes all
prior progress in that call.

To bound the blast radius, documents are indexed in batches (default 200
docs/batch). Each batch is a separate index() call, so a failure only costs
that batch's chunks, not the whole corpus. Already-indexed docs are skipped
automatically by HyperRAG's own dedup (filter_keys on doc/chunk hashes), so
re-running this script after a failure resumes roughly where it left off —
at batch granularity, not chunk granularity.

Usage
-----
    cd D:\Projects\hyperbolic
    python -m hyperscholar.eval.preindex --corpus demo --file demo_small.jsonl --backend both
    python -m hyperscholar.eval.preindex --corpus neurology --file D:\Datasets\neurology\neurology.jsonl --backend both --batch-size 200
"""
from __future__ import annotations

import argparse
import asyncio
import time
from pathlib import Path


async def preindex(corpus: str, namespace: str, file: str, backend: str,
                   batch_size: int = 200) -> None:
    from hyperscholar.core.config import load_config
    from hyperscholar.core.embedder import build_embedder
    from hyperscholar.core.llm import build_llm_func
    from hyperscholar.ingestion import load_corpus, corpus_summary
    from hyperscholar.rag.hyperrag_backend import HyperRAGBackend
    from hyperscholar.rag.hierarchical_backend import HierarchicalRAGBackend
    from hyperscholar.rag.pure_cograg_backend import PureCogRAGBackend
    from hyperscholar.rag.cograg_flash_backend import CogRagFlashBackend
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

    hyperrag_kwargs = {
        "llm_model_max_async": cfg.hyperrag.max_async,
        "entity_extract_max_gleaning": cfg.hyperrag.entity_extract_max_gleaning,
    }
    print(f"[config] hyperrag max_async={cfg.hyperrag.max_async} "
          f"entity_extract_max_gleaning={cfg.hyperrag.entity_extract_max_gleaning}")

    backends = []
    if backend in ("hyperrag", "both", "all"):
        backends.append(("hyperrag", HyperRAGBackend(
            llm_func=llm, embedder=embedder, working_dir=cfg.working_dir,
            kv_cls=JsonKVStorage,
            vector_cls=NanoVectorDBStorage,
            hypergraph_cls=HypergraphStorage,
            pg_dsn=None, fail_markers=cfg.rag.fail_markers,
            hyperrag_kwargs=hyperrag_kwargs)))
    if backend in ("hierarchical", "both", "all"):
        backends.append(("hierarchical", HierarchicalRAGBackend(
            llm_func=llm, embedder=embedder,
            working_dir=cfg.working_dir,
            kv_cls=JsonKVStorage,
            vector_cls=NanoVectorDBStorage,
            pg_dsn=None, fail_markers=cfg.rag.fail_markers)))
    if backend in ("pure_cograg", "all"):
        backends.append(("pure_cograg", PureCogRAGBackend(
            llm_func=llm, embedder=embedder,
            working_dir=cfg.working_dir,
            kv_cls=JsonKVStorage,
            vector_cls=NanoVectorDBStorage,
            pg_dsn=None, fail_markers=cfg.rag.fail_markers)))
    if backend in ("cograg_flash", "all"):
        llm_fast_func = build_llm_func(cfg.llm_fast) if cfg.llm_fast else llm
        backends.append(("cograg_flash", CogRagFlashBackend(
            llm_func=llm, llm_fast_func=llm_fast_func, embedder=embedder,
            working_dir=cfg.working_dir,
            kv_cls=JsonKVStorage,
            vector_cls=NanoVectorDBStorage,
            pg_dsn=None, fail_markers=cfg.rag.fail_markers)))

    n_batches = (len(docs) + batch_size - 1) // batch_size

    for name, b in backends:
        print(f"\n[index] {name} -> namespace '{namespace}' "
              f"({len(docs)} docs in {n_batches} batches of {batch_size})...")
        t0 = time.time()
        total_chunks = 0
        failed_batches = []

        for i in range(0, len(docs), batch_size):
            batch_num = i // batch_size + 1
            batch = docs[i:i + batch_size]
            bt0 = time.time()
            try:
                result = await b.index(namespace, batch)
                total_chunks = result.chunks  # cumulative count from backend
                bdt = time.time() - bt0
                print(f"  [batch {batch_num}/{n_batches}] {name} ok — "
                      f"{result.chunks} total chunks so far ({bdt:.1f}s) "
                      f"detail={result.detail}")
            except Exception as e:
                bdt = time.time() - bt0
                print(f"  [batch {batch_num}/{n_batches}] {name} FAILED after "
                      f"{bdt:.1f}s: {type(e).__name__}: {e}")
                print(f"  [batch {batch_num}/{n_batches}] continuing to next "
                      f"batch — re-run this script later to retry failed batches "
                      f"(already-chunked docs are skipped automatically)")
                failed_batches.append(batch_num)
                continue

        dt = time.time() - t0
        status = "done" if not failed_batches else f"done WITH {len(failed_batches)} FAILED BATCHES {failed_batches}"
        print(f"[index] {name} {status} - {dt:.1f}s total")
        if failed_batches:
            print(f"[index] {name}: re-run the same command to retry. "
                  f"Already-indexed docs are skipped via content-hash dedup, "
                  f"so only the failed batches' docs will be reprocessed "
                  f"(though batch boundaries may shift slightly).")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--corpus", required=True)
    ap.add_argument("--namespace", default=None)
    ap.add_argument("--file", required=True)
    ap.add_argument("--backend", default="all",
                    choices=["hyperrag", "hierarchical", "pure_cograg", "cograg_flash", "both", "all"])
    ap.add_argument("--batch-size", type=int, default=200,
                    help="documents per index() call — bounds blast radius of "
                         "a single LLM failure during entity extraction")
    args = ap.parse_args()
    namespace = args.namespace or args.corpus
    asyncio.run(preindex(args.corpus, namespace, args.file, args.backend,
                        args.batch_size))


if __name__ == "__main__":
    main()
