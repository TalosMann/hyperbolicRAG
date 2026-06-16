"""Compare RAG backends on identical content — the Phase 1 deliverable.

Indexes a corpus once per backend family (HyperRAG + HyperRAG-light share one
index; HierarchicalRAG builds its tree over the same shared chunks), then runs
the same queries through every backend and prints results side by side.

Usage (on your Mac, with the real stack):

    export DEEPSEEK_API_KEY=...
    python -m hyperscholar.scripts.compare_backends \
        --corpus path/to/notes.txt path/to/paper.txt \
        --query "What is adaptive instance normalization?" \
        --namespace inst_demo

    # offline smoke run (hash embedder + stub LLM, no network):
    python -m hyperscholar.scripts.compare_backends --offline \
        --query "How do plants convert sunlight into energy?"

Config (config.yaml) controls the store DSN: memory:// or postgresql://...
"""
from __future__ import annotations

import argparse
import asyncio
import sys
from pathlib import Path

from ..core.config import load_config
from ..core.embedder import HashEmbedder, build_embedder
from ..core.types import Document
from ..rag.factory import BACKENDS, build_backend

OFFLINE_CORPUS = [
    Document(content=(
        "Photosynthesis is the process by which green plants convert sunlight "
        "into chemical energy. Chlorophyll inside chloroplasts absorbs light, "
        "driving the conversion of carbon dioxide and water into glucose."),
        title="Photosynthesis"),
    Document(content=(
        "Cellular respiration releases energy stored in glucose. It occurs in "
        "the mitochondria and produces ATP, the cell's energy currency."),
        title="Respiration"),
]


async def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--corpus", nargs="*", default=[], help="text files to index")
    ap.add_argument("--query", action="append", required=True)
    ap.add_argument("--namespace", default="inst_demo")
    ap.add_argument("--backends", nargs="*", default=list(BACKENDS))
    ap.add_argument("--offline", action="store_true",
                    help="hash embedder + stub LLM (CI / smoke test)")
    args = ap.parse_args()

    cfg = load_config()
    if args.offline:
        from ..tests.hyperrag_stub import HyperRAGStubLLM
        embedder = HashEmbedder(dim=128)
        llm = HyperRAGStubLLM()
    else:
        embedder = build_embedder(cfg.embedding)
        from ..core.llm import make_deepseek_complete
        llm = make_deepseek_complete(cfg.llm)

    docs = ([Document(content=Path(p).read_text(), title=Path(p).name)
             for p in args.corpus] if args.corpus else OFFLINE_CORPUS)

    indexed_families = set()
    results = {}
    for name in args.backends:
        cfg.rag.backend = name                      # ← the one-line swap
        be = build_backend(cfg, llm_func=llm, embedder=embedder)
        if args.offline:
            be._cosine_threshold = -1.0             # permissive for hash embedder
            if hasattr(be, "_instances"):
                be._instances.clear()

        family = "hyperrag" if name.startswith("hyperrag") else name
        if family not in indexed_families:          # hyper + light share one index
            print(f"\n=== indexing [{family}] ns={args.namespace} "
                  f"({len(docs)} docs) ===")
            ir = await be.index(args.namespace, docs)
            print(f"    chunks={ir.chunks} detail={ir.detail}")
            indexed_families.add(family)

        for q in args.query:
            r = await be.query(args.namespace, q)
            results.setdefault(q, {})[name] = r

    for q, by_backend in results.items():
        print(f"\n{'='*72}\nQUERY: {q}\n{'='*72}")
        for name, r in by_backend.items():
            print(f"\n--- {name}  (ok={r.ok}, mode={r.mode}, "
                  f"sources={len(r.sources)}) ---")
            print(r.answer[:600])


if __name__ == "__main__":
    sys.exit(asyncio.run(main()))
