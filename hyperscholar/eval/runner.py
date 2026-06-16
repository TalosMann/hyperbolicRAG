r"""eval/runner.py

Runs every generated question through both backends, capturing answers and
retrieval provenance. Uses file-backed storage to read from persisted indexes.

Usage
-----
    cd D:\Projects\hyperbolic
    python -m hyperscholar.eval.runner --corpus demo
    python -m hyperscholar.eval.runner --corpus neurology
"""
from __future__ import annotations

import argparse
import asyncio
import json
from pathlib import Path


async def run_corpus(corpus: str, namespace: str, results_dir: Path,
                     top_k: int = 60) -> Path:
    from hyperscholar.core.config import load_config
    from hyperscholar.core.embedder import build_embedder
    from hyperscholar.core.llm import build_llm_func
    from hyperscholar.rag.hyperrag_backend import HyperRAGBackend
    from hyperscholar.rag.hierarchical_backend import HierarchicalRAGBackend
    from hyperscholar.eval.provenance import (
        hyperrag_query_with_provenance,
        hierarchical_query_with_provenance,
    )
    from hyperrag.storage import JsonKVStorage, NanoVectorDBStorage, HypergraphStorage

    cfg = load_config()
    embedder = build_embedder(cfg.embedding)
    llm = build_llm_func(cfg.llm)

    hyper = HyperRAGBackend(
        llm_func=llm, embedder=embedder, working_dir=cfg.working_dir,
        kv_cls=JsonKVStorage,
        vector_cls=NanoVectorDBStorage,
        hypergraph_cls=HypergraphStorage,
        pg_dsn=None, fail_markers=cfg.rag.fail_markers)

    hier = HierarchicalRAGBackend(
        llm_func=llm, embedder=embedder,
        kv_cls=JsonKVStorage,
        vector_cls=NanoVectorDBStorage,
        pg_dsn=None, fail_markers=cfg.rag.fail_markers)

    q_path = results_dir / corpus / "questions.json"
    if not q_path.exists():
        raise FileNotFoundError(
            f"{q_path} not found. Run question_generator first.")
    qdata = json.loads(q_path.read_text(encoding="utf-8"))
    questions = qdata["questions"]

    out = {"corpus": corpus, "namespace": namespace, "results": []}
    for q in questions:
        qid, text = q["id"], q["question"]
        print(f"  Q{qid}: {text[:70]}")

        hyper_res = await hyperrag_query_with_provenance(
            hyper, namespace, text, top_k=top_k)
        print(f"    hyperrag     ok={hyper_res['ok']} "
              f"ents={hyper_res['provenance'].get('counts', {}).get('entities', 0)} "
              f"edges={hyper_res['provenance'].get('counts', {}).get('hyperedges', 0)}")

        hier_res = await hierarchical_query_with_provenance(
            hier, namespace, text, top_k=top_k)
        print(f"    hierarchical ok={hier_res['ok']} "
              f"nodes={hier_res['provenance'].get('counts', {}).get('tree_nodes', 0)} "
              f"levels={hier_res['provenance'].get('levels_accessed', [])}")

        out["results"].append({
            "id": qid,
            "question": text,
            "source_chunk_id": q["source_chunk_id"],
            "hyperrag": hyper_res,
            "hierarchical": hier_res,
        })

    out_path = results_dir / corpus / "answers.json"
    out_path.write_text(json.dumps(out, indent=2, ensure_ascii=False),
                        encoding="utf-8")
    print(f"\n✓ {len(out['results'])} answered → {out_path}")
    return out_path


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--corpus", required=True)
    ap.add_argument("--namespace", default=None)
    ap.add_argument("--top-k", type=int, default=60)
    ap.add_argument("--results-dir",
                    default=str(Path(__file__).resolve().parent / "results"))
    args = ap.parse_args()
    namespace = args.namespace or args.corpus
    asyncio.run(run_corpus(args.corpus, namespace,
                           Path(args.results_dir), args.top_k))


if __name__ == "__main__":
    main()
