r"""eval/runner.py

Runs every generated question through both backends, capturing answers and
retrieval provenance.

Features:
- Saves after every question (checkpoint) — safe to interrupt and resume
- Resume: skips questions already in answers.json on restart
- Uses file-backed storage to read from persisted indexes

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
        working_dir=cfg.working_dir,
        kv_cls=JsonKVStorage,
        vector_cls=NanoVectorDBStorage,
        pg_dsn=None, fail_markers=cfg.rag.fail_markers)

    q_path = results_dir / corpus / "questions.json"
    if not q_path.exists():
        raise FileNotFoundError(
            f"{q_path} not found. Run question_generator first.")
    qdata = json.loads(q_path.read_text(encoding="utf-8"))
    questions = qdata["questions"]

    out_path = results_dir / corpus / "answers.json"

    # Resume: load already-completed results if file exists
    if out_path.exists():
        existing = json.loads(out_path.read_text(encoding="utf-8"))
        done_ids = {r["id"] for r in existing.get("results", [])}
        out = existing
        print(f"  resuming — {len(done_ids)} already answered, "
              f"{len(questions) - len(done_ids)} remaining")
    else:
        done_ids = set()
        out = {"corpus": corpus, "namespace": namespace, "results": []}

    for q in questions:
        qid, text = q["id"], q["question"]
        if qid in done_ids:
            continue

        print(f"  Q{qid}: {text[:70]}")

        try:
            hyper_res = await hyperrag_query_with_provenance(
                hyper, namespace, text, top_k=top_k)
        except Exception as e:
            print(f"    hyperrag ERROR: {e}")
            hyper_res = {"answer": "", "ok": False,
                         "provenance": {"entities": [], "hyperedges": [],
                                        "text_units": [], "error": str(e)}}

        try:
            hier_res = await hierarchical_query_with_provenance(
                hier, namespace, text, top_k=top_k)
        except Exception as e:
            print(f"    hierarchical ERROR: {e}")
            hier_res = {"answer": "", "ok": False,
                        "provenance": {"nodes_accessed": [], "chunks_accessed": [],
                                       "levels_accessed": [], "error": str(e)}}

        print(f"    hyperrag     ok={hyper_res['ok']} "
              f"ents={hyper_res['provenance'].get('counts', {}).get('entities', 0)} "
              f"edges={hyper_res['provenance'].get('counts', {}).get('hyperedges', 0)}")
        print(f"    hierarchical ok={hier_res['ok']} "
              f"nodes={hier_res['provenance'].get('counts', {}).get('tree_nodes', 0)} "
              f"levels={hier_res['provenance'].get('levels_accessed', [])}")

        # Carry through every source_* field generically (source_chunk_id,
        # source_hyperedge_id, source_entities, source_topic, source_degree,
        # etc.) so downstream tools like fact_check.py have whatever
        # ground-truth anchor this question's style provides — without
        # hardcoding field names here that would silently drop new ones.
        source_fields = {k: v for k, v in q.items() if k.startswith("source_")}

        out["results"].append({
            "id": qid,
            "style": q.get("style", "fact"),
            "question": text,
            **source_fields,
            "hyperrag": hyper_res,
            "hierarchical": hier_res,
        })

        # Checkpoint — write after every question
        out_path.write_text(json.dumps(out, indent=2, ensure_ascii=False),
                            encoding="utf-8")

    total = len(out["results"])
    print(f"\n✓ {total} answered → {out_path}")
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
