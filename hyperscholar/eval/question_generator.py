r"""eval/question_generator.py

Generates N evaluation questions per corpus, anchored to specific source
chunks — matching the iMoonLab Hyper-RAG paper's protocol.

Reads chunks from the file-backed JsonKVStorage written by preindex.py.

Usage
-----
    cd D:\Projects\hyperbolic
    python -m hyperscholar.eval.question_generator --corpus demo --n 5
    python -m hyperscholar.eval.question_generator --corpus neurology --domain medicine --n 50
"""
from __future__ import annotations

import argparse
import asyncio
import json
import os
import random
from pathlib import Path

QUESTION_PROMPT = """You are creating an exam question to test understanding of a specific passage from a {domain} corpus.

Read the passage below and write ONE clear, self-contained question whose answer is found in the passage. The question must:
- be answerable from the passage alone
- not reference "the passage" or "the text" (ask about the subject matter directly)
- require understanding, not just keyword matching
- be a single sentence

PASSAGE:
{passage}

Output ONLY the question, nothing else."""


async def generate_questions(corpus: str, n: int, namespace: str,
                             results_dir: Path, domain: str = "academic",
                             seed: int = 42) -> Path:
    from hyperscholar.core.config import load_config
    from hyperscholar.core.llm import build_llm_func
    from hyperrag.storage import JsonKVStorage

    cfg = load_config()
    llm = build_llm_func(cfg.llm)

    # Read chunks from the file-backed store written by preindex.py
    workdir = os.path.join(cfg.working_dir, "hyperrag", namespace)
    gcfg = {"working_dir": workdir, "addon_params": {}, "embedding_batch_num": 8}
    chunks = JsonKVStorage(namespace="text_chunks", global_config=gcfg)

    chunk_ids = await chunks.all_keys()
    if not chunk_ids:
        raise RuntimeError(
            f"No indexed chunks for namespace '{namespace}'. "
            f"Run preindex.py first.\n"
            f"Expected store at: {workdir}")

    rng = random.Random(seed)
    sample_ids = rng.sample(chunk_ids, min(n, len(chunk_ids)))
    rows = await chunks.get_by_ids(sample_ids)

    questions = []
    for i, (cid, row) in enumerate(zip(sample_ids, rows)):
        content = (row or {}).get("content", "")
        if not content.strip():
            continue
        prompt = QUESTION_PROMPT.format(domain=domain, passage=content[:2000])
        q = await llm(prompt)
        q = (q or "").strip().strip('"')
        if q:
            questions.append({
                "id": i + 1,
                "question": q,
                "source_chunk_id": cid,
                "source_excerpt": content[:500],
            })
        print(f"  [{len(questions)}/{n}] {q[:80]}")

    out_dir = results_dir / corpus
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / "questions.json"
    out_path.write_text(json.dumps({
        "corpus": corpus,
        "namespace": namespace,
        "domain": domain,
        "n_requested": n,
        "n_generated": len(questions),
        "questions": questions,
    }, indent=2, ensure_ascii=False), encoding="utf-8")
    print(f"\n✓ {len(questions)} questions → {out_path}")
    return out_path


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--corpus", required=True)
    ap.add_argument("--namespace", default=None)
    ap.add_argument("--n", type=int, default=50)
    ap.add_argument("--domain", default="academic")
    ap.add_argument("--results-dir",
                    default=str(Path(__file__).resolve().parent / "results"))
    ap.add_argument("--seed", type=int, default=42)
    args = ap.parse_args()
    namespace = args.namespace or args.corpus
    asyncio.run(generate_questions(
        corpus=args.corpus, n=args.n, namespace=namespace,
        results_dir=Path(args.results_dir), domain=args.domain, seed=args.seed))


if __name__ == "__main__":
    main()
