r"""eval/run_all.py

One-command orchestrator: runs the full evaluation pipeline for a corpus that
has ALREADY been indexed (use preindex.py first).

Pipeline:
    1. corpus_export   (hyperrag + hierarchical structure dumps)
    2. question_gen    (N questions anchored to chunks)
    3. runner          (answers + provenance from both backends)
    4. judge           (LLM-as-judge, 5 metrics)
    5. report          (markdown comparison)

Usage
-----
    cd D:\Projects\hyperbolic
    python -m hyperscholar.eval.run_all --corpus demo --n 5
    python -m hyperscholar.eval.run_all --corpus neurology --domain medicine --n 50

Prereq: corpus is indexed under namespace == corpus name:
    python -m hyperscholar.eval.preindex --corpus neurology --file <path> --backend both
r"""
from __future__ import annotations

import argparse
import asyncio
from pathlib import Path

RESULTS = Path(__file__).resolve().parent / "results"


async def run_all(corpus: str, namespace: str, n: int, domain: str, top_k: int):
    from hyperscholar.eval.corpus_export import export_hyperrag, export_hierarchical
    from hyperscholar.eval.question_generator import generate_questions
    from hyperscholar.eval.runner import run_corpus
    from hyperscholar.eval.judge import judge_corpus
    from hyperscholar.eval.report import build_report

    print("=" * 60)
    print(f"EVAL PIPELINE — {corpus} (namespace={namespace})")
    print("=" * 60)

    print("\n[1/5] Exporting corpus structures…")
    await export_hyperrag(corpus, namespace, RESULTS)
    await export_hierarchical(corpus, namespace, RESULTS)

    print("\n[2/5] Generating questions…")
    await generate_questions(corpus, n, namespace, RESULTS, domain=domain)

    print("\n[3/5] Running both backends…")
    await run_corpus(corpus, namespace, RESULTS, top_k=top_k)

    print("\n[4/5] Judging…")
    await judge_corpus(corpus, RESULTS)

    print("\n[5/5] Building report…")
    build_report(RESULTS, [corpus])

    print("\n✓ pipeline complete. See:")
    print(f"   {RESULTS / corpus / 'eval_results.json'}")
    print(f"   {RESULTS / 'eval_report.md'}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--corpus", required=True)
    ap.add_argument("--namespace", default=None)
    ap.add_argument("--n", type=int, default=50)
    ap.add_argument("--domain", default="academic")
    ap.add_argument("--top-k", type=int, default=60)
    args = ap.parse_args()
    namespace = args.namespace or args.corpus
    asyncio.run(run_all(args.corpus, namespace, args.n, args.domain, args.top_k))


if __name__ == "__main__":
    main()
