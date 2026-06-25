r"""eval/run_all.py

One-command orchestrator for the full evaluation pipeline on an
ALREADY-INDEXED corpus.

Two ways to specify questions:

  --n + --domain          single style (default "fact"), backward compatible
  --styles "fact:15,relational:10,synthesis:10,overview:5,negative:5"
                           multi-style mix — recommended, since a single
                           aggregate score is dominated by whichever style
                           you happen to generate. See question_generator.py
                           for what each style tests.

Pipeline:
    1. corpus_export   (hyperrag + hierarchical structure dumps)
    2. question_gen    (one or more styles, appended into questions.json)
    3. runner          (answers + provenance from both backends)
    4. judge           (LLM-as-judge for scored styles; refusal-check for negative)
    5. report          (markdown comparison, broken down by style)

Usage
-----
    cd D:\Projects\hyperbolic
    python -m hyperscholar.eval.run_all --corpus demo --n 5
    python -m hyperscholar.eval.run_all --corpus neurology --domain medicine ^
        --styles "fact:15,relational:10,synthesis:10,overview:5,negative:5"

Prereq: corpus is indexed under namespace == corpus name:
    python -m hyperscholar.eval.preindex --corpus neurology --file <path> --backend both
"""
from __future__ import annotations

import argparse
import asyncio
from pathlib import Path

RESULTS = Path(__file__).resolve().parent / "results"


def _parse_styles(spec: str) -> list[tuple[str, int]]:
    out = []
    for part in spec.split(","):
        part = part.strip()
        if not part:
            continue
        name, _, count = part.partition(":")
        out.append((name.strip(), int(count.strip()) if count.strip() else 10))
    return out


async def run_all(corpus: str, namespace: str, n: int, domain: str, top_k: int,
                  styles: str | None) -> None:
    from hyperscholar.eval.corpus_export import export_hyperrag, export_hierarchical
    from hyperscholar.eval.question_generator import generate_questions
    from hyperscholar.eval.runner import run_corpus
    from hyperscholar.eval.judge import judge_corpus
    from hyperscholar.eval.report import build_report

    print("=" * 60)
    print(f"EVAL PIPELINE - {corpus} (namespace={namespace})")
    print("=" * 60)

    print("\n[1/5] Exporting corpus structures…")
    await export_hyperrag(corpus, namespace, RESULTS)
    await export_hierarchical(corpus, namespace, RESULTS)

    print("\n[2/5] Generating questions…")
    if styles:
        style_plan = _parse_styles(styles)
        print(f"  style plan: {style_plan}")
        for style_name, count in style_plan:
            print(f"\n  -- style: {style_name} (n={count}) --")
            await generate_questions(corpus, count, namespace, RESULTS,
                                     style=style_name, domain=domain)
    else:
        await generate_questions(corpus, n, namespace, RESULTS,
                                 style="fact", domain=domain)

    print("\n[3/5] Running both backends…")
    await run_corpus(corpus, namespace, RESULTS, top_k=top_k)

    print("\n[4/5] Judging…")
    await judge_corpus(corpus, RESULTS)

    print("\n[5/5] Building report…")
    build_report(RESULTS, [corpus])

    print("\n* pipeline complete. See:")
    print(f"   {RESULTS / corpus / 'eval_results.json'}")
    print(f"   {RESULTS / 'eval_report.md'}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--corpus", required=True)
    ap.add_argument("--namespace", default=None)
    ap.add_argument("--n", type=int, default=10,
                    help="used only if --styles is not given (single 'fact' style)")
    ap.add_argument("--domain", default="academic")
    ap.add_argument("--top-k", type=int, default=60)
    ap.add_argument("--styles", default=None,
                    help='e.g. "fact:15,relational:10,synthesis:10,overview:5,negative:5". '
                         'If given, overrides --n and generates a mixed-style set.')
    args = ap.parse_args()
    namespace = args.namespace or args.corpus
    asyncio.run(run_all(args.corpus, namespace, args.n, args.domain,
                       args.top_k, args.styles))


if __name__ == "__main__":
    main()
