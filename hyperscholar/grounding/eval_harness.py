r"""Labeled-eval harness for the grounding layer.

Scores serve/decline decisions against labels:
  should_serve   (answerable)  -> answered = good, declined = OVER-REFUSAL
  should_decline (negative)    -> declined = good, served fabrication = LEAK
                                  (a serve that says "not in the source" is an
                                   honest non-answer, NOT a leak)

Labeled set is built once and cached so before/after runs use the same questions:
  answerable : reused 'fact' questions from eval/results/<corpus>/questions.json
  negative   : freshly generated, each verified absent from its source chunk

Usage:
    python -m hyperscholar.grounding.eval_harness --corpus christmas_carol --backend cograg_official
"""
from __future__ import annotations

import argparse
import asyncio
import json
import random
from pathlib import Path

from hyperscholar import grounding

RESULTS = Path(__file__).resolve().parent.parent / "eval" / "results"
ABSENCE = ["no_answer_in_source", "does not", "do not", "doesn't", "don't", "not provided",
           "not stated", "not mention", "not given", "not specify", "not contain",
           "no specific", "cannot be determined", "not answer", "does not appear",
           "not explicitly", "no information"]


def _honest_nonanswer(text: str) -> bool:
    t = text.lower()
    return any(m in t for m in ABSENCE)


async def _build_backend(cfg, namespace, query_mode):
    from hyperscholar.core.embedder import build_embedder
    from hyperscholar.core.llm import build_llm_func
    from hyperscholar.rag.cograg_official_backend import CogRAGOfficialBackend
    embedder = build_embedder(cfg.embedding)
    llm = build_llm_func(cfg.llm)
    be = CogRAGOfficialBackend(llm_func=llm, embedder=embedder,
                               working_dir=cfg.working_dir, query_mode=query_mode,
                               fail_markers=cfg.rag.fail_markers)
    return be, be._rag(namespace), llm


async def _gen_negatives(llm, rag, n, domain, seed=123):
    from hyperscholar.eval.question_generator import NEGATIVE_PROMPT, NEGATIVE_VERIFY_PROMPT
    ids = await rag.text_chunks.all_keys()
    rng = random.Random(seed)
    sample = rng.sample(ids, min(n * 3, len(ids)))
    rows = await rag.text_chunks.get_by_ids(sample)
    out = []
    for row in rows:
        if len(out) >= n:
            break
        content = (row or {}).get("content", "")
        if not content.strip():
            continue
        passage = content[:2000]
        for _ in range(3):
            cand = (await llm(NEGATIVE_PROMPT.format(domain=domain, passage=passage))).strip().strip('"')
            if not cand:
                continue
            verdict = (await llm(NEGATIVE_VERIFY_PROMPT.format(passage=passage, question=cand))).strip().upper()
            if not verdict.startswith("Y"):
                out.append({"question": cand, "label": "should_decline"})
                break
    return out


async def _gen_facts(llm, rag, n, domain, seed=77):
    """Generate answerable fact questions from the backend's own chunks — used
    when a corpus has no pre-existing eval questions.json."""
    from hyperscholar.eval.question_generator import FACT_PROMPT
    ids = await rag.text_chunks.all_keys()
    rng = random.Random(seed)
    sample = rng.sample(ids, min(n, len(ids)))
    rows = await rag.text_chunks.get_by_ids(sample)
    out = []
    for row in rows:
        content = (row or {}).get("content", "")
        if not content.strip():
            continue
        q = (await llm(FACT_PROMPT.format(domain=domain, passage=content[:2000]))).strip().strip('"')
        if q:
            out.append({"question": q, "label": "should_serve"})
    return out


async def _labeled_set(corpus, llm, rag, n_fact, n_neg, domain):
    cache = RESULTS / corpus / "grounding_labeled_set.json"
    if cache.exists():
        return json.loads(cache.read_text(encoding="utf-8"))
    qpath = RESULTS / corpus / "questions.json"
    if qpath.exists():
        qs = json.loads(qpath.read_text(encoding="utf-8"))
        answerable = [{"question": q["question"], "label": "should_serve"}
                      for q in qs["questions"] if q.get("style") == "fact"]
    else:
        print(f"  no questions.json for {corpus}; generating {n_fact} fact questions from chunks")
        answerable = await _gen_facts(llm, rag, n_fact, domain)
    negatives = await _gen_negatives(llm, rag, n_neg, domain)
    dataset = answerable + negatives
    cache.parent.mkdir(parents=True, exist_ok=True)
    cache.write_text(json.dumps(dataset, indent=2, ensure_ascii=False), encoding="utf-8")
    return dataset


async def run(corpus, namespace, query_mode, n_fact, n_neg, k, domain):
    from hyperscholar.core.config import load_config
    cfg = load_config()
    be, rag, llm = await _build_backend(cfg, namespace, query_mode)
    dataset = await _labeled_set(corpus, llm, rag, n_fact, n_neg, domain)
    print(f"labeled set: {sum(d['label']=='should_serve' for d in dataset)} should_serve + "
          f"{sum(d['label']=='should_decline' for d in dataset)} should_decline")

    cm = {"true_serve": 0, "false_decline": 0, "true_decline": 0,
          "honest_nonanswer": 0, "leak": 0}
    rows = []
    for i, item in enumerate(dataset, 1):
        d = await grounding.grounded_answer(be, rag, llm, namespace, item["question"], k=k)
        served = " ".join(d["clean_supported"] + d["clean_interp"])
        surviving = grounding.flagged_specifics(
            served, grounding.build_source_index(
                await grounding.retrieve_raw_chunks(rag, item["question"], k))) if served else []
        if item["label"] == "should_serve":
            outcome = "true_serve" if d["answered"] else "false_decline"
        else:
            if not d["answered"]:
                outcome = "true_decline"
            elif surviving or (served and not _honest_nonanswer(served)):
                outcome = "leak"
            else:
                outcome = "honest_nonanswer"
        cm[outcome] += 1
        print(f"  Q{i:2d} [{item['label']:14s}] {d['decision']:17s} "
              f"reason={d.get('reason',''):16s} -> {outcome}")
        rows.append({**item, "decision": d["decision"], "reason": d.get("reason", ""),
                     "outcome": outcome, "served": served[:400]})

    n_serve = cm["true_serve"] + cm["false_decline"]
    n_decl = cm["true_decline"] + cm["honest_nonanswer"] + cm["leak"]
    report = {"backend": query_mode, "confusion": cm,
              "over_refusal_rate": round(cm["false_decline"] / n_serve, 3) if n_serve else None,
              "fabrication_leak_rate": round(cm["leak"] / n_decl, 3) if n_decl else None,
              "serve_recall": round(cm["true_serve"] / n_serve, 3) if n_serve else None,
              "rows": rows}
    out = RESULTS / corpus / "grounding_eval_results.json"
    out.write_text(json.dumps(report, indent=2, ensure_ascii=False), encoding="utf-8")

    print("\n" + "=" * 56)
    print(f"GROUNDING LAYER — {corpus} / {query_mode}")
    print("=" * 56)
    print(f"  should_serve   : true_serve={cm['true_serve']}  false_decline={cm['false_decline']}")
    print(f"  should_decline : true_decline={cm['true_decline']}  "
          f"honest_nonanswer={cm['honest_nonanswer']}  leak={cm['leak']}")
    print(f"\n  over-refusal rate     : {report['over_refusal_rate']}")
    print(f"  fabrication-leak rate : {report['fabrication_leak_rate']}")
    print(f"  serve recall          : {report['serve_recall']}")
    print(f"\n  -> {out}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--corpus", default="christmas_carol")
    ap.add_argument("--namespace", default=None)
    ap.add_argument("--backend", default="cograg_official", help="query mode / backend id")
    ap.add_argument("--n-fact", type=int, default=10,
                    help="answerable questions to generate if no questions.json exists")
    ap.add_argument("--n-neg", type=int, default=10)
    ap.add_argument("--k", type=int, default=24)
    ap.add_argument("--domain", default="academic")
    args = ap.parse_args()
    mode = "cog-entity" if args.backend == "cograg_official" else args.backend
    asyncio.run(run(args.corpus, args.namespace or args.corpus, mode,
                    args.n_fact, args.n_neg, args.k, args.domain))


if __name__ == "__main__":
    main()
