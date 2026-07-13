r"""eval/eval_pure_cograg_qa.py

Direct QA evaluation for pure_cograg against a labeled QA jsonl (like
agriculture_20.jsonl) that already ships real gold answers per question.

Unlike judge.py — which does blind pairwise comparison between hyperrag and
hierarchical and has no concept of pure_cograg or of a ground-truth answer —
this scores pure_cograg's actual answer against each row's own "answers"
field directly. No question_generator/corpus_export step needed.

Saves after every question (checkpoint) — safe to interrupt and resume.

Usage
-----
    python -m hyperscholar.eval.eval_pure_cograg_qa \
        --file hyperscholar/agriculture_5.jsonl --namespace agriculture_5
"""
from __future__ import annotations

import argparse
import asyncio
import json
import re
from pathlib import Path

QA_JUDGE_PROMPT = """You are evaluating whether a system's answer correctly addresses a question, compared to a known correct reference answer.

QUESTION:
{question}

REFERENCE ANSWER (ground truth):
{reference}

SYSTEM ANSWER:
{answer}

Score the system answer from 1 (completely wrong/irrelevant) to 10 (fully correct and complete, matching the reference). Also classify it as one of: "correct", "partially_correct", "incorrect".

Respond with ONLY a JSON object in exactly this form, no other text:
{{"score": int, "verdict": "correct"|"partially_correct"|"incorrect", "rationale": "one sentence"}}"""


def _parse_verdict(raw: str) -> dict | None:
    if not raw:
        return None
    cleaned = raw.replace("```json", "").replace("```", "").strip()
    m = re.search(r"\{.*\}", cleaned, re.DOTALL)
    if not m:
        return None
    try:
        obj = json.loads(m.group(0))
    except json.JSONDecodeError:
        return None
    if "score" not in obj or "verdict" not in obj:
        return None
    return obj


def _load_qa_rows(file: str) -> list[dict]:
    rows = []
    with open(file, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    return rows


async def evaluate(file: str, namespace: str, results_dir: Path,
                   top_k: int = 60) -> Path:
    from hyperscholar.core.config import load_config
    from hyperscholar.core.embedder import build_embedder
    from hyperscholar.core.llm import build_llm_func
    from hyperscholar.rag.factory import storage_classes
    from hyperscholar.rag.pure_cograg_backend import PureCogRAGBackend

    cfg = load_config()
    kv_cls, vector_cls, _hg_cls, pg_dsn = storage_classes(cfg)
    embedder = build_embedder(cfg.embedding)
    llm = build_llm_func(cfg.llm)

    backend = PureCogRAGBackend(
        llm_func=llm, embedder=embedder, working_dir=cfg.working_dir,
        kv_cls=kv_cls, vector_cls=vector_cls,
        pg_dsn=pg_dsn, fail_markers=cfg.rag.fail_markers)

    rows = _load_qa_rows(file)
    print(f"[qa-eval] {len(rows)} questions from {file}, namespace='{namespace}'")

    out_dir = results_dir / namespace
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / "pure_cograg_qa.json"

    if out_path.exists():
        existing = json.loads(out_path.read_text(encoding="utf-8"))
        results = existing.get("results", [])
        done = {r["id"] for r in results}
        print(f"  resuming — {len(done)} already evaluated")
    else:
        results = []
        done = set()

    for i, row in enumerate(rows):
        if i in done:
            continue
        question = row["input"]
        reference = row["answers"][0] if isinstance(row.get("answers"), list) else row.get("answers", "")
        print(f"  Q{i}: {question[:70]}")

        try:
            qres = await backend.query(namespace, question, top_k=top_k)
            answer = qres.answer
        except Exception as e:
            print(f"    query ERROR: {e}")
            results.append({"id": i, "question": question, "reference": reference,
                            "answer": "", "ok": False, "score": None,
                            "verdict": "error", "rationale": str(e)})
            out_path.write_text(json.dumps({"results": results}, indent=2,
                                           ensure_ascii=False), encoding="utf-8")
            continue

        try:
            reply = await llm(QA_JUDGE_PROMPT.format(
                question=question, reference=reference, answer=answer))
            parsed = _parse_verdict(reply)
        except Exception as e:
            print(f"    judge ERROR: {e}")
            parsed = None

        entry = {
            "id": i, "question": question, "reference": reference,
            "answer": answer, "ok": qres.ok,
            "score": parsed["score"] if parsed else None,
            "verdict": parsed["verdict"] if parsed else "judge_failed",
            "rationale": parsed.get("rationale", "") if parsed else "",
        }
        print(f"    ok={qres.ok} score={entry['score']} verdict={entry['verdict']}")
        results.append(entry)
        out_path.write_text(json.dumps({"results": results}, indent=2,
                                       ensure_ascii=False), encoding="utf-8")

    scored = [r["score"] for r in results if isinstance(r.get("score"), (int, float))]
    avg = round(sum(scored) / len(scored), 2) if scored else None
    verdicts = {}
    for r in results:
        verdicts[r["verdict"]] = verdicts.get(r["verdict"], 0) + 1
    summary = {"n": len(results), "avg_score": avg, "verdicts": verdicts}
    out_path.write_text(json.dumps({"results": results, "summary": summary},
                                   indent=2, ensure_ascii=False), encoding="utf-8")
    print(f"\n✓ {len(results)} questions evaluated → {out_path}")
    print(f"  avg_score={avg}  verdicts={verdicts}")
    return out_path


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--file", required=True)
    ap.add_argument("--namespace", required=True)
    ap.add_argument("--top-k", type=int, default=60)
    ap.add_argument("--results-dir",
                    default=str(Path(__file__).resolve().parent / "results"))
    args = ap.parse_args()
    asyncio.run(evaluate(args.file, args.namespace, Path(args.results_dir), args.top_k))


if __name__ == "__main__":
    main()
