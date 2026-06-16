r"""eval/judge.py

LLM-as-judge on the five iMoonLab paper metrics, blind and position-randomized.

Usage
-----
    cd D:\Projects\hyperbolic
    python -m hyperscholar.eval.judge --corpus demo
    python -m hyperscholar.eval.judge --corpus neurology
r"""
from __future__ import annotations

import argparse
import asyncio
import json
import random
import re
from pathlib import Path

METRICS = ["comprehensiveness", "diversity", "empowerment", "logical", "readability"]

JUDGE_PROMPT = """You are an impartial expert evaluator comparing two answers to the same question. Score each answer from 1 (poor) to 10 (excellent) on five dimensions:

- Comprehensiveness: covers all relevant aspects of the question
- Diversity: presents varied, rich perspectives and detail
- Empowerment: helps the reader genuinely understand and act on the topic
- Logical: reasoning is sound, coherent, and well-ordered
- Readability: clear, well-structured, easy to follow

QUESTION:
{question}

ANSWER A:
{answer_a}

ANSWER B:
{answer_b}

Respond with ONLY a JSON object in exactly this form, no other text:
{{
  "A": {{"comprehensiveness": int, "diversity": int, "empowerment": int, "logical": int, "readability": int}},
  "B": {{"comprehensiveness": int, "diversity": int, "empowerment": int, "logical": int, "readability": int}}
}}"""


def _parse_scores(raw: str) -> dict | None:
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
    if "A" not in obj or "B" not in obj:
        return None
    for side in ("A", "B"):
        for metric in METRICS:
            if metric not in obj[side]:
                return None
    return obj


def _mean(scores: dict) -> float:
    return round(sum(scores[m] for m in METRICS) / len(METRICS), 2)


async def judge_corpus(corpus: str, results_dir: Path, seed: int = 7) -> Path:
    from hyperscholar.core.config import load_config
    from hyperscholar.core.llm import build_llm_func

    cfg = load_config()
    llm = build_llm_func(cfg.llm)

    ans_path = results_dir / corpus / "answers.json"
    if not ans_path.exists():
        raise FileNotFoundError(f"{ans_path} not found. Run runner first.")
    data = json.loads(ans_path.read_text(encoding="utf-8"))

    rng = random.Random(seed)
    judged = []
    agg = {b: {m: 0.0 for m in METRICS} for b in ("hyperrag", "hierarchical")}
    wins = {"hyperrag": 0, "hierarchical": 0, "tie": 0}
    n_scored = 0

    for item in data["results"]:
        qid = item["id"]
        hyper_ans = item["hyperrag"]["answer"]
        hier_ans = item["hierarchical"]["answer"]

        flip = rng.random() < 0.5
        a_ans, b_ans = (hier_ans, hyper_ans) if flip else (hyper_ans, hier_ans)
        a_backend, b_backend = (
            ("hierarchical", "hyperrag") if flip else ("hyperrag", "hierarchical"))

        reply = await llm(JUDGE_PROMPT.format(
            question=item["question"], answer_a=a_ans, answer_b=b_ans))
        parsed = _parse_scores(reply)
        if parsed is None:
            print(f"  Q{qid}: judge parse failed, skipping")
            judged.append({**item, "scores": None})
            continue

        scores = {a_backend: parsed["A"], b_backend: parsed["B"]}
        hyper_mean = _mean(scores["hyperrag"])
        hier_mean = _mean(scores["hierarchical"])
        winner = ("hyperrag" if hyper_mean > hier_mean
                  else "hierarchical" if hier_mean > hyper_mean else "tie")
        wins[winner] += 1
        n_scored += 1
        for b in ("hyperrag", "hierarchical"):
            for m in METRICS:
                agg[b][m] += scores[b][m]

        print(f"  Q{qid}: hyperrag={hyper_mean} hierarchical={hier_mean} → {winner}")
        judged.append({
            **item,
            "scores": {
                "hyperrag": {**scores["hyperrag"], "mean": hyper_mean},
                "hierarchical": {**scores["hierarchical"], "mean": hier_mean},
                "winner": winner,
            },
        })

    aggregate = {}
    for b in ("hyperrag", "hierarchical"):
        per_metric = {m: round(agg[b][m] / n_scored, 2) if n_scored else 0.0
                      for m in METRICS}
        per_metric["mean"] = round(sum(per_metric.values()) / len(METRICS), 2)
        aggregate[b] = per_metric
    aggregate["wins"] = wins
    aggregate["n_scored"] = n_scored

    out = {
        "corpus": corpus,
        "namespace": data.get("namespace", corpus),
        "judge_model": (cfg.llm.providers[0].model
                        if cfg.llm.providers else "unknown"),
        "questions": judged,
        "aggregate": aggregate,
    }
    out_path = results_dir / corpus / "eval_results.json"
    out_path.write_text(json.dumps(out, indent=2, ensure_ascii=False),
                        encoding="utf-8")
    print(f"\n✓ scored {n_scored} → {out_path}")
    print(f"  wins: {wins}")
    return out_path


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--corpus", required=True)
    ap.add_argument("--results-dir",
                    default=str(Path(__file__).resolve().parent / "results"))
    ap.add_argument("--seed", type=int, default=7)
    args = ap.parse_args()
    asyncio.run(judge_corpus(args.corpus, Path(args.results_dir), args.seed))


if __name__ == "__main__":
    main()
