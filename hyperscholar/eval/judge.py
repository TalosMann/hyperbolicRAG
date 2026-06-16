r"""eval/judge.py

LLM-as-judge on the five iMoonLab paper metrics, blind and position-randomized.

Features:
- Saves after every question (checkpoint) — safe to interrupt and resume
- Resume: skips questions already scored in eval_results.json on restart
- Gracefully skips unparseable judge responses rather than crashing

Usage
-----
    cd D:\Projects\hyperbolic
    python -m hyperscholar.eval.judge --corpus demo
    python -m hyperscholar.eval.judge --corpus neurology
"""
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


def _compute_aggregate(judged: list) -> dict:
    agg = {b: {m: 0.0 for m in METRICS} for b in ("hyperrag", "hierarchical")}
    wins = {"hyperrag": 0, "hierarchical": 0, "tie": 0}
    n_scored = 0
    for item in judged:
        s = item.get("scores")
        if not s:
            continue
        n_scored += 1
        for b in ("hyperrag", "hierarchical"):
            for m in METRICS:
                agg[b][m] += s[b].get(m, 0)
        wins[s.get("winner", "tie")] += 1
    aggregate = {}
    for b in ("hyperrag", "hierarchical"):
        per_metric = {m: round(agg[b][m] / n_scored, 2) if n_scored else 0.0
                      for m in METRICS}
        per_metric["mean"] = round(sum(per_metric.values()) / len(METRICS), 2)
        aggregate[b] = per_metric
    aggregate["wins"] = wins
    aggregate["n_scored"] = n_scored
    return aggregate


async def judge_corpus(corpus: str, results_dir: Path, seed: int = 7) -> Path:
    from hyperscholar.core.config import load_config
    from hyperscholar.core.llm import build_llm_func

    cfg = load_config()
    llm = build_llm_func(cfg.llm)

    ans_path = results_dir / corpus / "answers.json"
    if not ans_path.exists():
        raise FileNotFoundError(f"{ans_path} not found. Run runner first.")
    data = json.loads(ans_path.read_text(encoding="utf-8"))

    out_path = results_dir / corpus / "eval_results.json"

    # Resume: load already-scored results
    if out_path.exists():
        existing = json.loads(out_path.read_text(encoding="utf-8"))
        judged = existing.get("questions", [])
        done_ids = {item["id"] for item in judged if item.get("scores") is not None}
        print(f"  resuming — {len(done_ids)} already scored, "
              f"{len(data['results']) - len(done_ids)} remaining")
    else:
        judged = []
        done_ids = set()

    # Index existing judged items by id for merging
    judged_by_id = {item["id"]: item for item in judged}

    rng = random.Random(seed)

    for item in data["results"]:
        qid = item["id"]
        if qid in done_ids:
            continue

        hyper_ans = item["hyperrag"]["answer"]
        hier_ans = item["hierarchical"]["answer"]

        flip = rng.random() < 0.5
        a_ans, b_ans = (hier_ans, hyper_ans) if flip else (hyper_ans, hier_ans)
        a_backend, b_backend = (
            ("hierarchical", "hyperrag") if flip else ("hyperrag", "hierarchical"))

        try:
            reply = await llm(JUDGE_PROMPT.format(
                question=item["question"], answer_a=a_ans, answer_b=b_ans))
            parsed = _parse_scores(reply)
        except Exception as e:
            print(f"  Q{qid}: judge error ({e}), skipping")
            judged_by_id[qid] = {**item, "scores": None}
            judged = list(judged_by_id.values())
            _save(out_path, corpus, data, judged)
            continue

        if parsed is None:
            print(f"  Q{qid}: judge parse failed, skipping")
            judged_by_id[qid] = {**item, "scores": None}
        else:
            scores = {a_backend: parsed["A"], b_backend: parsed["B"]}
            hyper_mean = _mean(scores["hyperrag"])
            hier_mean = _mean(scores["hierarchical"])
            winner = ("hyperrag" if hyper_mean > hier_mean
                      else "hierarchical" if hier_mean > hyper_mean else "tie")
            print(f"  Q{qid}: hyperrag={hyper_mean} hierarchical={hier_mean} → {winner}")
            judged_by_id[qid] = {
                **item,
                "scores": {
                    "hyperrag": {**scores["hyperrag"], "mean": hyper_mean},
                    "hierarchical": {**scores["hierarchical"], "mean": hier_mean},
                    "winner": winner,
                },
            }

        judged = list(judged_by_id.values())
        # Checkpoint — write after every question
        _save(out_path, corpus, data, judged)

    aggregate = _compute_aggregate(judged)
    _save(out_path, corpus, data, judged, aggregate)
    print(f"\n✓ scored {aggregate['n_scored']} → {out_path}")
    print(f"  wins: {aggregate['wins']}")
    return out_path


def _save(out_path: Path, corpus: str, data: dict,
          judged: list, aggregate: dict | None = None) -> None:
    out = {
        "corpus": corpus,
        "namespace": data.get("namespace", corpus),
        "questions": judged,
    }
    if aggregate:
        out["aggregate"] = aggregate
    out_path.write_text(json.dumps(out, indent=2, ensure_ascii=False),
                        encoding="utf-8")


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
