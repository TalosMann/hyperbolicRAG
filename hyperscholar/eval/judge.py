r"""eval/judge.py

LLM-as-judge on the five iMoonLab paper metrics, blind and position-randomized
— for fact / relational / synthesis / overview style questions. Scores all
four backends (hyperrag, hierarchical, pure_cograg, cograg_flash) together in
a single judge call per question, labeled A-D in randomized order so the
judge can't infer identity from position.

NEGATIVE-style questions are scored differently: no LLM comparison, just a
refusal check against each backend's own fail_markers. A "refused" answer is
the CORRECT behavior here (the asked-for detail is intentionally absent from
the source). A non-refusal is NOT proof of hallucination — it only means the
canned fail-marker wasn't triggered; the answer could still be hedged,
partially correct, or genuinely fabricated. Manually spot-check non-refused
negative answers if you need a true hallucination rate.

Features:
- Saves after every question (checkpoint) — safe to interrupt and resume
- Resume: skips questions already scored/checked in eval_results.json
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
import string
from pathlib import Path

BACKENDS = ["hyperrag", "hierarchical", "cograg_official"]
METRICS = ["comprehensiveness", "diversity", "empowerment", "logical", "readability"]

JUDGE_PROMPT_HEADER = """You are an impartial expert evaluator comparing {n} answers to the same question. Score EACH answer from 1 (poor) to 10 (excellent) on five dimensions:

- Comprehensiveness: covers all relevant aspects of the question
- Diversity: presents varied, rich perspectives and detail
- Empowerment: helps the reader genuinely understand and act on the topic
- Logical: reasoning is sound, coherent, and well-ordered
- Readability: clear, well-structured, easy to follow

QUESTION:
{question}
"""

JUDGE_PROMPT_FOOTER = """
Respond with ONLY a JSON object in exactly this form, no other text:
{{
{fields}
}}"""


def _build_judge_prompt(question: str, labeled_answers: list[tuple[str, str]]) -> str:
    """labeled_answers: [(label, answer_text), ...] in the order to present them."""
    parts = [JUDGE_PROMPT_HEADER.format(n=len(labeled_answers), question=question)]
    for label, answer in labeled_answers:
        parts.append(f"\nANSWER {label}:\n{answer}\n")
    fields = ",\n".join(
        f'  "{label}": {{"comprehensiveness": int, "diversity": int, '
        f'"empowerment": int, "logical": int, "readability": int}}'
        for label, _ in labeled_answers)
    parts.append(JUDGE_PROMPT_FOOTER.format(fields=fields))
    return "".join(parts)


def _parse_scores(raw: str, labels: list[str]) -> dict | None:
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
    for label in labels:
        if label not in obj:
            return None
        for metric in METRICS:
            if metric not in obj[label]:
                return None
    return obj


def _mean(scores: dict) -> float:
    return round(sum(scores[m] for m in METRICS) / len(METRICS), 2)


def _compute_aggregate(judged: list) -> dict:
    """Aggregate over scored (non-negative-style) questions only."""
    agg = {b: {m: 0.0 for m in METRICS} for b in BACKENDS}
    wins = {b: 0 for b in BACKENDS}
    wins["tie"] = 0
    n_scored = 0
    for item in judged:
        s = item.get("scores")
        if not s:
            continue
        n_scored += 1
        for b in BACKENDS:
            bs = s.get(b)
            if not bs:
                continue
            for m in METRICS:
                agg[b][m] += bs.get(m, 0)
        wins[s.get("winner", "tie")] = wins.get(s.get("winner", "tie"), 0) + 1
    aggregate = {}
    for b in BACKENDS:
        per_metric = {m: round(agg[b][m] / n_scored, 2) if n_scored else 0.0
                      for m in METRICS}
        per_metric["mean"] = round(sum(per_metric.values()) / len(METRICS), 2)
        aggregate[b] = per_metric
    aggregate["wins"] = wins
    aggregate["n_scored"] = n_scored
    return aggregate


def _compute_negative_summary(judged: list) -> dict | None:
    items = [j for j in judged if j.get("style") == "negative"
             and j.get("negative_check") is not None]
    if not items:
        return None
    out = {"n": len(items), **{f"{b}_refused": 0 for b in BACKENDS}}
    for j in items:
        nc = j["negative_check"]
        for b in BACKENDS:
            if nc.get(f"{b}_refused"):
                out[f"{b}_refused"] += 1
    return out


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

    if out_path.exists():
        existing = json.loads(out_path.read_text(encoding="utf-8"))
        judged = existing.get("questions", [])
        done_ids = {item["id"] for item in judged
                   if item.get("scores") is not None
                   or item.get("negative_check") is not None}
        print(f"  resuming — {len(done_ids)} already processed, "
              f"{len(data['results']) - len(done_ids)} remaining")
    else:
        judged = []
        done_ids = set()

    judged_by_id = {item["id"]: item for item in judged}
    rng = random.Random(seed)
    letters = list(string.ascii_uppercase[:len(BACKENDS)])

    for item in data["results"]:
        qid = item["id"]
        if qid in done_ids:
            continue

        style = item.get("style", "fact")

        # ── negative style: refusal check, no LLM comparison ─────────────────
        if style == "negative":
            nc = {f"{b}_refused": not item.get(b, {}).get("ok", True) for b in BACKENDS}
            print(f"  Q{qid} [negative]: " +
                  " ".join(f"{b}={nc[f'{b}_refused']}" for b in BACKENDS))
            judged_by_id[qid] = {**item, "scores": None, "negative_check": nc}
            judged = list(judged_by_id.values())
            _save(out_path, corpus, data, judged)
            continue

        # ── all other styles: blind, position-randomized LLM judge ──────────
        answers = {b: item.get(b, {}).get("answer", "") for b in BACKENDS}
        order = list(BACKENDS)
        rng.shuffle(order)
        label_to_backend = dict(zip(letters, order))
        labeled_answers = [(label, answers[b]) for label, b in label_to_backend.items()]

        try:
            reply = await llm(_build_judge_prompt(item["question"], labeled_answers))
            parsed = _parse_scores(reply, letters)
        except Exception as e:
            print(f"  Q{qid} [{style}]: judge error ({e}), skipping")
            judged_by_id[qid] = {**item, "scores": None}
            judged = list(judged_by_id.values())
            _save(out_path, corpus, data, judged)
            continue

        if parsed is None:
            print(f"  Q{qid} [{style}]: judge parse failed, skipping")
            judged_by_id[qid] = {**item, "scores": None}
        else:
            scores = {label_to_backend[label]: parsed[label] for label in letters}
            means = {b: _mean(scores[b]) for b in BACKENDS}
            top = max(means.values())
            winners = [b for b in BACKENDS if means[b] == top]
            winner = winners[0] if len(winners) == 1 else "tie"
            print(f"  Q{qid} [{style}]: " +
                  " ".join(f"{b}={means[b]}" for b in BACKENDS) + f" → {winner}")
            judged_by_id[qid] = {
                **item,
                "scores": {
                    **{b: {**scores[b], "mean": means[b]} for b in BACKENDS},
                    "winner": winner,
                },
            }

        judged = list(judged_by_id.values())
        _save(out_path, corpus, data, judged)

    aggregate = _compute_aggregate(judged)
    negative_summary = _compute_negative_summary(judged)
    _save(out_path, corpus, data, judged, aggregate, negative_summary)
    print(f"\n✓ scored {aggregate['n_scored']} → {out_path}")
    print(f"  wins: {aggregate['wins']}")
    if negative_summary:
        print(f"  negative checks: {negative_summary}")
    return out_path


def _save(out_path: Path, corpus: str, data: dict, judged: list,
          aggregate: dict | None = None,
          negative_summary: dict | None = None) -> None:
    out = {
        "corpus": corpus,
        "namespace": data.get("namespace", corpus),
        "questions": judged,
    }
    if aggregate:
        out["aggregate"] = aggregate
    if negative_summary:
        out["negative_summary"] = negative_summary
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
