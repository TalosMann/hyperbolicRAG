r"""eval/report.py

Reads <corpus>/eval_results.json and renders a Markdown comparison report
across all four backends (hyperrag, hierarchical, pure_cograg, cograg_flash):
  - overall scores (all non-negative-style questions combined)
  - PER-STYLE breakdown (fact / relational / synthesis / overview) — this is
    the table that actually means something, since a single aggregate number
    is dominated by whichever question style happens to be most numerous
  - hallucination-resistance section from negative-style refusal checks
  - cross-corpus summary if multiple corpora are present

Usage
-----
    cd D:\Projects\hyperbolic
    python -m hyperscholar.eval.report                       # all corpora found
    python -m hyperscholar.eval.report --corpus neurology    # one corpus
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

BACKENDS = ["hyperrag", "hierarchical", "cograg_official"]
BACKEND_LABELS = {
    "hyperrag": "HyperRAG",
    "hierarchical": "HierarchicalRAG",
    "cograg_official": "CogRAG",
}
METRICS = ["comprehensiveness", "diversity", "empowerment", "logical", "readability"]
STYLE_ORDER = ["fact", "relational", "synthesis", "overview"]
STYLE_LABELS = {
    "fact": "Fact retrieval",
    "relational": "Relational / multi-hop",
    "synthesis": "Cross-corpus synthesis",
    "overview": "Broad overview",
}


def _load_results(results_dir: Path, corpora: list[str] | None) -> dict:
    found = {}
    for sub in sorted(results_dir.iterdir()):
        if not sub.is_dir():
            continue
        if corpora and sub.name not in corpora:
            continue
        ep = sub / "eval_results.json"
        if ep.exists():
            found[sub.name] = json.loads(ep.read_text(encoding="utf-8"))
    return found


def _aggregate_for(questions: list) -> dict:
    """Compute wins/means over whatever scored questions are passed in."""
    agg = {b: {m: 0.0 for m in METRICS} for b in BACKENDS}
    wins = {b: 0 for b in BACKENDS}
    wins["tie"] = 0
    n = 0
    for item in questions:
        s = item.get("scores")
        if not s:
            continue
        n += 1
        for b in BACKENDS:
            bs = s.get(b)
            if not bs:
                continue
            for m in METRICS:
                agg[b][m] += bs.get(m, 0)
        wins[s.get("winner", "tie")] = wins.get(s.get("winner", "tie"), 0) + 1
    out = {}
    for b in BACKENDS:
        pm = {m: round(agg[b][m] / n, 2) if n else 0.0 for m in METRICS}
        pm["mean"] = round(sum(pm.values()) / len(METRICS), 2)
        out[b] = pm
    out["wins"] = wins
    out["n_scored"] = n
    return out


def _metric_table(agg: dict) -> list[str]:
    header = "| Metric | " + " | ".join(BACKEND_LABELS[b] for b in BACKENDS) + " |"
    sep = "|--------|" + "|".join("---:" for _ in BACKENDS) + "|"
    rows = [header, sep]
    for m in METRICS + ["mean"]:
        label = m.capitalize() if m != "mean" else "**Mean**"
        vals = " | ".join(str(agg[b].get(m, 0)) for b in BACKENDS)
        rows.append(f"| {label} | {vals} |")
    return rows


def _negative_table(neg: dict) -> list[str]:
    n = neg["n"]
    rows = [
        "| | Correctly declined | Did not decline |",
        "|---|---:|---:|",
    ]
    for b in BACKENDS:
        refused = neg.get(f"{b}_refused", 0)
        rows.append(f"| {BACKEND_LABELS[b]} | {refused}/{n} "
                    f"({100*refused/n:.0f}%) | {n-refused}/{n} |")
    return rows


def _wins_line(wins: dict) -> str:
    parts = [f"{BACKEND_LABELS[b]} {wins.get(b, 0)}" for b in BACKENDS]
    parts.append(f"tie {wins.get('tie', 0)}")
    return " · ".join(parts)


def build_report(results_dir: Path, corpora: list[str] | None) -> Path:
    data = _load_results(results_dir, corpora)
    if not data:
        raise RuntimeError(f"No eval_results.json found under {results_dir}")

    md = ["# RAG backend comparison — evaluation report\n",
          "Scoring follows the iMoonLab Hyper-RAG protocol: an LLM judge rates "
          "each answer 1–10 on five dimensions, blind and position-randomized, "
          "across all four backends in a single judge call per question. "
          "Higher is better.\n",
          "Questions span multiple **styles** targeting different retrieval "
          "capabilities — see the per-style breakdown below before drawing "
          "conclusions from the overall number alone, since a single aggregate "
          "is dominated by whichever style has the most questions.\n"]

    overall_running = {b: {m: 0.0 for m in METRICS + ["mean"]} for b in BACKENDS}
    overall_wins = {b: 0 for b in BACKENDS}
    overall_wins["tie"] = 0
    n_corpora = 0

    for corpus, d in data.items():
        questions = d.get("questions", [])
        agg = d.get("aggregate") or _aggregate_for(questions)
        wins = agg.get("wins", {})
        n_scored = agg.get("n_scored", 0)

        md.append(f"\n## {corpus}\n")
        md.append(f"Judge: `{d.get('judge_model', 'unknown')}` · "
                  f"non-negative questions scored: {n_scored}\n")
        md.append("### Overall (all styles combined)\n")
        md.extend(_metric_table(agg))
        md.append(f"\n**Wins:** {_wins_line(wins)}\n")

        # ── per-style breakdown ──────────────────────────────────────────────
        by_style: dict = {}
        for q in questions:
            s = q.get("style", "fact")
            by_style.setdefault(s, []).append(q)

        present_styles = [s for s in STYLE_ORDER if s in by_style]
        if present_styles:
            md.append("\n### By question style\n")
            for s in present_styles:
                qs = by_style[s]
                s_agg = _aggregate_for(qs)
                if s_agg["n_scored"] == 0:
                    continue
                md.append(f"\n**{STYLE_LABELS.get(s, s)}** "
                          f"({s_agg['n_scored']} questions)\n")
                md.extend(_metric_table(s_agg))
                md.append(f"\nWins: {_wins_line(s_agg['wins'])}\n")

        # ── negative / hallucination-resistance ──────────────────────────────
        neg = d.get("negative_summary")
        if neg:
            md.append("\n### Hallucination resistance (negative-style questions)\n")
            md.append("Each question asks for a specific detail that is "
                      "intentionally absent from the source passage. Correct "
                      "behavior is to decline rather than fabricate an answer. "
                      "*Caveat: \"did not decline\" is not proof of "
                      "hallucination — it only means the canned refusal "
                      "wasn't triggered; spot-check those answers manually "
                      "for a true hallucination rate.*\n")
            md.extend(_negative_table(neg))
            md.append("")

        # accumulate for cross-corpus summary (overall metrics only)
        n_corpora += 1
        for b in BACKENDS:
            for m in METRICS + ["mean"]:
                overall_running[b][m] += agg[b].get(m, 0)
        for k in overall_wins:
            overall_wins[k] += wins.get(k, 0)

    if n_corpora > 1:
        for b in BACKENDS:
            for m in METRICS + ["mean"]:
                overall_running[b][m] = round(overall_running[b][m] / n_corpora, 2)
        md.append("\n## Cross-corpus summary (overall, all styles)\n")
        md.append(f"Averaged across {n_corpora} corpora.\n")
        md.extend(_metric_table(overall_running))
        md.append(f"\n**Total wins:** {_wins_line(overall_wins)}\n")

    out_path = results_dir / "eval_report.md"
    out_path.write_text("\n".join(md), encoding="utf-8")
    print(f"✓ report → {out_path}")
    return out_path


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--corpus", action="append", default=None,
                    help="restrict to corpus (repeatable); default = all")
    ap.add_argument("--results-dir",
                    default=str(Path(__file__).resolve().parent / "results"))
    args = ap.parse_args()
    build_report(Path(args.results_dir), args.corpus)


if __name__ == "__main__":
    main()
