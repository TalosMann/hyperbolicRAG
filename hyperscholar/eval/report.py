r"""eval/report.py

Reads <corpus>/eval_results.json and renders a Markdown comparison report:
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
    agg = {b: {m: 0.0 for m in METRICS} for b in ("hyperrag", "hierarchical")}
    wins = {"hyperrag": 0, "hierarchical": 0, "tie": 0}
    n = 0
    for item in questions:
        s = item.get("scores")
        if not s:
            continue
        n += 1
        for b in ("hyperrag", "hierarchical"):
            for m in METRICS:
                agg[b][m] += s[b].get(m, 0)
        wins[s.get("winner", "tie")] += 1
    out = {}
    for b in ("hyperrag", "hierarchical"):
        pm = {m: round(agg[b][m] / n, 2) if n else 0.0 for m in METRICS}
        pm["mean"] = round(sum(pm.values()) / len(METRICS), 2)
        out[b] = pm
    out["wins"] = wins
    out["n_scored"] = n
    return out


def _metric_table(agg: dict) -> list[str]:
    rows = ["| Metric | HyperRAG | HierarchicalRAG | Δ |",
            "|--------|---------:|----------------:|---:|"]
    for m in METRICS + ["mean"]:
        h = agg["hyperrag"].get(m, 0)
        r = agg["hierarchical"].get(m, 0)
        delta = round(h - r, 2)
        sign = "+" if delta > 0 else ""
        label = m.capitalize() if m != "mean" else "**Mean**"
        rows.append(f"| {label} | {h} | {r} | {sign}{delta} |")
    return rows


def _negative_table(neg: dict) -> list[str]:
    n = neg["n"]
    hr = neg["hyperrag_refused"]
    hir = neg["hierarchical_refused"]
    rows = [
        "| | Correctly declined | Did not decline |",
        "|---|---:|---:|",
        f"| HyperRAG | {hr}/{n} ({100*hr/n:.0f}%) | {n-hr}/{n} |",
        f"| HierarchicalRAG | {hir}/{n} ({100*hir/n:.0f}%) | {n-hir}/{n} |",
    ]
    return rows


def build_report(results_dir: Path, corpora: list[str] | None) -> Path:
    data = _load_results(results_dir, corpora)
    if not data:
        raise RuntimeError(f"No eval_results.json found under {results_dir}")

    md = ["# HyperRAG vs HierarchicalRAG — evaluation report\n",
          "Scoring follows the iMoonLab Hyper-RAG protocol: an LLM judge rates "
          "each answer 1–10 on five dimensions, blind and position-randomized. "
          "Higher is better; Δ is HyperRAG minus HierarchicalRAG.\n",
          "Questions span multiple **styles** targeting different retrieval "
          "capabilities — see the per-style breakdown below before drawing "
          "conclusions from the overall number alone, since a single aggregate "
          "is dominated by whichever style has the most questions.\n"]

    overall_running = {b: {m: 0.0 for m in METRICS + ["mean"]}
                       for b in ("hyperrag", "hierarchical")}
    overall_wins = {"hyperrag": 0, "hierarchical": 0, "tie": 0}
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
        md.append(f"\n**Wins:** HyperRAG {wins.get('hyperrag', 0)} · "
                  f"HierarchicalRAG {wins.get('hierarchical', 0)} · "
                  f"tie {wins.get('tie', 0)}\n")

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
                w = s_agg["wins"]
                md.append(f"\nWins: HyperRAG {w['hyperrag']} · "
                          f"HierarchicalRAG {w['hierarchical']} · tie {w['tie']}\n")

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
        for b in ("hyperrag", "hierarchical"):
            for m in METRICS + ["mean"]:
                overall_running[b][m] += agg[b].get(m, 0)
        for k in overall_wins:
            overall_wins[k] += wins.get(k, 0)

    if n_corpora > 1:
        for b in ("hyperrag", "hierarchical"):
            for m in METRICS + ["mean"]:
                overall_running[b][m] = round(overall_running[b][m] / n_corpora, 2)
        md.append("\n## Cross-corpus summary (overall, all styles)\n")
        md.append(f"Averaged across {n_corpora} corpora.\n")
        md.extend(_metric_table(overall_running))
        md.append(f"\n**Total wins:** HyperRAG {overall_wins['hyperrag']} · "
                  f"HierarchicalRAG {overall_wins['hierarchical']} · "
                  f"tie {overall_wins['tie']}\n")

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
