"""eval/report.py

Reads one or more <corpus>/eval_results.json files and renders a human-readable
Markdown comparison report — per-corpus metric tables plus a cross-corpus
summary, in the style of the iMoonLab paper's Figure 4.

Usage
-----
    python -m eval.report                       # all corpora found
    python -m eval.report --corpus neurology    # one corpus

Writes: eval/results/eval_report.md
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

METRICS = ["comprehensiveness", "diversity", "empowerment", "logical", "readability"]


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


def build_report(results_dir: Path, corpora: list[str] | None) -> Path:
    data = _load_results(results_dir, corpora)
    if not data:
        raise RuntimeError(f"No eval_results.json found under {results_dir}")

    md = ["# HyperRAG vs HierarchicalRAG — evaluation report\n",
          "Scoring follows the iMoonLab Hyper-RAG protocol: an LLM judge rates "
          "each answer 1–10 on five dimensions, blind and position-randomized. "
          "Higher is better; Δ is HyperRAG minus HierarchicalRAG.\n"]

    # Per-corpus sections.
    overall = {b: {m: 0.0 for m in METRICS + ["mean"]}
               for b in ("hyperrag", "hierarchical")}
    overall_wins = {"hyperrag": 0, "hierarchical": 0, "tie": 0}
    n_corpora = 0

    for corpus, d in data.items():
        agg = d["aggregate"]
        wins = agg.get("wins", {})
        n_scored = agg.get("n_scored", 0)
        md.append(f"\n## {corpus}\n")
        md.append(f"Judge: `{d.get('judge_model', 'unknown')}` · "
                  f"questions scored: {n_scored}\n")
        md.extend(_metric_table(agg))
        md.append(f"\n**Wins:** HyperRAG {wins.get('hyperrag', 0)} · "
                  f"HierarchicalRAG {wins.get('hierarchical', 0)} · "
                  f"tie {wins.get('tie', 0)}\n")

        n_corpora += 1
        for b in ("hyperrag", "hierarchical"):
            for m in METRICS + ["mean"]:
                overall[b][m] += agg[b].get(m, 0)
        for k in overall_wins:
            overall_wins[k] += wins.get(k, 0)

    # Cross-corpus summary.
    if n_corpora > 1:
        for b in ("hyperrag", "hierarchical"):
            for m in METRICS + ["mean"]:
                overall[b][m] = round(overall[b][m] / n_corpora, 2)
        md.append("\n## Cross-corpus summary\n")
        md.append(f"Averaged across {n_corpora} corpora.\n")
        md.extend(_metric_table(overall))
        md.append(f"\n**Total wins:** HyperRAG {overall_wins['hyperrag']} · "
                  f"HierarchicalRAG {overall_wins['hierarchical']} · "
                  f"tie {overall_wins['tie']}\n")

        winner = ("HyperRAG" if overall["hyperrag"]["mean"] > overall["hierarchical"]["mean"]
                  else "HierarchicalRAG" if overall["hierarchical"]["mean"] > overall["hyperrag"]["mean"]
                  else "neither (tie)")
        md.append(f"\n**Overall higher mean score:** {winner}\n")

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
