r"""eval/visualize_results.py

Generates a fixed set of labeled charts from eval_results.json and
fact_check.json, for embedding directly into a written report.

Charts produced (PNG, 200dpi) in <results_dir>/<corpus>/charts/:

  01_overall_mean_scores.png     overall 5-metric means, HyperRAG vs Hierarchical
  02_per_style_mean_scores.png   mean score per question style, both backends
  03_per_style_wins.png          win/tie counts per style (stacked bar)
  04_radar_five_metrics.png      spider chart of the 5 judge metrics, overall
  05_fact_check_verdicts.png     SUPPORTED/CONTRADICTED/UNVERIFIABLE — the
                                  ground-truth grounding comparison
  06_negative_refusal_rate.png   refusal rate on negative-style questions
  07_per_question_score_diff.png per-question score delta, grouped by style
                                  (shows individual question variance/outliers)

Any chart whose required input file is missing is skipped with a printed
note rather than failing the whole run.

Usage
-----
    cd D:\Projects\hyperbolic
    python -m hyperscholar.eval.visualize_results --corpus neurology
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

METRICS = ["comprehensiveness", "diversity", "empowerment", "logical", "readability"]
STYLE_ORDER = ["fact", "relational", "synthesis", "overview"]
STYLE_LABELS = {
    "fact": "Fact retrieval",
    "relational": "Relational",
    "synthesis": "Synthesis",
    "overview": "Overview",
}

HYPER_COLOR = "#2C6E8C"   # teal-blue
HIER_COLOR = "#D9774B"    # warm orange
TIE_COLOR = "#9B9B9B"     # neutral gray
SUPPORTED_COLOR = "#4A7C59"
CONTRADICTED_COLOR = "#B5414A"
UNVERIFIABLE_COLOR = "#B8AC8E"

plt.rcParams.update({
    "font.size": 11,
    "axes.spines.top": False,
    "axes.spines.right": False,
    "figure.facecolor": "white",
    "axes.facecolor": "white",
})


def _load(results_dir: Path, corpus: str, filename: str) -> dict | None:
    p = results_dir / corpus / filename
    if not p.exists():
        print(f"  [skip] {filename} not found at {p}")
        return None
    return json.loads(p.read_text(encoding="utf-8"))


def _aggregate_for(questions: list) -> dict:
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


def chart_01_overall_means(eval_data: dict, out_dir: Path) -> None:
    agg = eval_data.get("aggregate") or _aggregate_for(eval_data.get("questions", []))
    labels = [m.capitalize() for m in METRICS] + ["Mean"]
    hyper_vals = [agg["hyperrag"][m] for m in METRICS] + [agg["hyperrag"]["mean"]]
    hier_vals = [agg["hierarchical"][m] for m in METRICS] + [agg["hierarchical"]["mean"]]

    x = np.arange(len(labels))
    w = 0.35
    fig, ax = plt.subplots(figsize=(9, 5))
    b1 = ax.bar(x - w/2, hyper_vals, w, label="HyperRAG", color=HYPER_COLOR)
    b2 = ax.bar(x + w/2, hier_vals, w, label="HierarchicalRAG", color=HIER_COLOR)
    ax.set_xticks(x)
    ax.set_xticklabels(labels)
    ax.set_ylim(0, 10.5)
    ax.set_ylabel("Mean score (1-10)")
    ax.set_title("Overall judge scores — all non-negative questions combined\n"
                f"n={agg['n_scored']} scored questions", fontsize=12)
    ax.legend(frameon=False)
    for bars in (b1, b2):
        for rect in bars:
            h = rect.get_height()
            ax.annotate(f"{h:.1f}", (rect.get_x() + rect.get_width()/2, h),
                       textcoords="offset points", xytext=(0, 3),
                       ha="center", fontsize=9)
    fig.tight_layout()
    fig.savefig(out_dir / "01_overall_mean_scores.png", dpi=200)
    plt.close(fig)
    print("  ✓ 01_overall_mean_scores.png")


def chart_02_per_style_means(eval_data: dict, out_dir: Path) -> None:
    questions = eval_data.get("questions", [])
    by_style: dict = {}
    for q in questions:
        s = q.get("style", "fact")
        by_style.setdefault(s, []).append(q)

    present = [s for s in STYLE_ORDER if s in by_style and
              _aggregate_for(by_style[s])["n_scored"] > 0]
    if not present:
        print("  [skip] 02_per_style_mean_scores.png — no scored styles found")
        return

    hyper_means, hier_means, labels = [], [], []
    for s in present:
        agg = _aggregate_for(by_style[s])
        hyper_means.append(agg["hyperrag"]["mean"])
        hier_means.append(agg["hierarchical"]["mean"])
        labels.append(f"{STYLE_LABELS.get(s, s)}\n(n={agg['n_scored']})")

    x = np.arange(len(labels))
    w = 0.35
    fig, ax = plt.subplots(figsize=(9, 5))
    b1 = ax.bar(x - w/2, hyper_means, w, label="HyperRAG", color=HYPER_COLOR)
    b2 = ax.bar(x + w/2, hier_means, w, label="HierarchicalRAG", color=HIER_COLOR)
    ax.set_xticks(x)
    ax.set_xticklabels(labels)
    ax.set_ylim(0, 10.5)
    ax.set_ylabel("Mean score (1-10)")
    ax.set_title("Mean judge score by question style", fontsize=12)
    ax.legend(frameon=False)
    for bars in (b1, b2):
        for rect in bars:
            h = rect.get_height()
            ax.annotate(f"{h:.1f}", (rect.get_x() + rect.get_width()/2, h),
                       textcoords="offset points", xytext=(0, 3),
                       ha="center", fontsize=9)
    fig.tight_layout()
    fig.savefig(out_dir / "02_per_style_mean_scores.png", dpi=200)
    plt.close(fig)
    print("  ✓ 02_per_style_mean_scores.png")


def chart_03_per_style_wins(eval_data: dict, out_dir: Path) -> None:
    questions = eval_data.get("questions", [])
    by_style: dict = {}
    for q in questions:
        s = q.get("style", "fact")
        by_style.setdefault(s, []).append(q)

    present = [s for s in STYLE_ORDER if s in by_style and
              _aggregate_for(by_style[s])["n_scored"] > 0]
    if not present:
        print("  [skip] 03_per_style_wins.png — no scored styles found")
        return

    hyper_wins, hier_wins, tie_wins, labels = [], [], [], []
    for s in present:
        agg = _aggregate_for(by_style[s])
        w = agg["wins"]
        hyper_wins.append(w["hyperrag"])
        hier_wins.append(w["hierarchical"])
        tie_wins.append(w["tie"])
        labels.append(STYLE_LABELS.get(s, s))

    x = np.arange(len(labels))
    fig, ax = plt.subplots(figsize=(9, 5))
    ax.bar(x, hyper_wins, label="HyperRAG wins", color=HYPER_COLOR)
    ax.bar(x, hier_wins, bottom=hyper_wins, label="HierarchicalRAG wins", color=HIER_COLOR)
    bottom2 = [h + r for h, r in zip(hyper_wins, hier_wins)]
    ax.bar(x, tie_wins, bottom=bottom2, label="Ties", color=TIE_COLOR)
    ax.set_xticks(x)
    ax.set_xticklabels(labels)
    ax.set_ylabel("Number of questions")
    ax.set_title("Win counts by question style", fontsize=12)
    ax.legend(frameon=False)
    fig.tight_layout()
    fig.savefig(out_dir / "03_per_style_wins.png", dpi=200)
    plt.close(fig)
    print("  ✓ 03_per_style_wins.png")


def chart_04_radar(eval_data: dict, out_dir: Path) -> None:
    agg = eval_data.get("aggregate") or _aggregate_for(eval_data.get("questions", []))
    n = len(METRICS)
    angles = np.linspace(0, 2 * np.pi, n, endpoint=False).tolist()
    angles += angles[:1]

    hyper_vals = [agg["hyperrag"][m] for m in METRICS]
    hyper_vals += hyper_vals[:1]
    hier_vals = [agg["hierarchical"][m] for m in METRICS]
    hier_vals += hier_vals[:1]

    fig, ax = plt.subplots(figsize=(7, 7), subplot_kw=dict(polar=True))
    ax.plot(angles, hyper_vals, color=HYPER_COLOR, linewidth=2, label="HyperRAG")
    ax.fill(angles, hyper_vals, color=HYPER_COLOR, alpha=0.15)
    ax.plot(angles, hier_vals, color=HIER_COLOR, linewidth=2, label="HierarchicalRAG")
    ax.fill(angles, hier_vals, color=HIER_COLOR, alpha=0.15)
    ax.set_xticks(angles[:-1])
    ax.set_xticklabels([m.capitalize() for m in METRICS])
    ax.set_ylim(0, 10)
    ax.set_title("Overall score profile across the five judge metrics", fontsize=12, pad=20)
    ax.legend(loc="upper right", bbox_to_anchor=(1.25, 1.1), frameon=False)
    fig.tight_layout()
    fig.savefig(out_dir / "04_radar_five_metrics.png", dpi=200)
    plt.close(fig)
    print("  ✓ 04_radar_five_metrics.png")


def chart_05_fact_check(fact_check_data: dict, out_dir: Path) -> None:
    summary = fact_check_data.get("summary")
    if not summary:
        print("  [skip] 05_fact_check_verdicts.png — no summary in fact_check.json")
        return

    backends = ["hyperrag", "hierarchical"]
    backend_labels = ["HyperRAG", "HierarchicalRAG"]
    verdicts = ["SUPPORTED", "CONTRADICTED", "UNVERIFIABLE"]
    colors = [SUPPORTED_COLOR, CONTRADICTED_COLOR, UNVERIFIABLE_COLOR]

    totals = [sum(summary[b].values()) for b in backends]
    fig, ax = plt.subplots(figsize=(8, 5.5))
    x = np.arange(len(backends))
    bottoms = np.zeros(len(backends))
    for verdict, color in zip(verdicts, colors):
        vals = [summary[b].get(verdict, 0) for b in backends]
        bars = ax.bar(x, vals, bottom=bottoms, label=verdict.capitalize(), color=color)
        for rect, v, tot, bot in zip(bars, vals, totals, bottoms):
            if v > 0:
                pct = 100 * v / tot if tot else 0
                ax.annotate(f"{v} ({pct:.0f}%)",
                           (rect.get_x() + rect.get_width()/2, bot + v/2),
                           ha="center", va="center", fontsize=10, color="white",
                           fontweight="bold")
        bottoms += np.array(vals)
    ax.set_xticks(x)
    ax.set_xticklabels([f"{lbl}\n(n={t})" for lbl, t in zip(backend_labels, totals)])
    ax.set_ylabel("Number of fact-style questions")
    ax.set_title("Ground-truth grounding check — fact-style questions only\n"
                "Answer's key claim verified directly against its own source chunk",
                fontsize=12)
    ax.legend(frameon=False, loc="upper center", bbox_to_anchor=(0.5, -0.12), ncol=3)
    fig.tight_layout()
    fig.savefig(out_dir / "05_fact_check_verdicts.png", dpi=200)
    plt.close(fig)
    print("  ✓ 05_fact_check_verdicts.png")


def chart_06_negative_refusal(eval_data: dict, out_dir: Path) -> None:
    neg = eval_data.get("negative_summary")
    if not neg:
        print("  [skip] 06_negative_refusal_rate.png — no negative_summary found")
        return

    n = neg["n"]
    hyper_pct = 100 * neg["hyperrag_refused"] / n if n else 0
    hier_pct = 100 * neg["hierarchical_refused"] / n if n else 0

    fig, ax = plt.subplots(figsize=(6.5, 5))
    bars = ax.bar(["HyperRAG", "HierarchicalRAG"], [hyper_pct, hier_pct],
                  color=[HYPER_COLOR, HIER_COLOR], width=0.5)
    for rect, pct, refused in zip(bars, [hyper_pct, hier_pct],
                                  [neg["hyperrag_refused"], neg["hierarchical_refused"]]):
        ax.annotate(f"{refused}/{n}\n({pct:.0f}%)",
                   (rect.get_x() + rect.get_width()/2, rect.get_height()),
                   textcoords="offset points", xytext=(0, 6),
                   ha="center", fontsize=11, fontweight="bold")
    ax.set_ylim(0, 100)
    ax.set_ylabel("% of negative-style questions correctly declined")
    ax.set_title("Refusal rate on questions with no answer in the corpus\n"
                "(higher = more often correctly admitted \"not in the corpus\")",
                fontsize=11)
    fig.tight_layout()
    fig.savefig(out_dir / "06_negative_refusal_rate.png", dpi=200)
    plt.close(fig)
    print("  ✓ 06_negative_refusal_rate.png")


def chart_07_per_question_diff(eval_data: dict, out_dir: Path) -> None:
    questions = eval_data.get("questions", [])
    points = []  # (style, diff)
    for q in questions:
        s = q.get("scores")
        if not s:
            continue
        style = q.get("style", "fact")
        diff = s["hyperrag"]["mean"] - s["hierarchical"]["mean"]
        points.append((style, diff))

    present = [s for s in STYLE_ORDER if any(p[0] == s for p in points)]
    if not present:
        print("  [skip] 07_per_question_score_diff.png — no scored questions found")
        return

    fig, ax = plt.subplots(figsize=(9, 5.5))
    rng = np.random.default_rng(0)
    for i, s in enumerate(present):
        diffs = [d for st, d in points if st == s]
        jitter = rng.uniform(-0.12, 0.12, size=len(diffs))
        colors = [HYPER_COLOR if d > 0 else (HIER_COLOR if d < 0 else TIE_COLOR)
                 for d in diffs]
        ax.scatter([i + j for j in jitter], diffs, c=colors, s=50,
                  alpha=0.8, edgecolors="white", linewidths=0.5)
    ax.axhline(0, color="gray", linewidth=1, linestyle="--")
    ax.set_xticks(range(len(present)))
    ax.set_xticklabels([STYLE_LABELS.get(s, s) for s in present])
    ax.set_ylabel("Score difference (HyperRAG mean − HierarchicalRAG mean)")
    ax.set_title("Per-question score gap, grouped by style\n"
                "Points below the line = HierarchicalRAG scored higher on that question",
                fontsize=11)
    fig.tight_layout()
    fig.savefig(out_dir / "07_per_question_score_diff.png", dpi=200)
    plt.close(fig)
    print("  ✓ 07_per_question_score_diff.png")


def visualize(corpus: str, results_dir: Path) -> Path:
    out_dir = results_dir / corpus / "charts"
    out_dir.mkdir(parents=True, exist_ok=True)

    eval_data = _load(results_dir, corpus, "eval_results.json")
    fact_check_data = _load(results_dir, corpus, "fact_check.json")

    print(f"Writing charts to {out_dir}\n")
    if eval_data:
        chart_01_overall_means(eval_data, out_dir)
        chart_02_per_style_means(eval_data, out_dir)
        chart_03_per_style_wins(eval_data, out_dir)
        chart_04_radar(eval_data, out_dir)
        chart_06_negative_refusal(eval_data, out_dir)
        chart_07_per_question_diff(eval_data, out_dir)
    if fact_check_data:
        chart_05_fact_check(fact_check_data, out_dir)

    print(f"\n✓ done — charts in {out_dir}")
    return out_dir


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--corpus", required=True)
    ap.add_argument("--results-dir",
                    default=str(Path(__file__).resolve().parent / "results"))
    args = ap.parse_args()
    visualize(args.corpus, Path(args.results_dir))


if __name__ == "__main__":
    main()
