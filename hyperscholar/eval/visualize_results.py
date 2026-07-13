r"""eval/visualize_results.py

Generates a fixed set of labeled charts from eval_results.json comparing
all four RAG backends, saved as PNG (200dpi) in <results_dir>/<corpus>/charts/:

  01_overall_mean_scores.png     overall 5-metric means, all 4 backends
  02_per_style_mean_scores.png   mean score per question style, all 4 backends
  03_per_style_wins.png          win counts per style (grouped bar)
  04_radar_five_metrics.png      spider chart of the 5 judge metrics, overall
  05_negative_refusal_rate.png   refusal rate on negative-style questions
  06_per_question_scores.png     per-question scores for each backend,
                                  grouped by style (shows individual variance)

Any chart whose required input is missing is skipped with a note.

Usage
-----
    cd /path/to/hyperbolic
    python -m hyperscholar.eval.visualize_results --corpus agriculture_5
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

BACKENDS = ["hyperrag", "hierarchical", "cograg_official"]
BACKEND_LABELS = {
    "hyperrag":       "HyperRAG",
    "hierarchical":   "HierarchicalRAG",
    "cograg_official": "CogRAG",
}
BACKEND_COLORS = {
    "hyperrag":       "#2C6E8C",
    "hierarchical":   "#D9774B",
    "cograg_official": "#6B4C9A",
}

METRICS = ["comprehensiveness", "diversity", "empowerment", "logical", "readability"]
STYLE_ORDER = ["fact", "relational", "synthesis", "overview"]
STYLE_LABELS = {
    "fact":       "Fact retrieval",
    "relational": "Relational",
    "synthesis":  "Synthesis",
    "overview":   "Overview",
}

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
        print(f"  [skip] {filename} not found")
        return None
    return json.loads(p.read_text(encoding="utf-8"))


def _aggregate_for(questions: list) -> dict:
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


def chart_01_overall_means(eval_data: dict, out_dir: Path) -> None:
    agg = eval_data.get("aggregate") or _aggregate_for(eval_data.get("questions", []))
    labels = [m.capitalize() for m in METRICS] + ["Mean"]
    x = np.arange(len(labels))
    n_b = len(BACKENDS)
    total_w = 0.75
    w = total_w / n_b

    fig, ax = plt.subplots(figsize=(11, 5))
    for i, b in enumerate(BACKENDS):
        vals = [agg[b][m] for m in METRICS] + [agg[b]["mean"]]
        offset = (i - n_b / 2 + 0.5) * w
        bars = ax.bar(x + offset, vals, w, label=BACKEND_LABELS[b],
                      color=BACKEND_COLORS[b])
        for rect in bars:
            h = rect.get_height()
            ax.annotate(f"{h:.1f}", (rect.get_x() + rect.get_width() / 2, h),
                        textcoords="offset points", xytext=(0, 3),
                        ha="center", fontsize=7.5)
    ax.set_xticks(x)
    ax.set_xticklabels(labels)
    ax.set_ylim(0, 11.5)
    ax.set_ylabel("Mean score (1–10)")
    ax.set_title(f"Overall judge scores — all non-negative questions combined\n"
                 f"n={agg['n_scored']} scored questions", fontsize=12)
    ax.legend(frameon=False, ncol=4, loc="upper center",
              bbox_to_anchor=(0.5, -0.08))
    fig.tight_layout()
    fig.savefig(out_dir / "01_overall_mean_scores.png", dpi=200)
    plt.close(fig)
    print("  ✓ 01_overall_mean_scores.png")


def chart_02_per_style_means(eval_data: dict, out_dir: Path) -> None:
    questions = eval_data.get("questions", [])
    by_style: dict = {}
    for q in questions:
        by_style.setdefault(q.get("style", "fact"), []).append(q)

    present = [s for s in STYLE_ORDER if s in by_style and
               _aggregate_for(by_style[s])["n_scored"] > 0]
    if not present:
        print("  [skip] 02_per_style_mean_scores.png")
        return

    x = np.arange(len(present))
    n_b = len(BACKENDS)
    total_w = 0.75
    w = total_w / n_b

    fig, ax = plt.subplots(figsize=(10, 5))
    for i, b in enumerate(BACKENDS):
        means, xlabels = [], []
        for s in present:
            agg = _aggregate_for(by_style[s])
            means.append(agg[b]["mean"])
            if i == 0:
                xlabels.append(f"{STYLE_LABELS.get(s, s)}\n(n={agg['n_scored']})")
        offset = (i - n_b / 2 + 0.5) * w
        bars = ax.bar(x + offset, means, w, label=BACKEND_LABELS[b],
                      color=BACKEND_COLORS[b])
        for rect in bars:
            h = rect.get_height()
            ax.annotate(f"{h:.1f}", (rect.get_x() + rect.get_width() / 2, h),
                        textcoords="offset points", xytext=(0, 3),
                        ha="center", fontsize=8)
    ax.set_xticks(x)
    ax.set_xticklabels(xlabels)
    ax.set_ylim(0, 11.5)
    ax.set_ylabel("Mean score (1–10)")
    ax.set_title("Mean judge score by question style", fontsize=12)
    ax.legend(frameon=False, ncol=4, loc="upper center",
              bbox_to_anchor=(0.5, -0.12))
    fig.tight_layout()
    fig.savefig(out_dir / "02_per_style_mean_scores.png", dpi=200)
    plt.close(fig)
    print("  ✓ 02_per_style_mean_scores.png")


def chart_03_per_style_wins(eval_data: dict, out_dir: Path) -> None:
    questions = eval_data.get("questions", [])
    by_style: dict = {}
    for q in questions:
        by_style.setdefault(q.get("style", "fact"), []).append(q)

    present = [s for s in STYLE_ORDER if s in by_style and
               _aggregate_for(by_style[s])["n_scored"] > 0]
    if not present:
        print("  [skip] 03_per_style_wins.png")
        return

    x = np.arange(len(present))
    n_b = len(BACKENDS)
    total_w = 0.7
    w = total_w / n_b

    fig, ax = plt.subplots(figsize=(10, 5))
    for i, b in enumerate(BACKENDS):
        wins = []
        for s in present:
            agg = _aggregate_for(by_style[s])
            wins.append(agg["wins"].get(b, 0))
        offset = (i - n_b / 2 + 0.5) * w
        ax.bar(x + offset, wins, w, label=BACKEND_LABELS[b],
               color=BACKEND_COLORS[b])
    ax.set_xticks(x)
    ax.set_xticklabels([STYLE_LABELS.get(s, s) for s in present])
    ax.set_ylabel("Number of questions won")
    ax.yaxis.set_major_locator(plt.MaxNLocator(integer=True))
    ax.set_title("Win counts by question style", fontsize=12)
    ax.legend(frameon=False, ncol=4, loc="upper center",
              bbox_to_anchor=(0.5, -0.1))
    fig.tight_layout()
    fig.savefig(out_dir / "03_per_style_wins.png", dpi=200)
    plt.close(fig)
    print("  ✓ 03_per_style_wins.png")


def chart_04_radar(eval_data: dict, out_dir: Path) -> None:
    agg = eval_data.get("aggregate") or _aggregate_for(eval_data.get("questions", []))
    n = len(METRICS)
    angles = np.linspace(0, 2 * np.pi, n, endpoint=False).tolist()
    angles += angles[:1]

    fig, ax = plt.subplots(figsize=(7, 7), subplot_kw=dict(polar=True))
    for b in BACKENDS:
        vals = [agg[b][m] for m in METRICS] + [agg[b][METRICS[0]]]
        ax.plot(angles, vals, color=BACKEND_COLORS[b], linewidth=2,
                label=BACKEND_LABELS[b])
        ax.fill(angles, vals, color=BACKEND_COLORS[b], alpha=0.08)
    ax.set_xticks(angles[:-1])
    ax.set_xticklabels([m.capitalize() for m in METRICS])
    ax.set_ylim(0, 10)
    ax.set_title("Score profile across the five judge metrics (overall)",
                 fontsize=12, pad=20)
    ax.legend(loc="upper right", bbox_to_anchor=(1.35, 1.1), frameon=False)
    fig.tight_layout()
    fig.savefig(out_dir / "04_radar_five_metrics.png", dpi=200)
    plt.close(fig)
    print("  ✓ 04_radar_five_metrics.png")


def chart_05_negative_refusal(eval_data: dict, out_dir: Path) -> None:
    neg = eval_data.get("negative_summary")
    if not neg:
        print("  [skip] 05_negative_refusal_rate.png — no negative_summary found")
        return
    n = neg["n"]
    refused = [neg.get(f"{b}_refused", 0) for b in BACKENDS]
    pcts = [100 * r / n if n else 0 for r in refused]

    fig, ax = plt.subplots(figsize=(8, 5))
    bars = ax.bar([BACKEND_LABELS[b] for b in BACKENDS], pcts,
                  color=[BACKEND_COLORS[b] for b in BACKENDS], width=0.5)
    for rect, pct, r in zip(bars, pcts, refused):
        ax.annotate(f"{r}/{n}\n({pct:.0f}%)",
                    (rect.get_x() + rect.get_width() / 2, rect.get_height()),
                    textcoords="offset points", xytext=(0, 6),
                    ha="center", fontsize=11, fontweight="bold")
    ax.set_ylim(0, 115)
    ax.set_ylabel("% correctly declined")
    ax.set_title("Hallucination resistance — refusal rate on unanswerable questions\n"
                 "(higher = more often correctly admits \"not in the corpus\")",
                 fontsize=11)
    fig.tight_layout()
    fig.savefig(out_dir / "05_negative_refusal_rate.png", dpi=200)
    plt.close(fig)
    print("  ✓ 05_negative_refusal_rate.png")


def chart_06_per_question_scores(eval_data: dict, out_dir: Path) -> None:
    questions = eval_data.get("questions", [])
    by_style: dict = {}
    for q in questions:
        s = q.get("scores")
        if not s:
            continue
        style = q.get("style", "fact")
        by_style.setdefault(style, []).append(
            {b: s.get(b, {}).get("mean", 0) for b in BACKENDS})

    present = [s for s in STYLE_ORDER if s in by_style]
    if not present:
        print("  [skip] 06_per_question_scores.png")
        return

    fig, axes = plt.subplots(1, len(present), figsize=(4 * len(present), 5),
                             sharey=True)
    if len(present) == 1:
        axes = [axes]

    for ax, style in zip(axes, present):
        items = by_style[style]
        rng = np.random.default_rng(42)
        for j, b in enumerate(BACKENDS):
            scores = [item[b] for item in items]
            jitter = rng.uniform(-0.15, 0.15, size=len(scores))
            ax.scatter([j + jit for jit in jitter], scores,
                       color=BACKEND_COLORS[b], s=60, alpha=0.85,
                       edgecolors="white", linewidths=0.5,
                       label=BACKEND_LABELS[b])
            ax.plot([j - 0.25, j + 0.25],
                    [np.mean(scores)] * 2, color=BACKEND_COLORS[b],
                    linewidth=2.5)
        ax.set_xticks(range(len(BACKENDS)))
        ax.set_xticklabels([BACKEND_LABELS[b][:6] for b in BACKENDS],
                           fontsize=8, rotation=20, ha="right")
        ax.set_ylim(0, 10.5)
        ax.set_title(f"{STYLE_LABELS.get(style, style)}\n(n={len(items)})",
                     fontsize=10)
        if ax is axes[0]:
            ax.set_ylabel("Score (1–10)")

    handles = [plt.Line2D([0], [0], marker='o', color='w',
                           markerfacecolor=BACKEND_COLORS[b], markersize=9,
                           label=BACKEND_LABELS[b])
               for b in BACKENDS]
    fig.legend(handles=handles, frameon=False, ncol=4,
               loc="lower center", bbox_to_anchor=(0.5, -0.06))
    fig.suptitle("Per-question scores by backend and style\n"
                 "Horizontal line = style mean", fontsize=12, y=1.01)
    fig.tight_layout()
    fig.savefig(out_dir / "06_per_question_scores.png", dpi=200,
                bbox_inches="tight")
    plt.close(fig)
    print("  ✓ 06_per_question_scores.png")


def visualize(corpus: str, results_dir: Path) -> Path:
    out_dir = results_dir / corpus / "charts"
    out_dir.mkdir(parents=True, exist_ok=True)

    eval_data = _load(results_dir, corpus, "eval_results.json")
    print(f"Writing charts to {out_dir}\n")
    if eval_data:
        chart_01_overall_means(eval_data, out_dir)
        chart_02_per_style_means(eval_data, out_dir)
        chart_03_per_style_wins(eval_data, out_dir)
        chart_04_radar(eval_data, out_dir)
        chart_05_negative_refusal(eval_data, out_dir)
        chart_06_per_question_scores(eval_data, out_dir)
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
