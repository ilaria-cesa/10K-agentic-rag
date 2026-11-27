"""Figures of the report (docs/figures/*.png), drawn from the files in results/.

Usage:  python -m src.eval.make_figures [--results results] [--out docs/figures]

Nothing here is typed in by hand: grades come from the hand-grading sheets (`grades_report`) and the
retrieval numbers from `results/rerank_experiment.jsonl`. Colours: one fixed colour per system, taken from
the Okabe-Ito set (readable with common colour-vision deficiencies); the lighter tone of a colour is the
local reranker, the full tone is the Cohere reranker.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd
from matplotlib.colors import to_rgb
from matplotlib.patches import Patch

from src.eval.grades_report import SYSTEMS, TYPES, load_grades, paired, wilson

INK, MUTED, GRID = "#1F2933", "#6B7280", "#E5E7EB"
COLOUR = {"rag_base": "#7F8790", "rag5": "#0072B2", "rag8": "#E69F00", "agent_v3": "#009E73"}
BLUE, ORANGE = "#0072B2", "#D55E00"
RERANKERS = ["bge-reranker-v2-m3", "rerank-v3.5", "rerank-v4.0-pro"]
RERANKER_LABEL = {
    "bge-reranker-v2-m3": "bge-reranker-v2-m3 (local)",
    "rerank-v3.5": "Cohere rerank-v3.5",
    "rerank-v4.0-pro": "Cohere rerank-v4.0-pro",
}
RERANKER_COLOUR = {
    "bge-reranker-v2-m3": "#B8BEC6",
    "rerank-v3.5": "#7BA7CC",
    "rerank-v4.0-pro": BLUE,
}


def tint(colour: str, amount: float = 0.55) -> tuple[float, float, float]:
    r, g, b = to_rgb(colour)
    return (r + (1 - r) * amount, g + (1 - g) * amount, b + (1 - b) * amount)


def setup() -> None:
    plt.rcParams.update(
        {
            "font.family": "DejaVu Sans",
            "font.size": 10,
            "text.color": INK,
            "axes.edgecolor": GRID,
            "axes.labelcolor": MUTED,
            "xtick.color": MUTED,
            "ytick.color": MUTED,
            "axes.spines.top": False,
            "axes.spines.right": False,
            "axes.grid": True,
            "grid.color": GRID,
            "grid.linewidth": 0.8,
            "axes.axisbelow": True,
            "figure.dpi": 100,
            "savefig.dpi": 200,
            "savefig.bbox": "tight",
            "savefig.facecolor": "white",
        }
    )


def heading(fig, title: str, subtitle: str) -> None:
    fig.text(0.0, 1.02, title, fontsize=13, fontweight="bold", ha="left", va="bottom")
    fig.text(0.0, 0.985, subtitle, fontsize=9.5, color=MUTED, ha="left", va="top")


def figure_accuracy(grades, ids, out: Path) -> None:
    fig, ax = plt.subplots(figsize=(8.4, 4.6))
    width, gap = 0.36, 0.02
    labels: list[str] = []
    for x, system in enumerate(SYSTEMS):
        local = [grades[("text", system, i)] == "correct" for i in ids]
        cohere = [grades[("cohere4", system, i)] == "correct" for i in ids]
        bars = [("text", local, tint(COLOUR[system]))]
        if system != "rag_base":
            bars.append(("cohere4", cohere, COLOUR[system]))
        offsets = [0.0] if len(bars) == 1 else [-(width + gap) / 2, (width + gap) / 2]
        for off, (_, hits, colour) in zip(offsets, bars):
            k, n = sum(hits), len(hits)
            lo, hi = wilson(k, n)
            ax.bar(x + off, k / n, width, color=colour, zorder=3)
            ax.vlines(x + off, lo, hi, color=INK, linewidth=1.1, zorder=4)
            ax.text(x + off + 0.01, hi + 0.015, f"{k / n:.1%}", ha="center", fontsize=9, zorder=5)
        gain = (sum(cohere) - sum(local)) / len(ids) * 100 if system != "rag_base" else None
        labels.append(system if gain is None else f"{system}\n{gain:+.1f} points")
    ax.set_xticks(range(len(SYSTEMS)))
    labels[0] = "rag_base\n(no reranker)"
    ax.set_xticklabels(labels)
    ax.set_ylim(0, 1.08)
    ax.set_yticks([0, 0.25, 0.5, 0.75, 1.0])
    ax.set_yticklabels(["0%", "25%", "50%", "75%", "100%"])
    ax.set_ylabel("answers fully correct")
    ax.grid(axis="x", visible=False)
    ax.legend(
        handles=[
            Patch(color="#C9CED6", label="local reranker (bge-reranker-v2-m3)"),
            Patch(color="#5E6670", label="Cohere rerank-v4.0-pro"),
        ],
        loc="lower right",
        frameon=False,
        fontsize=9,
        bbox_to_anchor=(1.0, 1.0),
        ncol=2,
        borderaxespad=0.2,
    )
    heading(
        fig,
        "Answer accuracy with a stronger reranker",
        f"{len(ids)} hand-graded questions, strict (only 'correct'); thin line = 95% Wilson interval",
    )
    fig.savefig(out / "fig_accuracy.png")
    plt.close(fig)


def figure_retrieval(df: pd.DataFrame, out: Path) -> None:
    metrics = [
        ("hit@5", "any right chunk\nin top 5"),
        ("all@5", "all right chunks\nin top 5"),
        ("all@8", "all right chunks\nin top 8"),
        ("mrr", "mean reciprocal\nrank"),
    ]
    ceiling = df[df["reranker"] == RERANKERS[0]]["pool_has_all"].mean()
    fig, ax = plt.subplots(figsize=(8.4, 4.6))
    width = 0.25
    for j, name in enumerate(RERANKERS):
        part = df[df["reranker"] == name]
        for i, (metric, _) in enumerate(metrics):
            value = part[metric].mean()
            x = i + (j - 1) * (width + 0.02)
            ax.bar(
                x,
                value,
                width,
                color=RERANKER_COLOUR[name],
                zorder=3,
                label=RERANKER_LABEL[name] if i == 0 else None,
            )
            ax.text(
                x,
                value + 0.012,
                f"{value:.2f}",
                ha="center",
                fontsize=8.5,
                zorder=6,
                bbox={"fc": "white", "ec": "none", "pad": 0.6},
            )
    ax.hlines(ceiling, 0.55, 2.45, color=INK, linestyle=(0, (4, 3)), linewidth=1.1, zorder=4)
    ax.set_xticks(range(len(metrics)))
    ax.set_xticklabels([m[1] for m in metrics])
    ax.set_ylim(0, 1.12)
    ax.set_ylabel("share of questions")
    ax.grid(axis="x", visible=False)
    ax.legend(
        loc="lower right",
        frameon=False,
        fontsize=9,
        bbox_to_anchor=(1.0, 1.0),
        ncol=3,
        borderaxespad=0.2,
        columnspacing=1.2,
        handlelength=1.2,
    )
    n = df["id"].nunique()
    heading(
        fig,
        "Retrieval: three rerankers on the same candidates",
        f"{n} answerable questions, same ~{df['pool_size'].mean():.0f}-chunk pool. "
        f"Dashed line = ceiling for 'all' ({ceiling:.1%} have every right chunk in the pool)",
    )
    fig.savefig(out / "fig_retrieval.png")
    plt.close(fig)


def figure_by_type(grades, ids, questions, out: Path) -> None:
    systems = SYSTEMS[1:]
    rows = []
    for t in TYPES:
        sub = [i for i in ids if questions[i]["type"] == t]
        n = len(sub) * len(systems)
        local = sum(grades[("text", s, i)] == "correct" for s in systems for i in sub) / n
        cohere = sum(grades[("cohere4", s, i)] == "correct" for s in systems for i in sub) / n
        rows.append((t, len(sub), local, cohere))
    fig, ax = plt.subplots(figsize=(8.4, 4.4))
    for y, (t, n, local, cohere) in enumerate(rows):
        ax.hlines(y, local, cohere, color="#C9CED6", linewidth=3, zorder=2)
        ax.scatter(local, y, s=70, color="white", edgecolor="#5E6670", linewidth=1.8, zorder=3)
        ax.scatter(cohere, y, s=70, color=BLUE, zorder=4)
        if abs(cohere - local) > 1e-9:
            hi = max(local, cohere)
            ax.text(hi + 0.025, y, f"{local:.0%} -> {cohere:.0%}", va="center", fontsize=9)
        else:
            ax.text(
                local + 0.025, y, f"{local:.0%} (no change)", va="center", fontsize=9, color=MUTED
            )
    ax.set_yticks(range(len(rows)))
    ax.set_yticklabels([f"{t.replace('_', ' ')}  (n={n})" for t, n, _, _ in rows])
    ax.invert_yaxis()
    ax.set_xlim(0.4, 1.17)
    ax.set_xticks([0.4, 0.6, 0.8, 1.0])
    ax.set_xticklabels(["40%", "60%", "80%", "100%"])
    ax.set_xlabel("answers fully correct (rag5, rag8 and agent_v3 pooled)")
    ax.grid(axis="y", visible=False)
    ax.legend(
        handles=[
            plt.Line2D(
                [],
                [],
                marker="o",
                linestyle="",
                markerfacecolor="white",
                markeredgecolor="#5E6670",
                markeredgewidth=1.8,
                markersize=8,
                label="local reranker",
            ),
            plt.Line2D(
                [],
                [],
                marker="o",
                linestyle="",
                color=BLUE,
                markersize=8,
                label="Cohere rerank-v4.0-pro",
            ),
        ],
        loc="lower right",
        frameon=False,
        fontsize=9,
        bbox_to_anchor=(1.0, 1.0),
        ncol=2,
        borderaxespad=0.2,
    )
    heading(
        fig,
        "Where the better reranker helps",
        "Comparison questions gain most; refusals and numeric questions do not change",
    )
    fig.savefig(out / "fig_by_type.png")
    plt.close(fig)


def figure_paired(grades, ids, out: Path) -> None:
    systems = SYSTEMS[1:]
    fig, ax = plt.subplots(figsize=(8.4, 3.4))
    for y, s in enumerate(systems):
        _, better, worse, p = paired(grades, ids, ("cohere4", s), ("text", s))
        tied = len(ids) - better - worse
        ax.barh(y, -worse, height=0.55, color=ORANGE, zorder=3)
        ax.barh(y, better, height=0.55, color=BLUE, zorder=3, left=0)
        ax.text(
            -worse - 0.6, y, f"{worse}", ha="right", va="center", fontsize=10, fontweight="bold"
        )
        ax.text(
            better + 0.6, y, f"{better}", ha="left", va="center", fontsize=10, fontweight="bold"
        )
        ax.text(
            17.5,
            y,
            f"{tied} unchanged\nsign test p = {p:.3f}",
            ha="right",
            va="center",
            fontsize=8.5,
            color=MUTED,
        )
    ax.axvline(0, color=INK, linewidth=1, zorder=4)
    ax.set_yticks(range(len(systems)))
    ax.set_yticklabels(systems)
    ax.invert_yaxis()
    ax.set_xlim(-6, 18)
    ax.set_xticks([-4, 0, 4, 8, 12])
    ax.set_xticklabels(["4", "0", "4", "8", "12"])
    ax.set_xlabel("number of questions")
    ax.grid(axis="y", visible=False)
    ax.set_ylim(2.85, -0.5)
    ax.text(
        -0.3, 2.62, "worse with Cohere", ha="right", fontsize=9, color=ORANGE, fontweight="bold"
    )
    ax.text(0.3, 2.62, "better with Cohere", ha="left", fontsize=9, color=BLUE, fontweight="bold")
    heading(
        fig,
        "Question by question: Cohere against the local reranker",
        f"{len(ids)} hand-graded questions; correct = 1, partial = 0.5, wrong = 0",
    )
    fig.savefig(out / "fig_paired.png")
    plt.close(fig)


SEARCH_COLOUR = {"dense": "#0072B2", "bm25": "#D55E00", "hybrid": "#CC79A7"}
SEARCH_LABEL = {"dense": "dense", "bm25": "BM25", "hybrid": "hybrid"}


def figure_ablation(df: pd.DataFrame, out: Path) -> None:
    """Search type with and without the company/year filter; top row: candidates only, bottom row:
    the same candidates reordered by the reranker."""
    metrics = [
        ("hit@1", "right chunk in first place"),
        ("all@5", "all right chunks in top 5"),
        ("all@10", "all right chunks in top 10"),
    ]
    rows = [(False, "candidates only (no reranker)"), (True, "then reordered by the reranker")]
    fig, axes = plt.subplots(2, 3, figsize=(12.5, 7.2), sharey=True)
    width = 0.34
    for r, (rerank, row_title) in enumerate(rows):
        for c, (metric, title) in enumerate(metrics):
            ax = axes[r][c]
            for i, source in enumerate(SEARCH_COLOUR):
                for k, use_filter in enumerate((False, True)):
                    part = df[
                        (df["source"] == source)
                        & (df["filter"] == use_filter)
                        & (df["rerank"] == rerank)
                    ]
                    value = part[metric].mean()
                    colour = SEARCH_COLOUR[source]
                    x = i + (k - 0.5) * (width + 0.02)
                    ax.bar(
                        x, value, width, color=tint(colour) if not use_filter else colour, zorder=3
                    )
                    ax.text(
                        x,
                        value + 0.012,
                        f"{value:.2f}".lstrip("0"),
                        ha="center",
                        fontsize=8,
                        zorder=5,
                    )
            ax.set_xticks(range(3))
            ax.set_xticklabels([SEARCH_LABEL[s] for s in SEARCH_COLOUR], fontsize=9.5)
            ax.grid(axis="x", visible=False)
            ax.set_ylim(0, 1.0)
            if r == 0:
                ax.set_title(title, fontsize=10, color=INK, loc="left")
            if c == 0:
                ax.set_ylabel(row_title, fontsize=9.5)
    handles = [
        Patch(color="#C9CED6", label="without filter"),
        Patch(color="#5E6670", label="with company/year filter"),
    ]
    fig.legend(
        handles=handles,
        loc="upper right",
        bbox_to_anchor=(0.995, 1.04),
        frameon=False,
        ncol=2,
        fontsize=9,
        handlelength=1.2,
    )
    fig.subplots_adjust(left=0.07, right=0.995, wspace=0.08, hspace=0.22)
    heading(
        fig,
        "Retrieval: search type, with and without the company/year filter",
        f"{df['id'].nunique()} answerable questions. Light bar = no filter, full colour = filter. "
        "Without a reranker, hybrid interleaves the two lists (first place = dense)",
    )
    fig.savefig(out / "fig_ablation.png")
    plt.close(fig)


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--results", type=Path, default=Path("results"))
    parser.add_argument("--questions", type=Path, default=Path("questions/questions.jsonl"))
    parser.add_argument("--out", type=Path, default=Path("docs/figures"))
    args = parser.parse_args(argv)
    args.out.mkdir(parents=True, exist_ok=True)
    questions = {
        q["id"]: q for q in map(json.loads, args.questions.read_text(encoding="utf-8").splitlines())
    }
    grades = load_grades(args.results)
    ids = sorted({i for (v, _, i) in grades if v == "text"})
    df = pd.read_json(args.results / "rerank_experiment.jsonl", lines=True)
    setup()
    figure_accuracy(grades, ids, args.out)
    figure_retrieval(df, args.out)
    figure_by_type(grades, ids, questions, args.out)
    figure_paired(grades, ids, args.out)
    ablation = args.results / "retrieval_ablation.jsonl"
    if ablation.exists():
        figure_ablation(pd.read_json(ablation, lines=True), args.out)
    print("figures written to", args.out)


if __name__ == "__main__":
    main()
