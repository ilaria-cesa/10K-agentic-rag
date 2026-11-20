"""Summaries of the hand grades: accuracy with Wilson intervals, paired comparisons, text vs tables.

Usage:  python -m src.eval.grades_report [--results results] [--questions questions/questions.jsonl]

Reads results/hand_grades/ (the grading sheets, shuffled and blind, with their keys). A row of a sheet
can stand for several answers (identical answers share a row), and the keys say which ones.
Grades: correct = 1, partial = 0.5, wrong = 0 for the paired comparisons; "strict" = only correct.
"""

from __future__ import annotations

import argparse
import json
import math
from collections import defaultdict
from math import comb
from pathlib import Path

import openpyxl

SYSTEMS = ["rag_base", "rag5", "rag8", "agent_v3"]
SCORE = {"correct": 1.0, "partial": 0.5, "wrong": 0.0}
TYPES = ["fact", "numeric", "comparison", "multi_hop", "false_premise", "unanswerable"]


def wilson(k: int, n: int, z: float = 1.96) -> tuple[float, float]:
    p = k / n
    denom = 1 + z * z / n
    centre = (p + z * z / (2 * n)) / denom
    half = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / denom
    return centre - half, centre + half


def sign_test(better: int, worse: int) -> float:
    """Two-sided exact sign test (ties are ignored)."""
    n = better + worse
    if n == 0:
        return 1.0
    return min(1.0, 2 * sum(comb(n, i) for i in range(min(better, worse) + 1)) / 2**n)


def read_sheet(
    sheet: Path, key: Path, variant: str | None = None
) -> dict[tuple[str, str, str], str]:
    """{(variant, system, question id): grade}. `variant` overrides the one written in the key."""
    keys = json.loads(key.read_text(encoding="utf-8"))
    ws = openpyxl.load_workbook(sheet, data_only=True)["grading"]
    grades = {}
    for row in ws.iter_rows(min_row=2, values_only=True):
        for run in keys[str(row[0])]["runs"]:
            grades[(variant or run["variant"], run["system"], row[1])] = row[6]
    return grades


def load_grades(results: Path) -> dict[tuple[str, str, str], str]:
    hand = results / "hand_grades"
    grades = read_sheet(hand / "grading_all.xlsx", hand / "grading_all_key.json")
    grades.update(
        read_sheet(
            hand / "grading_tables_final.xlsx", hand / "grading_tables_final_key.json", "final"
        )
    )
    reused = json.loads((hand / "grading_tables_final_reused.json").read_text(encoding="utf-8"))
    for name, item in reused.items():  # "system|question id"
        system, qid = name.split("|")
        grades[("final", system, qid)] = item["grade"]
    sheet = hand / "grading_text_cohere4.xlsx"  # the rerun with the Cohere reranker (optional)
    if sheet.exists():
        grades.update(read_sheet(sheet, hand / "grading_text_cohere4_key.json", "cohere4"))
        reused = json.loads((hand / "grading_text_cohere4_reused.json").read_text(encoding="utf-8"))
        for name, item in reused.items():
            system, qid = name.split("|")
            grades[("cohere4", system, qid)] = item["grade"]
        for (variant, system, qid), grade in list(grades.items()):
            if (
                variant == "text" and system == "rag_base"
            ):  # no reranker: not run again, same answers
                grades[("cohere4", system, qid)] = grade
    return grades


def accuracy_rows(grades, ids, variant, label):
    print(f"\n{label}: {len(ids)} questions")
    print("system     correct partial wrong  strict (95% CI)       lenient")
    for system in SYSTEMS:
        g = [grades[(variant, system, i)] for i in ids]
        c, p = g.count("correct"), g.count("partial")
        lo, hi = wilson(c, len(g))
        print(
            f"{system:10s} {c:5d} {p:7d} {g.count('wrong'):5d}  "
            f"{c / len(g):.3f} ({lo:.2f}-{hi:.2f})    {(c + p) / len(g):.3f}"
        )


def paired(grades, ids, a, b):
    """(mean score difference a - b, a better, b better, sign test p)."""
    diffs = [SCORE[grades[a + (i,)]] - SCORE[grades[b + (i,)]] for i in ids]
    better, worse = sum(d > 0 for d in diffs), sum(d < 0 for d in diffs)
    return sum(diffs) / len(diffs), better, worse, sign_test(better, worse)


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--results", type=Path, default=Path("results"))
    parser.add_argument("--questions", type=Path, default=Path("questions/questions.jsonl"))
    args = parser.parse_args(argv)
    questions = {
        q["id"]: q for q in map(json.loads, args.questions.read_text(encoding="utf-8").splitlines())
    }
    grades = load_grades(args.results)
    ids = sorted({i for (v, _, i) in grades if v == "text"})
    split = {i: questions[i]["split"] for i in ids}

    accuracy_rows(grades, ids, "text", "Text chunks, all hand-graded")
    for name in ["test", "agentic"]:
        accuracy_rows(grades, [i for i in ids if split[i] == name], "text", f"Text chunks, {name}")

    print("\nFully correct answers by question type (text chunks)")
    print("type            n  " + "  ".join(f"{s:9s}" for s in SYSTEMS))
    for t in TYPES:
        sub = [i for i in ids if questions[i]["type"] == t]
        counts = [sum(grades[("text", s, i)] == "correct" for i in sub) for s in SYSTEMS]
        print(f"{t:14s} {len(sub):3d}  " + "  ".join(f"{c:9d}" for c in counts))

    print("\nPaired comparisons, text chunks (mean score difference, better, worse, sign test p)")
    pairs = [("rag5", "rag_base"), ("rag8", "rag5"), ("rag8", "rag_base")]
    pairs += [("agent_v3", "rag8"), ("agent_v3", "rag5"), ("agent_v3", "rag_base")]
    for a, b in pairs:
        d, w, ll, p = paired(grades, ids, ("text", a), ("text", b))
        print(f"{a} - {b}: {d:+.3f}  {w}/{ll}  p={p:.3f}")
    agentic = [i for i in ids if split[i] == "agentic"]
    for a, b in [("agent_v3", "rag8"), ("agent_v3", "rag_base")]:
        d, w, ll, p = paired(grades, agentic, ("text", a), ("text", b))
        print(f"agentic only, {a} - {b}: {d:+.3f}  {w}/{ll}  p={p:.3f}")

    accuracy_rows(grades, ids, "final", "Table-aware chunks, all hand-graded")
    print("\nTable-aware chunks minus text chunks (mean score difference, better, worse, p)")
    for s in SYSTEMS:
        d, w, ll, p = paired(grades, ids, ("final", s), ("text", s))
        print(f"{s:10s} {d:+.3f}  {w}/{ll}  p={p:.3f}")

    if ("cohere4", "rag5", ids[0]) in grades:
        accuracy_rows(
            grades, ids, "cohere4", "Text chunks + Cohere reranker (rag_base is unchanged)"
        )
        print("\nCohere reranker minus local reranker (mean score difference, better, worse, p)")
        for s in SYSTEMS[1:]:
            d, w, ll, p = paired(grades, ids, ("cohere4", s), ("text", s))
            print(f"{s:10s} {d:+.3f}  {w}/{ll}  p={p:.3f}")
        for name in ["test", "agentic"]:
            sub = [i for i in ids if split[i] == name]
            for s in SYSTEMS[1:]:
                d, w, ll, p = paired(grades, sub, ("cohere4", s), ("text", s))
                print(f"  {name:8s} {s:10s} {d:+.3f}  {w}/{ll}  p={p:.3f}")
        print("\nFully correct by question type: local -> Cohere reranker (rag5, rag8, agent_v3)")
        for t in TYPES:
            sub = [i for i in ids if questions[i]["type"] == t]
            row = [
                f"{sum(grades[('text', s, i)] == 'correct' for i in sub)}->"
                f"{sum(grades[('cohere4', s, i)] == 'correct' for i in sub)}"
                for s in SYSTEMS[1:]
            ]
            print(f"{t:14s} {len(sub):3d}  " + "  ".join(f"{c:7s}" for c in row))

    print("\nMean score by question type, all four systems pooled: text -> table-aware")
    by_type = defaultdict(lambda: ([], []))
    for s in SYSTEMS:
        for i in ids:
            by_type[questions[i]["type"]][0].append(SCORE[grades[("text", s, i)]])
            by_type[questions[i]["type"]][1].append(SCORE[grades[("final", s, i)]])
    for t in TYPES:
        a, b = by_type[t]
        print(f"{t:14s} n={len(a):3d}  {sum(a) / len(a):.3f} -> {sum(b) / len(b):.3f}")


if __name__ == "__main__":
    main()
