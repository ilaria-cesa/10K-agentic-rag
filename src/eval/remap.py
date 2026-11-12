"""Translate the gold evidence of each question from one chunking to another.

The questions point to chunk ids of the text version (`chunks_1600.jsonl`). The table-aware
version cuts the same filings differently (tables are separate chunks and are written value by
value), so the ids no longer mean the same thing. To compare the two versions on the same
questions, each old evidence chunk is matched to the new chunks of the same filing and section
that contain its content.

Matching is by shared words and numbers, not by line equality: a table row such as
`Total 254,098 209,586` and its structured form `iPhone: 2026 = $254,098; 2025 = $209,586` share
the numbers but not the line. A new chunk counts as evidence when it holds at least `min_cover`
of the old chunk's tokens, where rare tokens (numbers, names) weigh more than common words. When none does, the best one is used (and the question is
listed as weak), so every answerable question keeps at least one gold chunk.

Chunk boundaries move when tables are cut out, so the part of an old chunk that really answers
the question can land in a new chunk that holds only a small share of the old chunk. With
`focus_on_answer` only the lines of the old chunk that contain a number written in the gold
answer are matched (the whole chunk is matched when none does), and a new chunk counts when it
covers at least `min_cover` of any one of those lines.

This is an approximation: it was not checked by hand against every question.
"""

from __future__ import annotations

import math
import re
from collections import Counter

_TOKEN = re.compile(r"[a-z0-9]+")


def tokens(text: str) -> set[str]:
    return set(_TOKEN.findall(text.lower()))


def token_weights(chunks: list[dict]) -> dict[str, float]:
    """Inverse document frequency of each token over `chunks`."""
    df: Counter[str] = Counter()
    for chunk in chunks:
        df.update(tokens(chunk["text"]))
    return {t: math.log(1 + len(chunks) / n) for t, n in df.items()}


def remap_chunk(
    old_chunk: dict,
    new_chunks: list[dict],
    min_cover: float = 0.5,
    weights: dict[str, float] | None = None,
) -> tuple[list[str], float]:
    """(new chunk ids that hold the old evidence, best coverage found)."""
    weights = weights or token_weights(new_chunks)
    wanted = tokens(old_chunk["text"])
    total = sum(weights.get(t, 1.0) for t in wanted)
    if not total:
        return [], 0.0
    scored = []
    for chunk in new_chunks:
        if (chunk["ticker"], chunk["report_date"], chunk["section"]) != (
            old_chunk["ticker"],
            old_chunk["report_date"],
            old_chunk["section"],
        ):
            continue
        shared = wanted & tokens(chunk["text"])
        scored.append((sum(weights.get(t, 1.0) for t in shared) / total, chunk["chunk_id"]))
    if not scored:
        return [], 0.0
    best = max(score for score, _ in scored)
    picked = [cid for score, cid in scored if score >= min_cover]
    return (picked or [max(scored)[1]]), best


_NUMBER = re.compile(r"\d[\d,]*\.?\d*")


def answer_lines(old_text: str, answer: str, min_chars: int = 15) -> list[str]:
    """Lines of the old chunk that contain, as a whole number, a number (3+ characters, so not
    "5" or "1") written in the gold answer. Empty when the answer has no such number."""
    numbers = {n.rstrip(".,") for n in _NUMBER.findall(answer) if len(n.rstrip(".,")) >= 3}
    patterns = [re.compile(rf"(?<![\d,.]){re.escape(n)}(?![\d]|,\d)") for n in numbers]
    lines = [line.strip() for line in old_text.splitlines() if len(line.strip()) >= min_chars]
    return [line for line in lines if any(p.search(line) for p in patterns)]


def remap_lines(
    old_chunk: dict,
    lines: list[str],
    new_chunks: list[dict],
    min_cover: float,
    weights: dict[str, float],
) -> tuple[list[str], float]:
    """New chunks (same filing and section) that cover at least `min_cover` of any one line."""
    key = (old_chunk["ticker"], old_chunk["report_date"], old_chunk["section"])
    pool = [c for c in new_chunks if (c["ticker"], c["report_date"], c["section"]) == key]
    pool_tokens = [(c["chunk_id"], tokens(c["text"])) for c in pool]
    picked: dict[str, float] = {}
    best = 0.0
    for line in lines:
        wanted = tokens(line)
        total = sum(weights.get(t, 1.0) for t in wanted)
        if not total:
            continue
        for chunk_id, have in pool_tokens:
            cover = sum(weights.get(t, 1.0) for t in wanted & have) / total
            best = max(best, cover)
            if cover >= min_cover:
                picked[chunk_id] = max(cover, picked.get(chunk_id, 0.0))
    if not picked and pool_tokens:  # nothing reached the threshold: keep the best single match
        line_tokens = tokens(" ".join(lines))
        picked = {max(pool_tokens, key=lambda p: len(p[1] & line_tokens))[0]: best}
    return list(picked), best


def remap_questions(
    questions: list[dict],
    old_chunks: list[dict],
    new_chunks: list[dict],
    min_cover: float = 0.5,
    focus_on_answer: bool = False,
) -> tuple[list[dict], list[str]]:
    """Copy of `questions` with `evidence_chunk_ids` in the new chunking, and the ids of the
    questions whose best match covered less than `min_cover` of the old evidence (weak)."""
    old_by_id = {c["chunk_id"]: c for c in old_chunks}
    weights = token_weights(new_chunks)
    out, weak = [], []
    for q in questions:
        ids, is_weak = [], False
        for old_id in q["evidence_chunk_ids"]:
            old = old_by_id[old_id]
            lines = answer_lines(old["text"], q["answer"]) if focus_on_answer else []
            if lines:
                new_ids, best = remap_lines(old, lines, new_chunks, min_cover, weights)
            else:
                new_ids, best = remap_chunk(old, new_chunks, min_cover, weights)
            ids.extend(i for i in new_ids if i not in ids)
            is_weak = is_weak or best < min_cover
        if is_weak:
            weak.append(q["id"])
        out.append({**q, "evidence_chunk_ids": ids})
    return out, weak
