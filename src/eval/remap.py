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


def remap_questions(
    questions: list[dict],
    old_chunks: list[dict],
    new_chunks: list[dict],
    min_cover: float = 0.5,
) -> tuple[list[dict], list[str]]:
    """Copy of `questions` with `evidence_chunk_ids` in the new chunking, and the ids of the
    questions whose best match covered less than `min_cover` of the old evidence (weak)."""
    old_by_id = {c["chunk_id"]: c for c in old_chunks}
    weights = token_weights(new_chunks)
    out, weak = [], []
    for q in questions:
        ids, is_weak = [], False
        for old_id in q["evidence_chunk_ids"]:
            new_ids, best = remap_chunk(old_by_id[old_id], new_chunks, min_cover, weights)
            ids.extend(i for i in new_ids if i not in ids)
            is_weak = is_weak or best < min_cover
        if is_weak:
            weak.append(q["id"])
        out.append({**q, "evidence_chunk_ids": ids})
    return out, weak
