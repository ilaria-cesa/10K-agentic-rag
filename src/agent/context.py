"""Pick the passages that go to the answering model from everything the agent collected."""

from __future__ import annotations

from collections.abc import Callable

ScoreFn = Callable[[str, list[dict]], list[float]]


def select_context(
    question: str, evidence: list[dict], score_fn: ScoreFn | None, max_chunks: int
) -> list[dict]:
    """Keep the best passage of every filing found, then fill with the best of the rest.

    Scores come from the reranker applied to the ORIGINAL question, so a query that was rewritten
    too loosely cannot push the useful passage out. Keeping one passage per filing guarantees that
    a comparison between two years still sees both years. The result is ordered best first.
    """
    if score_fn is None or len(evidence) <= 1:
        return evidence[:max_chunks]
    scores = score_fn(question, evidence)
    ranked = sorted(range(len(evidence)), key=lambda i: -scores[i])
    chosen: list[int] = []
    seen_filings: set[tuple[str, str]] = set()
    for i in ranked:
        filing = (evidence[i]["ticker"], evidence[i]["report_date"])
        if filing not in seen_filings and len(chosen) < max_chunks:
            seen_filings.add(filing)
            chosen.append(i)
    for i in ranked:
        if i not in chosen and len(chosen) < max_chunks:
            chosen.append(i)
    return [evidence[i] for i in sorted(chosen, key=lambda i: -scores[i])]


def select_context_by_search(
    question: str,
    evidence: list[dict],
    groups: list[list[str]],
    score_fn: ScoreFn | None,
    max_chunks: int,
    per_group: int = 2,
) -> list[dict]:
    """Keep the best passages of EVERY search first, then fill with the best of the rest.

    `groups` holds, for each search that aims at one company and year, the chunk ids in the order
    that search ranked them (its own query, so the passage with the exact number comes first).
    Ranking those passages against the original question instead can drop them: for "which
    company has the largest workforce" a general passage on people policy can outrank the one that
    states the headcount. The remaining places go to the best passages for the original question.
    """
    by_id = {c["chunk_id"]: c for c in evidence}
    if not groups:
        return select_context(question, evidence, score_fn, max_chunks)
    share = max(1, min(per_group, max_chunks // len(groups)))
    chosen: list[str] = []
    for rank in range(share):
        for ids in groups:
            if rank < len(ids) and ids[rank] not in chosen and len(chosen) < max_chunks:
                chosen.append(ids[rank])
    rest = [cid for cid in by_id if cid not in chosen]
    if score_fn is not None and rest:
        scores = score_fn(question, [by_id[cid] for cid in rest])
        rest = [cid for _, cid in sorted(zip(scores, rest), key=lambda t: -t[0])]
    chosen += rest[: max_chunks - len(chosen)]
    if score_fn is not None and len(chosen) > 1:
        scores = score_fn(question, [by_id[cid] for cid in chosen])
        chosen = [cid for _, cid in sorted(zip(scores, chosen), key=lambda t: -t[0])]
    return [by_id[cid] for cid in chosen]
