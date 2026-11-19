"""Retrieval-only comparison of rerankers: no generation, so it is cheap to repeat.

For every answerable question the candidate pool of `rag8` (company/year filter, dense + BM25
candidates) is built once. Each reranker then orders the same pool, and the order is scored with
the usual retrieval metrics. Because the pool is shared, any difference comes from the reranker.

`pool_has_all` is the ceiling: if a gold chunk is not in the pool, no reranker can bring it up.
"""

from __future__ import annotations

import numpy as np

from src.eval.metrics import retrieval_metrics
from src.retrieval.pipeline import Corpus, RetrievalConfig, candidate_pool

KS = (1, 3, 5, 8, 10)
POOL_CONFIG = RetrievalConfig("filter + hybrid", use_filter=True, hybrid=True, rerank=False)


def rank_pool(scorer, question: str, passages: list[str]) -> list[int]:
    """Positions in `passages`, best first (ties keep the pool order)."""
    scores = scorer.score(question, passages)
    return sorted(range(len(passages)), key=lambda i: -scores[i])


def evaluate(
    corpus: Corpus,
    questions: list[dict],
    query_vectors: np.ndarray,
    scorers: dict,
    config: RetrievalConfig = POOL_CONFIG,
) -> list[dict]:
    """One row per (question, reranker) with the retrieval metrics and the ranked chunk ids."""
    rows = []
    for q, vector in zip(questions, query_vectors):
        pool = candidate_pool(corpus, q["question"], vector, config)
        ids = [corpus.chunks[i]["chunk_id"] for i in pool]
        passages = [corpus.chunks[i]["embed_text"] for i in pool]
        gold = set(q["evidence_chunk_ids"])
        base = {
            "id": q["id"],
            "split": q["split"],
            "type": q["type"],
            "pool_size": len(pool),
            "pool_has_all": float(gold <= set(ids)),
        }
        for name, scorer in scorers.items():
            order = rank_pool(scorer, q["question"], passages)
            ranked = [ids[i] for i in order]
            rows.append(
                base
                | {"reranker": name, "ranked": ranked[:10]}
                | retrieval_metrics(q["evidence_chunk_ids"], ranked, KS)
            )
    return rows
