"""Retrieval ablation: every combination of search type, company/year filter and reranker.

Search type: dense (embeddings), bm25 (keywords only) or hybrid (both, candidates joined). With the
filter off the whole corpus is searched; with it on, only the filings the question names. Each
combination is run with and without the cross-encoder reranker. No language model is involved.

All searches give the same number of candidates (30 per list) so that the combinations differ only
in what they are meant to differ in. Without a reranker the hybrid lists are interleaved.
"""

from __future__ import annotations

import numpy as np

from src.eval.metrics import r_precision, retrieval_metrics
from src.retrieval.pipeline import Corpus, RetrievalConfig, retrieve

KS = (1, 3, 5, 8, 10)
POOL = 30
SOURCES = ("dense", "bm25", "hybrid")


def grid() -> list[RetrievalConfig]:
    configs = []
    for source in SOURCES:
        for use_filter in (False, True):
            for rerank in (False, True):
                name = (
                    source + (" + filter" if use_filter else "") + (" + rerank" if rerank else "")
                )
                configs.append(
                    RetrievalConfig(
                        name,
                        use_filter=use_filter,
                        rerank=rerank,
                        pool=POOL,
                        dense=source != "bm25",
                        hybrid=source != "dense",
                        bm25_pool=POOL if source == "bm25" else 20,
                    )
                )
    return configs


class CachedScorer:
    """Remembers the score of every (question, passage) pair, so a pair that appears in the
    candidates of several combinations is scored by the model only once."""

    def __init__(self, scorer):
        self.scorer = scorer
        self.cache: dict[tuple[str, str], float] = {}

    def score(self, question: str, passages: list[str]) -> list[float]:
        new = [p for p in dict.fromkeys(passages) if (question, p) not in self.cache]
        if new:
            for passage, value in zip(new, self.scorer.score(question, new)):
                self.cache[(question, passage)] = value
        return [self.cache[(question, p)] for p in passages]


def run_ablation(
    corpus: Corpus,
    questions: list[dict],
    query_vectors: np.ndarray,
    reranker,
    configs: list[RetrievalConfig] | None = None,
) -> list[dict]:
    """One row per (question, combination) with the retrieval metrics and the ranked chunk ids."""
    rows = []
    scorer = CachedScorer(reranker)
    for config in configs or grid():
        for q, vector in zip(questions, query_vectors):
            idx, _ = retrieve(corpus, q["question"], vector, config, scorer)
            ranked = [corpus.chunks[i]["chunk_id"] for i in idx]
            rows.append(
                {
                    "config": config.name,
                    "source": config.name.split(" ")[0],
                    "filter": config.use_filter,
                    "rerank": config.rerank,
                    "id": q["id"],
                    "split": q["split"],
                    "type": q["type"],
                    "ranked": ranked,
                }
                | retrieval_metrics(q["evidence_chunk_ids"], ranked, KS)
                | {"r_precision": r_precision(q["evidence_chunk_ids"], ranked)}
            )
    return rows
