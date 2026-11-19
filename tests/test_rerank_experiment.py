import numpy as np

from src.eval.rerank_experiment import evaluate
from src.retrieval.pipeline import Corpus, RetrievalConfig


class ByLength:
    def __init__(self, reverse=False):
        self.sign = -1 if reverse else 1

    def score(self, question, passages):
        return [self.sign * len(p) for p in passages]


def test_same_pool_different_order_and_ceiling():
    chunks = [
        {
            "chunk_id": f"c{i}",
            "embed_text": "x" * (i + 1),
            "ticker": "A",
            "report_date": "2023-12-31",
        }
        for i in range(6)
    ]
    vectors = np.eye(6, dtype=np.float32)
    q = {
        "id": "Q1",
        "split": "dev",
        "type": "lookup",
        "question": "what?",
        "evidence_chunk_ids": ["c5"],
    }
    config = RetrievalConfig("pool", use_filter=False, rerank=False, pool=6)
    rows = evaluate(
        Corpus(chunks, vectors),
        [q],
        vectors[:1],
        {"long": ByLength(), "short": ByLength(True)},
        config,
    )
    by = {r["reranker"]: r for r in rows}
    assert by["long"]["pool_size"] == by["short"]["pool_size"] == 6
    assert by["long"]["pool_has_all"] == 1.0
    assert by["long"]["hit@1"] == 1.0 and by["short"]["hit@1"] == 0.0
    assert by["short"]["hit@8"] == 1.0
