import numpy as np
import pytest

from src.eval.ablation import CachedScorer, grid, run_ablation
from src.retrieval.pipeline import Corpus, RetrievalConfig, retrieve


def make_corpus():
    texts = ["apple repurchased shares", "banana loans", "cherry deposits", "zebra insurance"]
    chunks = [
        {"chunk_id": f"c{i}", "embed_text": t, "ticker": "A", "report_date": "2023-12-31"}
        for i, t in enumerate(texts)
    ]
    return Corpus(chunks, np.eye(4, dtype=np.float32))


class Same:
    def score(self, question, passages):
        return [0.0] * len(passages)


def test_the_grid_has_every_combination_once():
    names = [c.name for c in grid()]
    assert len(names) == len(set(names)) == 12
    assert "bm25 + filter + rerank" in names and "dense" in names and "hybrid + rerank" in names


def test_keyword_only_search_ignores_the_vector():
    corpus = make_corpus()
    config = RetrievalConfig("bm25", hybrid=True, dense=False, pool=2, bm25_pool=2)
    wrong_vector = np.array([0, 0, 0, 1], dtype=np.float32)  # points at "zebra insurance"
    idx, _ = retrieve(corpus, "apple shares", wrong_vector, config)
    assert idx == [0]  # only the chunk that has the words; the dense list would have said 3


def test_a_config_needs_some_search():
    with pytest.raises(ValueError):
        RetrievalConfig("none", dense=False, hybrid=False)


def test_cached_scorer_asks_the_model_only_for_new_pairs():
    calls = []

    class Counting:
        def score(self, question, passages):
            calls.append(list(passages))
            return [float(len(p)) for p in passages]

    scorer = CachedScorer(Counting())
    assert scorer.score("q", ["aa", "b"]) == [2.0, 1.0]
    assert scorer.score("q", ["b", "ccc"]) == [1.0, 3.0]
    assert calls == [["aa", "b"], ["ccc"]]


def test_run_ablation_rows():
    corpus = make_corpus()
    q = {
        "id": "Q1",
        "split": "dev",
        "type": "fact",
        "question": "apple shares",
        "evidence_chunk_ids": ["c0"],
    }
    rows = run_ablation(corpus, [q], np.eye(4, dtype=np.float32)[:1], Same())
    by = {r["config"]: r for r in rows}
    assert len(rows) == 12
    assert by["bm25"]["hit@1"] == 1.0 and by["bm25"]["source"] == "bm25"
    assert by["dense"]["hit@1"] == 1.0 and by["hybrid + rerank"]["rerank"] is True
