import numpy as np

from src.retrieval.bm25 import BM25, tokenize


def test_tokenize_lowercases_and_drops_stopwords():
    assert tokenize("What was the LCR in 2025?") == ["lcr", "2025"]


def test_rows_with_the_query_words_rank_first_and_scope_is_respected():
    docs = [
        "Adjusted earnings 200 55 Other revenues 932 718",
        "Risk factors about climate and weather",
        "Other revenues grew in the segment",
        "Adjusted earnings other revenues table",
    ]
    bm = BM25(docs)
    top = bm.top("adjusted earnings other revenues", np.array([0, 1, 2]), 2)
    assert next(i for i, _ in top) == 0
    assert 3 not in [i for i, _ in top]
    assert bm.top("zzz unknown", np.array([0, 1]), 3) == []
