import numpy as np

from src.retrieval.embed import embed_texts, length_report, load_chunks, token_lengths
from src.retrieval.index import build_index, search


def test_search_returns_the_nearest_vector_first_and_is_cosine_based():
    rng = np.random.default_rng(0)
    vectors = rng.normal(size=(50, 16)).astype(np.float32)
    index = build_index(vectors)
    scores, ids = search(index, vectors[[7, 21]] * 5.0, k=3)  # scaling must not matter
    assert ids[:, 0].tolist() == [7, 21]
    assert np.allclose(scores[:, 0], 1.0, atol=1e-5)
    assert (np.diff(scores, axis=1) <= 1e-6).all()


def test_build_index_does_not_modify_the_input():
    vectors = np.arange(12, dtype=np.float32).reshape(3, 4) + 1
    before = vectors.copy()
    build_index(vectors)
    assert np.array_equal(vectors, before)


class FakeTokenizer:
    def __call__(self, texts, add_special_tokens, truncation):
        return {"input_ids": [[0] * (len(t.split()) + 2) for t in texts]}


def test_token_lengths_and_report_count_texts_over_the_limit():
    texts = ["a b c", "a " * 10, "a " * 100]
    lengths = token_lengths(texts, FakeTokenizer(), batch_size=2)
    assert lengths == [5, 12, 102]
    report = length_report(lengths, limit=50)
    assert report["n_over_limit"] == 1 and report["max"] == 102 and report["n_texts"] == 3


class FakeModel:
    def encode(self, texts, **kwargs):
        assert kwargs["normalize_embeddings"] is True
        return np.ones((len(texts), 4), dtype=np.float16)


def test_embed_texts_returns_float32():
    out = embed_texts(FakeModel(), ["x", "y"])
    assert out.dtype == np.float32 and out.shape == (2, 4)


def test_load_chunks_reads_jsonl(tmp_path):
    f = tmp_path / "c.jsonl"
    f.write_text('{"a": 1}\n{"a": 2}\n', encoding="utf-8")
    assert [c["a"] for c in load_chunks(f)] == [1, 2]
