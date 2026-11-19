from types import SimpleNamespace

import pytest

from src.retrieval.rerank_api import CohereReranker


class FakeClient:
    """Returns the passages in reverse order of their position, like a real API would shuffle."""

    def __init__(self, fail_first: int = 0):
        self.calls = 0
        self.fail_first = fail_first

    def rerank(self, model, query, documents, top_n):
        self.calls += 1
        if self.calls <= self.fail_first:
            raise ConnectionError("rate limit")
        n = len(documents)
        items = [SimpleNamespace(index=i, relevance_score=(i + 1) / n) for i in reversed(range(n))]
        return SimpleNamespace(results=items)


def make(tmp_path=None, **kwargs):
    path = tmp_path / "cache.jsonl" if tmp_path else None
    client = kwargs.pop("client", None) or FakeClient()
    return CohereReranker(client=client, cache_path=path, sleep=lambda s: None, **kwargs), client


def test_scores_come_back_in_the_order_of_the_passages():
    ranker, _ = make()
    assert ranker.score("q", ["a", "b", "c", "d"]) == [0.25, 0.5, 0.75, 1.0]


def test_cache_avoids_a_second_call_and_survives_a_restart(tmp_path):
    ranker, client = make(tmp_path)
    first = ranker.score("q", ["a", "b"])
    assert ranker.score("q", ["a", "b"]) == first and client.calls == 1
    again, client2 = make(tmp_path)  # a new object reads the file
    assert again.score("q", ["a", "b"]) == first and client2.calls == 0
    assert again.score("q", ["a", "c"]) != [] and client2.calls == 1  # other passages: new call


def test_max_calls_protects_the_quota():
    ranker, _ = make(max_calls=1)
    ranker.score("q1", ["a"])
    with pytest.raises(RuntimeError):
        ranker.score("q2", ["a"])


def test_a_failed_call_is_retried_and_then_given_up():
    ranker, client = make(client=FakeClient(fail_first=1))
    assert ranker.score("q", ["a", "b"]) == [0.5, 1.0] and client.calls == 2
    ranker, _ = make(client=FakeClient(fail_first=10), max_retries=2)
    with pytest.raises(ConnectionError):
        ranker.score("q", ["a"])


def test_the_key_is_not_written_to_the_cache(tmp_path):
    ranker, _ = make(tmp_path)
    ranker.score("q", ["a"])
    assert "api" not in (tmp_path / "cache.jsonl").read_text().lower()
