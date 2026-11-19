"""Reranking through the Cohere Rerank API, with the same `score(question, passages)` interface
as the local cross-encoder (`rerank.Reranker`), so it can be plugged into `retrieve` unchanged.

Why a cache: a free trial key allows only a small number of calls (1,000 a month at the time of
writing, 10 a minute for rerank). Every answer is written to a file as soon as it arrives, so a
stopped run restarts without paying for the same call twice. `max_calls` stops the run before the
quota is used up.

The key is read from the environment or passed by the caller; it is never written to the cache.
"""

from __future__ import annotations

import hashlib
import json
import time
from pathlib import Path


class CohereReranker:
    def __init__(
        self,
        api_key: str | None = None,
        model: str = "rerank-v3.5",
        cache_path: str | Path | None = None,
        min_interval: float = 6.5,  # seconds between calls (10 a minute on a trial key)
        max_calls: int | None = None,
        max_retries: int = 3,
        client=None,  # for tests
        sleep=time.sleep,
    ):
        if client is None:
            import cohere  # lazy: the package is only needed for this experiment

            client = cohere.ClientV2(api_key=api_key)
        self.client = client
        self.model = model
        self.min_interval = min_interval
        self.max_calls = max_calls
        self.max_retries = max_retries
        self.sleep = sleep
        self.calls = 0  # real API calls made by this object (cache hits do not count)
        self._last_call = 0.0
        self.cache_path = Path(cache_path) if cache_path else None
        self._cache: dict[str, list[float]] = {}
        if self.cache_path and self.cache_path.exists():
            for line in self.cache_path.read_text(encoding="utf-8").splitlines():
                if line.strip():
                    row = json.loads(line)
                    self._cache[row["key"]] = row["scores"]

    def _key(self, question: str, passages: list[str]) -> str:
        digest = hashlib.sha1()
        for part in [self.model, question, *passages]:
            digest.update(part.encode("utf-8"))
            digest.update(b"\x00")
        return digest.hexdigest()

    def score(self, question: str, passages: list[str]) -> list[float]:
        """One relevance score per passage, in the order given. One API call per question."""
        key = self._key(question, passages)
        if key in self._cache:
            return self._cache[key]
        if self.max_calls is not None and self.calls >= self.max_calls:
            raise RuntimeError(f"stopped: {self.calls} API calls made (max_calls)")
        for attempt in range(self.max_retries + 1):
            wait = self.min_interval - (time.monotonic() - self._last_call)
            if wait > 0:
                self.sleep(wait)
            self._last_call = time.monotonic()
            try:
                response = self.client.rerank(
                    model=self.model, query=question, documents=passages, top_n=len(passages)
                )
                break
            except Exception:  # rate limit or network error: wait and try again
                if attempt == self.max_retries:
                    raise
                self.sleep(self.min_interval * (attempt + 2))
        self.calls += 1
        scores = [0.0] * len(passages)
        for item in response.results:
            scores[item.index] = float(item.relevance_score)
        self._cache[key] = scores
        if self.cache_path:
            self.cache_path.parent.mkdir(parents=True, exist_ok=True)
            with open(self.cache_path, "a", encoding="utf-8") as f:
                f.write(json.dumps({"key": key, "scores": scores}) + "\n")
        return scores
