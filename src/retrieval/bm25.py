"""Keyword search (BM25) over the chunks, used next to the dense search.

Dense embeddings match meaning, but many numeric tables have almost no prose: their only handle
is the row labels ("Adjusted earnings", "Other revenues", "LCR"). BM25 scores exact words, so it
finds those tables. No extra library: the corpus is small (about 10k chunks).
"""

from __future__ import annotations

import math
import re
from collections import Counter

import numpy as np

_TOKEN = re.compile(r"[a-z0-9]+")
STOPWORDS = frozenset(
    [
        "a",
        "an",
        "and",
        "are",
        "as",
        "at",
        "be",
        "by",
        "did",
        "do",
        "does",
        "for",
        "from",
        "how",
        "in",
        "is",
        "it",
        "its",
        "of",
        "on",
        "or",
        "that",
        "the",
        "this",
        "to",
        "was",
        "were",
        "what",
        "which",
        "who",
        "whose",
        "with",
        "between",
        "than",
        "year",
        "years",
        "fiscal",
        "change",
        "changed",
    ]
)


def tokenize(text: str) -> list[str]:
    return [t for t in _TOKEN.findall(text.lower()) if t not in STOPWORDS]


class BM25:
    def __init__(self, documents: list[str], k1: float = 1.5, b: float = 0.75):
        self.k1, self.b = k1, b
        self.counts = [Counter(tokenize(d)) for d in documents]
        self.lengths = np.array([sum(c.values()) for c in self.counts], dtype=np.float64)
        self.avg_length = float(self.lengths.mean()) if len(documents) else 0.0
        df: Counter[str] = Counter()
        for c in self.counts:
            df.update(c.keys())
        n = len(documents)
        self.idf = {t: math.log(1 + (n - f + 0.5) / (f + 0.5)) for t, f in df.items()}

    def top(self, query: str, idx: np.ndarray, k: int) -> list[tuple[int, float]]:
        """The `k` best documents among the positions in `idx`, as (position, score)."""
        terms = [t for t in dict.fromkeys(tokenize(query)) if t in self.idf]
        if not terms:
            return []
        scores = np.zeros(len(idx))
        for pos, i in enumerate(idx):
            counts = self.counts[i]
            norm = self.k1 * (1 - self.b + self.b * self.lengths[i] / self.avg_length)
            for t in terms:
                f = counts.get(t, 0)
                if f:
                    scores[pos] += self.idf[t] * f * (self.k1 + 1) / (f + norm)
        order = [p for p in np.argsort(-scores)[:k] if scores[p] > 0]
        return [(int(idx[p]), float(scores[p])) for p in order]
