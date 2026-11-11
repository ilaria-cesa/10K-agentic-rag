"""Exact vector search with FAISS (inner product on L2-normalised vectors = cosine similarity).

Why a flat (exact) index: with about 10 thousand vectors an exact search takes milliseconds,
so approximate indexes (IVF, HNSW) would only add tuning work and a small recall loss.
"""

from __future__ import annotations

import faiss
import numpy as np


def _as_unit_float32(vectors: np.ndarray) -> np.ndarray:
    vectors = np.ascontiguousarray(vectors, dtype=np.float32)
    faiss.normalize_L2(vectors)  # in place; harmless if the vectors are already normalised
    return vectors


def build_index(embeddings: np.ndarray) -> faiss.IndexFlatIP:
    vectors = _as_unit_float32(embeddings.copy())
    index = faiss.IndexFlatIP(vectors.shape[1])
    index.add(vectors)
    return index


def search(
    index: faiss.IndexFlatIP, queries: np.ndarray, k: int = 10
) -> tuple[np.ndarray, np.ndarray]:
    """Return (scores, ids), both of shape (n_queries, k), best match first."""
    return index.search(_as_unit_float32(queries.copy()), k)
