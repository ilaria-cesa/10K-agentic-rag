"""Helpers for the embedding step. Heavy libraries are passed in, never imported here,
so these functions can be tested locally without a GPU or a model download."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np


def load_chunks(path: str | Path) -> list[dict]:
    with open(path, encoding="utf-8") as f:
        return [json.loads(line) for line in f]


def token_lengths(texts: list[str], tokenizer, batch_size: int = 1000) -> list[int]:
    """Real token count of each text (special tokens included, no truncation)."""
    lengths: list[int] = []
    for start in range(0, len(texts), batch_size):
        batch = texts[start : start + batch_size]
        encoded = tokenizer(batch, add_special_tokens=True, truncation=False)["input_ids"]
        lengths.extend(len(ids) for ids in encoded)
    return lengths


def length_report(lengths: list[int], limit: int) -> dict:
    """Summary of token lengths, and how many texts the model would silently truncate."""
    ordered = sorted(lengths)
    over = sum(n > limit for n in ordered)
    return {
        "n_texts": len(ordered),
        "limit": limit,
        "min": ordered[0],
        "median": ordered[len(ordered) // 2],
        "p95": ordered[int(len(ordered) * 0.95)],
        "max": ordered[-1],
        "n_over_limit": over,
        "share_over_limit": round(over / len(ordered), 4),
    }


def embed_texts(model, texts: list[str], batch_size: int = 32, progress: bool = True) -> np.ndarray:
    """Dense, L2-normalised embeddings as float32 (n_texts, dim)."""
    vectors = model.encode(
        texts,
        batch_size=batch_size,
        normalize_embeddings=True,
        convert_to_numpy=True,
        show_progress_bar=progress,
    )
    return np.asarray(vectors, dtype=np.float32)
