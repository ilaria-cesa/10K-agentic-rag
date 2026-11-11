"""Cross-encoder reranker (BAAI/bge-reranker-v2-m3).

A bi-encoder (bge-m3) embeds question and passage separately, which is fast but coarse. A
cross-encoder reads question and passage together and scores the pair, which is slower but much
better at telling near-duplicate passages apart. We use it only on a short list of candidates.
Imports are lazy so that the package works without torch (CI).
"""

from __future__ import annotations

MODEL_NAME = "BAAI/bge-reranker-v2-m3"


class Reranker:
    def __init__(self, name: str = MODEL_NAME, max_length: int = 640, device: str = "cuda"):
        import torch
        from sentence_transformers import CrossEncoder

        self.model = CrossEncoder(
            name, max_length=max_length, device=device, model_kwargs={"dtype": torch.float16}
        )

    def score(self, question: str, passages: list[str], batch_size: int = 16) -> list[float]:
        pairs = [(question, passage) for passage in passages]
        return [
            float(s)
            for s in self.model.predict(pairs, batch_size=batch_size, show_progress_bar=False)
        ]
