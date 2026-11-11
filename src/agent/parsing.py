"""Turn raw model output into something we can score: clean answer, refusal flag, citations."""

from __future__ import annotations

import re

from src.agent.prompts import NOT_FOUND

_THINK_BLOCK = re.compile(r"<think>.*?</think>", re.DOTALL)
_THINK_OPEN = re.compile(r"<think>.*", re.DOTALL)  # thinking cut off before it was closed
_CITATION = re.compile(r"\[([A-Z0-9.\-]+_\d{4}-\d{2}-\d{2}_[a-z_]+_\d{4})\]")


def strip_thinking(text: str) -> tuple[str, bool]:
    """Remove <think>...</think> blocks. Returns (clean_text, thinking_was_present)."""
    had = "<think>" in text or "</think>" in text
    text = _THINK_BLOCK.sub("", text)
    text = _THINK_OPEN.sub("", text)
    return text.replace("</think>", "").strip(), had


def parse_answer(raw: str, provided_ids: list[str]) -> dict:
    answer, had_thinking = strip_thinking(raw)
    # A refusal is the fixed token on its own, not a longer answer that happens to contain it.
    refused = answer.strip().strip(".").upper() == NOT_FOUND
    cited = list(dict.fromkeys(_CITATION.findall(answer)))
    provided = set(provided_ids)
    return {
        "answer": answer,
        "refused": refused,
        "cited_ids": cited,
        "invalid_citations": [c for c in cited if c not in provided],
        "n_citations": len(cited),
        "had_thinking": had_thinking,
    }
