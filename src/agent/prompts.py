"""Prompt for answering questions from retrieved 10-K passages, with citations and refusal.

The refusal token is fixed (NOT_FOUND) so that refusals can be detected by code, not by guessing.
"""

from __future__ import annotations

NOT_FOUND = "NOT_FOUND"

SYSTEM_PROMPT = (
    "You answer questions about company annual reports (SEC Form 10-K) using ONLY the context "
    "passages provided by the user.\n"
    "Rules:\n"
    "1. Use only information that appears in the passages. Do not use outside knowledge.\n"
    "2. After every claim, cite the passage it comes from by copying its id exactly as it is "
    "written in square brackets at the start of that passage. Never invent or alter an id.\n"
    f"3. If the passages do not contain the answer, reply with exactly {NOT_FOUND} and nothing else.\n"
    "4. Answer in the language of the question. Be concise."
)

_SECTION_NAMES = {
    "business": "Item 1 Business",
    "risk_factors": "Item 1A Risk Factors",
    "mdna": "Item 7 MD&A",
}


def format_context(chunks: list[dict]) -> str:
    blocks = []
    for chunk in chunks:
        header = (
            f"{chunk['company']}, period ended {chunk['report_date']}, "
            f"{_SECTION_NAMES[chunk['section']]}"
        )
        blocks.append(f"[{chunk['chunk_id']}] {header}\n{chunk['text']}")
    return "\n\n".join(blocks)


def build_messages(question: str, chunks: list[dict]) -> list[dict]:
    user = f"Context passages:\n\n{format_context(chunks)}\n\nQuestion: {question}"
    return [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": user},
    ]
