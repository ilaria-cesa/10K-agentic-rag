from src.agent.parsing import parse_answer, strip_thinking
from src.agent.prompts import NOT_FOUND, build_messages

CHUNKS = [
    {
        "chunk_id": "TRV_2025-12-31_mdna_0073",
        "company": "TRAVELERS",
        "report_date": "2025-12-31",
        "section": "mdna",
        "text": "About 37% of catastrophe losses occur in Q2.",
    },
    {
        "chunk_id": "TRV_2024-12-31_mdna_0081",
        "company": "TRAVELERS",
        "report_date": "2024-12-31",
        "section": "mdna",
        "text": "About 38%.",
    },
]
IDS = [c["chunk_id"] for c in CHUNKS]


def test_prompt_contains_every_chunk_id_and_the_question():
    messages = build_messages("What share?", CHUNKS)
    assert messages[0]["role"] == "system" and NOT_FOUND in messages[0]["content"]
    user = messages[1]["content"]
    assert all(f"[{i}]" in user for i in IDS) and user.endswith("Question: What share?")


def test_strip_thinking_handles_closed_and_unclosed_blocks():
    assert strip_thinking("<think>hmm</think>\n\nAnswer") == ("Answer", True)
    assert strip_thinking("<think>still thinking and cut off") == ("", True)
    assert strip_thinking("plain") == ("plain", False)


def test_valid_and_invalid_citations_are_separated():
    raw = "It is 37% [TRV_2025-12-31_mdna_0073] and 40% [TRV_2023-12-31_mdna_0001]."
    parsed = parse_answer(raw, IDS)
    assert parsed["cited_ids"] == ["TRV_2025-12-31_mdna_0073", "TRV_2023-12-31_mdna_0001"]
    assert parsed["invalid_citations"] == ["TRV_2023-12-31_mdna_0001"]
    assert not parsed["refused"]


def test_refusal_is_only_the_exact_token():
    assert parse_answer("NOT_FOUND", IDS)["refused"]
    assert parse_answer("<think>x</think> NOT_FOUND.", IDS)["refused"]
    assert not parse_answer("The answer is NOT_FOUND in the text, but roughly 37%.", IDS)["refused"]
