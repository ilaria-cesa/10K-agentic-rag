from src.eval.remap import remap_chunk, remap_questions

KEY = {"ticker": "AAPL", "report_date": "2026-09-26", "section": "mdna"}


def chunk(cid, text, **over):
    return {"chunk_id": cid, "text": text, **KEY, **over}


OLD = chunk("old_1", "iPhone 254,098 209,586 Mac 30,000 29,000")
NEW = [
    chunk("new_prose", "Net sales rose because of services and strong demand overall."),
    chunk(
        "new_table",
        "Table: Net sales:\niPhone: 2026 = $254,098; 2025 = $209,586\nMac: 2026 = 30,000",
    ),
    chunk("other_filing", "iPhone 254,098 209,586 Mac 30,000", report_date="2025-09-27"),
    chunk("other_section", "iPhone 254,098 209,586 Mac 30,000", section="business"),
]


def test_numbers_link_a_table_row_to_its_structured_form():
    ids, best = remap_chunk(OLD, NEW)
    assert ids == ["new_table"] and best > 0.5


def test_best_chunk_is_used_when_nothing_reaches_the_threshold():
    ids, best = remap_chunk(OLD, [NEW[0], NEW[1]], min_cover=0.99)
    assert ids == ["new_table"] and best < 0.99


def test_questions_are_remapped_and_weak_ones_listed():
    questions = [
        {"id": "Q1", "evidence_chunk_ids": ["old_1"]},
        {"id": "Q2", "evidence_chunk_ids": []},
    ]
    mapped, weak = remap_questions(questions, [OLD], NEW)
    assert (
        mapped[0]["evidence_chunk_ids"] == ["new_table"] and mapped[1]["evidence_chunk_ids"] == []
    )
    assert weak == []
    _, weak = remap_questions(questions, [OLD], [NEW[0]])
    assert weak == ["Q1"]
