from src.eval.browse import search
from src.eval.golden import validate

CHUNKS = [
    {
        "chunk_id": "AAA_2025-12-31_mdna_0000",
        "ticker": "AAA",
        "report_date": "2025-12-31",
        "section": "mdna",
        "text": "Catastrophe losses rose sharply in 2025.",
    },
    {
        "chunk_id": "AAA_2025-12-31_mdna_0001",
        "ticker": "AAA",
        "report_date": "2025-12-31",
        "section": "mdna",
        "text": "Net income grew.",
    },
    {
        "chunk_id": "BBB_2025-12-31_mdna_0000",
        "ticker": "BBB",
        "report_date": "2025-12-31",
        "section": "mdna",
        "text": "Catastrophe exposure is limited.",
    },
]


def good(**overrides):
    row = {
        "id": "q1",
        "question": "What happened to AAA catastrophe losses in 2025?",
        "language": "en",
        "type": "fact",
        "tickers": ["AAA"],
        "report_dates": ["2025-12-31"],
        "expected": "answer",
        "answer": "They rose sharply.",
        "evidence_chunk_ids": ["AAA_2025-12-31_mdna_0000"],
    }
    row.update(overrides)
    return row


def test_a_correct_row_has_no_errors():
    errors, warnings = validate([good()], CHUNKS)
    assert errors == []
    assert any("unanswerable" in w for w in warnings)  # 0% unanswerable is flagged


def test_unknown_chunk_wrong_company_and_missing_filing_are_errors():
    rows = [
        good(id="a", evidence_chunk_ids=["AAA_missing"]),
        good(id="b", evidence_chunk_ids=["BBB_2025-12-31_mdna_0000"]),
        good(id="c", report_dates=["2019-12-31"]),
    ]
    errors, _ = validate(rows, CHUNKS)
    assert any("does not exist" in e for e in errors)
    assert any("not listed in tickers" in e for e in errors)
    assert any("no filing for AAA 2019-12-31" in e for e in errors)


def test_refusal_rules():
    refuse = good(id="r", type="unanswerable", expected="refuse", answer="", evidence_chunk_ids=[])
    assert validate([refuse], CHUNKS)[0] == []
    bad = good(
        id="r2",
        type="unanswerable",
        expected="refuse",
        answer="x",
        evidence_chunk_ids=["AAA_2025-12-31_mdna_0000"],
    )
    assert validate([bad], CHUNKS)[0]
    wrong_type = good(id="r3", type="fact", expected="refuse", answer="", evidence_chunk_ids=[])
    assert any("requires type" in e for e in validate([wrong_type], CHUNKS)[0])


def test_comparison_needs_two_evidence_chunks_and_ids_must_be_unique():
    errors, warnings = validate([good(id="x", type="comparison"), good(id="x")], CHUNKS)
    assert any("single chunk" in w for w in warnings)
    errors_mh, _ = validate([good(id="y", type="multi_hop")], CHUNKS)
    assert any("at least two" in e for e in errors_mh)
    assert any("duplicate id" in e for e in errors)


def test_missing_year_is_only_a_warning():
    errors, warnings = validate([good(question="What happened to AAA catastrophe losses?")], CHUNKS)
    assert errors == [] and any("no year" in w for w in warnings)


def test_browse_search_filters_and_ranks():
    hits = search(CHUNKS, "catastrophe losses", ticker="AAA")
    assert [h["chunk_id"] for h in hits] == ["AAA_2025-12-31_mdna_0000"]
    assert search(CHUNKS, "zzz nothing") == []


def test_export_readable_shows_chunk_ids_in_section_order():
    from src.eval.export_readable import render

    chunks = [
        {**CHUNKS[1], "company": "AAA INC", "position": 1, "section": "mdna"},
        {**CHUNKS[0], "company": "AAA INC", "position": 0, "section": "mdna"},
        {
            **CHUNKS[0],
            "company": "AAA INC",
            "chunk_id": "AAA_2025-12-31_business_0000",
            "position": 0,
            "section": "business",
        },
    ]
    text = render(chunks)
    assert text.index("ITEM 1 BUSINESS") < text.index("ITEM 7 MD&A")
    assert text.index("[AAA_2025-12-31_mdna_0000]") < text.index("[AAA_2025-12-31_mdna_0001]")


def test_several_companies_may_close_their_year_on_different_dates():
    chunks = [
        *CHUNKS,
        {**CHUNKS[2], "chunk_id": "BBB_2026-01-31_mdna_0000", "report_date": "2026-01-31"},
    ]
    row = good(
        tickers=["AAA", "BBB"],
        report_dates=["2025-12-31", "2026-01-31"],
        evidence_chunk_ids=["AAA_2025-12-31_mdna_0000", "BBB_2026-01-31_mdna_0000"],
        type="comparison",
    )
    assert validate([row], chunks)[0] == []
    missing = good(tickers=["AAA", "BBB"], report_dates=["2027-01-01"])
    assert any("no filing for AAA / BBB 2027-01-01" in e for e in validate([missing], chunks)[0])


def test_split_must_be_known():
    rows = [
        good(id="Q1", split="dev"),
        good(id="Q2", split="holdout"),
        good(id="Q3", split="test"),
    ]
    errors, _ = validate(rows, CHUNKS)
    assert any("Q2: split must be one of" in e for e in errors)
    assert not any(e.startswith("Q1") for e in errors)


def test_load_golden_can_filter_by_split(tmp_path):
    import json

    from src.eval.golden import load_golden

    path = tmp_path / "q.jsonl"
    path.write_text(
        "\n".join(json.dumps(good(id=i, split=s)) for i, s in [("Q1", "dev"), ("Q2", "test")])
    )
    assert [r["id"] for r in load_golden(path)] == ["Q1", "Q2"]
    assert [r["id"] for r in load_golden(path, "test")] == ["Q2"]
